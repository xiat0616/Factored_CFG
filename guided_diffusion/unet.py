import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from nn_utils import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
    EmbedSequential,
    ResBlock,
    AttentionBlock,
    Upsample,
    Downsample,
)


class UNetModel(nn.Module):
    """
    UNetModel for diffusion autoencoder.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # === Config ===
        self.image_size = args.image_size
        self.in_channels = args.in_channels
        self.model_channels = args.model_channels
        self.out_channels = args.out_channels
        self.num_res_blocks = args.num_res_blocks
        self.attention_resolutions = tuple(
            self.image_size // int(res) for res in args.attention_resolutions.split(",")
        )
        self.dropout = args.dropout
        self.channel_mult = args.channel_mult
        self.conv_resample = args.conv_resample
        self.use_checkpoint = args.use_checkpoint
        self.dtype = th.float32
        self.num_heads = args.num_heads
        self.num_head_channels = args.num_head_channels
        self.num_heads_upsample = args.num_heads_upsample
        self.dims = args.dims
        self.resblock_updown = args.resblock_updown
        self.use_new_attention_order = args.use_new_attention_order

        self.cfg_train = args.cfg_train
        self.cfg_dropout = args.cfg_dropout

        # === Embeddings ===
        self.embedding_mode = getattr(args, "embedding_mode", None)

        self.time_embed_dim = getattr(args, "time_embed_dim", self.model_channels * 4)
        self.cond_embed_dim = getattr(args, "cond_embed_dim", None)
        if self.cond_embed_dim is not None and self.cond_embed_dim <= 0:
            self.cond_embed_dim = None
        self.sem_embed_dim = getattr(args, "sem_embed_dim", None)
        if self.sem_embed_dim is not None and self.sem_embed_dim <= 0:
            self.sem_embed_dim = None

        # === Time Embed MLP ===
        self.time_embed = nn.Sequential(
            linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        # === Input blocks ===
        input_ch = int(self.channel_mult[0] * self.model_channels)
        ch = input_ch
        self.input_blocks = nn.ModuleList([
            EmbedSequential(conv_nd(self.dims, self.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        dropout=self.dropout,
                        tdim=self.time_embed_dim,
                        cdim=self.cond_embed_dim,
                        sdim=self.sem_embed_dim,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        embedding_mode=self.embedding_mode,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(
                        ch,
                        use_checkpoint=self.use_checkpoint,
                        num_heads=self.num_heads,
                        num_head_channels=self.num_head_channels,
                        use_new_attention_order=self.use_new_attention_order,
                    ))
                self.input_blocks.append(EmbedSequential(*layers))
                input_block_chans.append(ch)
                self._feature_size += ch

            if level != len(self.channel_mult) - 1:
                out_ch = ch
                if self.resblock_updown:
                    down = ResBlock(
                        channels=ch,
                        dropout=self.dropout,
                        tdim=self.time_embed_dim,
                        cdim=self.cond_embed_dim,
                        sdim=self.sem_embed_dim,
                        out_channels=out_ch,
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        down=True,
                        embedding_mode=self.embedding_mode,
                    )
                else:
                    down = Downsample(
                        ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                    )
                self.input_blocks.append(EmbedSequential(down))
                input_block_chans.append(out_ch)
                ch = out_ch
                ds *= 2
                self._feature_size += ch

        # === Middle block ===
        self.middle_block = EmbedSequential(
            ResBlock(
                channels=ch,
                dropout=self.dropout,
                tdim=self.time_embed_dim,
                cdim=self.cond_embed_dim,
                sdim=self.sem_embed_dim,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                embedding_mode=self.embedding_mode,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            ResBlock(
                channels=ch,
                dropout=self.dropout,
                tdim=self.time_embed_dim,
                cdim=self.cond_embed_dim,
                sdim=self.sem_embed_dim,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                embedding_mode=self.embedding_mode,
            ),
        )
        self._feature_size += ch

        # === Output blocks ===
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        dropout=self.dropout,
                        tdim=self.time_embed_dim,
                        cdim=self.cond_embed_dim,
                        sdim=self.sem_embed_dim,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        embedding_mode=self.embedding_mode,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(
                        ch,
                        use_checkpoint=self.use_checkpoint,
                        num_heads=self.num_heads_upsample if self.num_heads_upsample != -1 else self.num_heads,
                        num_head_channels=self.num_head_channels,
                        use_new_attention_order=self.use_new_attention_order,
                    ))
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    if self.resblock_updown:
                        layers.append(
                            ResBlock(
                                channels=ch,
                                dropout=self.dropout,
                                tdim=self.time_embed_dim,
                                cdim=self.cond_embed_dim,
                                sdim=self.sem_embed_dim,
                                out_channels=out_ch,
                                dims=self.dims,
                                use_checkpoint=self.use_checkpoint,
                                up=True,
                                embedding_mode=self.embedding_mode,
                            )
                        )
                    else:
                        layers.append(Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        # === Final output ===
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, cemb=None, semb=None):
        """
        Apply the model to an input batch.

        :param x: [N x C x ...] Input tensor
        :param timesteps: 1-D batch of diffusion steps
        :param cemb: conditional embedding
        :param semb: semantic embedding
        :return: [N x C x ...] Output tensor
        """
        hs = []
        temb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, temb=temb, cemb=cemb, semb=semb)
            hs.append(h)

        h = self.middle_block(h, temb=temb, cemb=cemb, semb=semb)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, temb=temb, cemb=cemb, semb=semb)

        h = h.type(x.dtype)
        return self.out(h)
