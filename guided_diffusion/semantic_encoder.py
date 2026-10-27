import torch
import torch.nn as nn
from nn_utils import (
    EmbedSequential,
    conv_nd,
    ResBlock,
    AttentionBlock,
    Downsample,
)
import torch.nn.functional as F

class SemanticEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = args.image_size
        self.in_channels = args.in_channels
        self.model_channels = args.model_channels
        self.dropout = args.dropout
        self.enc_channel_mult = args.enc_channel_mult
        self.dims = args.dims
        self.use_checkpoint = args.use_checkpoint
        self.resblock_updown = args.resblock_updown
        self.conv_resample = args.conv_resample
        self.embedding_mode = "none"
        self.attention_resolutions = tuple(
            self.image_size // int(res) for res in args.attention_resolutions.split(",")
        )
        self.use_new_attention_order = args.use_new_attention_order
        self.dtype = torch.float32
        self.sem_embed_dim = getattr(self.args, "sem_embed_dim", self.model_channels * 4)
        assert self.embedding_mode == "none", "Semantic encoders do not use embedding modes other than 'none'."
        ch = int(self.enc_channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList([
            EmbedSequential(conv_nd(self.dims, self.in_channels, ch, 3, padding=1))
        ])
        ds = 1

        for level, mult in enumerate(self.enc_channel_mult):
            for _ in range(self.args.num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        dropout=self.dropout,
                        tdim=None,
                        cdim=None,
                        sdim=None,
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
                        num_heads=self.args.num_heads,
                        num_head_channels=self.args.num_head_channels,
                        use_new_attention_order=self.use_new_attention_order,
                    ))
                self.input_blocks.append(EmbedSequential(*layers))
            if level != len(self.enc_channel_mult) - 1:
                out_ch = ch
                downsample_layer = (
                    ResBlock(
                        channels=ch,
                        dropout=self.dropout,
                        tdim=None,
                        cdim=None,
                        sdim=None,
                        out_channels=out_ch,
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        down=True,
                        embedding_mode=self.embedding_mode,
                    )
                    if self.resblock_updown
                    else Downsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                )
                self.input_blocks.append(EmbedSequential(downsample_layer))
                ch = out_ch
                ds *= 2

        self.middle_block = EmbedSequential(
            ResBlock(
                channels=ch,
                dropout=self.dropout,
                tdim=None,
                cdim=None,
                sdim=None,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                embedding_mode=self.embedding_mode,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.args.num_heads,
                num_head_channels=self.args.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            ResBlock(
                channels=ch,
                dropout=self.dropout,
                tdim=None,
                cdim=None,
                sdim=None,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                embedding_mode=self.embedding_mode,
            ),
        )

        self.flatten_block = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(ch, self.sem_embed_dim, 1, 1, 0),
            nn.Flatten(),
        )

    def forward(self, x):
        out = x.type(self.dtype)
        for module in self.input_blocks:
            out = module(out, temb=None, cemb=None, semb=None)
        out = self.middle_block(out, temb=None, cemb=None, semb=None)
        out = self.flatten_block(out)

        out = F.normalize(out, p=2, dim=1)

        # # Normalize each sample to have mean 0, std 1
        # mean = out.mean(dim=1, keepdim=True)
        # std = out.std(dim=1, keepdim=True) + 1e-6
        # out = (out - mean) / std
        return out

class VAESemanticEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = args.image_size
        self.in_channels = args.in_channels
        self.model_channels = args.model_channels
        self.dropout = args.dropout
        self.enc_channel_mult = args.enc_channel_mult
        self.dims = args.dims
        self.use_checkpoint = args.use_checkpoint
        self.resblock_updown = args.resblock_updown
        self.conv_resample = args.conv_resample
        self.embedding_mode = "none"
        self.attention_resolutions = tuple(
            self.image_size // int(res) for res in args.attention_resolutions.split(",")
        )
        self.use_new_attention_order = args.use_new_attention_order
        self.dtype = torch.float32
        self.sem_embed_dim = getattr(args, "sem_embed_dim", self.model_channels * 4)
        assert self.embedding_mode == "none", "Semantic encoders do not use embedding modes other than 'none'."
        ch = int(self.enc_channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList([
            EmbedSequential(conv_nd(self.dims, self.in_channels, ch, 3, padding=1))
        ])
        ds = 1

        for level, mult in enumerate(self.enc_channel_mult):
            for _ in range(self.args.num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        dropout=self.dropout,
                        tdim=None,
                        cdim=None,
                        sdim=None,
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
                        num_heads=self.args.num_heads,
                        num_head_channels=self.args.num_head_channels,
                        use_new_attention_order=self.use_new_attention_order,
                    ))
                self.input_blocks.append(EmbedSequential(*layers))

            if level != len(self.enc_channel_mult) - 1:
                out_ch = ch
                down = (
                    ResBlock(
                        channels=ch,
                        dropout=self.dropout,
                        tdim=None,
                        cdim=None,
                        sdim=None,
                        out_channels=out_ch,
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        down=True,
                        embedding_mode=self.embedding_mode,
                    )
                    if self.resblock_updown
                    else Downsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                )
                self.input_blocks.append(EmbedSequential(down))
                ch = out_ch
                ds *= 2

        self.middle_block = EmbedSequential(
            ResBlock(
                channels=ch,
                dropout=self.dropout,
                tdim=None,
                cdim=None,
                sdim=None,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                embedding_mode=self.embedding_mode,
            ),
            AttentionBlock(
                channels=ch,
                use_checkpoint=self.use_checkpoint,
                num_heads=self.args.num_heads,
                num_head_channels=self.args.num_head_channels,
                use_new_attention_order=self.use_new_attention_order,
            ),
            ResBlock(
                channels=ch,
                dropout=self.dropout,
                tdim=None,
                cdim=None,
                sdim=None,
                dims=self.dims,
                use_checkpoint=self.use_checkpoint,
                embedding_mode=self.embedding_mode,
            ),
        )

        self.shared = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(ch, self.sem_embed_dim)
        self.fc_logvar = nn.Linear(ch, self.sem_embed_dim)

    def forward(self, x):
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, temb=None, cemb=None, semb=None)
        h = self.middle_block(h, temb=None, cemb=None, semb=None)
        h = self.shared(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
