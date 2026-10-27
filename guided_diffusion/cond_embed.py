import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CondEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.parents = args.parents  # list of conditioning keys
        self.parent_dims = args.parent_dims  # dict: key -> input dim (e.g., {'sex': 1, 'race': 3})
        self.cond_embed_dim = args.cond_embed_dim  # final output embedding dim

        assert self.cond_embed_dim % len(self.parents) == 0, (
            f"cond_embed_dim ({self.cond_embed_dim}) must be divisible by number of parents ({len(self.parents)})"
        )
        # Individual embedding layers per parent
        # Per-parent embedding dim
        self.parent_embed_dim = self.cond_embed_dim // len(self.parents)

        self.embeddings = nn.ModuleDict()
        for key in self.parents:
            in_dim = self.parent_dims[key]
            self.embeddings[key] = nn.Sequential(
                nn.Linear(in_dim, self.parent_embed_dim),
                nn.SiLU(),
                nn.Linear(self.parent_embed_dim, self.parent_embed_dim)
            )

    def forward(self, y_dict):
        # Get individual embeddings
        embs = []
        for key in self.parents:
            assert key in y_dict, f"Missing key in y_dict: {key}"
            val = y_dict[key]  # shape [B, in_dim]
            emb = self.embeddings[key](val)  # shape [B, embed_dim]
            embs.append(emb)

        # Concatenate and project to final cond embedding
        out = torch.cat(embs, dim=1)  # shape [B, embed_dim * num_parents]


        out = F.normalize(out, p=2, dim=1)

        # Normalize each sample to have mean 0, std 1
        # mean = out.mean(dim=1, keepdim=True)
        # std = out.std(dim=1, keepdim=True) + 1e-6
        # out = (out - mean) / std
        return out



class GlobalCondEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.parents = args.parents  # list of keys for conditioning input
        self.parent_dims = args.parent_dims  # dict: key -> input dim (e.g., {'sex': 1, 'race': 3})
        self.cond_embed_dim = args.cond_embed_dim  # final output embedding dim

        # Total input dim is sum of all parent input dims
        self.input_dim = sum([self.parent_dims[k] for k in self.parents])

        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, self.cond_embed_dim),
            nn.SiLU(),
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim),
        )

    def forward(self, y_dict):
        # Concatenate all parent values into a single [B, input_dim] tensor
        vals = []
        for key in self.parents:
            assert key in y_dict, f"Missing key in y_dict: {key}"
            vals.append(y_dict[key])  # shape [B, input_dim_k]
        concat_y = torch.cat(vals, dim=1)  # shape [B, total_input_dim]

        # Apply global embedding MLP
        out = self.embed(concat_y)
        out = F.normalize(out, p=2, dim=1)
        return out