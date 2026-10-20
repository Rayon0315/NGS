from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, _, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, height, width)
        return self.dwconv(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(_init_module)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, height, width))
        x = self.drop(x)
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        ca_num_heads: int,
        sa_num_heads: int,
        qkv_bias: bool,
        ca_attention: int,
        expand_ratio: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.ca_attention = int(ca_attention)
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.ca_attention == 1:
            if dim % ca_num_heads != 0:
                raise ValueError("dim must be divisible by ca_num_heads")
            self.split_groups = dim // ca_num_heads
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(ca_num_heads):
                k = 3 + 2 * i
                setattr(
                    self,
                    f"local_conv_{i+1}",
                    nn.Conv2d(
                        self.split_groups,
                        self.split_groups,
                        k,
                        padding=k // 2,
                        groups=self.split_groups,
                    ),
                )
            self.proj0 = nn.Conv2d(dim, dim * expand_ratio, 1, groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim * expand_ratio)
            self.proj1 = nn.Conv2d(dim * expand_ratio, dim, 1)
        else:
            if dim % sa_num_heads != 0:
                raise ValueError("dim must be divisible by sa_num_heads")
            self.scale = (dim // sa_num_heads) ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.local_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.apply(_init_module)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, n, c = x.shape
        if self.ca_attention == 1:
            value = self.v(x)
            spatial = (
                self.s(x)
                .reshape(b, height, width, self.ca_num_heads, c // self.ca_num_heads)
                .permute(3, 0, 4, 1, 2)
            )
            outs = []
            for i in range(self.ca_num_heads):
                conv = getattr(self, f"local_conv_{i+1}")
                outs.append(conv(spatial[i]).reshape(b, self.split_groups, -1, height, width))
            spatial_out = torch.cat(outs, dim=2).reshape(b, c, height, width)
            spatial_out = self.proj1(self.act(self.bn(self.proj0(spatial_out))))
            spatial_out = spatial_out.reshape(b, c, n).permute(0, 2, 1)
            x = spatial_out * value
        else:
            q = (
                self.q(x)
                .reshape(b, n, self.sa_num_heads, c // self.sa_num_heads)
                .permute(0, 2, 1, 3)
            )
            kv = (
                self.kv(x)
                .reshape(b, n, 2, self.sa_num_heads, c // self.sa_num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = kv[0], kv[1]
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(b, n, c)
            local_value = (
                v.transpose(1, 2)
                .reshape(b, n, c)
                .transpose(1, 2)
                .reshape(b, c, height, width)
            )
            local_value = self.local_conv(local_value).reshape(b, c, n).transpose(1, 2)
            x = x + local_value
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        ca_num_heads: int,
        sa_num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ca_attention: int,
        expand_ratio: int,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            ca_num_heads,
            sa_num_heads,
            qkv_bias,
            ca_attention,
            expand_ratio,
            attn_drop,
            drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        self.apply(_init_module)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), height, width))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 3, 2, 1)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(_init_module)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, h, w


class Head(nn.Module):
    def __init__(self, dim: int, head_conv: int = 3) -> None:
        super().__init__()
        padding = 3 if head_conv == 7 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim, head_conv, 2, padding, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 2, 2),
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(_init_module)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        _, _, h, w = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class SMTTinyBackbone(nn.Module):
    out_channels = (64, 128, 256, 512)
    out_strides = (4, 8, 16, 32)

    def __init__(
        self,
        pretrained_path: str | Path | None = None,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        embed_dims = [64, 128, 256, 512]
        ca_heads = [4, 4, 4, -1]
        sa_heads = [-1, -1, 8, 16]
        mlp_ratios = [4, 4, 4, 2]
        depths = [2, 2, 8, 1]
        ca_attentions = [1, 1, 1, 0]
        dpr = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        current = 0
        for s in range(4):
            patch = (
                Head(embed_dims[s], 3)
                if s == 0
                else OverlapPatchEmbed(embed_dims[s - 1], embed_dims[s])
            )
            blocks = nn.ModuleList()
            for i in range(depths[s]):
                ca = 0 if (s == 2 and i % 2 == 1) else ca_attentions[s]
                blocks.append(
                    Block(
                        embed_dims[s],
                        ca_heads[s],
                        sa_heads[s],
                        mlp_ratios[s],
                        True,
                        ca,
                        2,
                        0.0,
                        0.0,
                        dpr[current + i],
                    )
                )
            current += depths[s]
            setattr(self, f"patch_embed{s+1}", patch)
            setattr(self, f"block{s+1}", blocks)
            setattr(self, f"norm{s+1}", nn.LayerNorm(embed_dims[s], eps=1e-6))
        self.apply(_init_module)
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, pretrained_path: str | Path) -> None:
        p = Path(pretrained_path)
        if not p.is_file():
            raise FileNotFoundError(f"SMT-T pretrained checkpoint not found: {p}")
        ckpt: Any = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if not isinstance(sd, dict):
            raise TypeError("Unsupported SMT checkpoint format")
        norm = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("head."):
                continue
            norm[k] = v
        incompatible = self.load_state_dict(norm, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"SMT-T checkpoint mapping failed. Missing={incompatible.missing_keys}; "
                f"Unexpected={incompatible.unexpected_keys}"
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b = x.shape[0]
        outputs = []
        for s in range(4):
            tokens, h, w = getattr(self, f"patch_embed{s+1}")(x)
            for block in getattr(self, f"block{s+1}"):
                tokens = block(tokens, h, w)
            tokens = getattr(self, f"norm{s+1}")(tokens)
            feat = tokens.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(feat)
            x = feat
        return tuple(outputs)


def smt_tiny(
    pretrained_path: str | Path | None = None, **kwargs: Any
) -> SMTTinyBackbone:
    return SMTTinyBackbone(pretrained_path=pretrained_path, **kwargs)


smt_t = smt_tiny