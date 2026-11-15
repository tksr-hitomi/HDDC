"""Fuse DDM-guided Mamba and GDFN degradation experts."""

from typing import Mapping
import torch
from torch import nn
from einops import rearrange
from timm.models.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba as MambaSSM
from .dcw import DEGRADATION_KEYS, DegradationCompositionRouter


class GDFN(nn.Module):
    def __init__(
        self, dim: int, ffn_expansion_factor: float = 2.0, bias: bool = False
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        if hidden_dim % 2 != 0:
            raise ValueError("The expanded GDFN dimension must be even.")
        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_dim // 2,
            hidden_dim // 2,
            kernel_size=3,
            padding=1,
            groups=hidden_dim // 2,
            bias=bias,
        )
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_dim // 2, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(self.project_in(x), chunks=2, dim=1)
        return self.project_out(self.act(self.dwconv(x1)) * x2)


class MambaBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        ssm_cfg: dict | None = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(channels * patch_size * patch_size, embed_dim)
        rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    MambaSSM(d_model=embed_dim, **ssm_cfg or {}),
                    DropPath(rates[index]) if rates[index] > 0.0 else nn.Identity(),
                )
                for index in range(depth)
            ]
        )
        self.output_projection = nn.Linear(
            embed_dim, channels * patch_size * patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        patch = self.patch_size
        if height % patch != 0 or width % patch != 0:
            raise ValueError(
                f"Feature size {(height, width)} must be divisible by patch size {patch}."
            )
        patches = (
            x.unfold(2, patch, patch)
            .unfold(3, patch, patch)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        hidden = self.input_projection(
            patches.view(batch, -1, channels * patch * patch)
        )
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        sequence = self.output_projection(hidden)
        return rearrange(
            sequence,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=height // patch,
            w=width // patch,
            p1=patch,
            p2=patch,
        )


class Mamba(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 4,
        ssm_cfg: dict | None = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if depth % 2 != 0:
            raise ValueError("Mamba depth must be even for regular/shifted pairs.")
        self.depth = depth
        self.patch_size = patch_size
        self.sft = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels * 2, kernel_size=3, padding=1),
        )
        self.mamba_blocks = nn.ModuleList(
            [
                MambaBlock(
                    channels=channels,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    depth=1,
                    ssm_cfg=ssm_cfg,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(depth)
            ]
        )
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x_in: torch.Tensor, prior_map: torch.Tensor) -> torch.Tensor:
        residual = x_in
        gamma, beta = torch.chunk(self.sft(prior_map), chunks=2, dim=1)
        x = x_in * (1.0 + gamma) + beta
        shift = self.patch_size // 2
        for pair_index in range(self.depth // 2):
            regular = self.mamba_blocks[pair_index * 2]
            shifted = self.mamba_blocks[pair_index * 2 + 1]
            x = x + regular(x)
            shifted_x = torch.roll(x, shifts=(-shift, -shift), dims=(2, 3))
            shifted_x = shifted(shifted_x)
            x = x + torch.roll(shifted_x, shifts=(shift, shift), dims=(2, 3))
        return residual + self.output_projection(x)


class DegradationExpert(nn.Module):
    def __init__(
        self,
        channels: int,
        mamba_config: dict,
        num_gdfn_blocks: int = 2,
        gdfn_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.mamba = Mamba(channels=channels, **mamba_config)
        self.gdfn_blocks = nn.ModuleList(
            [GDFN(dim=channels, **gdfn_config or {}) for _ in range(num_gdfn_blocks)]
        )
        self.drop_path = nn.Identity()

    def forward(self, x_in: torch.Tensor, prior_map: torch.Tensor) -> torch.Tensor:
        residual = x_in
        refined = self.mamba(x_in, prior_map)
        for block in self.gdfn_blocks:
            refined = refined + self.drop_path(block(refined))
        return residual + refined


class HDDCBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expert_config: dict,
        num_experts: int = 4,
        router_hidden_factor: float = 0.5,
    ) -> None:
        super().__init__()
        if num_experts != len(DEGRADATION_KEYS):
            raise ValueError(f"HDDC requires exactly {len(DEGRADATION_KEYS)} experts.")
        self.num_experts = num_experts
        self.router = DegradationCompositionRouter(
            visual_channels=channels,
            num_experts=num_experts,
            mlp_hidden_factor=router_hidden_factor,
        )
        self.experts = nn.ModuleList(
            [
                DegradationExpert(channels=channels, **expert_config)
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        clip_prior_logits: torch.Tensor,
        quality_maps: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        weights = self.router(features, clip_prior_logits)
        outputs = [
            expert(features, quality_maps[key])
            for expert, key in zip(self.experts, DEGRADATION_KEYS)
        ]
        fused = torch.zeros_like(features)
        for index, output in enumerate(outputs):
            fused = fused + output * weights[:, index : index + 1]
        return fused
