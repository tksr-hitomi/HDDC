"""Build the multiscale HDDC enhancement network."""

from typing import Mapping
import torch
import torch.nn.functional as functional
from torch import nn
from PIL import Image
from .dcw import DEGRADATION_KEYS, RankingPromptEncoder
from .block import HDDCBlock


class ImagePyramidBlock(nn.Module):
    def __init__(
        self, in_channels: int, mid1: int, mid2: int, out_channels: int
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid1, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid1, mid1, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid1, mid2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid2, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HDDC(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        base_channels: int = 32,
        router_hidden_factor: float = 0.5,
        clip_encoder: RankingPromptEncoder | object | None = None,
        prompt_checkpoint: str | None = None,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        clip_device: str = "cpu",
        clip_cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        if clip_encoder is None:
            if prompt_checkpoint is None:
                raise ValueError(
                    "prompt_checkpoint is required when clip_encoder is not provided."
                )
            clip_encoder = RankingPromptEncoder(
                prompt_checkpoint=prompt_checkpoint,
                model_name=clip_model_name,
                pretrained=clip_pretrained,
                device=clip_device,
                cache_dir=clip_cache_dir,
            )
        self.clip_encoder = clip_encoder
        middle_channels = max(input_channels, base_channels // 2)
        self.image_stem = nn.Sequential(
            nn.Conv2d(
                input_channels, middle_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                middle_channels, base_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        channels1 = base_channels
        channels2 = channels1 * 2
        channels3 = channels2 * 2
        stage1 = {
            "mamba_config": {"patch_size": 8, "embed_dim": 192, "depth": 2},
            "num_gdfn_blocks": 2,
        }
        stage2 = {
            "mamba_config": {"patch_size": 4, "embed_dim": 256, "depth": 2},
            "num_gdfn_blocks": 2,
        }
        stage3 = {
            "mamba_config": {"patch_size": 2, "embed_dim": 384, "depth": 4},
            "num_gdfn_blocks": 2,
        }
        self.encoder_stage1 = HDDCBlock(
            channels1, stage1, router_hidden_factor=router_hidden_factor
        )
        self.downsample1 = nn.Conv2d(
            channels1, channels2, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.down_activation1 = nn.ReLU(inplace=True)
        self.half_image_features = ImagePyramidBlock(
            input_channels, channels1, channels2, channels2
        )
        self.half_fusion = nn.Sequential(
            nn.Conv2d(
                channels2 + channels2, channels2, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.encoder_stage2 = HDDCBlock(
            channels2, stage2, router_hidden_factor=router_hidden_factor
        )
        self.downsample2 = nn.Conv2d(
            channels2, channels3, kernel_size=3, stride=2, padding=1, bias=True
        )
        self.down_activation2 = nn.ReLU(inplace=True)
        self.quarter_image_features = ImagePyramidBlock(
            input_channels, channels2, channels3, channels3
        )
        self.quarter_fusion = nn.Sequential(
            nn.Conv2d(
                channels3 + channels3, channels3, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = HDDCBlock(
            channels3, stage3, router_hidden_factor=router_hidden_factor
        )
        self.upsample2 = nn.ConvTranspose2d(
            channels3, channels2, kernel_size=2, stride=2, bias=True
        )
        self.decoder_fusion2 = nn.Sequential(
            nn.Conv2d(
                channels2 + channels2, channels2, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder_stage2 = HDDCBlock(
            channels2, stage2, router_hidden_factor=router_hidden_factor
        )
        self.upsample1 = nn.ConvTranspose2d(
            channels2, channels1, kernel_size=2, stride=2, bias=True
        )
        self.decoder_fusion1 = nn.Sequential(
            nn.Conv2d(
                channels1 + channels1, channels1, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
        )
        self.decoder_stage1 = HDDCBlock(
            channels1, stage1, router_hidden_factor=router_hidden_factor
        )
        self.output_head = nn.Conv2d(
            channels1, output_channels, kernel_size=3, padding=1, bias=True
        )

    @staticmethod
    def _resize_maps(
        maps: Mapping[str, torch.Tensor], size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        return {
            key: functional.interpolate(
                value, size=size, mode="bilinear", align_corners=False
            )
            for key, value in maps.items()
        }

    def forward(
        self,
        raw_image: torch.Tensor,
        clip_images: Image.Image | list[Image.Image],
        quality_maps: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        height, width = raw_image.shape[-2:]
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"Input size {(height, width)} must be divisible by 8.")
        missing = [key for key in DEGRADATION_KEYS if key not in quality_maps]
        if missing:
            raise KeyError(f"Missing quality maps: {missing}")
        with torch.no_grad():
            clip_logits = self.clip_encoder.get_logits(clip_images).to(raw_image.device)
            if clip_logits.shape[0] != raw_image.shape[0]:
                if clip_logits.shape[0] == 1 and raw_image.shape[0] > 1:
                    clip_logits = clip_logits.repeat(raw_image.shape[0], 1)
                else:
                    raise ValueError(
                        f"Batch mismatch: images={raw_image.shape[0]}, CLIP={clip_logits.shape[0]}"
                    )
        maps1 = self._resize_maps(quality_maps, (height, width))
        maps2 = self._resize_maps(quality_maps, (height // 2, width // 2))
        maps3 = self._resize_maps(quality_maps, (height // 4, width // 4))
        raw_half = functional.interpolate(
            raw_image,
            size=(height // 2, width // 2),
            mode="bilinear",
            align_corners=False,
        )
        raw_quarter = functional.interpolate(
            raw_image,
            size=(height // 4, width // 4),
            mode="bilinear",
            align_corners=False,
        )
        features1 = self.image_stem(raw_image)
        skip1 = self.encoder_stage1(features1, clip_logits, maps1)
        down1 = self.down_activation1(self.downsample1(skip1))
        features2 = self.half_fusion(
            torch.cat([down1, self.half_image_features(raw_half)], dim=1)
        )
        skip2 = self.encoder_stage2(features2, clip_logits, maps2)
        down2 = self.down_activation2(self.downsample2(skip2))
        features3 = self.quarter_fusion(
            torch.cat([down2, self.quarter_image_features(raw_quarter)], dim=1)
        )
        bottleneck = self.bottleneck(features3, clip_logits, maps3)
        up2 = self.upsample2(bottleneck)
        decoded2 = self.decoder_fusion2(torch.cat([up2, skip2], dim=1))
        decoded2 = self.decoder_stage2(decoded2, clip_logits, maps2)
        up1 = self.upsample1(decoded2)
        decoded1 = self.decoder_fusion1(torch.cat([up1, skip1], dim=1))
        decoded1 = self.decoder_stage1(decoded1, clip_logits, maps1)
        output = self.output_head(decoded1)
        if output.shape == raw_image.shape:
            output = output + raw_image
        return output
