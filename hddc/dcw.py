"""Compute global CLIP guidance and layer-wise expert weights."""

from pathlib import Path
import open_clip
import torch
from torch import nn
from PIL import Image
from prompt_tuning.model import LearnablePrompts, patched_token_embedding

DEGRADATION_KEYS = ("green_cast", "blue_cast", "haziness", "illumination")


class RankingPromptEncoder:
    def __init__(
        self,
        prompt_checkpoint: str | Path,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | torch.device = "cpu",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model, _, self.preprocess_image = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        self.prompt_learner = LearnablePrompts(self.model, tokenizer).to(self.device)
        checkpoint_path = Path(prompt_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Ranking-prompt checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        self.prompt_learner.load_state_dict(state_dict, strict=True)
        self.prompt_learner.eval()

    @torch.no_grad()
    def get_logits(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        if (
            not isinstance(images, list)
            or not images
            or (not all((isinstance(image, Image.Image) for image in images)))
        ):
            raise TypeError(
                "CLIP input must be a PIL image or a non-empty list of PIL images."
            )
        image_batch = torch.stack(
            [self.preprocess_image(image) for image in images]
        ).to(self.device)
        image_features = self.model.encode_image(image_batch, normalize=True)
        with patched_token_embedding(
            self.model, self.prompt_learner.learnable_embeddings
        ):
            text_features = self.model.encode_text(
                self.prompt_learner.tokenized_prompts_buffer.to(self.device),
                normalize=True,
            )
        return image_features @ text_features.T


class DegradationCompositionRouter(nn.Module):
    def __init__(
        self, visual_channels: int, num_experts: int = 4, mlp_hidden_factor: float = 0.5
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        hidden_channels = max(1, int(visual_channels * mlp_hidden_factor))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.visual_projection = nn.Sequential(
            nn.Conv2d(visual_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, visual_features: torch.Tensor, prior_logits: torch.Tensor
    ) -> torch.Tensor:
        visual_logits = self.visual_projection(self.pool(visual_features))
        prior_logits = (prior_logits / 0.1).unsqueeze(-1).unsqueeze(-1)
        return self.softmax(prior_logits + visual_logits * 0.8)
