"""Learn the four CLIP prompt embedding sequences."""

from contextlib import contextmanager
from typing import Iterator
import torch
from torch import nn

PROMPTS = (
    "A photo of a green-tinted underwater scene",
    "A photo of a blue-tinted underwater scene",
    "A photo of a hazy underwater scene",
    "A photo of a low-light underwater scene",
)


class LearnablePrompts(nn.Module):
    def __init__(self, clip_model: nn.Module, tokenizer) -> None:
        super().__init__()
        tokenized = tokenizer(list(PROMPTS))
        self.register_buffer("tokenized_prompts_buffer", tokenized)
        with torch.no_grad():
            device = clip_model.token_embedding.weight.device
            embeddings = (
                clip_model.token_embedding(tokenized.to(device)).detach().clone()
            )
        self.learnable_embeddings = nn.Parameter(embeddings)


@contextmanager
def patched_token_embedding(
    clip_model: nn.Module, learnable_embeddings: torch.Tensor
) -> Iterator[None]:
    original_forward = clip_model.token_embedding.forward

    def embedding_forward(_token_ids: torch.Tensor) -> torch.Tensor:
        return learnable_embeddings

    try:
        clip_model.token_embedding.forward = embedding_forward
        yield
    finally:
        clip_model.token_embedding.forward = original_forward
