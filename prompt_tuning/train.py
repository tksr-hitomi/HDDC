"""Train CLIP prompt embeddings with ranking supervision."""

from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
import open_clip
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from hddc.utils import atomic_torch_save, load_yaml, resolve_path, value
from prompt_tuning.model import LearnablePrompts, patched_token_embedding
from prompt_tuning.data import RankingAnnotationDataset
from prompt_tuning.loss import build_ranking_pairs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "prompt_tuning/config.yaml"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = config_parser.parse_known_args()
    config = load_yaml(known.config)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--annotations", default=value(config, "data", "annotations"))
    parser.add_argument("--image-root", default=value(config, "data", "image_root"))
    parser.add_argument(
        "--batch-size", type=int, default=value(config, "data", "batch_size")
    )
    parser.add_argument(
        "--num-workers", type=int, default=value(config, "data", "num_workers")
    )
    parser.add_argument(
        "--epochs", type=int, default=value(config, "training", "epochs")
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=value(config, "training", "learning_rate"),
    )
    parser.add_argument(
        "--margin", type=float, default=value(config, "training", "margin")
    )
    parser.add_argument("--seed", type=int, default=value(config, "training", "seed"))
    parser.add_argument("--device", default=value(config, "training", "device"))
    parser.add_argument("--clip-model", default=value(config, "clip", "model_name"))
    parser.add_argument(
        "--clip-pretrained", default=value(config, "clip", "pretrained")
    )
    parser.add_argument("--clip-cache-dir", default=value(config, "clip", "cache_dir"))
    parser.add_argument(
        "--output-dir",
        default=value(config, "output", "output_dir"),
        help="Directory in which ranking_prompts.pth will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=value(config, "output", "overwrite", default=False),
        help="Allow replacement of an existing output checkpoint.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate annotation structure and image paths without loading CLIP.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.margin < 0:
        raise ValueError("--margin cannot be negative.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    validate_args(args)
    annotation_path = resolve_path(args.annotations, PROJECT_ROOT)
    image_root = resolve_path(args.image_root, PROJECT_ROOT)
    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    cache_dir = resolve_path(args.clip_cache_dir, PROJECT_ROOT)
    assert annotation_path is not None
    assert image_root is not None
    if output_dir is None:
        raise ValueError(
            "Specify output.output_dir in the config or pass --output-dir."
        )
    output_path = output_dir / "ranking_prompts.pth"
    dataset = RankingAnnotationDataset(
        annotation_file=annotation_path, image_root=image_root, transform=None
    )
    print(f"Validated {len(dataset)} ranking annotations.")
    if args.validate_only:
        print("Validation finished. CLIP was not loaded and no checkpoint was written.")
        return
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output directory is not a directory: {output_dir}")
    if output_path.exists() and (not args.overwrite):
        raise FileExistsError(
            f"Output checkpoint already exists: {output_path}. Review it, then pass --overwrite to replace it."
        )
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is not available.")
    set_seed(args.seed)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    dataset.transform = preprocess
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    prompt_learner = LearnablePrompts(clip_model, tokenizer).to(device)
    optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=args.learning_rate)
    ranking_loss = nn.MarginRankingLoss(margin=args.margin)
    best_train_loss = float("inf")
    print(
        f"Training {len(dataset)} samples for {args.epochs} epoch(s) on {device}; output={output_path}"
    )
    for epoch in range(1, args.epochs + 1):
        prompt_learner.train()
        weighted_loss = 0.0
        pair_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels, primary_indices in progress:
            images = images.to(device)
            labels = labels.to(device)
            primary_indices = primary_indices.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                image_features = clip_model.encode_image(images, normalize=True)
            with patched_token_embedding(
                clip_model, prompt_learner.learnable_embeddings
            ):
                text_features = clip_model.encode_text(
                    prompt_learner.tokenized_prompts_buffer.to(device), normalize=True
                )
            logits = image_features @ text_features.T
            higher, lower = build_ranking_pairs(logits, labels, primary_indices)
            if higher is None or lower is None:
                continue
            targets = torch.ones_like(higher)
            loss = ranking_loss(higher, lower, targets)
            loss.backward()
            optimizer.step()
            current_pairs = int(higher.numel())
            weighted_loss += float(loss.item()) * current_pairs
            pair_count += current_pairs
            progress.set_postfix(loss=f"{loss.item():.6f}", pairs=pair_count)
        if pair_count == 0:
            raise RuntimeError(
                "No ranking pairs were produced. Check the annotation labels."
            )
        average_loss = weighted_loss / pair_count
        print(f"Epoch {epoch}: train_loss={average_loss:.6f}, pairs={pair_count}")
        if average_loss < best_train_loss:
            best_train_loss = average_loss
            atomic_torch_save(prompt_learner.state_dict(), output_path)
            print(f"Saved new best ranking prompts: {output_path}")
    print("Training finished.")
    print(f"Best training ranking loss: {best_train_loss:.6f}")
    print(f"Checkpoint: {output_path}")


if __name__ == "__main__":
    main()
