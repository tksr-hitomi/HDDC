"""Enhance images using the released HDDC and prompt weights."""

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import load_model_checkpoint, load_yaml, resolve_path, value
from .dcw import RankingPromptEncoder
from .data import InferenceUIEBDataset, build_inference_records, collate_samples
from .model import HDDC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "hddc/configs/inference.yaml"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = config_parser.parse_known_args()
    config = load_yaml(known.config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--raw-dir", default=value(config, "dataset", "raw_dir"))
    parser.add_argument(
        "--ddm-dir",
        "--prior-dir",
        dest="ddm_dir",
        default=value(config, "dataset", "ddm_dir"),
    )
    parser.add_argument(
        "--test-ratio", type=float, default=value(config, "dataset", "ratio")
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=value(config, "dataset", "expected_samples"),
    )
    parser.add_argument(
        "--image-size", type=int, default=value(config, "dataset", "image_size")
    )
    parser.add_argument(
        "--batch-size", type=int, default=value(config, "dataset", "batch_size")
    )
    parser.add_argument(
        "--num-workers", type=int, default=value(config, "dataset", "num_workers")
    )
    parser.add_argument(
        "--input-channels", type=int, default=value(config, "model", "input_channels")
    )
    parser.add_argument(
        "--output-channels", type=int, default=value(config, "model", "output_channels")
    )
    parser.add_argument(
        "--base-channels", type=int, default=value(config, "model", "base_channels")
    )
    parser.add_argument(
        "--router-hidden-factor",
        type=float,
        default=value(config, "model", "router_hidden_factor"),
    )
    parser.add_argument("--checkpoint", default=value(config, "model", "checkpoint"))
    parser.add_argument("--clip-model", default=value(config, "clip", "model_name"))
    parser.add_argument(
        "--clip-pretrained", default=value(config, "clip", "pretrained")
    )
    parser.add_argument(
        "--prompt-checkpoint", default=value(config, "clip", "prompt_checkpoint")
    )
    parser.add_argument("--clip-cache-dir", default=value(config, "clip", "cache_dir"))
    parser.add_argument("--output-dir", default=value(config, "output", "directory"))
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=value(config, "output", "overwrite"),
    )
    parser.add_argument("--device", default=value(config, "runtime", "device"))
    parser.add_argument(
        "--clip-device", default=value(config, "runtime", "clip_device")
    )
    return parser.parse_args()


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(
        np.uint8
    )
    Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    if args.image_size % 8 != 0:
        raise ValueError("image_size must be divisible by 8.")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is not available.")
    raw_dir = resolve_path(args.raw_dir, PROJECT_ROOT)
    ddm_dir = resolve_path(args.ddm_dir, PROJECT_ROOT)
    checkpoint = resolve_path(args.checkpoint, PROJECT_ROOT)
    prompt_checkpoint = resolve_path(args.prompt_checkpoint, PROJECT_ROOT)
    clip_cache_dir = resolve_path(args.clip_cache_dir, PROJECT_ROOT)
    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    assert output_dir is not None
    records = build_inference_records(
        raw_dir, ddm_dir, ratio=args.test_ratio, expected_count=args.expected_samples
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_entries = list(output_dir.iterdir())
    if existing_entries and (not args.overwrite):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. Review it, then pass --overwrite if replacement is intended."
        )
    dataset = InferenceUIEBDataset(records, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
    )
    clip_encoder = RankingPromptEncoder(
        prompt_checkpoint=prompt_checkpoint,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=args.clip_device,
        cache_dir=clip_cache_dir,
    )
    model = HDDC(
        input_channels=args.input_channels,
        output_channels=args.output_channels,
        base_channels=args.base_channels,
        router_hidden_factor=args.router_hidden_factor,
        clip_encoder=clip_encoder,
    ).to(device)
    load_model_checkpoint(model, checkpoint, map_location="cpu")
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Enhancing images"):
            raw = batch["raw_tensor"].to(device)
            quality_maps = {
                key: tensor.to(device) for key, tensor in batch["quality_maps"].items()
            }
            enhanced = model(raw, batch["raw_pil"], quality_maps)
            for tensor, filename in zip(enhanced, batch["filename"]):
                destination = output_dir / filename
                if destination.exists() and (not args.overwrite):
                    raise FileExistsError(f"Output already exists: {destination}")
                save_tensor(tensor, destination)
                saved += 1
    print(f"Saved {saved} enhanced images to {output_dir}")


if __name__ == "__main__":
    main()
