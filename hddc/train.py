"""Train HDDC and keep only the best and latest checkpoints."""

import argparse
import time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from .utils import (
    atomic_torch_save,
    load_training_checkpoint,
    load_yaml,
    resolve_path,
    value,
    PSNRAggregator,
)
from .dcw import RankingPromptEncoder
from .data import build_paired_records, create_training_loaders
from .loss import ReconstructionLoss
from .model import HDDC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "hddc/configs/train.yaml"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = config_parser.parse_known_args()
    config = load_yaml(known.config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config)
    parser.add_argument(
        "--train-raw", default=value(config, "dataset", "train", "raw_dir")
    )
    parser.add_argument(
        "--train-reference", default=value(config, "dataset", "train", "reference_dir")
    )
    parser.add_argument(
        "--train-ddm",
        "--train-prior",
        dest="train_ddm",
        default=value(config, "dataset", "train", "ddm_dir"),
    )
    parser.add_argument(
        "--train-ratio", type=float, default=value(config, "dataset", "train", "ratio")
    )
    parser.add_argument(
        "--expected-train-samples",
        type=int,
        default=value(config, "dataset", "train", "expected_samples"),
    )
    parser.add_argument(
        "--test-raw", default=value(config, "dataset", "test", "raw_dir")
    )
    parser.add_argument(
        "--test-reference", default=value(config, "dataset", "test", "reference_dir")
    )
    parser.add_argument(
        "--test-ddm",
        "--test-prior",
        dest="test_ddm",
        default=value(config, "dataset", "test", "ddm_dir"),
    )
    parser.add_argument(
        "--test-ratio", type=float, default=value(config, "dataset", "test", "ratio")
    )
    parser.add_argument(
        "--expected-test-samples",
        type=int,
        default=value(config, "dataset", "test", "expected_samples"),
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
    parser.add_argument("--seed", type=int, default=value(config, "dataset", "seed"))
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
    parser.add_argument("--clip-model", default=value(config, "clip", "model_name"))
    parser.add_argument(
        "--clip-pretrained", default=value(config, "clip", "pretrained")
    )
    parser.add_argument(
        "--prompt-checkpoint", default=value(config, "clip", "prompt_checkpoint")
    )
    parser.add_argument("--clip-cache-dir", default=value(config, "clip", "cache_dir"))
    parser.add_argument(
        "--epochs", type=int, default=value(config, "training", "epochs")
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=value(config, "training", "learning_rate"),
    )
    parser.add_argument(
        "--scheduler-start-epoch",
        type=int,
        default=value(config, "training", "scheduler_start_epoch"),
    )
    parser.add_argument(
        "--scheduler-eta-min",
        type=float,
        default=value(config, "training", "scheduler_eta_min"),
    )
    parser.add_argument(
        "--smooth-l1-weight",
        type=float,
        default=value(config, "training", "smooth_l1_weight"),
    )
    parser.add_argument(
        "--smooth-l1-beta",
        type=float,
        default=value(config, "training", "smooth_l1_beta"),
    )
    parser.add_argument(
        "--ssim-weight", type=float, default=value(config, "training", "ssim_weight")
    )
    parser.add_argument(
        "--ssim-data-range",
        type=float,
        default=value(config, "training", "ssim_data_range"),
    )
    parser.add_argument(
        "--ssim-window-size",
        type=int,
        default=value(config, "training", "ssim_window_size"),
    )
    parser.add_argument("--output-dir", default=value(config, "training", "output_dir"))
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__LATEST__",
        default=value(config, "training", "resume"),
        help="Resume from PATH, or from output/latest.pth when PATH is omitted.",
    )
    parser.add_argument("--device", default=value(config, "runtime", "device"))
    parser.add_argument(
        "--clip-device", default=value(config, "runtime", "clip_device")
    )
    return parser.parse_args()


def checked_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def prepare_output(output_dir: Path, resume_path: Path | None) -> Path:
    allowed_root_files = {"best_psnr.pth", "latest.pth"}
    output_dir.mkdir(parents=True, exist_ok=True)
    record_dir = output_dir / "record_txt"
    record_dir.mkdir(exist_ok=True)
    unexpected: list[Path] = []
    for path in output_dir.iterdir():
        if path == record_dir:
            continue
        if path.is_file() and path.name in allowed_root_files:
            continue
        unexpected.append(path)
    for path in record_dir.iterdir():
        if not (path.is_file() and path.name == "eval_metrics_log.txt"):
            unexpected.append(path)
    if unexpected:
        raise RuntimeError(
            "Output directory contains unexpected artifacts; refusing to alter them: "
            + ", ".join((str(path) for path in unexpected))
        )
    existing_runtime_files = [
        path
        for path in (
            output_dir / "best_psnr.pth",
            output_dir / "latest.pth",
            record_dir / "eval_metrics_log.txt",
        )
        if path.exists()
    ]
    if existing_runtime_files and resume_path is None:
        raise RuntimeError(
            "Existing training artifacts found. Use --resume or choose a new --output-dir."
        )
    return record_dir / "eval_metrics_log.txt"


def move_quality_maps(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in batch["quality_maps"].items()}


def train_one_epoch(
    model: HDDC,
    loader,
    optimizer: AdamW,
    criterion: ReconstructionLoss,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    components = {"smooth_l1": 0.0, "ssim": 0.0}
    processed_batches = 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} train")
    for batch in progress:
        raw = batch["raw_tensor"].to(device)
        reference = batch["reference_tensor"].to(device)
        quality_maps = move_quality_maps(batch, device)
        optimizer.zero_grad()
        generated = model(raw, batch["raw_pil"], quality_maps)
        loss, current = criterion(generated, reference)
        loss.backward()
        optimizer.step()
        processed_batches += 1
        total_loss += loss.item()
        for key in components:
            components[key] += current[key].item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    if processed_batches == 0:
        raise RuntimeError("No training batches were processed.")
    return (
        total_loss / processed_batches,
        {key: amount / processed_batches for key, amount in components.items()},
    )


@torch.no_grad()
def evaluate_psnr(
    model: HDDC, loader, device: torch.device, epoch: int, epochs: int
) -> float:
    model.eval()
    aggregator = PSNRAggregator()
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} PSNR")
    for batch in progress:
        raw = batch["raw_tensor"].to(device)
        quality_maps = move_quality_maps(batch, device)
        generated = model(raw, batch["raw_pil"], quality_maps)
        aggregator.update(generated, batch["reference_tensor"])
    return aggregator.mean


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.image_size % 8 != 0:
        raise ValueError("image_size must be divisible by 8.")
    device = checked_device(args.device)
    checked_device(args.clip_device)
    train_raw = resolve_path(args.train_raw, PROJECT_ROOT)
    train_reference = resolve_path(args.train_reference, PROJECT_ROOT)
    train_ddm = resolve_path(args.train_ddm, PROJECT_ROOT)
    test_raw = resolve_path(args.test_raw, PROJECT_ROOT)
    test_reference = resolve_path(args.test_reference, PROJECT_ROOT)
    test_ddm = resolve_path(args.test_ddm, PROJECT_ROOT)
    prompt_checkpoint = resolve_path(args.prompt_checkpoint, PROJECT_ROOT)
    clip_cache_dir = resolve_path(args.clip_cache_dir, PROJECT_ROOT)
    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    assert output_dir is not None
    resume_path: Path | None
    if args.resume == "__LATEST__":
        resume_path = output_dir / "latest.pth"
    else:
        resume_path = resolve_path(args.resume, PROJECT_ROOT)
    metric_log = prepare_output(output_dir, resume_path)
    train_records = build_paired_records(
        train_raw,
        train_reference,
        train_ddm,
        ratio=args.train_ratio,
        expected_count=args.expected_train_samples,
    )
    test_records = build_paired_records(
        test_raw,
        test_reference,
        test_ddm,
        ratio=args.test_ratio,
        expected_count=args.expected_test_samples,
    )
    train_loader, test_loader = create_training_loaders(
        train_records,
        test_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Training images={len(train_records)}, test images={len(test_records)}")
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
    criterion = ReconstructionLoss(
        smooth_l1_weight=args.smooth_l1_weight,
        smooth_l1_beta=args.smooth_l1_beta,
        ssim_weight=args.ssim_weight,
        ssim_data_range=args.ssim_data_range,
        ssim_window_size=args.ssim_window_size,
    ).to(device)
    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.scheduler_eta_min
    )
    start_epoch = 0
    best_psnr = float("-inf")
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_psnr = load_training_checkpoint(
            model, optimizer, scheduler, resume_path, device
        )
        print(f"Resuming at epoch {start_epoch + 1}; best PSNR={best_psnr:.4f} dB")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss, loss_components = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        psnr = evaluate_psnr(model, test_loader, device, epoch, args.epochs)
        if psnr > best_psnr:
            best_psnr = psnr
            atomic_torch_save(
                {
                    "format_version": 1,
                    "model_name": "HDDC",
                    "epoch": epoch + 1,
                    "psnr": psnr,
                    "model_state_dict": model.state_dict(),
                },
                output_dir / "best_psnr.pth",
            )
        if epoch + 1 >= args.scheduler_start_epoch:
            scheduler.step()
        atomic_torch_save(
            {
                "format_version": 1,
                "model_name": "HDDC",
                "epoch": epoch + 1,
                "best_psnr": best_psnr,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            output_dir / "latest.pth",
        )
        with metric_log.open("a", encoding="utf-8") as stream:
            stream.write(f"Epoch {epoch + 1:03d}: PSNR={psnr:.6f} dB\n")
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1:03d}: loss={train_loss:.6f}, SmoothL1={loss_components['smooth_l1']:.6f}, SSIMLoss={loss_components['ssim']:.6f}, PSNR={psnr:.6f} dB, best={best_psnr:.6f} dB, time={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
