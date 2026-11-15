"""Handle configuration, lightweight checkpoints, and PSNR."""

import os
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping
import yaml
import torch
from torch import nn


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"The YAML root must be a mapping: {config_path}")
    return data


def value(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def resolve_path(path: str | Path | None, project_root: Path) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def model_state_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(
        (isinstance(key, str) for key in checkpoint)
    ):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def load_model_checkpoint(
    model: nn.Module, path: str | Path, map_location: str | torch.device = "cpu"
) -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(model_state_from_checkpoint(checkpoint), strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def load_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: str | Path,
    map_location: str | torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    required = {
        "epoch",
        "best_psnr",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint is missing fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (int(checkpoint["epoch"]), float(checkpoint["best_psnr"]))


class PSNRAggregator:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.infinite_count = 0

    def update(self, generated: torch.Tensor, reference: torch.Tensor) -> None:
        generated_u8 = (generated.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
        reference_u8 = (reference.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8)
        for predicted, target in zip(generated_u8, reference_u8):
            difference = predicted.to(torch.float64) - target.to(torch.float64)
            mse = difference.square().mean().item()
            if mse == 0.0:
                self.infinite_count += 1
                continue
            self.values.append(10.0 * math.log10(255.0 * 255.0 / mse))

    @property
    def mean(self) -> float:
        if not self.values:
            raise ValueError("PSNR has no finite image-pair scores to average.")
        return sum(self.values) / len(self.values)
