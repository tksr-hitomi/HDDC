"""Pair images and DDMs with synchronized preprocessing."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
import torch
import torchvision.transforms.functional as functional
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .dcw import DEGRADATION_KEYS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}


@dataclass(frozen=True)
class PairedRecord:
    stem: str
    raw_path: Path
    reference_path: Path
    prior_paths: dict[str, Path]


@dataclass(frozen=True)
class InferenceRecord:
    stem: str
    filename: str
    raw_path: Path
    prior_paths: dict[str, Path]


def _image_index(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    index: dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in index:
            raise ValueError(f"Duplicate image stem '{path.stem}' in {root}")
        index[path.stem] = path
    if not index:
        raise ValueError(f"No supported images found in: {root}")
    return index


def _prior_paths(prior_dir: str | Path, stem: str) -> dict[str, Path]:
    root = Path(prior_dir)
    paths = {key: root / f"{stem}_CLIP_{key}_smooth.png" for key in DEGRADATION_KEYS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prior map(s) for '{stem}': {missing}")
    return paths


def build_paired_records(
    raw_dir: str | Path,
    reference_dir: str | Path,
    prior_dir: str | Path,
    ratio: float,
    expected_count: int | None,
) -> list[PairedRecord]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"Dataset ratio must be in (0, 1], got {ratio}.")
    raw_index = _image_index(raw_dir)
    reference_index = _image_index(reference_dir)
    if set(raw_index) != set(reference_index):
        raw_only = sorted(set(raw_index) - set(reference_index))
        reference_only = sorted(set(reference_index) - set(raw_index))
        raise ValueError(
            f"Raw/reference pairing mismatch; raw-only={raw_only[:5]}, reference-only={reference_only[:5]}."
        )
    records = [
        PairedRecord(
            stem=stem,
            raw_path=raw_index[stem],
            reference_path=reference_index[stem],
            prior_paths=_prior_paths(prior_dir, stem),
        )
        for stem in sorted(raw_index)
    ]
    selected_count = int(len(records) * ratio)
    records = records[:selected_count]
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} paired samples after ratio={ratio}, found {len(records)}."
        )
    return records


def build_inference_records(
    raw_dir: str | Path, prior_dir: str | Path, ratio: float, expected_count: int | None
) -> list[InferenceRecord]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"Dataset ratio must be in (0, 1], got {ratio}.")
    raw_index = _image_index(raw_dir)
    records = [
        InferenceRecord(
            stem=stem,
            filename=raw_index[stem].name,
            raw_path=raw_index[stem],
            prior_paths=_prior_paths(prior_dir, stem),
        )
        for stem in sorted(raw_index)
    ]
    records = records[: int(len(records) * ratio)]
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} inference samples after ratio={ratio}, found {len(records)}."
        )
    return records


class TrainTransform:
    def __init__(self, image_size: int) -> None:
        target = (image_size, image_size)
        self.resize_rgb = transforms.Resize(
            target, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.resize_map = transforms.Resize(
            target, interpolation=transforms.InterpolationMode.NEAREST
        )

    def __call__(
        self, raw: Image.Image, reference: Image.Image, maps: Mapping[str, Image.Image]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.resize_rgb(raw)
        reference = self.resize_rgb(reference)
        transformed_maps = {key: self.resize_map(image) for key, image in maps.items()}
        horizontal_flip = random.random() > 0.5
        vertical_flip = random.random() > 0.5
        angle = random.choice([0, 90, 180, 270])
        if horizontal_flip:
            raw = functional.hflip(raw)
            reference = functional.hflip(reference)
            transformed_maps = {
                key: functional.hflip(image) for key, image in transformed_maps.items()
            }
        if vertical_flip:
            raw = functional.vflip(raw)
            reference = functional.vflip(reference)
            transformed_maps = {
                key: functional.vflip(image) for key, image in transformed_maps.items()
            }
        if angle > 0:
            raw = functional.rotate(raw, angle)
            reference = functional.rotate(reference, angle)
            transformed_maps = {
                key: functional.rotate(image, angle)
                for key, image in transformed_maps.items()
            }
        return (
            functional.to_tensor(raw),
            functional.to_tensor(reference),
            {
                key: functional.to_tensor(image)
                for key, image in transformed_maps.items()
            },
        )


class EvalTransform:
    def __init__(self, image_size: int) -> None:
        target = (image_size, image_size)
        self.resize_rgb = transforms.Resize(
            target, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.resize_map = transforms.Resize(
            target, interpolation=transforms.InterpolationMode.NEAREST
        )

    def __call__(
        self, raw: Image.Image, reference: Image.Image, maps: Mapping[str, Image.Image]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return (
            functional.to_tensor(self.resize_rgb(raw)),
            functional.to_tensor(self.resize_rgb(reference)),
            {
                key: functional.to_tensor(self.resize_map(image))
                for key, image in maps.items()
            },
        )


class PairedUIEBDataset(Dataset):
    def __init__(self, records: list[PairedRecord], transform: Callable) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record.raw_path) as image:
            raw = image.convert("RGB")
        with Image.open(record.reference_path) as image:
            reference = image.convert("RGB")
        maps: dict[str, Image.Image] = {}
        for key, path in record.prior_paths.items():
            with Image.open(path) as image:
                maps[key] = image.convert("L")
        clip_image = raw.copy()
        raw_tensor, reference_tensor, quality_maps = self.transform(
            raw, reference, maps
        )
        return {
            "raw_pil": clip_image,
            "raw_tensor": raw_tensor,
            "reference_tensor": reference_tensor,
            "quality_maps": quality_maps,
            "stem": record.stem,
        }


class InferenceUIEBDataset(Dataset):
    def __init__(self, records: list[InferenceRecord], image_size: int) -> None:
        self.records = records
        target = (image_size, image_size)
        self.resize_rgb = transforms.Resize(
            target, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.resize_map = transforms.Resize(
            target, interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record.raw_path) as image:
            raw = image.convert("RGB")
        maps: dict[str, torch.Tensor] = {}
        for key, path in record.prior_paths.items():
            with Image.open(path) as image:
                maps[key] = functional.to_tensor(self.resize_map(image.convert("L")))
        return {
            "raw_pil": raw.copy(),
            "raw_tensor": functional.to_tensor(self.resize_rgb(raw)),
            "quality_maps": maps,
            "filename": record.filename,
        }


def collate_samples(batch: list[dict]) -> dict:
    quality_maps = {
        key: torch.stack([sample["quality_maps"][key] for sample in batch])
        for key in DEGRADATION_KEYS
    }
    output = {
        "raw_pil": [sample["raw_pil"] for sample in batch],
        "raw_tensor": torch.stack([sample["raw_tensor"] for sample in batch]),
        "quality_maps": quality_maps,
    }
    if "reference_tensor" in batch[0]:
        output["reference_tensor"] = torch.stack(
            [sample["reference_tensor"] for sample in batch]
        )
        output["stem"] = [sample["stem"] for sample in batch]
    if "filename" in batch[0]:
        output["filename"] = [sample["filename"] for sample in batch]
    return output


def create_training_loaders(
    train_records: list[PairedRecord],
    test_records: list[PairedRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    random.seed(seed)
    train_dataset = PairedUIEBDataset(train_records, TrainTransform(image_size))
    test_dataset = PairedUIEBDataset(test_records, EvalTransform(image_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    return (train_loader, test_loader)
