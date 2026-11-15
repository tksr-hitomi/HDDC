"""Read private ranking annotations and their images."""

import json
from pathlib import Path
from typing import Mapping
import torch
from PIL import Image
from torch.utils.data import Dataset

DEGRADATION_NAMES = ("green_cast", "blue_cast", "haziness", "low_light")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class RankingAnnotationDataset(Dataset):
    def __init__(
        self, annotation_file: str | Path, image_root: str | Path, transform=None
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.transform = transform
        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"Annotation JSON not found: {self.annotation_file}"
            )
        with self.annotation_file.open("r", encoding="utf-8") as stream:
            annotations = json.load(stream)
        if not isinstance(annotations, Mapping) or not annotations:
            raise ValueError("Annotation JSON must be a non-empty object.")
        self.records: list[tuple[Path, tuple[int, ...], int]] = []
        errors: list[str] = []
        for image_key, annotation in annotations.items():
            try:
                record = self._validate_record(image_key, annotation)
                self.records.append(record)
            except (TypeError, ValueError, FileNotFoundError) as exc:
                errors.append(f"{image_key!r}: {exc}")
        if errors:
            preview = "\n".join((f"- {message}" for message in errors[:10]))
            remainder = len(errors) - 10
            suffix = f"\n... and {remainder} more" if remainder > 0 else ""
            raise ValueError(
                f"Invalid ranking annotations ({len(errors)} record(s)):\n{preview}{suffix}"
            )
        if not self.records:
            raise ValueError("No valid annotation records were found.")

    def _validate_record(
        self, image_key: object, annotation: object
    ) -> tuple[Path, tuple[int, ...], int]:
        if not isinstance(image_key, str) or not image_key.strip():
            raise TypeError("Image path key must be a non-empty string.")
        if not isinstance(annotation, Mapping):
            raise TypeError("Annotation value must be an object.")
        labels = annotation.get("labels")
        if (
            not isinstance(labels, list)
            or len(labels) != len(DEGRADATION_NAMES)
            or any(
                (
                    isinstance(label, bool)
                    or not isinstance(label, int)
                    or label not in (0, 1)
                    for label in labels
                )
            )
        ):
            raise ValueError("labels must contain exactly four integer 0/1 values.")
        if sum(labels) == 0:
            raise ValueError("At least one degradation label must be positive.")
        primary_index = annotation.get("primary_index")
        if (
            isinstance(primary_index, bool)
            or not isinstance(primary_index, int)
            or primary_index not in range(len(DEGRADATION_NAMES))
        ):
            raise ValueError("primary_index must be an integer from 0 to 3.")
        if labels[primary_index] != 1:
            raise ValueError("primary_index must point to a positive label.")
        image_path = Path(image_key).expanduser()
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        image_path = image_path.resolve()
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {image_path.suffix}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return (image_path, tuple(labels), primary_index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, labels, primary_index = self.records[index]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        if self.transform is not None:
            rgb = self.transform(rgb)
        return (
            rgb,
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(primary_index, dtype=torch.long),
        )
