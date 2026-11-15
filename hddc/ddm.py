"""Generate the original four CLIP degradation degree maps."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import open_clip
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm
from torchvision import transforms
from .utils import load_yaml, resolve_path, value

PROMPT_PAIRS = {
    "green_cast": {
        "bad": "A photo of an underwater scene with a heavy green color cast.",
        "good": "A photo of an underwater scene with a slight green color cast.",
    },
    "blue_cast": {
        "bad": "A photo of an underwater scene with a heavy blue color cast.",
        "good": "A photo of an underwater scene with a slight blue color cast.",
    },
    "haziness": {
        "bad": "A photo of an underwater scene with heavy haze.",
        "good": "A photo of an underwater scene with slight haze.",
    },
    "illumination": {
        "bad": "A photo of a dark underwater scene.",
        "good": "A photo of a bright underwater scene.",
    },
}


class SpatialPriorGenerator:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        grid_size: tuple[int, int] = (32, 32),
        blur_radius_factor: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.grid_size = grid_size
        self.blur_radius_factor = blur_radius_factor
        self.alpha = 1.0
        self.model, _, self.preprocess_image = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        text_list: list[str] = []
        self.prompt_indices: defaultdict[str, dict[str, int]] = defaultdict(dict)
        index = 0
        for attribute, states in PROMPT_PAIRS.items():
            for state, text in states.items():
                text_list.append(text)
                self.prompt_indices[attribute][state] = index
                index += 1
        tokenized = tokenizer(text_list).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(tokenized)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

    def _patchify(self, image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        grid_width, grid_height = self.grid_size
        patch_width = width // grid_width
        patch_height = height // grid_height
        if patch_width < 1 or patch_height < 1:
            raise ValueError(
                f"Image size {image.size} is smaller than grid {self.grid_size}."
            )
        return [
            image.crop(
                (
                    column * patch_width,
                    row * patch_height,
                    (column + 1) * patch_width,
                    (row + 1) * patch_height,
                )
            )
            for row in range(grid_height)
            for column in range(grid_width)
        ]

    @torch.no_grad()
    def _similarity_logits(self, images: list[Image.Image]) -> torch.Tensor:
        image_batch = torch.stack(
            [self.preprocess_image(image) for image in images]
        ).to(self.device)
        image_features = self.model.encode_image(image_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.model.logit_scale.exp() * image_features @ self.text_features.T

    def _patch_rows(self, logits: torch.Tensor) -> list[dict[str, int | str]]:
        rows: list[dict[str, int | str]] = []
        grid_width = self.grid_size[0]
        for patch_index in range(logits.shape[0]):
            row: dict[str, int | str] = {
                "patch_row": patch_index // grid_width,
                "patch_col": patch_index % grid_width,
            }
            patch_logits = logits[patch_index]
            for attribute in PROMPT_PAIRS:
                bad_index = self.prompt_indices[attribute]["bad"]
                good_index = self.prompt_indices[attribute]["good"]
                bad = patch_logits[bad_index].item()
                good = patch_logits[good_index].item()
                score = torch.sigmoid(torch.tensor((bad - good) / self.alpha)).item()
                row[f"{attribute}_score"] = f"{score:.4f}"
                row[f"{attribute}_logit_bad"] = f"{bad:.2f}"
                row[f"{attribute}_logit_good"] = f"{good:.2f}"
            rows.append(row)
        return rows

    def _save_map(
        self,
        score_map: np.ndarray,
        output_size: tuple[int, int],
        path: Path,
        smooth: bool,
    ) -> None:
        small_map = Image.fromarray((score_map * 255).astype(np.uint8), mode="L")
        output = small_map.resize(output_size, Image.Resampling.NEAREST)
        if smooth:
            patch_width = output_size[0] / score_map.shape[1]
            output = output.filter(
                ImageFilter.GaussianBlur(radius=patch_width * self.blur_radius_factor)
            )
        output.save(path)

    def generate_directory(
        self,
        raw_image_dir: str | Path,
        output_ddm_dir: str | Path,
        target_size: tuple[int, int] = (256, 256),
        save_smooth: bool = True,
        save_blocky: bool = False,
        save_csv: bool = False,
        overwrite: bool = False,
    ) -> None:
        raw_root = Path(raw_image_dir)
        output_root = Path(output_ddm_dir)
        if not raw_root.is_dir():
            raise FileNotFoundError(f"Raw-image directory not found: {raw_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(
            (
                path
                for path in raw_root.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
        )
        if not image_paths:
            raise ValueError(f"No images found in: {raw_root}")
        resize = transforms.Resize(target_size)
        for image_path in tqdm(image_paths, desc="Generating DDMs"):
            stem = image_path.stem
            expected_outputs = [
                output_root / f"{stem}_CLIP_{attribute}_smooth.png"
                for attribute in PROMPT_PAIRS
            ]
            if save_blocky:
                expected_outputs.extend(
                    (
                        output_root / f"{stem}_CLIP_{attribute}_blocky.png"
                        for attribute in PROMPT_PAIRS
                    )
                )
            if save_csv:
                expected_outputs.append(output_root / f"{stem}_CLIP_patch_report.csv")
            existing = [path for path in expected_outputs if path.exists()]
            if existing and (not overwrite):
                if len(existing) == len(expected_outputs):
                    continue
                raise FileExistsError(
                    f"Partial DDM outputs already exist for '{stem}'. Use --overwrite only after reviewing them."
                )
            with Image.open(image_path) as image:
                standardized = resize(image.convert("RGB"))
            logits = self._similarity_logits(self._patchify(standardized))
            rows = self._patch_rows(logits)
            if save_csv:
                header = ["patch_row", "patch_col"]
                for attribute in PROMPT_PAIRS:
                    header.extend(
                        [
                            f"{attribute}_score",
                            f"{attribute}_logit_bad",
                            f"{attribute}_logit_good",
                        ]
                    )
                with (output_root / f"{stem}_CLIP_patch_report.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(rows)
            for attribute in PROMPT_PAIRS:
                score_map = np.array(
                    [float(row[f"{attribute}_score"]) for row in rows]
                ).reshape(self.grid_size[::-1])
                if save_smooth:
                    self._save_map(
                        score_map,
                        standardized.size,
                        output_root / f"{stem}_CLIP_{attribute}_smooth.png",
                        smooth=True,
                    )
                if save_blocky:
                    self._save_map(
                        score_map,
                        standardized.size,
                        output_root / f"{stem}_CLIP_{attribute}_blocky.png",
                        smooth=False,
                    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "hddc/configs/ddm.yaml"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = config_parser.parse_known_args()
    config = load_yaml(known.config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--raw-dir", default=value(config, "input", "raw_dir"))
    parser.add_argument(
        "--ddm-dir",
        "--prior-dir",
        dest="ddm_dir",
        default=value(config, "output", "ddm_dir"),
    )
    parser.add_argument(
        "--save-smooth",
        action=argparse.BooleanOptionalAction,
        default=value(config, "output", "save_smooth"),
    )
    parser.add_argument(
        "--save-blocky",
        action=argparse.BooleanOptionalAction,
        default=value(config, "output", "save_blocky"),
    )
    parser.add_argument(
        "--save-csv",
        action=argparse.BooleanOptionalAction,
        default=value(config, "output", "save_csv"),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=value(config, "output", "overwrite"),
    )
    parser.add_argument(
        "--target-size", type=int, default=value(config, "generation", "target_size")
    )
    parser.add_argument(
        "--grid-size", type=int, default=value(config, "generation", "grid_size")
    )
    parser.add_argument(
        "--blur-radius-factor",
        type=float,
        default=value(config, "generation", "blur_radius_factor"),
    )
    parser.add_argument("--clip-model", default=value(config, "clip", "model_name"))
    parser.add_argument(
        "--clip-pretrained", default=value(config, "clip", "pretrained")
    )
    parser.add_argument("--clip-cache-dir", default=value(config, "clip", "cache_dir"))
    parser.add_argument("--device", default=value(config, "runtime", "device"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = resolve_path(args.raw_dir, PROJECT_ROOT)
    ddm_dir = resolve_path(args.ddm_dir, PROJECT_ROOT)
    cache_dir = resolve_path(args.clip_cache_dir, PROJECT_ROOT)
    generator = SpatialPriorGenerator(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=args.device,
        cache_dir=cache_dir,
        grid_size=(args.grid_size, args.grid_size),
        blur_radius_factor=args.blur_radius_factor,
    )
    generator.generate_directory(
        raw_image_dir=raw_dir,
        output_ddm_dir=ddm_dir,
        target_size=(args.target_size, args.target_size),
        save_smooth=args.save_smooth,
        save_blocky=args.save_blocky,
        save_csv=args.save_csv,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
