#!/usr/bin/env python3
"""Evaluate an EnsExam-GAN checkpoint on selected hard SCUT pages."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config  # noqa: E402
from networks.generator import Generator  # noqa: E402
from utils.page_inference import infer_full_page  # noqa: E402
from utils.path_utils import normalize_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local/config.local-hardcase-mps.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--baseline-pred-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--page-overlap", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mask-threshold", type=int, default=None)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument(
        "--copy-input-outside-mask",
        choices=("none", "ms", "mb"),
        default="none",
        help="Postprocess prediction by copying input outside the selected predicted mask.",
    )
    parser.add_argument("--copy-mask-threshold", type=int, default=32)
    parser.add_argument(
        "--copy-mask-threshold-map",
        default="",
        help=(
            "Optional comma-separated per-file threshold overrides, for example "
            "'123.jpg:16,156.jpg:160'. This is intended for controlled hardcase "
            "experiments and is empty by default."
        ),
    )
    parser.add_argument(
        "--copy-mask-threshold-auto",
        choices=("none", "mb_cov8_step"),
        default="none",
        help=(
            "Optional inference-time threshold heuristic. 'mb_cov8_step' uses only "
            "predicted mb coverage at threshold 8 and is intended for controlled "
            "hardcase experiments. Explicit threshold-map entries take precedence."
        ),
    )
    parser.add_argument("--copy-mask-dilate", type=int, default=0)
    return parser.parse_args()


def parse_threshold_map(value: str) -> dict[str, int]:
    if not value.strip():
        return {}
    result: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid threshold override {item!r}; expected '<filename>:<threshold>'")
        name, threshold = item.split(":", 1)
        result[name.strip()] = int(threshold.strip())
    return result


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_generator(config_path: str, weights_path: str, device: torch.device) -> Generator:
    cfg = load_config(config_path)
    generator = Generator(cfg=cfg["model"]).to(device)
    ckpt = torch.load(normalize_path(weights_path), map_location=device, weights_only=False)
    if "G_state_dict" in ckpt:
        generator.load_state_dict(ckpt["G_state_dict"])
    elif "state_dict" in ckpt:
        generator.load_state_dict(ckpt["state_dict"])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()
    return generator


def read_sample_paths(samples_file: Path) -> list[Path]:
    paths: list[Path] = []
    for line in samples_file.read_text().splitlines():
        value = line.strip()
        if not value:
            continue
        paths.append(Path(value))
    return paths


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("all_images")
    except ValueError as exc:
        raise ValueError(f"Cannot derive label path from {image_path}") from exc
    parts[idx] = "all_labels"
    return Path(*parts)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def ensure_same_size(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    if image.shape[:2] == target.shape[:2]:
        return image
    h, w = target.shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def build_changed_mask(input_bgr: np.ndarray, label_bgr: np.ndarray, threshold: int) -> np.ndarray:
    delta = cv2.absdiff(input_bgr, label_bgr).mean(axis=2)
    mask = (delta >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask > 0


def compute_residual_metrics(
    input_bgr: np.ndarray,
    label_bgr: np.ndarray,
    pred_bgr: np.ndarray,
    change_threshold: int,
    eval_threshold: int,
) -> dict[str, float | int]:
    changed = build_changed_mask(input_bgr, label_bgr, change_threshold)
    outside = ~changed
    residual_delta = cv2.absdiff(pred_bgr, label_bgr).mean(axis=2)
    over_delta = cv2.absdiff(pred_bgr, input_bgr).mean(axis=2)
    residual = changed & (residual_delta >= eval_threshold)
    overerase = outside & (over_delta >= eval_threshold)

    changed_px = int(changed.sum())
    outside_px = int(outside.sum())
    residual_px = int(residual.sum())
    over_px = int(overerase.sum())

    return {
        "changed_px": changed_px,
        "outside_px": outside_px,
        "residual_px": residual_px,
        "over_px": over_px,
        "residual_ratio": residual_px / max(changed_px, 1),
        "overerase_ratio": over_px / max(outside_px, 1),
        "mean_residual_delta": float(residual_delta[changed].mean()) if changed_px else 0.0,
        "mean_over_delta": float(over_delta[outside].mean()) if outside_px else 0.0,
    }


def copy_input_outside_mask(
    pred_bgr: np.ndarray,
    input_bgr: np.ndarray,
    mask_u8: np.ndarray,
    threshold: int,
    dilate: int,
) -> np.ndarray:
    """Restrict edits to a predicted mask by restoring input outside it."""
    edit_mask = mask_u8 >= threshold
    if dilate > 0:
        kernel = np.ones((3, 3), np.uint8)
        edit_mask = cv2.dilate(edit_mask.astype(np.uint8), kernel, iterations=dilate) > 0
    restricted = input_bgr.copy()
    restricted[edit_mask] = pred_bgr[edit_mask]
    return restricted


def auto_copy_mask_threshold(mask_name: str, mask_u8: np.ndarray, mode: str, default: int) -> tuple[int, float]:
    if mode == "none":
        return default, 0.0
    if mode != "mb_cov8_step":
        raise ValueError(f"Unsupported copy mask threshold auto mode: {mode}")
    if mask_name != "mb":
        raise ValueError("--copy-mask-threshold-auto mb_cov8_step requires --copy-input-outside-mask mb")

    cov8 = float((mask_u8 >= 8).mean())
    if cov8 <= 0.129:
        return 8, cov8
    if cov8 <= 0.421:
        return 76, cov8
    return 160, cov8


def crop_bounds(mask: np.ndarray, width: int, height: int, pad: int = 80) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, width, height)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, width)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, height)
    return (x0, y0, x1, y1)


def panel(image_bgr: np.ndarray, title: str, max_width: int = 420) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if img.width > max_width:
        scale = max_width / img.width
        img = img.resize((max_width, max(1, int(img.height * scale))), Image.Resampling.LANCZOS)

    title_h = 28
    canvas = Image.new("RGB", (img.width, img.height + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, img.width, title_h), fill=(245, 245, 245))
    draw.text((8, 7), title, fill=(20, 20, 20), font=ImageFont.load_default())
    canvas.paste(img, (0, title_h))
    return canvas


def hstack(images: list[Image.Image], gap: int = 8) -> Image.Image:
    height = max(img.height for img in images)
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + gap
    return canvas


def save_sheet(rows: list[Image.Image], path: Path, gap: int = 12) -> None:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap
    canvas.save(path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = pick_device(args.device)
    page_overlap = args.page_overlap
    if page_overlap is None:
        page_overlap = int(cfg.get("evaluation", {}).get("page_overlap", 32))
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(cfg["train"]["batch_size"])
    eval_threshold = args.mask_threshold
    if eval_threshold is None:
        eval_threshold = int(cfg["data"].get("mask_threshold", 12))

    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred"
    crop_dir = output_dir / "crops"
    pred_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    generator = load_generator(args.config, args.weights, device)
    sample_paths = read_sample_paths(Path(args.samples_file))
    baseline_dir = Path(args.baseline_pred_dir) if args.baseline_pred_dir else None
    threshold_map = parse_threshold_map(args.copy_mask_threshold_map)

    rows: list[dict[str, str | float | int]] = []
    crop_rows: list[Image.Image] = []

    for index, image_path in enumerate(sample_paths, start=1):
        label_path = label_path_for(image_path)
        input_bgr = read_bgr(image_path)
        label_bgr = read_bgr(label_path)
        label_bgr = ensure_same_size(label_bgr, input_bgr)
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

        outputs = infer_full_page(
            generator,
            input_rgb,
            device,
            overlap=page_overlap,
            batch_size=batch_size,
        )
        pred_bgr = cv2.cvtColor(outputs["icomp"], cv2.COLOR_RGB2BGR)
        copy_mask_cov8 = 0.0
        if args.copy_input_outside_mask != "none":
            selected_mask = outputs[args.copy_input_outside_mask]
            if image_path.name in threshold_map:
                copy_mask_threshold = threshold_map[image_path.name]
            else:
                copy_mask_threshold, copy_mask_cov8 = auto_copy_mask_threshold(
                    args.copy_input_outside_mask,
                    selected_mask,
                    args.copy_mask_threshold_auto,
                    args.copy_mask_threshold,
                )
            pred_bgr = copy_input_outside_mask(
                pred_bgr,
                input_bgr,
                selected_mask,
                threshold=copy_mask_threshold,
                dilate=args.copy_mask_dilate,
            )
        else:
            copy_mask_threshold = args.copy_mask_threshold
        stem = image_path.stem
        pred_path = pred_dir / f"{stem}.png"
        cv2.imwrite(str(pred_path), pred_bgr)

        metrics = compute_residual_metrics(
            input_bgr,
            label_bgr,
            pred_bgr,
            change_threshold=args.change_threshold,
            eval_threshold=eval_threshold,
        )
        row: dict[str, str | float | int] = {
            "file": image_path.name,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "pred_path": str(pred_path),
            "copy_input_outside_mask": args.copy_input_outside_mask,
            "copy_mask_threshold": copy_mask_threshold,
            "copy_mask_threshold_map": args.copy_mask_threshold_map,
            "copy_mask_threshold_auto": args.copy_mask_threshold_auto,
            "copy_mask_cov8": copy_mask_cov8,
            "copy_mask_dilate": args.copy_mask_dilate,
            **metrics,
        }

        panels = [
            panel(input_bgr, f"{stem} input"),
            panel(label_bgr, "target"),
        ]
        if baseline_dir is not None:
            baseline_path = baseline_dir / image_path.name
            if not baseline_path.exists():
                baseline_path = baseline_dir / f"{stem}.png"
            if baseline_path.exists():
                baseline_bgr = ensure_same_size(read_bgr(baseline_path), input_bgr)
                base_metrics = compute_residual_metrics(
                    input_bgr,
                    label_bgr,
                    baseline_bgr,
                    change_threshold=args.change_threshold,
                    eval_threshold=eval_threshold,
                )
                row.update({
                    "baseline_pred_path": str(baseline_path),
                    "baseline_residual_ratio": base_metrics["residual_ratio"],
                    "baseline_overerase_ratio": base_metrics["overerase_ratio"],
                    "delta_residual_ratio": float(base_metrics["residual_ratio"]) - float(metrics["residual_ratio"]),
                    "delta_overerase_ratio": float(metrics["overerase_ratio"]) - float(base_metrics["overerase_ratio"]),
                })
                panels.append(panel(baseline_bgr, "baseline"))
        panels.append(panel(pred_bgr, "hardcase"))

        changed = build_changed_mask(input_bgr, label_bgr, args.change_threshold)
        x0, y0, x1, y1 = crop_bounds(changed, input_bgr.shape[1], input_bgr.shape[0])
        crop_panels = [panel(img[y0:y1, x0:x1], title) for img, title in [
            (input_bgr, f"{stem} input"),
            (label_bgr, "target"),
            (pred_bgr, "hardcase"),
        ]]
        crop_row = hstack(crop_panels)
        crop_row.save(crop_dir / f"{stem}_crop_compare.png")
        crop_rows.append(hstack(panels))
        rows.append(row)
        print(f"[{index}/{len(sample_paths)}] {image_path.name} residual={metrics['residual_ratio']:.4f} over={metrics['overerase_ratio']:.4f}")

    csv_path = output_dir / "hardcase_worst_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_sheet(crop_rows, output_dir / "hardcase_worst_sheet.png")
    print(f"metrics: {csv_path}")
    print(f"sheet: {output_dir / 'hardcase_worst_sheet.png'}")


if __name__ == "__main__":
    main()
