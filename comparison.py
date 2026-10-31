from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from inference import (
    CLASS_COLORS, CLASS_NAMES, IMAGE_SIZE, MODE_LABELS, MODEL_SPECS,
    PINNED_ULTRALYTICS, PredictionOutput, file_sha256, label_font, model_revision, render_overlay,
)


def experiment_key(data: bytes, mode: str, conf: Any, iou: Any) -> str:
    revisions = [model_revision(spec.weights) if spec.weights.exists() else str(spec.weights) for spec in MODEL_SPECS]
    config = {"mode": mode, "confidence": conf, "iou_nms": iou if mode == "nms" else None,
              "imgsz": IMAGE_SIZE, "weights": revisions, "ultralytics": PINNED_ULTRALYTICS}
    return hashlib.sha256(data + json.dumps(config, sort_keys=True).encode()).hexdigest()


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def run_metadata(data: bytes, file_name: str, mode: str, conf: Any, iou: Any,
                 device: str | int) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_name": file_name, "image_sha256": hashlib.sha256(data).hexdigest(),
        "scope": "single_image_exploration_without_ground_truth", "mode": mode,
        "confidence": conf, "iou_nms": iou if mode == "nms" else None,
        "imgsz": IMAGE_SIZE, "rect": False, "device": str(device),
        "ultralytics": PINNED_ULTRALYTICS,
        "models": [{"id": spec.id, "weights": str(spec.weights),
                    "revision": model_revision(spec.weights) if spec.weights.exists() else None,
                    "sha256": file_sha256(str(spec.weights), model_revision(spec.weights)) if spec.weights.exists() else None}
                   for spec in MODEL_SPECS],
    }


def summary_row(spec: Any, output: PredictionOutput | None, error: str = "") -> dict[str, Any]:
    return {
        "Model": spec.label,
        "Status": output.status if output else "Gagal diproses",
        "Kelas terdeteksi": output.detected_classes if output else "-",
        "Jumlah deteksi": len(output.detections) if output else None,
        "Confidence tertinggi": max((d["confidence"] for d in output.detections), default=None) if output else None,
        "Luas mask (%)": output.mask_area_percent if output and output.masks_available else None,
        "Waktu inferensi (s)": output.inference_time if output else None,
        "Error": error,
    }


def class_prediction_frame(outputs: dict[str, PredictionOutput]) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        if spec.id not in outputs:
            continue
        output = outputs[spec.id]
        for name in CLASS_NAMES:
            items = [d for d in output.detections if d["kelas"] == name]
            rows.append({
                "Model": spec.label, "Kelas": name, "Jumlah deteksi": len(items),
                "Confidence (%)": max((d["confidence"] * 100 for d in items), default=None),
                "Luas mask (%)": float(output.class_masks[name].mean() * 100) if output.masks_available else None,
            })
    return pd.DataFrame(rows)


def colored_mask(output: PredictionOutput) -> Image.Image:
    first = next(iter(output.class_masks.values()))
    pixels = np.zeros((*first.shape, 3), dtype=np.uint8)
    for name in CLASS_NAMES:
        pixels[output.class_masks[name]] = CLASS_COLORS[name]
    return Image.fromarray(pixels)


def mask_difference(image: Image.Image, left: PredictionOutput, right: PredictionOutput,
                    class_name: str | None = None) -> tuple[Image.Image, dict[str, int]]:
    if not left.masks_available or not right.masks_available:
        raise ValueError("Mask lengkap kedua model diperlukan untuk membuat peta perbedaan.")
    names = [class_name] if class_name else CLASS_NAMES
    a = np.logical_or.reduce([left.class_masks[n] for n in names])
    b = np.logical_or.reduce([right.class_masks[n] for n in names])
    equal_classes = np.logical_and.reduce([left.class_masks[n] == right.class_masks[n] for n in names])
    same_class = a & b & equal_classes
    different_class = a & b & ~equal_classes
    regions = {"Hanya tanpa negative": a & ~b, "Hanya dengan negative": b & ~a,
               "Keduanya, kelas sama": same_class, "Keduanya, kelas berbeda": different_class}
    colors = [(217, 119, 6), (59, 130, 246), (20, 184, 166), (225, 64, 130)]
    pixels = np.asarray(image.convert("RGB")).copy()
    for mask, color in zip(regions.values(), colors):
        pixels[mask] = (pixels[mask] * 0.3 + np.asarray(color) * 0.7).astype(np.uint8)
    return Image.fromarray(pixels), {name: int(mask.sum()) for name, mask in regions.items()}


def comparison_image(image: Image.Image, outputs: dict[str, PredictionOutput],
                     mode: str, conf: float, iou: float, file_name: str) -> bytes:
    panel_width, panel_height, gutter = 520, 520, 24
    sheet = Image.new("RGB", (panel_width * 2 + gutter * 3, 680), "#FFFFFF")
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 16), "Perbandingan YOLO26s-Seg", fill="#18212D", font=label_font(23))
    detail = f"{MODE_LABELS[mode]} | confidence {conf:.2f}"
    if mode == "nms":
        detail += f" | IoU NMS {iou:.2f}"
    draw.text((gutter, 48), detail, fill="#52606D", font=label_font(16))
    for index, spec in enumerate(MODEL_SPECS):
        x = gutter + index * (panel_width + gutter)
        draw.text((x, 83), spec.label, fill="#18212D", font=label_font(20))
        output = outputs.get(spec.id)
        if output is None:
            draw.text((x + 20, 300), "Prediksi tidak tersedia", fill="#52606D", font=label_font(18))
            continue
        thumbnail = ImageOps.contain(render_overlay(image, output), (panel_width, panel_height))
        background = Image.new("RGB", (panel_width, panel_height), "black")
        background.paste(thumbnail, ((panel_width - thumbnail.width) // 2, (panel_height - thumbnail.height) // 2))
        sheet.paste(background, (x, 116))
        label = f"{output.detected_classes} | {len(output.detections)} deteksi"
        draw.text((x, 646), label, fill="#52606D", font=label_font(16))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue()


def research_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    summary, per_class, fp, errors = [], [], [], []
    for spec in MODEL_SPECS:
        for path, rows in ((spec.summary, summary), (spec.per_class, per_class), (spec.fp_summary, fp)):
            if not path.is_file():
                continue
            try:
                for record in pd.read_csv(path).to_dict("records"):
                    rows.append({**record, "Model": spec.label})
            except (ValueError, OSError) as exc:
                errors.append(f"{path.name}: {exc}")
    return pd.DataFrame(summary), pd.DataFrame(per_class), pd.DataFrame(fp), errors
