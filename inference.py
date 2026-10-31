from __future__ import annotations

import hashlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
CLASS_COLORS = {"glioma": (225, 72, 89), "meningioma": (32, 166, 112), "pituitary": (67, 132, 245)}
MODEL_COLORS = ["#D97706", "#0D9488"]
MODE_LABELS = {"end2end": "End-to-end", "nms": "NMS"}
IMAGE_SIZE = 640
MAX_DETECTIONS = 300
PINNED_ULTRALYTICS = "8.4.126"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    weights: Path
    summary: Path
    per_class: Path
    fp_summary: Path


_WITHOUT = ROOT / "YOLO26s_seg/02_hasil_training/YOLO26s_seg_tanpa_no_tumor"
_WITH = ROOT / "YOLO26s_seg/03_hasil_evaluasi"
MODEL_SPECS = (
    ModelSpec(
        "tanpa_negative", "Tanpa negative samples", "Dilatih menggunakan citra bertumor.",
        _WITHOUT / "weights/best.pt",
        _WITHOUT / "03_hasil_evaluasi/yolo26s_seg_tanpa_no_tumor_official_test_metrics.csv",
        _WITHOUT / "03_hasil_evaluasi/yolo26s_seg_tanpa_no_tumor_per_class_metrics.csv",
        _WITHOUT / "03_hasil_evaluasi/yolo26s_seg_tanpa_no_tumor_no_tumor_fp_summary.csv",
    ),
    ModelSpec(
        "dengan_negative", "Dengan negative samples", "Dilatih dengan tambahan citra no_tumor berlabel kosong.",
        ROOT / "models/best.pt",
        _WITH / "YOLO26s_seg_official_test_metrics.csv",
        _WITH / "YOLO26s_seg_per_class_metrics.csv",
        _WITH / "YOLO26s_seg_no_tumor_fp_summary.csv",
    ),
)


@dataclass
class PredictionOutput:
    annotated_image: Image.Image
    detections: list[dict[str, Any]]
    inference_time: float
    mask_area_percent: float
    status: str
    primary_class: str
    detected_classes: str
    class_masks: dict[str, np.ndarray]
    masks_available: bool = True


def model_revision(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


@st.cache_data(show_spinner=False)
def file_sha256(path: str, revision: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_model_path() -> Path:
    return MODEL_SPECS[1].weights


def get_device() -> str | int:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def build_model(model_path: str, inference_mode: str = "end2end"):
    if inference_mode not in MODE_LABELS:
        raise ValueError("Mode inferensi harus end2end atau nms.")
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Bobot model tidak ditemukan: {path}")
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            f'Ultralytics belum tersedia. Jalankan "{sys.executable}" -m pip install -r requirements.txt'
        ) from exc
    if ultralytics.__version__ != PINNED_ULTRALYTICS:
        raise RuntimeError(
            f"Eksperimen ini menggunakan ultralytics=={PINNED_ULTRALYTICS}; "
            f"versi aktif {ultralytics.__version__}. Instal requirements.txt pada interpreter aktif."
        )
    model = YOLO(str(path))
    expected_names = {index: name for index, name in enumerate(CLASS_NAMES)}
    if model.task != "segment" or model.names != expected_names:
        raise ValueError("Bobot harus berupa model segmentasi dengan kelas glioma, meningioma, dan pituitary.")
    model.model.end2end = inference_mode == "end2end"
    model._app_inference_mode = inference_mode
    model._app_prediction_lock = threading.RLock()
    return model


@st.cache_resource(show_spinner=False, max_entries=4)
def load_model(model_path: str, inference_mode: str = "end2end", revision: str = ""):
    # End-to-end fusion removes one-to-many heads. Cache a separate checkpoint instance for each mode.
    return build_model(model_path, inference_mode)


def label_font(size: int = 14) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_overlay(image: Image.Image, output: PredictionOutput, opacity: float = 0.36,
                   show_boxes: bool = True) -> Image.Image:
    canvas = image.convert("RGBA")
    for name, mask in output.class_masks.items():
        if not np.any(mask):
            continue
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        layer = Image.new("RGBA", image.size, (*CLASS_COLORS[name], round(255 * opacity)))
        canvas.alpha_composite(Image.composite(layer, Image.new("RGBA", image.size), mask_image))
    if show_boxes:
        draw = ImageDraw.Draw(canvas)
        font = label_font(max(12, min(18, image.width // 32)))
        for item in output.detections:
            color = CLASS_COLORS[item["kelas"]]
            x1, y1, x2, y2 = item["bbox"]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.width // 220))
            text = f"{item['kelas']} {item['confidence']:.2f}"
            bounds = draw.textbbox((0, 0), text, font=font)
            tw, th = bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 8
            tx, ty = max(0, min(x1, image.width - tw)), max(0, y1 - th)
            draw.rectangle((tx, ty, tx + tw, ty + th), fill=color)
            draw.text((tx + 4, ty + 4 - bounds[1]), text, font=font, fill="white")
    return canvas.convert("RGB")


def annotate_result(image: Image.Image, result: Any, conf_threshold: float) -> PredictionOutput:
    image = image.convert("RGB")
    width, height = image.size
    class_masks = {name: np.zeros((height, width), dtype=bool) for name in CLASS_NAMES}
    detections: list[dict[str, Any]] = []
    boxes, masks = getattr(result, "boxes", None), getattr(result, "masks", None)
    masks_available = True
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        mask_values = None if masks is None else masks.data.cpu().numpy()
        for index, (box, class_id, score) in enumerate(zip(xyxy, class_ids, scores)):
            if float(score) <= conf_threshold:
                continue
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"ID kelas tidak dikenal: {class_id}")
            name = CLASS_NAMES[class_id]
            area = None
            if mask_values is not None and index < len(mask_values):
                mask_image = Image.fromarray((mask_values[index] > 0.5).astype(np.uint8) * 255)
                if mask_image.size != image.size:
                    raise ValueError("Ukuran mask tidak sesuai citra; gunakan retina_masks=True.")
                mask = np.asarray(mask_image) > 0
                class_masks[name] |= mask
                area = int(mask.sum())
            else:
                masks_available = False
            bounds = [max(0, min(round(float(v)), width if i % 2 == 0 else height)) for i, v in enumerate(box)]
            detections.append({"kelas": name, "confidence": float(score), "bbox": bounds, "luas_mask_px": area})
    union = np.logical_or.reduce(list(class_masks.values()))
    best = max(detections, key=lambda item: item["confidence"], default=None)
    output = PredictionOutput(
        annotated_image=image, detections=detections, inference_time=0.0,
        mask_area_percent=float(union.mean() * 100) if masks_available else float("nan"),
        status="Prediksi tumor ditemukan" if detections else "Tidak ada prediksi tumor",
        primary_class=best["kelas"] if best else "-",
        detected_classes=", ".join(sorted({d["kelas"] for d in detections})) or "-",
        class_masks=class_masks, masks_available=masks_available,
    )
    output.annotated_image = render_overlay(image, output)
    return output


def predict_image(model: Any, image: Image.Image, conf: float, iou: float | None,
                  device: str | int, inference_mode: str = "end2end") -> PredictionOutput:
    if not 0 < conf < 1:
        raise ValueError("Confidence harus berada di antara 0 dan 1.")
    if inference_mode == "nms" and (iou is None or not 0 < iou < 1):
        raise ValueError("IoU NMS harus berada di antara 0 dan 1.")
    if getattr(model, "_app_inference_mode", None) != inference_mode:
        raise ValueError("Mode berbeda dari instance bobot. Muat model terpisah untuk tiap mode inferensi.")
    with model._app_prediction_lock:
        start = time.perf_counter()
        # End-to-end outputs bypass NMS; its IoU threshold is not an application setting.
        options = {"iou": iou} if inference_mode == "nms" else {}
        results = model.predict(
            source=image.convert("RGB"), conf=conf, device=device, **options,
            imgsz=IMAGE_SIZE, rect=False, max_det=MAX_DETECTIONS,
            end2end=inference_mode == "end2end", retina_masks=True, verbose=False, save=False,
        )
        elapsed = time.perf_counter() - start
        active = bool(getattr(model.predictor.model, "end2end", False))
        if active != (inference_mode == "end2end"):
            raise RuntimeError("Mode inferensi aktual berbeda dari konfigurasi eksperimen.")
        output = annotate_result(image, results[0], conf)
        output.inference_time = elapsed
        return output
