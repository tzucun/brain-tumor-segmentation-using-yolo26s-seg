"""Reproducible validation operating points, using Ultralytics box/mask matching."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from ultralytics.models.yolo.segment import SegmentationValidator

from inference import CLASS_NAMES, IMAGE_SIZE, MAX_DETECTIONS, MODEL_SPECS, PINNED_ULTRALYTICS, build_model, get_device


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(data_path: Path, split: str) -> dict[str, Any]:
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("data.yaml harus berupa mapping YOLO segmentation.")
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != dict(enumerate(CLASS_NAMES)):
        raise ValueError("Urutan kelas harus 0=glioma, 1=meningioma, 2=pituitary.")
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    if not root.is_dir():
        raise ValueError(f"Folder dataset tidak ditemukan: {root}. Sesuaikan path pada data.yaml.")
    values = data.get(split)
    values = values if isinstance(values, list) else [values]
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"Split {split} tidak ditemukan.")
    folders = [(root / value).resolve() for value in values]
    if any(not folder.is_dir() for folder in folders):
        raise ValueError("Eksperimen ini memerlukan split berupa folder images/val atau images/test.")
    if split == "val" and isinstance(data.get("test"), str):
        test_folder = (root / data["test"]).resolve()
        if test_folder in folders:
            raise ValueError("Split validation dan test menunjuk folder yang sama. Pisahkan data tuning dari test.")
    images = sorted({path for folder in folders for path in folder.rglob("*")
                     if path.suffix.lower() in {".jpg", ".jpeg", ".png"}})
    if not images:
        raise ValueError("Split tidak berisi citra JPG/PNG.")
    from ultralytics.data.utils import img2label_paths

    records, negatives, instances = [], 0, 0
    for image, label_name in zip(images, img2label_paths([str(p) for p in images])):
        label = Path(label_name)
        if not label.is_file():
            raise ValueError(f"Ground truth hilang: {label}. Label kosong harus berupa file .txt yang ada.")
        lines = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        for fields in lines:
            values = np.asarray(fields, dtype=float)
            if (len(values) < 7 or len(values) % 2 != 1 or not np.isfinite(values).all()
                    or values[0] not in range(len(CLASS_NAMES)) or (values[1:] < 0).any() or (values[1:] > 1).any()):
                raise ValueError(f"Anotasi bukan polygon YOLO segmentation yang valid: {label}")
        negatives += not lines
        instances += len(lines)
        records.append({"image": image.relative_to(root).as_posix(), "image_sha256": hash_file(image),
                        "label": label.relative_to(root).as_posix(), "label_sha256": hash_file(label)})
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"image_count": len(images), "negative_image_count": int(negatives), "ground_truth_instances": instances,
            "dataset_sha256": fingerprint, "data_yaml_sha256": hash_file(data_path), "files": records}


def operating_metrics(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision_micro": precision, "recall_micro": recall, "f1_micro": f1}


class ThresholdValidator(SegmentationValidator):
    def __init__(self, *args, confidence_grid: list[float], **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_grid = confidence_grid
        self.counts = {
            conf: {"box_tp": np.zeros(3, dtype=int), "mask_tp": np.zeros(3, dtype=int),
                   "predicted": np.zeros(3, dtype=int), "targets": np.zeros(3, dtype=int),
                   "negative_images": 0, "false_positive_images": 0, "false_detections_on_negatives": 0}
            for conf in confidence_grid
        }

    def collect_operating_points(self, pred: dict, target: dict) -> None:
        target_classes = target["cls"].long().cpu().numpy()
        target_counts = np.bincount(target_classes, minlength=3)
        for conf in self.confidence_grid:
            keep = pred["conf"] > conf
            filtered = {key: value[keep] for key, value in pred.items()}
            # Re-match after filtering: a removed low-confidence match must not hide a valid remaining match.
            correct = self._process_batch(filtered, target)
            classes = filtered["cls"].long().cpu().numpy()
            counters = self.counts[conf]
            counters["targets"] += target_counts
            counters["predicted"] += np.bincount(classes, minlength=3)
            for prefix, key in (("box", "tp"), ("mask", "tp_m")):
                counters[f"{prefix}_tp"] += np.bincount(classes[correct[key][:, 0]], minlength=3)
            if len(target_classes) == 0:
                counters["negative_images"] += 1
                counters["false_positive_images"] += int(len(classes) > 0)
                counters["false_detections_on_negatives"] += len(classes)

    def update_metrics(self, preds: list[dict], batch: dict) -> None:
        for index, pred in enumerate(preds):
            self.collect_operating_points(self._prepare_pred(pred), self._prepare_batch(index, batch))
        super().update_metrics(preds, batch)

    def operating_rows(self) -> tuple[list[dict], list[dict]]:
        rows, per_class = [], []
        for conf, counts in self.counts.items():
            row = {"confidence": conf, "match_iou": 0.50, "negative_images": counts["negative_images"],
                   "false_positive_images": counts["false_positive_images"],
                   "false_detections_on_negatives": counts["false_detections_on_negatives"],
                   "negative_fp_image_rate": counts["false_positive_images"] / counts["negative_images"]
                   if counts["negative_images"] else None}
            for prefix in ("box", "mask"):
                tp = int(counts[f"{prefix}_tp"].sum())
                stats = operating_metrics(tp, int(counts["predicted"].sum()) - tp, int(counts["targets"].sum()) - tp)
                row.update({f"{prefix}_{key}": value for key, value in stats.items()})
            rows.append(row)
            for class_id, name in enumerate(CLASS_NAMES):
                detail = {"confidence": conf, "class_id": class_id, "class_name": name}
                for prefix in ("box", "mask"):
                    tp = int(counts[f"{prefix}_tp"][class_id])
                    stats = operating_metrics(tp, int(counts["predicted"][class_id]) - tp,
                                              int(counts["targets"][class_id]) - tp)
                    detail.update({f"{prefix}_{key}": value for key, value in stats.items()})
                per_class.append(detail)
        return rows, per_class


def rank_common_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "confidence", "iou_nms"]
    ranked = frame.groupby(keys, dropna=False).agg(
        models=("model_id", "nunique"), mean_mask_f1_micro=("mask_f1_micro", "mean"),
        mean_mask_recall_micro=("mask_recall_micro", "mean"),
        mean_negative_fp_image_rate=("negative_fp_image_rate", "mean"),
    ).reset_index()
    ranked = ranked[ranked["models"] == len(MODEL_SPECS)].dropna(subset=["mean_mask_f1_micro"])
    return ranked.sort_values(["mode", "mean_mask_f1_micro", "mean_mask_recall_micro", "mean_negative_fp_image_rate", "confidence"],
                              ascending=[True, False, False, True, True], na_position="last")


def save_figures(frame: pd.DataFrame, directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from inference import MODEL_COLORS

    directory.mkdir(exist_ok=True)
    labels = {"mask_precision_micro": "Mask precision (micro)",
              "mask_recall_micro": "Mask recall (micro)", "mask_f1_micro": "Mask F1 (micro)"}
    for mode in frame["mode"].unique():
        part = frame[frame["mode"] == mode]
        values = sorted(part["iou_nms"].dropna().unique()) if mode == "nms" else [None]
        for iou in values:
            subset = part[np.isclose(part["iou_nms"], iou)] if iou is not None else part
            figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
            for axis, (metric, label) in zip(axes, labels.items()):
                for spec, color in zip(MODEL_SPECS, MODEL_COLORS):
                    selected = subset[subset["model_id"] == spec.id].sort_values("confidence")
                    axis.plot(selected["confidence"], selected[metric], marker="o", color=color, label=spec.label)
                axis.set(xlabel="Confidence threshold", ylabel=label, xlim=(0, 1), ylim=(0, 1.05))
                axis.grid(alpha=.2)
            axes[0].legend(fontsize=8, loc="lower left")
            suffix = f"{mode}_iou_{iou}" if iou is not None else mode
            figure.suptitle(f"YOLO26s-Seg | {suffix} | matching IoU = 0.50")
            for extension in ("png", "svg"):
                figure.savefig(directory / f"{suffix}_operating_points.{extension}", dpi=200)
            plt.close(figure)
        if mode == "nms":
            figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for axis, spec in zip(axes, MODEL_SPECS):
                table = part[part["model_id"] == spec.id].pivot(index="iou_nms", columns="confidence", values="mask_f1_micro").sort_index()
                heatmap = axis.imshow(table.to_numpy(), origin="lower", aspect="auto", vmin=0, vmax=1, cmap="cividis")
                axis.set_xticks(range(len(table.columns)), [f"{c:.2f}" for c in table.columns])
                axis.set_yticks(range(len(table.index)), [f"{v:.2f}" for v in table.index])
                axis.set(xlabel="Confidence threshold", ylabel="IoU NMS threshold", title=spec.label)
                for row in range(len(table.index)):
                    for column in range(len(table.columns)):
                        value = table.iloc[row, column]
                        axis.text(column, row, f"{value:.3f}", ha="center", va="center",
                                  color="black" if value > .55 else "white", fontsize=9)
            figure.colorbar(heatmap, ax=axes, label="Mask F1 (micro)", shrink=.8)
            for extension in ("png", "svg"):
                figure.savefig(directory / f"{mode}_mask_f1_heatmap.{extension}", dpi=200)
            plt.close(figure)


def run_experiment(args: argparse.Namespace) -> Path:
    confs = sorted(set(args.confidence))
    ious = sorted(set(args.iou))
    if any(not 0.001 < value < 1 for value in [*confs, *ious]):
        raise ValueError("Confidence dan IoU grid harus lebih besar dari 0.001 dan lebih kecil dari 1.")
    if args.split == "test" and (len(confs) != 1 or len(ious) != 1 or len(args.modes) != 1):
        raise ValueError("Evaluasi test hanya menerima satu konfigurasi yang telah ditetapkan; tuning menggunakan split val.")
    data_path = Path(args.data).resolve()
    inventory = inspect_dataset(data_path, args.split)
    output_dir = Path(args.output).resolve() if args.output else Path("experiments") / datetime.now().strftime("threshold_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = get_device() if args.device == "auto" else args.device
    if "," in str(device):
        raise ValueError("Gunakan satu perangkat untuk eksperimen ini.")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "demo" if args.demo else "research",
        "split": args.split, "data": str(data_path), "ultralytics": PINNED_ULTRALYTICS,
        "confidence_grid": confs, "iou_nms_grid": ious, "modes": args.modes,
        "candidate_confidence_floor": 0.001, "matching_iou": 0.50, "imgsz": args.imgsz,
        "rect": False, "batch": args.batch, "mask_ratio": 1, "device": str(device),
        "max_det": MAX_DETECTIONS, "overlap_mask": False, "seed": 42,
        "metric_aggregation": "micro across instances; Ultralytics class-aware matching, recomputed per confidence",
        "selection_criterion": "Common threshold per mode: mean mask F1 micro of both models; ties: recall, negative FP rate, lower confidence",
        **{key: value for key, value in inventory.items() if key != "files"},
        "models": [{"id": spec.id, "label": spec.label, "weights": str(spec.weights),
                    "sha256": hash_file(spec.weights)} for spec in MODEL_SPECS],
        "status": "running", "errors": [],
    }
    def save_manifest():
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    save_manifest()
    (output_dir / "dataset_manifest.json").write_text(json.dumps(inventory["files"], indent=2), encoding="utf-8")
    rows, class_rows, map_rows = [], [], []
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(42, deterministic=True)
    for mode in args.modes:
        for spec in MODEL_SPECS:
            for nms_iou in (ious if mode == "nms" else [None]):
                base = {"model_id": spec.id, "Model": spec.label, "mode": mode, "split": args.split, "iou_nms": nms_iou}
                print(f"\n{spec.label} | {mode} | IoU NMS {nms_iou}", flush=True)
                try:
                    model = build_model(str(spec.weights), mode)
                    validators = []
                    def validator_factory(**kwargs):
                        validator = ThresholdValidator(confidence_grid=confs, **kwargs)
                        validators.append(validator)
                        return validator

                    metrics = model.val(
                        validator=validator_factory, data=str(data_path), split=args.split,
                        conf=0.001, iou=nms_iou if nms_iou is not None else 0.70,
                        end2end=mode == "end2end", imgsz=args.imgsz, batch=args.batch, workers=0,
                        rect=False, mask_ratio=1, overlap_mask=False, device=device, max_det=MAX_DETECTIONS,
                        plots=False, save_json=False, verbose=False,
                        project=str(output_dir / "validation_runs"),
                        name=f"{spec.id}_{mode}_{nms_iou}", exist_ok=False,
                    )
                    if validators[0].seen != inventory["image_count"]:
                        raise ValueError("Validator melewati sebagian citra; hasil tidak boleh dianggap evaluasi lengkap.")
                    operating, classes = validators[0].operating_rows()
                    rows.extend({**base, **row} for row in operating)
                    class_rows.extend({**base, **row} for row in classes)
                    record = {**base, "confidence_floor": 0.001}
                    for prefix, metric in (("box", metrics.box), ("mask", metrics.seg)):
                        record.update({f"{prefix}_precision_max_f1": metric.mp, f"{prefix}_recall_max_f1": metric.mr,
                                       f"{prefix}_mAP50": metric.map50, f"{prefix}_mAP50_95": metric.map})
                    map_rows.append(record)
                    pd.DataFrame(rows).to_csv(output_dir / "threshold_metrics.csv", index=False)
                    pd.DataFrame(class_rows).to_csv(output_dir / "threshold_per_class.csv", index=False)
                    pd.DataFrame(map_rows).to_csv(output_dir / "map_metrics.csv", index=False)
                except Exception as exc:
                    metadata["errors"].append({**base, "error": str(exc)})
                    print(f"Konfigurasi gagal: {exc}", flush=True)
                save_manifest()
    metadata["status"] = "incomplete" if metadata["errors"] else "complete"
    save_manifest()
    if rows and args.split == "val" and not args.demo and not metadata["errors"]:
        rank_common_thresholds(pd.DataFrame(rows)).to_csv(output_dir / "ranked_validation.csv", index=False)
    if not rows or metadata["errors"]:
        raise RuntimeError(f"Eksperimen belum lengkap; periksa {output_dir / 'metadata.json'}")
    save_figures(pd.DataFrame(rows), output_dir / "figures")
    print(f"Hasil tersimpan: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="data.yaml YOLO segmentation lokal")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--confidence", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75, 0.90])
    parser.add_argument("--iou", nargs="+", type=float, default=[0.30, 0.50, 0.70])
    parser.add_argument("--modes", nargs="+", choices=["end2end", "nms"], default=["end2end", "nms"])
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--demo", action="store_true", help="Tandai uji alur, jangan menghasilkan ranking tuning penelitian")
    args = parser.parse_args()
    try:
        run_experiment(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
