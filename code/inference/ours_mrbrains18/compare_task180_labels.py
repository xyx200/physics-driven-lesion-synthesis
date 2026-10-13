#!/usr/bin/env python3
"""Validate and compare Task180 predictions with MRBrainS18 references."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np


LABEL_NAMES = {
    0: "Background",
    1: "GM",
    2: "BG",
    3: "WM",
    4: "Lesion",
    5: "Ventricles",
    6: "Cerebellum",
    7: "Brainstem",
}
LABEL_COLORS = [
    (0.0, 0.0, 0.0, 0.0),
    (0.20, 0.75, 0.35, 0.72),
    (1.00, 0.62, 0.10, 0.78),
    (0.16, 0.48, 0.95, 0.70),
    (0.95, 0.10, 0.22, 0.92),
    (0.05, 0.85, 0.90, 0.78),
    (0.98, 0.86, 0.15, 0.78),
    (0.62, 0.26, 0.86, 0.80),
]
DIFF_COLORS = [
    (0.0, 0.0, 0.0, 0.0),
    (0.95, 0.10, 0.15, 0.92),  # missed foreground
    (0.05, 0.55, 1.00, 0.92),  # extra foreground
    (1.00, 0.82, 0.05, 0.92),  # wrong nonzero class
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-summary", type=Path)
    return parser.parse_args()


def load_image(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    return image, np.asanyarray(image.dataobj)


def case_ids(folder: Path) -> list[str]:
    return sorted(path.name.removesuffix(".nii.gz") for path in folder.glob("MRB18_*.nii.gz"))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size else finite
    if not values.size:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        high = low + 1
    return np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)


def slice_view(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    slices = [slice(None)] * 3
    slices[axis] = int(np.clip(index, 0, array.shape[axis] - 1))
    return np.rot90(array[tuple(slices)])


def selection(gt: np.ndarray, pred: np.ndarray) -> list[tuple[str, int, int]]:
    disagreement = gt != pred
    lesion_union = (gt == 4) | (pred == 4)
    foreground = (gt > 0) | (pred > 0)
    points = np.argwhere(disagreement)
    if not points.size:
        points = np.argwhere(foreground)
    center = np.rint(np.median(points, axis=0)).astype(int) if points.size else np.asarray(gt.shape) // 2
    disagree_z = int(np.argmax(disagreement.sum(axis=(0, 1))))
    lesion_z = int(np.argmax(lesion_union.sum(axis=(0, 1)))) if lesion_union.any() else int(center[2])
    return [
        ("Axial max disagreement", 2, disagree_z),
        ("Axial max lesion union", 2, lesion_z),
        ("Coronal disagreement center", 1, int(center[1])),
        ("Sagittal disagreement center", 0, int(center[0])),
    ]


def make_montage(
    case_id: str,
    t1: np.ndarray,
    flair: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    destination: Path,
) -> None:
    t1n, flairn = normalize(t1), normalize(flair)
    diff = np.zeros(gt.shape, dtype=np.uint8)
    diff[(gt > 0) & (pred == 0)] = 1
    diff[(gt == 0) & (pred > 0)] = 2
    diff[(gt > 0) & (pred > 0) & (gt != pred)] = 3
    label_cmap = ListedColormap(LABEL_COLORS)
    diff_cmap = ListedColormap(DIFF_COLORS)
    views = selection(gt, pred)
    figure, axes = plt.subplots(4, 5, figsize=(18, 13), constrained_layout=True)
    for row, (title, axis, index) in enumerate(views):
        bases = (t1n, flairn)
        axes[row, 0].imshow(slice_view(bases[0], axis, index), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(slice_view(bases[1], axis, index), cmap="gray", vmin=0, vmax=1)
        for column, labels in ((2, gt), (3, pred)):
            axes[row, column].imshow(slice_view(flairn, axis, index), cmap="gray", vmin=0, vmax=1)
            overlay = slice_view(labels, axis, index)
            axes[row, column].imshow(
                np.ma.masked_where(overlay == 0, overlay),
                cmap=label_cmap,
                vmin=0,
                vmax=7,
                interpolation="nearest",
            )
        axes[row, 4].imshow(slice_view(flairn, axis, index), cmap="gray", vmin=0, vmax=1)
        overlay_diff = slice_view(diff, axis, index)
        axes[row, 4].imshow(
            np.ma.masked_where(overlay_diff == 0, overlay_diff),
            cmap=diff_cmap,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        for column, heading in enumerate(("T1", "FLAIR", "Manual reference", "Task180 prediction", "Difference")):
            axes[row, column].set_title(f"{heading}\n{title}, index {index}", fontsize=9)
            axes[row, column].axis("off")
    counts = np.bincount(diff.ravel(), minlength=4)
    figure.suptitle(
        f"{case_id} | missed={counts[1]:,} extra={counts[2]:,} class-swap={counts[3]:,}",
        fontsize=14,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    plt.close(figure)


def make_contact_sheets(montage_dir: Path, destination: Path, ids: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(ids), 10), start=1):
        subset = ids[start : start + 10]
        figure, axes = plt.subplots(2, 5, figsize=(25, 12), constrained_layout=True)
        for axis, case_id in zip(axes.flat, subset):
            axis.imshow(plt.imread(montage_dir / f"{case_id}.png"))
            axis.set_title(case_id)
            axis.axis("off")
        for axis in axes.flat[len(subset) :]:
            axis.axis("off")
        figure.savefig(destination / f"contact_sheet_{page:02d}.png", dpi=100)
        plt.close(figure)


def normalize_evaluator(summary_path: Path, destination: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for item in payload.get("results", {}).get("all", []):
        prediction = item.get("test") or item.get("prediction") or ""
        case_id = Path(prediction).name.removesuffix(".nii.gz")
        for label in range(1, 8):
            metrics = item.get(str(label), {})
            row: dict[str, object] = {"case_id": case_id, "label": label, "label_name": LABEL_NAMES[label]}
            row.update({str(key): value for key, value in metrics.items()})
            rows.append(row)
    write_csv(destination / "evaluator_per_case.csv", rows)
    mean = payload.get("results", {}).get("mean", {})
    (destination / "evaluator_mean.json").write_text(
        json.dumps(mean, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"source": str(summary_path), "row_count": len(rows)}


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    predictions = args.predictions.resolve()
    output = args.output.resolve()
    references = source / "labels"
    images = source / "images"
    ids = case_ids(references)
    predicted_ids = case_ids(predictions)
    if ids != [f"MRB18_{index:03d}" for index in range(1, 31)]:
        raise RuntimeError(f"Unexpected reference IDs: {ids}")
    if predicted_ids != ids:
        raise RuntimeError(f"Prediction/reference mismatch: predictions={predicted_ids}, references={ids}")

    metrics_dir = output / "metrics"
    montage_dir = output / "qc" / "montages"
    contact_dir = output / "qc" / "contact_sheets"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    confusion = np.zeros((8, 8), dtype=np.int64)
    per_label: list[dict[str, object]] = []
    per_case: list[dict[str, object]] = []

    for case_id in ids:
        gt_image, gt_raw = load_image(references / f"{case_id}.nii.gz")
        pred_image, pred_raw = load_image(predictions / f"{case_id}.nii.gz")
        if gt_image.shape != pred_image.shape or not np.allclose(gt_image.affine, pred_image.affine, atol=1e-5):
            raise RuntimeError(f"{case_id}: prediction geometry mismatch")
        if not np.all(np.isfinite(pred_raw)) or not np.all(np.isclose(pred_raw, np.rint(pred_raw))):
            raise RuntimeError(f"{case_id}: prediction is non-finite or non-integral")
        gt = np.rint(gt_raw).astype(np.uint8)
        pred = np.rint(pred_raw).astype(np.uint8)
        if not set(np.unique(gt)).issubset(set(range(8))) or not set(np.unique(pred)).issubset(set(range(8))):
            raise RuntimeError(f"{case_id}: labels outside 0-7")
        np.add.at(confusion, (gt.ravel(), pred.ravel()), 1)
        voxel_volume = float(np.prod(gt_image.header.get_zooms()[:3]))
        lesion_gt, lesion_pred = gt == 4, pred == 4
        lesion_tp = int(np.count_nonzero(lesion_gt & lesion_pred))
        lesion_fn = int(np.count_nonzero(lesion_gt & ~lesion_pred))
        lesion_fp = int(np.count_nonzero(~lesion_gt & lesion_pred))
        per_case.append(
            {
                "case_id": case_id,
                "agreement_fraction": float(np.mean(gt == pred)),
                "disagreement_voxels": int(np.count_nonzero(gt != pred)),
                "lesion_tp": lesion_tp,
                "lesion_fn": lesion_fn,
                "lesion_fp": lesion_fp,
                "lesion_reference_empty": bool(not lesion_gt.any()),
                "lesion_prediction_empty": bool(not lesion_pred.any()),
            }
        )
        for label in range(1, 8):
            gt_mask, pred_mask = gt == label, pred == label
            gt_count, pred_count = int(gt_mask.sum()), int(pred_mask.sum())
            intersection = int(np.count_nonzero(gt_mask & pred_mask))
            if gt_count == 0 and pred_count == 0:
                presence = "both_empty"
                dice: float | None = None
            elif gt_count == 0:
                presence = "reference_empty_prediction_nonempty"
                dice = 0.0
            elif pred_count == 0:
                presence = "reference_nonempty_prediction_empty"
                dice = 0.0
            else:
                presence = "both_nonempty"
                dice = 2.0 * intersection / (gt_count + pred_count)
            per_label.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "presence_state": presence,
                    "reference_voxels": gt_count,
                    "prediction_voxels": pred_count,
                    "reference_volume_mm3": gt_count * voxel_volume,
                    "prediction_volume_mm3": pred_count * voxel_volume,
                    "volume_delta_mm3": (pred_count - gt_count) * voxel_volume,
                    "intersection_voxels": intersection,
                    "dice_nonempty": "" if dice is None else dice,
                }
            )
        _, t1 = load_image(images / f"{case_id}_0000.nii.gz")
        _, flair = load_image(images / f"{case_id}_0001.nii.gz")
        make_montage(case_id, t1, flair, gt, pred, montage_dir / f"{case_id}.png")

    write_csv(metrics_dir / "per_case.csv", per_case)
    write_csv(metrics_dir / "per_case_per_label.csv", per_label)
    confusion_rows = [
        {"reference_label": reference, **{f"pred_{prediction}": int(confusion[reference, prediction]) for prediction in range(8)}}
        for reference in range(8)
    ]
    write_csv(metrics_dir / "confusion_matrix.csv", confusion_rows)
    make_contact_sheets(montage_dir, contact_dir, ids)
    evaluator = None
    if args.evaluator_summary:
        evaluator = normalize_evaluator(args.evaluator_summary.resolve(), metrics_dir)
    summary = {
        "case_count": len(ids),
        "labels": LABEL_NAMES,
        "montage_count": len(list(montage_dir.glob("MRB18_*.png"))),
        "contact_sheet_count": len(list(contact_dir.glob("contact_sheet_*.png"))),
        "total_disagreement_voxels": int(sum(row["disagreement_voxels"] for row in per_case)),
        "evaluator": evaluator,
    }
    (metrics_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
