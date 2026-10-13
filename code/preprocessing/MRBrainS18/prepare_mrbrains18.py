#!/usr/bin/env python3
"""Prepare the released MRBrainS18 data for paper-compatible evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
SOURCE_TO_TARGET = np.asarray([0, 1, 2, 3, 4, 1, 5, 6, 7, 4, 4], dtype=np.uint8)
PILOT_IDS = ("1", "13", "24")
EXPECTED_SHAPE = (240, 240, 155)
EXPECTED_ZOOMS = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Case:
    subject_id: str
    source_split: str
    case_dir: Path

    @property
    def output_id(self) -> str:
        return f"MRB18_{int(self.subject_id):03d}"

    @property
    def t1(self) -> Path:
        return self.case_dir / "pre" / "reg_T1.nii.gz"

    @property
    def flair(self) -> Path:
        return self.case_dir / "pre" / "FLAIR.nii.gz"

    @property
    def segmentation(self) -> Path:
        return self.case_dir / "segm.nii.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("preflight", "pilot", "all", "validate-only"),
        help="Execution phase. Run in the listed order.",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hdbet-device", default="0")
    parser.add_argument(
        "--hdbet-executable",
        type=Path,
        default=Path("/path/to/environment/bin/hd-bet"),
    )
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def discover_cases(source: Path) -> list[Case]:
    cases: list[Case] = []
    for split in ("training", "test"):
        split_dir = source / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing source split: {split_dir}")
        for case_dir in split_dir.iterdir():
            if case_dir.is_dir() and case_dir.name.isdigit():
                cases.append(Case(case_dir.name, split, case_dir))
    cases.sort(key=lambda case: int(case.subject_id))
    if len(cases) != 30:
        raise RuntimeError(f"Expected exactly 30 cases, found {len(cases)}")
    if len({case.subject_id for case in cases}) != 30:
        raise RuntimeError("Duplicate subject IDs across source splits")
    return cases


def ensure_layout(output: Path) -> None:
    for name in (
        "images",
        "labels",
        "masks",
        "transforms",
        "qc",
        "qc/montages",
        "logs",
        "code",
        "work",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)


def nifti_details(path: Path) -> dict[str, object]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    return {
        "path": str(path),
        "shape": list(image.shape),
        "zooms": [float(value) for value in image.header.get_zooms()[:3]],
        "axcodes": list(nib.aff2axcodes(image.affine)),
        "affine": image.affine.tolist(),
        "dtype": str(data.dtype),
        "finite": bool(np.isfinite(data).all()),
        "nonzero_voxels": int(np.count_nonzero(data)),
    }


def integer_segmentation(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        raise RuntimeError(f"Non-integral segmentation values: {path}")
    return image, rounded.astype(np.int16)


def source_label_counts(data: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(data, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def map_labels(data: np.ndarray, subject_id: str) -> np.ndarray:
    labels = set(int(value) for value in np.unique(data))
    unexpected = labels.difference(range(11))
    if unexpected:
        raise RuntimeError(f"Case {subject_id}: unexpected source labels {sorted(unexpected)}")
    if np.count_nonzero(data == 10):
        raise RuntimeError(f"Case {subject_id}: source label 10 is nonzero")
    mapped = SOURCE_TO_TARGET[data]
    expected_lesion = int(np.count_nonzero((data == 4) | (data == 9)))
    actual_lesion = int(np.count_nonzero(mapped == 4))
    if expected_lesion != actual_lesion:
        raise RuntimeError(
            f"Case {subject_id}: lesion mapping mismatch {actual_lesion} != {expected_lesion}"
        )
    expected_gm = int(np.count_nonzero((data == 1) | (data == 5)))
    actual_gm = int(np.count_nonzero(mapped == 1))
    if expected_gm != actual_gm:
        raise RuntimeError(
            f"Case {subject_id}: GM mapping mismatch {actual_gm} != {expected_gm}"
        )
    return mapped


def template_3d(template: Path, output: Path) -> Path:
    destination = output / "work" / "sri24_template_3d.nii.gz"
    if destination.exists():
        return destination
    image = nib.load(str(template))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"Template is not 3D or singleton-4D: {image.shape}")
    result = nib.Nifti1Image(data.astype(np.float32), image.affine, image.header)
    result.set_qform(image.affine, code=1)
    result.set_sform(image.affine, code=1)
    nib.save(result, str(destination))
    return destination


def run_preflight(source: Path, output: Path, template: Path) -> None:
    cases = discover_cases(source)
    template_path = template_3d(template, output)
    template_info = nifti_details(template_path)
    if tuple(template_info["shape"]) != EXPECTED_SHAPE:
        raise RuntimeError(f"Unexpected SRI24 shape: {template_info['shape']}")
    if not np.allclose(template_info["zooms"], EXPECTED_ZOOMS, atol=1e-6):
        raise RuntimeError(f"Unexpected SRI24 spacing: {template_info['zooms']}")

    rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {str(template): sha256(template)}
    label_totals: dict[int, int] = {}
    label_case_counts: dict[int, int] = {}
    max_affine_difference = 0.0

    for case in cases:
        for path in (case.t1, case.flair, case.segmentation):
            if not path.is_file():
                raise FileNotFoundError(f"Missing required input: {path}")
            input_hashes[str(path)] = sha256(path)
        t1 = nib.load(str(case.t1))
        flair = nib.load(str(case.flair))
        segmentation_image, segmentation = integer_segmentation(case.segmentation)
        if t1.shape != flair.shape or t1.shape != segmentation_image.shape:
            raise RuntimeError(f"Case {case.subject_id}: input shapes do not match")
        if not np.isfinite(np.asanyarray(t1.dataobj)).all():
            raise RuntimeError(f"Case {case.subject_id}: non-finite T1 values")
        if not np.isfinite(np.asanyarray(flair.dataobj)).all():
            raise RuntimeError(f"Case {case.subject_id}: non-finite FLAIR values")
        difference = max(
            float(np.max(np.abs(t1.affine - flair.affine))),
            float(np.max(np.abs(t1.affine - segmentation_image.affine))),
        )
        max_affine_difference = max(max_affine_difference, difference)
        if difference > 0.001:
            raise RuntimeError(
                f"Case {case.subject_id}: affine mismatch {difference:.6g} mm exceeds 0.001 mm"
            )
        counts = source_label_counts(segmentation)
        if counts.get(10, 0):
            raise RuntimeError(f"Case {case.subject_id}: source label 10 is nonzero")
        map_labels(segmentation, case.subject_id)
        for label, count in counts.items():
            label_totals[label] = label_totals.get(label, 0) + count
            label_case_counts[label] = label_case_counts.get(label, 0) + 1
        rows.append(
            {
                "subject_id": case.subject_id,
                "output_id": case.output_id,
                "source_split": case.source_split,
                "shape": "x".join(str(value) for value in t1.shape),
                "spacing": "x".join(f"{value:.9g}" for value in t1.header.get_zooms()[:3]),
                "max_affine_difference_mm": difference,
                "source_labels": json.dumps(counts, sort_keys=True),
            }
        )

    manifest = {
        "created_unix": time.time(),
        "source": str(source),
        "output": str(output),
        "template": str(template),
        "case_count": len(cases),
        "source_split_counts": {
            "training": sum(case.source_split == "training" for case in cases),
            "test": sum(case.source_split == "test" for case in cases),
        },
        "source_to_target": {
            str(index): int(value) for index, value in enumerate(SOURCE_TO_TARGET)
        },
        "label_names": {str(key): value for key, value in LABEL_NAMES.items()},
        "source_label_totals": {str(key): value for key, value in sorted(label_totals.items())},
        "source_label_case_counts": {
            str(key): value for key, value in sorted(label_case_counts.items())
        },
        "max_affine_difference_mm": max_affine_difference,
        "template_info": template_info,
        "input_sha256": input_hashes,
    }
    if manifest["source_label_totals"].get("10", 0):
        raise RuntimeError("Source label 10 is nonzero")
    atomic_json(output / "qc" / "preflight_manifest.json", manifest)
    write_csv(output / "qc" / "case_manifest.csv", rows)
    print(
        f"Preflight passed: {len(cases)} cases; label9={label_totals.get(9, 0)} voxels; "
        f"label10={label_totals.get(10, 0)} voxels; max affine delta="
        f"{max_affine_difference:.6g} mm"
    )


def case_paths(case: Case, output: Path) -> dict[str, Path]:
    prefix = case.output_id
    work = output / "work" / prefix
    return {
        "work": work,
        "t1_bet": work / f"{prefix}_t1_bet.nii.gz",
        "t1_mask": work / f"{prefix}_t1_bet_mask.nii.gz",
        "flair_bet": work / f"{prefix}_flair_bet.nii.gz",
        "label_mapped": work / f"{prefix}_mapped_native.nii.gz",
        "t1_output": output / "images" / f"{prefix}_0000.nii.gz",
        "flair_output": output / "images" / f"{prefix}_0001.nii.gz",
        "label_output": output / "labels" / f"{prefix}.nii.gz",
        "mask_output": output / "masks" / f"{prefix}.nii.gz",
        "transform_output": output / "transforms" / f"{prefix}_0GenericAffine.mat",
        "case_stats": output / "qc" / f"{prefix}_stats.json",
        "montage": output / "qc" / "montages" / f"{prefix}.png",
    }


def completed(paths: dict[str, Path]) -> bool:
    required = (
        "t1_output",
        "flair_output",
        "label_output",
        "mask_output",
        "transform_output",
        "case_stats",
        "montage",
    )
    present = [paths[key].exists() for key in required]
    if any(present) and not all(present):
        missing = [key for key, value in zip(required, present) if not value]
        raise RuntimeError(f"Partial case output exists; refusing overwrite. Missing: {missing}")
    return all(present)


def prepare_native(
    case: Case,
    output: Path,
    hdbet_executable: Path,
    hdbet_device: str,
) -> None:
    paths = case_paths(case, output)
    if completed(paths):
        return
    paths["work"].mkdir(parents=True, exist_ok=True)
    native_complete = all(
        paths[key].exists() for key in ("t1_bet", "t1_mask", "flair_bet", "label_mapped")
    )
    native_partial = any(
        paths[key].exists() for key in ("t1_bet", "t1_mask", "flair_bet", "label_mapped")
    )
    if native_partial and not native_complete:
        raise RuntimeError(f"Case {case.subject_id}: partial native work exists")
    if native_complete:
        return
    if not hdbet_executable.is_file():
        raise FileNotFoundError(f"HD-BET executable not found: {hdbet_executable}")

    command = [
        str(hdbet_executable),
        "-i",
        str(case.t1),
        "-o",
        str(paths["t1_bet"]),
        "-mode",
        "accurate",
        "-device",
        str(hdbet_device),
        "-tta",
        "1",
        "-pp",
        "1",
        "-s",
        "1",
        "--overwrite_existing",
        "0",
        "-b",
        "1",
    ]
    print(f"Case {case.subject_id}: HD-BET")
    hdbet_environment = os.environ.copy()
    # Keep downloaded model parameters inside the derivative, not in the user's
    # home directory or the conda environment.
    hdbet_environment["HOME"] = str(output / "code")
    subprocess.run(command, check=True, env=hdbet_environment)
    if not paths["t1_bet"].is_file():
        raise RuntimeError(f"Case {case.subject_id}: HD-BET did not create the brain image")
    if not paths["t1_mask"].is_file():
        candidates = sorted(paths["work"].glob("*mask*.nii.gz"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Case {case.subject_id}: cannot identify HD-BET mask: {candidates}"
            )
        candidates[0].replace(paths["t1_mask"])

    flair_image = nib.load(str(case.flair))
    flair = np.asanyarray(flair_image.dataobj).astype(np.float32)
    mask = np.asanyarray(nib.load(str(paths["t1_mask"])).dataobj) > 0
    if flair.shape != mask.shape or not np.any(mask):
        raise RuntimeError(f"Case {case.subject_id}: invalid HD-BET mask geometry")
    flair_bet = nib.Nifti1Image(flair * mask, flair_image.affine, flair_image.header)
    flair_bet.set_data_dtype(np.float32)
    nib.save(flair_bet, str(paths["flair_bet"]))

    segmentation_image, segmentation = integer_segmentation(case.segmentation)
    mapped = map_labels(segmentation, case.subject_id)
    mapped_image = nib.Nifti1Image(
        mapped.astype(np.uint8), segmentation_image.affine, segmentation_image.header
    )
    mapped_image.set_data_dtype(np.uint8)
    nib.save(mapped_image, str(paths["label_mapped"]))


def class_volumes(data: np.ndarray, zooms: Iterable[float]) -> dict[int, float]:
    voxel_volume = float(np.prod(tuple(zooms)))
    return {
        label: float(np.count_nonzero(data == label) * voxel_volume)
        for label in range(1, 8)
    }


def make_montage(t1_path: Path, flair_path: Path, label_path: Path, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t1 = np.asanyarray(nib.load(str(t1_path)).dataobj)
    flair = np.asanyarray(nib.load(str(flair_path)).dataobj)
    label = np.asanyarray(nib.load(str(label_path)).dataobj)
    foreground = np.argwhere(label > 0)
    center = (
        np.rint(np.median(foreground, axis=0)).astype(int)
        if foreground.size
        else np.asarray(label.shape) // 2
    )
    views = (
        ("sagittal", 0, int(center[0])),
        ("coronal", 1, int(center[1])),
        ("axial", 2, int(center[2])),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for column, (title, axis, index) in enumerate(views):
        slices = [slice(None)] * 3
        slices[axis] = index
        key = tuple(slices)
        for row, image in enumerate((t1, flair)):
            base = np.rot90(image[key])
            overlay = np.rot90(label[key])
            axes[row, column].imshow(base, cmap="gray")
            axes[row, column].imshow(
                np.ma.masked_where(overlay == 0, overlay),
                cmap="turbo",
                vmin=1,
                vmax=7,
                alpha=0.38,
                interpolation="nearest",
            )
            axes[row, column].set_title(f"{'T1' if row == 0 else 'FLAIR'} {title} {index}")
            axes[row, column].axis("off")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def register_case(payload: tuple[str, str, str, str]) -> dict[str, object]:
    source_string, output_string, template_string, subject_id = payload
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    import ants

    source = Path(source_string)
    output = Path(output_string)
    template = Path(template_string)
    case = next(case for case in discover_cases(source) if case.subject_id == subject_id)
    paths = case_paths(case, output)
    if completed(paths):
        return {"subject_id": subject_id, "status": "skipped"}

    fixed = ants.image_read(str(template))
    moving_t1 = ants.image_read(str(paths["t1_bet"]))
    moving_flair = ants.image_read(str(paths["flair_bet"]))
    moving_label = ants.image_read(str(paths["label_mapped"]))
    moving_mask = ants.image_read(str(paths["t1_mask"]))
    registration = ants.registration(fixed=fixed, moving=moving_t1, type_of_transform="Rigid")
    transforms = registration["fwdtransforms"]
    affine_transforms = [Path(path) for path in transforms if str(path).endswith(".mat")]
    if len(affine_transforms) != 1:
        raise RuntimeError(f"Case {subject_id}: expected one rigid affine, got {transforms}")
    shutil.copy2(affine_transforms[0], paths["transform_output"])
    ants.image_write(registration["warpedmovout"], str(paths["t1_output"]))
    warped_flair = ants.apply_transforms(
        fixed=fixed,
        moving=moving_flair,
        transformlist=transforms,
        interpolator="linear",
    )
    warped_label = ants.apply_transforms(
        fixed=fixed,
        moving=moving_label,
        transformlist=transforms,
        interpolator="nearestNeighbor",
    )
    warped_mask = ants.apply_transforms(
        fixed=fixed,
        moving=moving_mask,
        transformlist=transforms,
        interpolator="nearestNeighbor",
    )
    ants.image_write(warped_flair, str(paths["flair_output"]))
    ants.image_write(warped_label, str(paths["label_output"]))
    ants.image_write(warped_mask, str(paths["mask_output"]))

    native_image, native = integer_segmentation(paths["label_mapped"])
    output_image, registered = integer_segmentation(paths["label_output"])
    output_labels = set(int(value) for value in np.unique(registered))
    if not output_labels.issubset(range(8)):
        raise RuntimeError(f"Case {subject_id}: invalid registered labels {sorted(output_labels)}")
    native_volumes = class_volumes(native, native_image.header.get_zooms()[:3])
    output_volumes = class_volumes(registered, output_image.header.get_zooms()[:3])
    relative_differences = {}
    threshold_flags = {}
    for label in range(1, 8):
        before = native_volumes[label]
        after = output_volumes[label]
        difference = abs(after - before) / before if before else (0.0 if after == 0 else float("inf"))
        relative_differences[label] = difference
        threshold = 0.15 if label == 4 else 0.05
        threshold_flags[label] = bool(difference > threshold)
    source_segmentation = integer_segmentation(case.segmentation)[1]
    stats = {
        "subject_id": subject_id,
        "output_id": case.output_id,
        "source_split": case.source_split,
        "source_label_counts": source_label_counts(source_segmentation),
        "native_mapped_label_counts": source_label_counts(native),
        "output_label_counts": source_label_counts(registered),
        "native_volumes_mm3": native_volumes,
        "output_volumes_mm3": output_volumes,
        "relative_volume_difference": relative_differences,
        "volume_threshold_flags": threshold_flags,
    }
    atomic_json(paths["case_stats"], stats)
    make_montage(paths["t1_output"], paths["flair_output"], paths["label_output"], paths["montage"])
    return {"subject_id": subject_id, "status": "processed"}


def selected_cases(source: Path, phase: str) -> list[Case]:
    cases = discover_cases(source)
    if phase == "pilot":
        return [case for case in cases if case.subject_id in PILOT_IDS]
    return cases


def run_processing(args: argparse.Namespace) -> None:
    manifest_path = args.output / "qc" / "preflight_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Preflight manifest is missing; run phase preflight first")
    cases = selected_cases(args.source, args.phase)
    template_path = template_3d(args.template, args.output)
    for case in cases:
        prepare_native(case, args.output, args.hdbet_executable, args.hdbet_device)
    payloads = [
        (str(args.source), str(args.output), str(template_path), case.subject_id)
        for case in cases
    ]
    workers = min(max(1, args.workers), 4)
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(register_case, payload): payload[3] for payload in payloads}
        for future in as_completed(futures):
            subject_id = futures[future]
            try:
                result = future.result()
                print(f"Case {subject_id}: {result['status']}")
            except Exception as error:
                errors.append(f"Case {subject_id}: {error}")
                print(errors[-1], file=sys.stderr)
    if errors:
        raise RuntimeError("Processing failures:\n" + "\n".join(errors))
    validate_outputs(args.source, args.output, template_path, pilot=args.phase == "pilot")


def validate_outputs(source: Path, output: Path, template: Path, pilot: bool = False) -> None:
    cases = selected_cases(source, "pilot" if pilot else "all")
    template_image = nib.load(str(template))
    template_affine = template_image.affine
    failures: list[str] = []
    stats_rows: list[dict[str, object]] = []
    output_hashes: dict[str, str] = {}
    source_label9_cases: list[str] = []
    volume_threshold_flags: list[dict[str, object]] = []

    for case in cases:
        paths = case_paths(case, output)
        try:
            if not completed(paths):
                raise RuntimeError("required outputs are incomplete")
            for key in ("t1_output", "flair_output", "label_output", "mask_output"):
                image = nib.load(str(paths[key]))
                if image.shape != EXPECTED_SHAPE:
                    raise RuntimeError(f"{key} shape is {image.shape}")
                if not np.allclose(image.header.get_zooms()[:3], EXPECTED_ZOOMS, atol=1e-6):
                    raise RuntimeError(f"{key} spacing is {image.header.get_zooms()[:3]}")
                if not np.allclose(image.affine, template_affine, atol=1e-5):
                    raise RuntimeError(f"{key} affine differs from template")
                data = np.asanyarray(image.dataobj)
                if not np.isfinite(data).all() or not np.any(data):
                    raise RuntimeError(f"{key} is non-finite or empty")
                output_hashes[str(paths[key])] = sha256(paths[key])
            _, label = integer_segmentation(paths["label_output"])
            values = set(int(value) for value in np.unique(label))
            if not values.issubset(range(8)):
                raise RuntimeError(f"invalid output labels {sorted(values)}")
            _, source_segmentation = integer_segmentation(case.segmentation)
            if np.any(source_segmentation == 9):
                source_label9_cases.append(case.subject_id)
                if not np.any(label == 4):
                    raise RuntimeError("infarction case lost the lesion class")
            if case.subject_id == "7" and not np.any(label == 4):
                raise RuntimeError("case 7 tiny WML disappeared")
            stats = json.loads(paths["case_stats"].read_text(encoding="utf-8"))
            for label_value in range(1, 8):
                threshold_flag = bool(stats["volume_threshold_flags"][str(label_value)])
                stats_rows.append(
                    {
                        "subject_id": case.subject_id,
                        "output_id": case.output_id,
                        "source_split": case.source_split,
                        "label": label_value,
                        "label_name": LABEL_NAMES[label_value],
                        "native_volume_mm3": stats["native_volumes_mm3"][str(label_value)],
                        "output_volume_mm3": stats["output_volumes_mm3"][str(label_value)],
                        "relative_volume_difference": stats["relative_volume_difference"][
                            str(label_value)
                        ],
                        "threshold_flag": threshold_flag,
                    }
                )
                if threshold_flag:
                    volume_threshold_flags.append(
                        {
                            "subject_id": case.subject_id,
                            "label": label_value,
                            "relative_volume_difference": stats[
                                "relative_volume_difference"
                            ][str(label_value)],
                        }
                    )
        except Exception as error:
            failures.append(f"Case {case.subject_id}: {error}")

    preflight = json.loads((output / "qc" / "preflight_manifest.json").read_text(encoding="utf-8"))
    source_hash_changes = []
    for path_string, expected_hash in preflight["input_sha256"].items():
        path = Path(path_string)
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            source_hash_changes.append(path_string)
    if source_hash_changes:
        failures.append(f"Source hashes changed: {source_hash_changes}")
    if not pilot and len(source_label9_cases) != 7:
        failures.append(f"Expected seven infarction cases, found {source_label9_cases}")
    manual_review_path = output / "qc" / "manual_review.json"
    manual_review_status = "not_required"
    if volume_threshold_flags:
        manual_review_status = "missing"
        accepted_pairs: set[tuple[str, int]] = set()
        if manual_review_path.is_file():
            manual_review = json.loads(manual_review_path.read_text(encoding="utf-8"))
            manual_review_status = str(manual_review.get("status", "invalid"))
            if manual_review_status == "accepted":
                for review in manual_review.get("volume_threshold_reviews", []):
                    if review.get("decision") == "accepted_after_manual_review":
                        accepted_pairs.add((str(review["subject_id"]), int(review["label"])))
        unreviewed = [
            flag
            for flag in volume_threshold_flags
            if (str(flag["subject_id"]), int(flag["label"])) not in accepted_pairs
        ]
        if unreviewed:
            failures.append(f"Unreviewed volume threshold flags: {unreviewed}")
    if stats_rows:
        write_csv(output / "qc" / ("pilot_label_statistics.csv" if pilot else "label_statistics.csv"), stats_rows)
    report = {
        "validated_unix": time.time(),
        "pilot": pilot,
        "case_count": len(cases),
        "infarction_cases": source_label9_cases,
        "volume_threshold_flags": volume_threshold_flags,
        "manual_review_path": str(manual_review_path) if volume_threshold_flags else None,
        "manual_review_status": manual_review_status,
        "failures": failures,
        "output_sha256": output_hashes,
    }
    atomic_json(
        output / "qc" / ("pilot_validation.json" if pilot else "final_validation.json"),
        report,
    )
    if failures:
        raise RuntimeError("Validation failed:\n" + "\n".join(failures))
    print(f"Validation passed for {len(cases)} cases")


def main() -> int:
    args = parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.template = args.template.resolve()
    if args.output.parent != args.source:
        raise RuntimeError(f"Output must be the selected source child: {args.source / 'processed_v1'}")
    if args.output.name != "processed_v1":
        raise RuntimeError("Output directory name must be processed_v1")
    if not args.source.is_dir() or not args.template.is_file():
        raise FileNotFoundError("Source or template does not exist")
    ensure_layout(args.output)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    if args.phase == "preflight":
        run_preflight(args.source, args.output, args.template)
    elif args.phase in ("pilot", "all"):
        run_processing(args)
    else:
        validate_outputs(
            args.source,
            args.output,
            template_3d(args.template, args.output),
            pilot=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
