#!/usr/bin/env python3
"""Align label storage orientation to a reference image without interpolation.

Only voxel-axis permutations and flips are permitted. The default checks and
prints a plan; --execute writes a new label and a provenance JSON. Existing
files are never overwritten. Native-space registration/resampling is excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def make_plan(label_path: Path, reference_path: Path, output_path: Path) -> dict:
    import nibabel as nib
    import numpy as np
    from nibabel.orientations import inv_ornt_aff, io_orientation, ornt_transform

    label = nib.load(str(label_path))
    reference = nib.load(str(reference_path))
    if len(label.shape) != 3 or len(reference.shape) != 3:
        raise ValueError("Labels and reference images must both be 3D")
    if output_path.resolve() in {label_path.resolve(), reference_path.resolve()}:
        raise ValueError("Output must be separate from both source files")
    sidecar = Path(str(output_path) + ".json")
    if output_path.exists() or sidecar.exists():
        raise FileExistsError(f"Output already exists: {output_path} or {sidecar}")
    transform = ornt_transform(io_orientation(label.affine), io_orientation(reference.affine))
    if not np.isfinite(transform).all():
        raise ValueError("Cannot determine all three image axes")
    aligned_shape = tuple(int(label.shape[i]) for i in np.argsort(transform[:, 0]))
    aligned_affine = label.affine @ inv_ornt_aff(transform, label.shape)
    if aligned_shape != reference.shape or not np.allclose(aligned_affine, reference.affine, rtol=0, atol=1e-5):
        raise ValueError(
            "Reference requires registration/resampling or another geometry change; refusing. "
            f"After axis-only alignment: shape={aligned_shape}, reference={reference.shape}, "
            f"max_affine_difference={float(np.max(np.abs(aligned_affine - reference.affine)))}"
        )
    return {"label_path": str(label_path.resolve()), "reference_path": str(reference_path.resolve()),
            "output_path": str(output_path.resolve()), "orientation_transform": transform.tolist(),
            "input_shape": list(label.shape), "output_shape": list(aligned_shape),
            "input_axcodes": list(nib.aff2axcodes(label.affine)),
            "output_axcodes": list(nib.aff2axcodes(aligned_affine)),
            "aligned_affine": aligned_affine.tolist(), "reference_affine": reference.affine.tolist(),
            "affine_tolerance": 1e-5, "interpolation": "none", "operation": "axis permutation/flip only"}


def execute_plan(plan: dict) -> dict:
    import nibabel as nib
    import numpy as np
    from nibabel.orientations import apply_orientation

    source = nib.load(plan["label_path"])
    values = np.asanyarray(source.dataobj)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError(f"Source label must contain finite integers: {plan['label_path']}")
    transform = np.asarray(plan["orientation_transform"])
    aligned = apply_orientation(values, transform)
    # as_reoriented performs the same index operation and updates the NIfTI
    # geometry; it never interpolates. Check against the explicit operation.
    output_image = source.as_reoriented(transform)
    if not np.array_equal(np.asanyarray(output_image.dataobj), aligned):
        raise RuntimeError("Unexpected orientation implementation mismatch")
    output = Path(plan["output_path"])
    sidecar = Path(str(output) + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_image, str(output))
    saved = nib.load(str(output))
    saved_values = np.asanyarray(saved.dataobj)
    if saved.shape != tuple(plan["output_shape"]) or not np.allclose(saved.affine, plan["reference_affine"], rtol=0, atol=1e-5):
        raise RuntimeError(f"Saved geometry did not match the reference: {output}")
    if not np.array_equal(saved_values, aligned):
        raise RuntimeError(f"Saved label values changed: {output}")
    labels, counts = np.unique(values, return_counts=True)
    plan = dict(plan)
    plan["label_voxel_counts"] = {str(int(label)): int(count) for label, count in zip(labels, counts)}
    plan["saved_validation"] = "shape, affine and exact reordered voxel values passed"
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    return plan


def safe_name(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
        raise ValueError(f"Invalid dataset/case filename component: {value!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=Path)
    parser.add_argument("--reference", type=Path, help="Processed T1 paired with this label")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="CSV with dataset,case_id,label_path,reference_path. Paths relative to CSV are supported.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    triples = []
    if args.manifest:
        if not args.output_root or any((args.label, args.reference, args.output)):
            parser.error("manifest mode requires output-root and excludes single-case arguments")
        with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            parser.error("manifest is empty")
        def source_path(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else args.manifest.parent / path
        for row in rows:
            output = args.output_root / safe_name(row["dataset"]) / (safe_name(row["case_id"]) + ".nii.gz")
            triples.append((source_path(row["label_path"]), source_path(row["reference_path"]), output))
    else:
        if not all((args.label, args.reference, args.output)) or args.output_root:
            parser.error("single-case mode requires label,reference,output")
        triples.append((args.label, args.reference, args.output))
    outputs = [output.resolve() for _, _, output in triples]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Duplicate output paths")
    # Precheck every geometry and output path before any write. This reads only
    # NIfTI headers. Voxel values are read one case at a time during execution.
    plans = [make_plan(*triple) for triple in triples]
    for plan in plans:
        result = execute_plan(plan) if args.execute else plan
        print(json.dumps(result, ensure_ascii=False))
    print(f"{'Wrote' if args.execute else 'Checked'} {len(plans)} labels; interpolation=none")


if __name__ == "__main__":
    main()
