#!/usr/bin/env python3
"""Stage downloaded BraTS images; optionally reproduce documented GLI filling.

The default is a read-only plan. --execute writes to a new output directory.
This entry point does not generate or change the released segmentation labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def find_image(root: Path, original_id: str, suffix: str) -> Path:
    name = f"{original_id}-{suffix}.nii.gz"
    direct = [root / original_id / name, root / name]
    matches = [path for path in direct if path.is_file()]
    if not matches:
        matches = list(root.rglob(name))
    matches = list(dict.fromkeys(path.resolve() for path in matches))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name}, found {len(matches)}: {matches}")
    return matches[0]


def plan_cases(args: argparse.Namespace) -> list[dict]:
    rows = read_rows(args.case_map)
    if rows and "dataset" in rows[0]:
        rows = [row for row in rows if row["dataset"] == args.dataset]
    if not rows or "case_id" not in rows[0]:
        raise ValueError("case-map needs column case_id and at least one row")
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate output case_id in case-map")
    if any(not value or Path(value).name != value or value in {".", ".."} or "\\" in value for value in ids):
        raise ValueError("case_id must be a nonempty filename without path components")
    metadata = {}
    if args.dataset == "GLI":
        if not args.gli_case_metadata:
            raise ValueError("GLI requires --gli-case-metadata to select image-repair cases")
        entries = read_rows(args.gli_case_metadata)
        metadata = {row["case_id"]: row for row in entries}
        if len(metadata) != len(entries):
            raise ValueError("Duplicate GLI case_id in metadata")
    output = []
    for row in rows:
        case_id = row["case_id"]
        original_id = row["case_id"]
        if not original_id or Path(original_id).name != original_id or "\\" in original_id:
            raise ValueError(f"Invalid original_id: {original_id!r}")
        repair = False
        repair_label = None
        if args.dataset == "GLI":
            if original_id not in metadata:
                raise ValueError(f"Missing GLI metadata for {original_id}")
            item = metadata[original_id]
            flag = item["image_repair_required"]
            if flag not in {"repair_required", "no_repair_required"}:
                raise ValueError(f"Unexpected image-repair status for {original_id}: {flag}")
            repair = flag == "repair_required"
            if repair:
                if not args.gli_release_root or not item["repair_label_path"]:
                    raise ValueError(f"Missing repair-label location for {original_id}")
                repair_label = args.gli_release_root / item["repair_label_path"]
                if not repair_label.is_file():
                    raise FileNotFoundError(repair_label)
        # The source inpainting protocol operates on all four modalities.
        modalities = ("t1n", "t1c", "t2w", "t2f") if repair or getattr(args, "four_modalities", False) else ("t1n", "t2f")
        sources = {name: str(find_image(args.images_root, original_id, name)) for name in modalities}
        output.append({"dataset": args.dataset, "case_id": case_id, "original_id": original_id,
                       "repair_required": repair, "repair_label": str(repair_label) if repair_label else "",
                       "sources": sources})
    return output


def command(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True)


def fill_case(case: dict, work: Path, seg_maths: str, fill_lesions: str) -> dict[str, Path]:
    """Replay run_inpainting.bash defaults; keep intermediate products."""
    work.mkdir()
    dilated_tumor = work / "tumor_dilated.nii.gz"
    brain = work / "brain_mask.nii.gz"
    lesion = work / "wmh_mask.nii.gz"
    label = case["repair_label"]
    command([seg_maths, label, "-thr", "0.7", "-uthr", "1.3", "-bin", "-dil", "3", str(dilated_tumor)])
    command([seg_maths, case["sources"]["t1n"], "-thr", "0.0001", "-bin", "-sub", str(dilated_tumor),
             "-thr", "0", "-bin", str(brain)])
    command([seg_maths, label, "-thr", "1.7", "-uthr", "2.3", "-bin", str(lesion)])
    results = {}
    for modality, source in case["sources"].items():
        result = work / f"{case['original_id']}-{modality}.nii.gz"
        command([fill_lesions, "-i", source, "-l", str(lesion), "-o", str(result), "-mask", str(brain),
                 "-dil", "2", "-size", "4", "-v"])
        if not result.is_file():
            raise FileNotFoundError(result)
        results[modality] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("GLI", "MEN", "MET", "PED", "SSA"))
    parser.add_argument("--case-map", type=Path, required=True,
                        help="case_manifest.csv; case_id is both the official BraTS ID and final label basename.")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gli-case-metadata", type=Path,
                        help="GLI repair CSV with case_id,image_repair_required,repair_label_path")
    parser.add_argument("--gli-release-root", type=Path,
                        help="Root against which metadata repair_label_path entries are resolved")
    parser.add_argument("--niftyseg-bin", type=Path)
    parser.add_argument("--four-modalities", action="store_true",
                        help="Also export T1ce as 0002 and T2 as 0003 for four-modality comparators")
    parser.add_argument("--execute", action="store_true", help="Write outputs; otherwise print the plan only")
    args = parser.parse_args()
    if not args.images_root.is_dir():
        parser.error("images-root must exist")
    cases = plan_cases(args)
    repair_count = sum(case["repair_required"] for case in cases)
    print(json.dumps({"cases": len(cases), "repair_cases": repair_count, "mode": "execute" if args.execute else "plan",
                      "output": str(args.output_dir), "first_case": cases[0]}, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    seg_maths = fill_lesions = ""
    if repair_count:
        def executable(name: str) -> str:
            result = str(args.niftyseg_bin / name) if args.niftyseg_bin else shutil.which(name)
            if not result or not Path(result).is_file():
                raise FileNotFoundError(f"Required NiftySeg executable: {name}")
            return result
        seg_maths, fill_lesions = executable("seg_maths"), executable("seg_FillLesions")
    args.output_dir.mkdir(parents=True)
    images = args.output_dir / "images"
    images.mkdir()
    work_root = args.output_dir / "repair_work"
    if repair_count:
        work_root.mkdir()
    with (args.output_dir / "input_plan.json").open("w", encoding="utf-8") as stream:
        json.dump(cases, stream, ensure_ascii=False, indent=2)
    for case in cases:
        sources = ({name: Path(path) for name, path in case["sources"].items()} if not case["repair_required"] else
                   fill_case(case, work_root / case["case_id"], seg_maths, fill_lesions))
        channels = [("0000", "t1n"), ("0001", "t2f")]
        if args.four_modalities:
            channels += [("0002", "t1c"), ("0003", "t2w")]
        for channel, modality in channels:
            shutil.copy2(sources[modality], images / f"{case['case_id']}_{channel}.nii.gz")
        print(f"Completed {case['case_id']}", flush=True)
    print(f"Completed {len(cases)} cases; source images and repair labels were read only.")


if __name__ == "__main__":
    main()
