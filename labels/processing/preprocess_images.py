#!/usr/bin/env python3
"""Portable WMH/MS/Stroke/MRBrainS18 image preprocessing from a case manifest.

Defaults to input preflight. --execute writes under a new output directory.
Saved transforms are used whenever supplied; missing transforms are calculated
by the registration pipeline and recorded per case. Final released
labels are never inputs to intensity processing and are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DATASETS = {"WMH", "MS", "Stroke", "MRBrainS18"}
STAGES = {"raw", "skullstripped", "n4", "intra"}
DEFAULT_HDBET = {"argv": ["hd-bet", "-i", "{input}", "-o", "{brain}", "--save_bet_mask"],
                 "mask_output": "{stem}_bet.nii.gz"}


def resolve_path(value: str, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_manifest(args: argparse.Namespace) -> list[dict]:
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    plans = []
    for row in rows:
        dataset, case_id, stage = row["dataset"], row["case_id"], row["input_stage"]
        if dataset not in DATASETS or stage not in STAGES:
            raise ValueError(f"Unknown dataset/input_stage: {dataset}/{stage}")
        if not case_id or Path(case_id).name != case_id or case_id in {".", ".."} or "\\" in case_id:
            raise ValueError(f"Invalid case_id: {case_id!r}")
        if (dataset, case_id) in seen:
            raise ValueError(f"Duplicate case: {dataset}/{case_id}")
        seen.add((dataset, case_id))
        if stage == "n4" and dataset not in {"MS", "Stroke"}:
            raise ValueError(f"N4 is not part of the documented {dataset} pipeline")
        if stage == "intra" and dataset != "Stroke":
            raise ValueError("intra stage is the saved Stroke T1-space intermediate only")
        paths = {key: resolve_path(row.get(key, ""), args.manifest.parent) for key in (
            "t1_path", "flair_path", "t1_mask", "flair_mask", "intra_transform", "atlas_transform",
            "expected_t1", "expected_flair")}
        for key in ("t1_path", "flair_path"):
            if paths[key] is None:
                raise ValueError(f"Missing {key}: {dataset}/{case_id}")
        if not paths["atlas_transform"]:
            suffix = "_Affine.mat" if dataset == "MS" else "_0GenericAffine.mat"
            candidate = args.transforms / (dataset + "_" + case_id + suffix)
            if candidate.is_file():
                paths["atlas_transform"] = candidate.resolve()
        for key, value in paths.items():
            if value is not None and not value.is_file():
                raise FileNotFoundError(f"{dataset}/{case_id} {key}: {value}")
        needs_intra = dataset in {"MS", "Stroke"} and stage != "intra" and not (dataset == "MS" and case_id == "P49_T2")
        estimated = []
        if not paths["atlas_transform"]:
            estimated.append("main_to_SRI24")
        if needs_intra and not paths["intra_transform"]:
            estimated.append("FLAIR_to_T1")
        if dataset == "Stroke" and stage == "skullstripped" and not (paths["t1_mask"] and paths["flair_mask"]):
            raise ValueError("Stroke skullstripped stage requires both original HD-BET masks for N4")
        plans.append({"dataset": dataset, "case_id": case_id, "input_stage": stage,
                      "paths": {key: str(value) if value else "" for key, value in paths.items()},
                      "registration_estimation": estimated,
                      "modality_order": {"0000": "T1", "0001": "FLAIR"}})
    return plans


def same_geometry(first, second, atol=1e-5) -> bool:
    import numpy as np
    return first.shape == second.shape and np.allclose(first.affine, second.affine, rtol=0, atol=atol)


def copy_or_lps(source: Path, destination: Path, lps: bool) -> Path:
    if not lps:
        shutil.copy2(source, destination)
    else:
        import nibabel as nib
        from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
        image = nib.load(str(source))
        transform = ornt_transform(io_orientation(image.affine), axcodes2ornt(("L", "P", "S")))
        nib.save(image.as_reoriented(transform), str(destination))
    return destination


def mask_brain(source: Path, mask_path: Path, destination: Path, atol=1e-5) -> Path:
    import nibabel as nib
    import numpy as np
    image, mask = nib.load(str(source)), nib.load(str(mask_path))
    if not same_geometry(image, mask, atol=atol):
        raise ValueError(f"Brain mask geometry differs from its source image: {mask_path}")
    data = np.asanyarray(image.dataobj).copy()
    data[np.asanyarray(mask.dataobj) <= 0] = 0
    nib.save(nib.Nifti1Image(data, image.affine, image.header), str(destination))
    return destination


def extract_brain(source: Path, work: Path, name: str, configuration: dict) -> tuple[Path, Path]:
    brain = work / f"{name}_brain.nii.gz"
    fields = {"input": str(source), "brain": str(brain), "stem": str(work / f"{name}_brain")}
    argv = [part.format(**fields) for part in configuration["argv"]]
    mask = Path(configuration["mask_output"].format(**fields))
    if mask.parent.resolve() != work.resolve():
        raise ValueError("HD-BET mask_output must stay in the per-case working directory")
    subprocess.run(argv, check=True)
    if not brain.is_file() or not mask.is_file():
        raise FileNotFoundError(f"HD-BET did not create the specified brain image and mask: {brain}, {mask}")
    return brain, mask


def resample_stroke_t1(source: Path, destination: Path) -> Path:
    import SimpleITK as sitk
    image = sitk.ReadImage(str(source))
    old_spacing = image.GetSpacing()
    spacing = [max(1.0, value) for value in old_spacing]
    if spacing == list(old_spacing):
        sitk.WriteImage(image, str(destination))
        return destination
    size = [int(round(n * old / new)) for n, old, new in zip(image.GetSize(), old_spacing, spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    result = sitk.Cast(resampler.Execute(image), image.GetPixelID())
    sitk.WriteImage(result, str(destination))
    return destination


def clean_image(source: Path, destination: Path) -> Path:
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import label
    image = nib.load(str(source))
    data = image.get_fdata()
    connected, number = label(data > 0)
    if number:
        sizes = np.bincount(connected.ravel())
        sizes[0] = 0
        data[connected != sizes.argmax()] = 0
    data[data < 0.1] = 0
    nib.save(nib.Nifti1Image(data, image.affine, image.header), str(destination))
    return destination


def check_output(output: Path, template: Path, expected: str = "") -> dict:
    import nibabel as nib
    import numpy as np
    image, fixed = nib.load(str(output)), nib.load(str(template))
    values = np.asanyarray(image.dataobj)
    if not same_geometry(image, fixed) or not np.isfinite(values).all():
        raise ValueError(f"Output is non-finite or not on the SRI24 grid: {output}")
    result = {"shape": list(image.shape), "affine": image.affine.tolist(), "finite": True}
    if expected:
        old = nib.load(expected)
        if not same_geometry(image, old):
            result["comparison"] = {"geometry_match": False, "voxel_exact": False}
        else:
            old_values = np.asanyarray(old.dataobj)
            result["comparison"] = {"geometry_match": True, "voxel_exact": bool(np.array_equal(values, old_values)),
                                    "maximum_absolute_difference": float(np.max(np.abs(values.astype(np.float64)-old_values)))}
    return result


def run_case(plan: dict, args: argparse.Namespace, template: Path, hdbet: dict | None) -> dict:
    import ants
    import nibabel as nib
    import numpy as np
    dataset, case_id, stage = plan["dataset"], plan["case_id"], plan["input_stage"]
    paths = {key: Path(value) if value else None for key, value in plan["paths"].items()}
    work = args.output_dir / "work" / dataset / case_id
    work.mkdir(parents=True)
    images = args.output_dir / "images" / dataset
    images.mkdir(parents=True, exist_ok=True)
    t1, flair = paths["t1_path"], paths["flair_path"]
    masks = {"T1": paths["t1_mask"], "FLAIR": paths["flair_mask"]}
    if stage == "raw":
        raw = {"T1": t1, "FLAIR": flair}
        brains = {}
        for modality in ("T1", "FLAIR"):
            local = copy_or_lps(raw[modality], work / f"{modality}_source.nii.gz", dataset == "Stroke")
            mask = masks[modality]
            if dataset == "MRBrainS18" and modality == "FLAIR":
                mask = masks["T1"]
            if dataset == "MS" and case_id == "P49_T2" and modality == "T1":
                if not np.any(np.asanyarray(nib.load(str(local)).dataobj)):
                    brains[modality], masks[modality] = local, None
                    continue
            if mask:
                mask = copy_or_lps(mask, work / f"{modality}_mask.nii.gz", dataset == "Stroke")
                brain = mask_brain(local, mask, work / f"{modality}_brain.nii.gz",
                                   atol=1e-3 if dataset == "MRBrainS18" else 1e-5)
            else:
                brain, mask = extract_brain(local, work, modality, hdbet)
            brains[modality], masks[modality] = brain, mask
        t1, flair = brains["T1"], brains["FLAIR"]
    if dataset in {"MS", "Stroke"} and stage in {"raw", "skullstripped"}:
        corrected = {}
        for modality, path in (("T1", t1), ("FLAIR", flair)):
            image = ants.image_read(str(path))
            if dataset == "MS" and case_id == "P49_T2" and modality == "T1" and not np.any(image.numpy()):
                corrected[modality] = path
                continue
            if dataset == "MS":
                mask = ants.get_mask(image, low_thresh=1e-5, high_thresh=None, cleanup=0)
                result = ants.n4_bias_field_correction(image, mask=mask)
            else:
                mask = ants.threshold_image(ants.image_read(str(masks[modality])), 0.5, 1e9)
                result = ants.n4_bias_field_correction(image, mask=mask, shrink_factor=2, verbose=False)
            output = work / f"{modality}_n4.nii.gz"
            ants.image_write(result, str(output))
            corrected[modality] = output
        t1, flair = corrected["T1"], corrected["FLAIR"]
    if dataset == "Stroke" and stage != "intra":
        t1 = resample_stroke_t1(t1, work / "T1_n4_resampled.nii.gz")
    fixed = ants.image_read(str(template))
    moving_t1, moving_flair = ants.image_read(str(t1)), ants.image_read(str(flair))
    special = dataset == "MS" and case_id == "P49_T2"
    main, secondary = (moving_flair, moving_t1) if special else (moving_t1, moving_flair)
    intra = [str(paths["intra_transform"])] if paths["intra_transform"] else []
    if "FLAIR_to_T1" in plan["registration_estimation"]:
        tx = ants.registration(fixed=main, moving=secondary, type_of_transform="Rigid", outprefix=str(work / "intra_"))
        intra = tx["fwdtransforms"]
    # Stroke historically saved FLAIR in T1 space before template registration;
    # preserve those two interpolations. MS applied the composite once.
    if dataset == "Stroke" and stage != "intra":
        moving_flair = ants.apply_transforms(fixed=moving_t1, moving=moving_flair, transformlist=intra, interpolator="linear")
        intermediate = work / "FLAIR_in_T1.nii.gz"
        ants.image_write(moving_flair, str(intermediate))
        moving_flair = ants.image_read(str(intermediate))
        secondary = moving_flair
    atlas = [str(paths["atlas_transform"])] if paths["atlas_transform"] else []
    if "main_to_SRI24" in plan["registration_estimation"]:
        tx = ants.registration(fixed=fixed, moving=main, type_of_transform="Rigid", outprefix=str(work / "atlas_"))
        atlas = tx["fwdtransforms"]
    warped_main = ants.apply_transforms(fixed=fixed, moving=main, transformlist=atlas, interpolator="linear")
    secondary_transforms = atlas + intra if dataset == "MS" and not special else atlas
    warped_secondary = ants.apply_transforms(fixed=fixed, moving=secondary, transformlist=secondary_transforms, interpolator="linear")
    warped_t1, warped_flair = (warped_secondary, warped_main) if special else (warped_main, warped_secondary)
    checks = {}
    for channel, modality, image in (("0000", "t1", warped_t1), ("0001", "flair", warped_flair)):
        output = images / f"{case_id}_{channel}.nii.gz"
        clean = dataset in {"MS", "Stroke"} and not (special and modality == "t1")
        registered = work / f"{modality}_registered.nii.gz" if clean else output
        ants.image_write(image, str(registered))
        if clean:
            clean_image(registered, output)
        checks[modality] = check_output(output, template, plan["paths"][f"expected_{modality}"])
    result = dict(plan)
    result.update({"atlas_transforms_used": atlas, "intra_transforms_used": intra,
                   "source_processing": "raw_and_masks" if stage == "raw" and not hdbet else stage,
                   "hdbet_configuration": hdbet if stage == "raw" else None, "outputs": checks,
                   "historical_voxel_identity_claimed": False})
    (work / "case_record.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.require_voxel_match:
        if any(not check.get("comparison", {}).get("voxel_exact", False) for check in checks.values()):
            raise ValueError(f"Reference voxel equality failed or was not supplied: {dataset}/{case_id}")
    return result


def main() -> None:
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=package / "reproduction_resources/SRI24_spgr.nii")
    parser.add_argument("--transforms", type=Path, default=package / "reproduction_resources/registration_transforms")
    parser.add_argument("--hdbet-config", type=Path,
                        help="JSON with explicit argv template and mask_output; placeholders input,brain,stem")
    parser.add_argument("--require-voxel-match", action="store_true",
                        help="Require equality with expected_t1 and expected_flair images supplied per case")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.template.is_file():
        parser.error("The exact SRI24 template is required")
    if args.output_dir.exists():
        parser.error("Use a new output directory; existing outputs are never overwritten")
    if args.threads < 1:
        parser.error("threads must be positive")
    hdbet = json.loads(args.hdbet_config.read_text("utf-8")) if args.hdbet_config else DEFAULT_HDBET
    if hdbet is not None and (not isinstance(hdbet.get("argv"), list) or not hdbet.get("mask_output")):
        parser.error("HD-BET configuration needs an argv array and mask_output string")
    plans = read_manifest(args)
    print(json.dumps({"cases": len(plans), "datasets": sorted({plan['dataset'] for plan in plans}),
                      "registration_estimated_cases": sum(bool(plan['registration_estimation']) for plan in plans),
                      "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    import nibabel as nib
    import numpy as np
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "work").mkdir()
    template_image = nib.load(str(args.template))
    if template_image.shape == (240, 240, 155, 1):
        template = args.output_dir / "work/SRI24_3d.nii.gz"
        data = np.asanyarray(template_image.dataobj)[..., 0]
        nib.save(nib.Nifti1Image(data, template_image.affine), str(template))
    elif template_image.shape == (240, 240, 155):
        template = args.template
    else:
        raise ValueError(f"Unexpected SRI24 geometry: {template_image.shape}")
    versions = {"python": sys.version}
    for dependency in ("antspyx", "nibabel", "numpy", "scipy", "SimpleITK"):
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = "not installed"
    (args.output_dir / "run_environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    (args.output_dir / "input_plan.json").write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")
    for plan in plans:
        result = run_case(plan, args, template, hdbet)
        print(json.dumps({"dataset": result['dataset'], "case_id": result['case_id'], "status": "complete",
                          "estimated": result['registration_estimation']}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
