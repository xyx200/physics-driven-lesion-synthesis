"""Compute single-case BrainSeg lesion metrics using the original experimental label rules and metric calculations."""
import argparse
import json
import math

import nibabel as nib
import numpy as np
from nnunet_mednext.evaluation.evaluator import Evaluator

SENTINEL = 374.0

def load(path):
    image = nib.load(path)
    array = np.asanyarray(image.dataobj)
    spacing = tuple(float(x) for x in image.header.get_zooms()[:3])
    return image, array, spacing

def check_geometry(base_image, base_spacing, image, spacing, role, case):
    if base_image.shape != image.shape:
        raise RuntimeError(f"{case}/{role}: shape mismatch {base_image.shape} != {image.shape}")
    if not np.allclose(base_image.affine, image.affine, rtol=0, atol=1e-5):
        raise RuntimeError(f"{case}/{role}: affine mismatch")
    if not np.allclose(base_spacing, spacing, rtol=0, atol=1e-5):
        raise RuntimeError(f"{case}/{role}: spacing mismatch {base_spacing} != {spacing}")

def evaluate(pred, ref, spacing):
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    p = int(pred.sum())
    r = int(ref.sum())
    intersection = int(np.count_nonzero(pred & ref))
    both_empty = p == 0 and r == 0
    one_empty = (p == 0) != (r == 0)
    all_full = bool(pred.all() or ref.all())
    dice = 1.0 if both_empty else (2.0 * intersection / (p + r))
    if both_empty:
        hd95, status = 0.0, "both_empty"
    elif one_empty:
        hd95, status = SENTINEL, "single_empty"
    elif all_full:
        hd95, status = SENTINEL, "all_full"
    else:
        try:
            result = Evaluator(
                test=pred.astype(np.uint8),
                reference=ref.astype(np.uint8),
                labels=[1],
                metrics=["Dice", "Hausdorff Distance 95"],
                advanced_metrics=[],
            ).evaluate(voxel_spacing=spacing, connectivity=3)["1"]
            evaluator_dice = float(result["Dice"])
            if not math.isclose(evaluator_dice, dice, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"Evaluator Dice mismatch {evaluator_dice} != {dice}")
            hd95 = float(result["Hausdorff Distance 95"])
            if not math.isfinite(hd95) or hd95 < 0 or hd95 > SENTINEL:
                hd95 = SENTINEL
            status = "normal"
        except Exception as error:
            hd95, status = SENTINEL, "distance_failure:" + type(error).__name__
    return {
        "dsc": float(dice),
        "hd95": float(hd95),
        "status": status,
        "prediction_voxels": p,
        "reference_voxels": r,
        "intersection_voxels": intersection,
        "prediction_empty": p == 0,
        "reference_empty": r == 0,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True, help='Prediction in the shared label space; lesion label is 4')
    parser.add_argument('--reference', required=True, help='Reference in the shared label space; lesion label is 4')
    parser.add_argument('--raw', action='store_true', help='Use when the input is a raw BrainSeg prediction')
    parser.add_argument('--support', help='Ours prediction used with --raw; values >0 define the brain region')
    args = parser.parse_args()
    if args.raw and not args.support:
        parser.error('--raw requires --support')

    ref_image, ref, ref_spacing = load(args.reference)
    pred_image, pred, pred_spacing = load(args.prediction)
    check_geometry(ref_image, ref_spacing, pred_image, pred_spacing, 'prediction', 'case')
    if args.raw:
        support_image, support, support_spacing = load(args.support)
        check_geometry(ref_image, ref_spacing, support_image, support_spacing, 'support', 'case')
        mask = (pred == 0) & (support > 0)
    else:
        mask = pred == 4
    print(json.dumps(evaluate(mask, ref == 4, ref_spacing), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
