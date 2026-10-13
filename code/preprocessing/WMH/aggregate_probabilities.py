"""Aggregate the original WMH-SynthSeg 33 probability channels into eight.

Extracted from the study's WMHSynthSeg/inference.py aggregation block. The
channel order below is label_list_segmentation for WMH-SynthSeg_v10_231110.
Input arrays are (33, X, Y, Z); output NIfTI arrays are (X, Y, Z, 8).
This step performs no spatial flips, matching DO_FLIP=False in that block.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


# Channel position is NOT the FreeSurfer label ID.
SOURCE_LABEL_IDS = (
    0, 14, 15, 16, 24, 77, 85, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17,
    18, 26, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
)

# Original experiment mapping. Preserve its grouping even where a generic
# anatomical regrouping would differ (e.g. ID 15 -> background, 24/85 -> GM).
# Some IDs in the original groups are absent from this 33-channel model.
GROUP_MAPPING = {
    0: (0, 1, 6, 40, 45, 15),
    1: (3, 24, 42, 85),  # Active IDs handled by the original default-GM rule.
    2: (9, 10, 11, 12, 13, 17, 18, 26, 48, 49, 50, 51, 52, 53, 54, 58),
    3: (2, 41, 251, 252, 253, 254, 255, 30, 62),
    4: (25, 57, 77),
    5: (4, 5, 14, 43, 44, 72, 31, 63),
    6: (7, 8, 46, 47),
    7: (16, 28, 60),
}
SOURCE_TO_SHARED = {
    source: target for target, sources in GROUP_MAPPING.items() for source in sources
}


def aggregate_probabilities(original_softmax):
    """Sum the original probability channels without argmax or renormalization."""
    original_softmax = np.asarray(original_softmax)
    if original_softmax.ndim != 4 or original_softmax.shape[0] != len(SOURCE_LABEL_IDS):
        raise ValueError("Expected WMH-SynthSeg probabilities with shape (33, X, Y, Z)")
    new_softmax = np.zeros((8,) + original_softmax.shape[1:], dtype=np.float32)
    for channel, source_label in enumerate(SOURCE_LABEL_IDS):
        new_softmax[SOURCE_TO_SHARED[source_label]] += original_softmax[channel]
    return new_softmax


def save_aggregated_probabilities(original_softmax, affine, output_path):
    """Save eight channels in NIfTI format for entropy_fusion_train.py/entropy_fusion_test.py."""
    new_softmax = aggregate_probabilities(original_softmax)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.moveaxis(new_softmax, 0, -1), affine)
    image.set_data_dtype(np.float32)
    nib.save(image, str(output_path))
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-probs", required=True, type=Path,
                        help="Native 33-channel NIfTI probabilities, shape (X, Y, Z, 33)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output, e.g. 11_0000_seg_aggregated_probs.nii.gz")
    args = parser.parse_args()
    image = nib.load(str(args.input_probs))
    probabilities = np.asanyarray(image.dataobj)
    if probabilities.ndim != 4 or probabilities.shape[-1] != 33:
        parser.error("Input must contain all 33 probability channels, not a hard label or lesion-only map")
    output = save_aggregated_probabilities(
        np.moveaxis(probabilities, -1, 0), image.affine, args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
