"""Aggregate the study's 19-channel TumorSynth probabilities into eight classes.

Mapping and channel-wise summation follow GLI-AL healthy label/aggregate_seg.py.
Input/output NPZ arrays use (C, Z, Y, X), as expected by GLI/entropy_fusion.py.
These are channel IDs of the study's TumorSynth output, not FreeSurfer IDs.
"""

import argparse
from pathlib import Path

import numpy as np


# Shared labels: background, GM, BG, WM, lesion, ventricles, cerebellum, brain stem.
SOURCE_GROUPS = {
    0: (0, 12),
    1: (2,),  # The remaining channel assigned to GM in the original LUT.
    2: (7, 8, 9, 10, 14, 15, 16),
    3: (1,),
    4: (18,),
    5: (3, 4, 11),
    6: (5, 6),
    7: (13, 17),
}
SOURCE_TO_SHARED = {
    source: target for target, sources in SOURCE_GROUPS.items() for source in sources
}


def aggregate_probabilities(original_softmax):
    """Sum probabilities before argmax; preserve the original spatial axes."""
    original_softmax = np.asarray(original_softmax)
    if original_softmax.ndim != 4 or original_softmax.shape[0] != 19:
        raise ValueError("Expected TumorSynth probabilities with shape (19, Z, Y, X)")
    new_softmax = np.zeros((8,) + original_softmax.shape[1:], dtype=np.float32)
    for source_class in range(19):
        new_softmax[SOURCE_TO_SHARED[source_class]] += original_softmax[source_class]
    return new_softmax


def aggregate_file(input_path, output_path):
    """Write the softmax NPZ consumed by the existing entropy-fusion script."""
    with np.load(input_path) as data:
        result = {"softmax": aggregate_probabilities(data["softmax"])}
        if "affine" in data:
            result["affine"] = data["affine"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    files = sorted(p for p in args.input_dir.glob("*.npz") if "aggregated" not in p.name)
    if not files:
        parser.error("No source probability NPZ files found")
    for path in files:
        stem = path.stem
        if stem.endswith("_softmax"):
            stem = stem[:-8]
        output = aggregate_file(path, args.output_dir / (stem + "_aggregated_probs.npz"))
        print(output)


if __name__ == "__main__":
    main()
