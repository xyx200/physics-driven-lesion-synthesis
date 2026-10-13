#!/usr/bin/env python3
import argparse
import json
import pathlib

import nibabel as nib
import numpy as np


SOURCE_GROUPS = {
    0: [0, 12],
    2: [7, 8, 9, 10, 14, 15, 16],
    3: [1],
    4: [18],
    5: [3, 4, 11],
    6: [5, 6],
    7: [13, 17],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    source = pathlib.Path(args.input)
    output = pathlib.Path(args.output)
    reference_path = pathlib.Path(args.reference)
    if output.exists():
        raise SystemExit(f"refuse existing output: {output}")

    image = nib.load(str(source))
    reference = nib.load(str(reference_path))
    raw = np.asanyarray(image.dataobj)
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5):
        raise SystemExit("raw TumorSynth geometry does not match reference")
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.rint(raw)):
        raise SystemExit("raw TumorSynth output is not finite integer data")

    raw_labels = set(np.unique(raw).astype(int).tolist())
    if not raw_labels <= set(range(19)):
        raise SystemExit(f"unexpected raw labels: {sorted(raw_labels)}")

    lut = np.ones(19, dtype=np.uint8)
    for target, labels in SOURCE_GROUPS.items():
        lut[labels] = target
    mapped = lut[raw.astype(np.int64)]
    mapped_labels = set(np.unique(mapped).astype(int).tolist())
    if not mapped_labels <= set(range(8)):
        raise SystemExit(f"unexpected mapped labels: {sorted(mapped_labels)}")

    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mapped, image.affine, header), str(output))
    print(
        json.dumps(
            {
                "input": str(source),
                "output": str(output),
                "raw_labels": sorted(raw_labels),
                "mapped_labels": sorted(mapped_labels),
                "raw_label_counts": {
                    str(label): int(np.count_nonzero(raw == label))
                    for label in sorted(raw_labels)
                },
                "mapped_label_counts": {
                    str(label): int(np.count_nonzero(mapped == label))
                    for label in sorted(mapped_labels)
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
