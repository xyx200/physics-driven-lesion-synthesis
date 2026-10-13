"""Small-array checks for the study's label mappings and fusion-file interfaces."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gli = load_module("gli_aggregation", ROOT / "GLI/aggregate_probabilities.py")
wmh = load_module("wmh_aggregation", ROOT / "WMH/aggregate_probabilities.py")
gli_fusion = load_module("gli_fusion", ROOT / "GLI/entropy_fusion.py")
wmh_fusion = load_module("wmh_fusion", ROOT / "WMH/entropy_fusion_train.py")


class ProbabilityAggregationTests(unittest.TestCase):
    def test_all_tumorsynth_channels(self):
        # One different source class per voxel, covering all 19 channels.
        probabilities = np.eye(19, dtype=np.float32).reshape(19, 19, 1, 1)
        expected_labels = [0, 3, 1, 5, 5, 6, 6, 2, 2, 2, 2, 5, 0, 7, 2, 2, 2, 7, 4]
        result = gli.aggregate_probabilities(probabilities)
        np.testing.assert_array_equal(result.argmax(axis=0).ravel(), expected_labels)
        np.testing.assert_array_equal(result.sum(axis=0), np.ones((19, 1, 1)))

    def test_all_wmh_channels_in_model_order(self):
        probabilities = np.eye(33, dtype=np.float32).reshape(33, 33, 1, 1)
        expected_labels = [
            0, 5, 0, 7, 1, 4, 1, 3, 1, 5, 6, 6, 2, 2, 2, 2, 2,
            2, 2, 7, 3, 1, 5, 6, 6, 2, 2, 2, 2, 2, 2, 2, 7,
        ]
        result = wmh.aggregate_probabilities(probabilities)
        np.testing.assert_array_equal(result.argmax(axis=0).ravel(), expected_labels)
        np.testing.assert_array_equal(result.sum(axis=0), np.ones((33, 1, 1)))

    def test_probabilities_are_summed_before_deciding_the_label(self):
        probabilities = np.zeros((19, 1, 1, 1), dtype=np.float32)
        probabilities[2] = 0.4  # Individually largest: GM.
        probabilities[7] = 0.3
        probabilities[8] = 0.3  # Together BG is larger than GM.
        result = gli.aggregate_probabilities(probabilities)
        self.assertEqual(int(result.argmax(axis=0).item()), 2)
        np.testing.assert_allclose(result[2], 0.6)
        np.testing.assert_allclose(result.sum(axis=0), 1.0)

    def test_probability_mass_is_preserved(self):
        rng = np.random.default_rng(42)
        for module, channels in [(gli, 19), (wmh, 33)]:
            probabilities = rng.random((channels, 2, 3, 4), dtype=np.float32)
            probabilities /= probabilities.sum(axis=0, keepdims=True)
            result = module.aggregate_probabilities(probabilities)
            self.assertEqual(result.shape, (8, 2, 3, 4))
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result.sum(axis=0), 1.0, atol=2e-7)

    def test_wrong_channel_count_and_hard_labels_are_rejected(self):
        for module in (gli, wmh):
            for shape in [(8, 2, 3, 4), (2, 3, 4)]:
                with self.assertRaises(ValueError):
                    module.aggregate_probabilities(np.zeros(shape, dtype=np.float32))

    def test_gli_cli_names_and_existing_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, outputs = root / "inputs", root / "outputs"
            inputs.mkdir()
            probabilities = np.zeros((19, 2, 3, 4), dtype=np.float32)
            probabilities[0] = 1.0
            probabilities[0, 1, 2, 3] = 0.0
            probabilities[18, 1, 2, 3] = 1.0
            affine = np.diag([1.2, 1.5, 2.0, 1.0])
            for name in ["0845_0000_softmax.npz", "0845_0001.npz"]:
                np.savez_compressed(inputs / name, softmax=probabilities, affine=affine)
            with patch.object(sys, "argv", ["aggregate", "--input-dir", str(inputs),
                                             "--output-dir", str(outputs)]):
                gli.main()
            for name in ["0845_0000_aggregated_probs.npz", "0845_0001_aggregated_probs.npz"]:
                path = outputs / name
                actual = gli_fusion.load_npz_prob(str(path))
                np.testing.assert_array_equal(actual, gli.aggregate_probabilities(probabilities))
                self.assertEqual(actual[4, 1, 2, 3], 1.0)
                with np.load(path) as saved:
                    np.testing.assert_array_equal(saved["affine"], affine)

    def test_wmh_cli_geometry_and_existing_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probabilities = np.zeros((33, 2, 3, 4), dtype=np.float32)
            probabilities[0] = 1.0
            probabilities[0, 1, 2, 3] = 0.0
            probabilities[5, 1, 2, 3] = 1.0  # Channel 5 is label ID 77.
            affine = np.array([[1.2, 0, 0, -11], [0, -1.5, 0, 17],
                               [0, 0, 2, 23], [0, 0, 0, 1]], dtype=float)
            source = root / "native_probabilities.nii.gz"
            output = root / "11_0000_seg_aggregated_probs.nii.gz"
            nib.save(nib.Nifti1Image(np.moveaxis(probabilities, 0, -1), affine), source)
            with patch.object(sys, "argv", ["aggregate", "--input-probs", str(source),
                                             "--output", str(output)]):
                wmh.main()
            image = nib.load(output)
            self.assertEqual(image.shape, (2, 3, 4, 8))
            self.assertEqual(image.get_data_dtype(), np.dtype("float32"))
            np.testing.assert_allclose(image.affine, affine, atol=1e-6)
            actual = wmh_fusion.load_nifti_prob(str(output))
            np.testing.assert_array_equal(actual, wmh.aggregate_probabilities(probabilities))
            self.assertEqual(actual[4, 1, 2, 3], 1.0)


if __name__ == "__main__":
    unittest.main()
