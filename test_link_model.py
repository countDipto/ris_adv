"""Executable geometry tests without PyTorch or the large RISnet data files."""

import unittest
from pathlib import Path
import json
import subprocess
import sys
import tempfile

import numpy as np

from link_model import LinkGeometry, build_reflective_links, vertical_upa


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.center = np.array([278.42, 576.97, 2.0])
        self.train = np.array([
            self.center + [20., 0., 0.],
            self.center + [30., 2., 0.],
            self.center + [20., -2., 0.],
        ])
        self.test = np.array([self.center + [15., -1., 0.]])

    def test_array_is_row_major_and_spaced(self):
        xyz = vertical_upa([0, 0, 0], 3, 4, 0.1, [1, 0, 0])
        self.assertEqual(xyz.shape, (12, 3))
        self.assertAlmostEqual(float(np.linalg.norm(xyz[1] - xyz[0])), 0.1)
        self.assertAlmostEqual(float(np.linalg.norm(xyz[4] - xyz[0])), 0.1)

    def test_channel_shapes_and_visibility(self):
        G, gt, ge, info = build_reflective_links(self.train, self.test)
        self.assertEqual(G.shape, (1296, 64))
        self.assertEqual(gt.shape, (3, 64))
        self.assertEqual(ge.shape, (1, 64))
        self.assertTrue(np.isfinite(G).all())
        self.assertGreater(np.abs(G).mean(), 0)
        self.assertGreater(np.abs(gt).mean(), 0)
        self.assertEqual(info["rear_side_training_users"], 0)
        self.assertAlmostEqual(info["separation_between_centers_m"],
                               np.sqrt(80), places=6)

    def test_rear_side_user_has_no_reflected_path(self):
        back = np.array([self.center + [20., 20., 0.]])
        _, _, g, info = build_reflective_links(self.train, back)
        self.assertEqual(info["rear_side_testing_users"], 1)
        self.assertTrue(np.allclose(g, 0))

    def test_reflection_efficiency_scales_physical_link(self):
        G1, _, _, _ = build_reflective_links(self.train, self.test,
                                              LinkGeometry(reflection_efficiency=1.0))
        G2, _, _, _ = build_reflective_links(self.train, self.test,
                                              LinkGeometry(reflection_efficiency=0.5))
        self.assertTrue(np.allclose(G2, 0.5 * G1, atol=1e-9))

    def test_generator_npz_roundtrip_without_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training = root / "training.npy"
            testing = root / "testing.npy"
            output = root / "links.npz"
            np.save(training, self.train)
            np.save(testing, self.test)
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("generate_links.py")),
                "--data_dir", temp, "--train_locations", str(training),
                "--test_locations", str(testing), "--out", str(output),
            ], check=True, capture_output=True, text=True)
            with np.load(output, allow_pickle=False) as obj:
                self.assertEqual(obj["G_ar"].shape, (1296, 64))
                self.assertEqual(obj["g_ua_training"].shape, (3, 64))
                meta = json.loads(str(obj["metadata"].item()))
                self.assertFalse(meta["units_aligned_with_h0"])
                self.assertEqual(meta["carrier_frequency_hz"], 3.5e9)


if __name__ == "__main__":
    unittest.main()
