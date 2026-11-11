"""NumPy-only regression tests for the train-only geometry screening bound."""

import unittest

import numpy as np

from link_model import LinkGeometry, build_reflective_links, vertical_upa
from sweep_geometry import SENSOR_IDX, _rank_key, optimistic_sensing_bound


class TestGeometrySweep(unittest.TestCase):
    def test_optimistic_bound_dominates_actual_one_bit_reflection(self):
        center_l = np.array([0., 0., 2.])
        normal_l = np.array([1., 0., 0.])
        center_a = np.array([2., 4., 2.])
        normal_a = np.array([0., -1., 0.])
        users = np.array([[10., 0., 2.], [14., 1., 2.], [20., -1., 2.]])
        cfg = LinkGeometry(
            legitimate_center_xyz=tuple(center_l),
            legitimate_normal_xyz=tuple(normal_l),
            malicious_rows=4, malicious_cols=4,
            attacker_forward_m=2., attacker_lateral_m=4.,
            malicious_normal_xyz=tuple(normal_a),
            reflection_efficiency=0.7)
        G, g, _, info = build_reflective_links(users, users[:1], cfg)
        self.assertEqual(info["rear_side_training_users"], 0)
        self.assertTrue(np.allclose(info["malicious_center_xyz"], center_a))
        lam = cfg.wavelength_m
        legit = vertical_upa(center_l, 36, 36, cfg.legitimate_spacing_lambda * lam,
                            normal_l)[SENSOR_IDX]
        attacker = vertical_upa(center_a, 4, 4, cfg.malicious_spacing_lambda * lam,
                                normal_a)
        rng = np.random.default_rng(7)
        phase = rng.choice(np.array([-1., 1.]), G.shape[1])
        actual_delta = (g * phase) @ G[SENSOR_IDX].T
        h0 = np.ones((len(users), len(SENSOR_IDX)), dtype=np.complex64) * 1e-4
        actual_power = np.abs(actual_delta).astype(np.float64) ** 2
        actual_db = 10 * np.log10(actual_power.sum(1) /
                                  (np.abs(h0) ** 2).sum(1))
        upper_db = []
        for position, baseline in zip(users, h0):
            m = optimistic_sensing_bound(legit, attacker, normal_l, normal_a,
                                        position[None, :], baseline[None, :],
                                        lam, cfg.reflection_efficiency)
            upper_db.append(m["median_db"])
            self.assertEqual(m["coverage"], 1.0)
        self.assertTrue(np.all(np.array(upper_db) + 1e-3 >= actual_db))

    def test_backward_panel_gives_zero_visibility_and_bound(self):
        center_l = np.array([0., 0., 2.])
        normal_l = np.array([1., 0., 0.])
        center_a = np.array([2., 4., 2.])
        normal_a = np.array([0., 1., 0.])  # opposite of the legitimate panel
        lam = LinkGeometry().wavelength_m
        legit = vertical_upa(center_l, 36, 36, 0.25 * lam, normal_l)[SENSOR_IDX]
        attacker = vertical_upa(center_a, 2, 2, 0.5 * lam, normal_a)
        users = np.array([[10., 0., 2.]])
        h = np.ones((1, 16), dtype=np.complex64) * 1e-4
        m = optimistic_sensing_bound(legit, attacker, normal_l, normal_a,
                                    users, h, lam)
        self.assertEqual(m["coverage"], 0.)
        self.assertEqual(m["median_db"], -300.)

    def test_explicit_normal_is_not_reinferred_from_user_locations(self):
        center = np.array([0., 0., 2.])
        users = np.array([[10., 15., 2.], [11., 15., 2.]])
        cfg = LinkGeometry(
            legitimate_center_xyz=tuple(center),
            legitimate_normal_xyz=(1., 0., 0.),
            malicious_normal_xyz=(0., -1., 0.))
        _, _, _, info = build_reflective_links(users, users, cfg)
        self.assertEqual(info["legitimate_normal_xyz"], [1., 0., 0.])
        self.assertEqual(info["malicious_normal_xyz"], [0., -1., 0.])

    def test_minimum_coverage_takes_priority_over_loose_bound(self):
        better_coverage = {"selection": {"coverage": .95, "median_db": -38.,
                                         "p10_db": -80.}}
        better_median = {"selection": {"coverage": .55, "median_db": -10.,
                                       "p10_db": -70.}}
        self.assertGreater(_rank_key(better_coverage, .9),
                           _rank_key(better_median, .9))


if __name__ == "__main__":
    unittest.main()
