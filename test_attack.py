"""Software smoke tests; small synthetic links are NOT research experiments."""

import unittest

import torch
from torch import nn

from attack import (AttackConfig, channel_with_malicious_ris,
                    optimize_malicious_ris, phases_from_angles, random_phases)


class FixedEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        b = x.shape[0]
        real = torch.ones((b, 1, 36, 36), device=x.device) + self.dummy * 0
        imag = torch.zeros_like(real)
        return torch.cat([real, imag], 1)


class TestPhysicalAttack(unittest.TestCase):
    def setUp(self):
        self.h0 = torch.ones(3, 36, 36, dtype=torch.complex64)
        self.G = torch.zeros(1296, 2, dtype=torch.complex64)
        self.G[0, 0] = 0.2
        self.G[1, 1] = 0.15j
        self.g = torch.ones(3, 2, dtype=torch.complex64)

    def test_physical_composition_and_shape(self):
        phi = torch.ones(2, dtype=torch.complex64)
        h = channel_with_malicious_ris(self.h0, self.G, self.g, phi)
        self.assertEqual(tuple(h.shape), (3, 36, 36))
        self.assertTrue(torch.allclose(h[:, 0, 0], torch.full((3,), 1.2 + 0j)))
        self.assertTrue(torch.allclose(h[:, 0, 1], torch.full((3,), 1 + 0.15j)))
        self.assertTrue(torch.allclose(h[:, 0, 2], self.h0[:, 0, 2]))

    def test_per_user_composition(self):
        phi = torch.tensor([[1, 1], [-1, 1], [1, -1]], dtype=torch.complex64)
        h = channel_with_malicious_ris(self.h0, self.G, self.g, phi)
        self.assertLess(h[1, 0, 0].real, h[0, 0, 0].real)
        self.assertAlmostEqual(h[2, 0, 1].imag, -0.15, places=5)

    def test_phase_quantization(self):
        theta = torch.tensor([0.1, 2.9, 4.7])
        phi = phases_from_angles(theta, bits=1)
        expected = torch.tensor([1, -1, -1], dtype=torch.complex64)
        self.assertTrue(torch.allclose(phi, expected, atol=2e-6))
        self.assertTrue(torch.allclose(phi.abs(), torch.ones(3)))
        self.assertTrue(torch.allclose(random_phases(20, 2, 8).abs(), torch.ones(20)))

    def test_rejects_nonphysical_magnitude(self):
        with self.assertRaises(ValueError):
            channel_with_malicious_ris(self.h0, self.G, self.g,
                                       torch.tensor([2 + 0j, 1 + 0j]))

    def test_universal_optimization_smoke(self):
        cfg = AttackConfig(steps=3, restarts=1, eot_samples=1,
                           pilot_snr_db=None, phase_bits=1, universal=True)
        result = optimize_malicious_ris(FixedEstimator(), self.h0, self.G,
                                        self.g, 1., config=cfg)
        self.assertEqual(tuple(result.phi.shape), (2,))
        self.assertTrue(torch.allclose(result.phi.abs(), torch.ones(2), atol=2e-6))
        self.assertTrue(torch.isfinite(result.nmse_linear).all())
        self.assertEqual(tuple(result.h_perturbed.shape), (3, 36, 36))

    def test_per_user_optimization_smoke(self):
        cfg = AttackConfig(steps=2, restarts=1, eot_samples=1,
                           pilot_snr_db=None, phase_bits=None, universal=False)
        result = optimize_malicious_ris(FixedEstimator(), self.h0, self.G,
                                        self.g, 1., config=cfg)
        self.assertEqual(tuple(result.phi.shape), (3, 2))
        self.assertTrue(torch.allclose(result.phi.abs(), torch.ones(3, 2), atol=2e-6))


if __name__ == "__main__":
    unittest.main()
