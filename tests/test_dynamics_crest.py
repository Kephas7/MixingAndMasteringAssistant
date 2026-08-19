import unittest
import numpy as np
from src.compression_model import analyze_dynamics

class DynamicsCrestTests(unittest.TestCase):
    def test_constant_signal_has_zero_crest_factor(self):
        metrics = analyze_dynamics(np.full(4096, 0.5), 48000)
        self.assertAlmostEqual(metrics["crest_factor_db"], 0.0, places=8)
