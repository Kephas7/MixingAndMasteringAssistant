import unittest
import numpy as np
from src.compression_model import analyze_dynamics

class DynamicsSilenceTests(unittest.TestCase):
    def test_silence_metrics_are_finite(self):
        metrics = analyze_dynamics(np.zeros(4096), 48000)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
