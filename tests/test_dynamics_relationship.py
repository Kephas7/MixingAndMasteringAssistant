import unittest
import numpy as np
from src.compression_model import analyze_dynamics

class DynamicsRelationshipTests(unittest.TestCase):
    def test_dynamic_range_matches_peak_minus_rms(self):
        metrics = analyze_dynamics(np.linspace(-1.0, 1.0, 4096), 48000)
        self.assertAlmostEqual(metrics["dynamic_range"], metrics["peak_db"] - metrics["rms_db"])
