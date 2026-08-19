import unittest
import numpy as np
from src.audio_processor import compute_eq_curve

class EqCurveFiniteTests(unittest.TestCase):
    def test_active_curve_contains_finite_values(self):
        _, magnitude = compute_eq_curve({"Air": -6.0}, 44100)
        self.assertTrue(np.isfinite(magnitude).all())
