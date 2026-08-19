import unittest
import numpy as np
from src.audio_processor import compute_eq_curve

class EqCurveFlatTests(unittest.TestCase):
    def test_empty_settings_produce_flat_curve(self):
        _, magnitude = compute_eq_curve({}, 48000)
        np.testing.assert_allclose(magnitude, 0.0, atol=1e-8)
