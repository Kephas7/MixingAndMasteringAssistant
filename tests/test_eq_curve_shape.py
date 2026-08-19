import unittest
from src.audio_processor import compute_eq_curve

class EqCurveShapeTests(unittest.TestCase):
    def test_curve_axes_have_matching_lengths(self):
        frequencies, magnitude = compute_eq_curve({"Bass": 3.0}, 48000)
        self.assertEqual(len(frequencies), len(magnitude))
        self.assertEqual(len(frequencies), 800)
