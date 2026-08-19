import unittest
import numpy as np
from src.audio_processor import apply_limiter

class LimiterCeilingTests(unittest.TestCase):
    def test_output_does_not_exceed_ceiling(self):
        output = apply_limiter(np.array([-2.0, 0.0, 2.0]), -6.0)
        self.assertLessEqual(np.max(np.abs(output)), 10 ** (-6.0 / 20.0))
