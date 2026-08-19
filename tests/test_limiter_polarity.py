import unittest
import numpy as np
from src.audio_processor import apply_limiter

class LimiterPolarityTests(unittest.TestCase):
    def test_clipping_is_symmetric(self):
        output = apply_limiter(np.array([-2.0, 2.0]), -3.0)
        self.assertAlmostEqual(output[0], -output[1])
