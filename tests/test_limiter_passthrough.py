import unittest
import numpy as np
from src.audio_processor import apply_limiter

class LimiterPassthroughTests(unittest.TestCase):
    def test_signal_below_ceiling_is_unchanged(self):
        signal = np.array([-0.2, 0.1, 0.3])
        np.testing.assert_array_equal(apply_limiter(signal, -0.1), signal)
