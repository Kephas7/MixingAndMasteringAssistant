import unittest
import numpy as np
from src.audio_processor import apply_eq

class EqIdentityTests(unittest.TestCase):
    def test_zero_gain_preserves_signal(self):
        signal = np.linspace(-0.8, 0.8, 512)
        output = apply_eq(signal, 48000, {"Mid": 0.0})
        np.testing.assert_array_equal(output, signal)
