import unittest
import numpy as np
from src.audio_processor import apply_compression

class CompressorIdentityTests(unittest.TestCase):
    def test_unity_ratio_without_makeup_preserves_signal(self):
        signal = np.linspace(-0.5, 0.5, 256)
        np.testing.assert_array_equal(apply_compression(signal, 48000, ratio=1.0), signal)
