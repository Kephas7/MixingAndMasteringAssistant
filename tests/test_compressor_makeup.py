import unittest
import numpy as np
from src.audio_processor import apply_compression

class CompressorMakeupTests(unittest.TestCase):
    def test_makeup_gain_increases_low_level_signal(self):
        signal = np.full(256, 0.001)
        output = apply_compression(signal, 48000, threshold_db=-10, ratio=1.0, makeup_db=6)
        self.assertGreater(np.mean(output), np.mean(signal))
