import unittest
import numpy as np
from src.audio_processor import apply_compression

class CompressorFiniteTests(unittest.TestCase):
    def test_silence_remains_finite(self):
        output = apply_compression(np.zeros(1024), 44100)
        self.assertTrue(np.isfinite(output).all())
