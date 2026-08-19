import unittest
import numpy as np
from src.audio_processor import apply_eq

class EqCopyTests(unittest.TestCase):
    def test_identity_returns_independent_array(self):
        signal = np.ones(32)
        output = apply_eq(signal, 48000, {})
        self.assertIsNot(output, signal)
