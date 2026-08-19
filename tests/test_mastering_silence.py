import unittest
import numpy as np
from src.mastering_model import analyze_loudness

class MasteringSilenceTests(unittest.TestCase):
    def test_silence_reports_floor_loudness(self):
        metrics = analyze_loudness(np.zeros(48000), 48000)
        self.assertEqual(metrics["integrated_lufs"], -70.0)
