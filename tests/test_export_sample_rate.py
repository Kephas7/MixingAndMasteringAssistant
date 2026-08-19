import unittest
import io
import numpy as np
from scipy.io.wavfile import read
from src.audio_processor import export_audio

class ExportSampleRateTests(unittest.TestCase):
    def test_export_preserves_requested_sample_rate(self):
        sample_rate, _ = read(io.BytesIO(export_audio(np.zeros(8), 22050)))
        self.assertEqual(sample_rate, 22050)
