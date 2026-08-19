import unittest
import io
import numpy as np
from scipy.io.wavfile import read
from src.audio_processor import export_audio

class ExportWavClippingTests(unittest.TestCase):
    def test_export_clips_float_signal_to_pcm_range(self):
        _, samples = read(io.BytesIO(export_audio(np.array([-2.0, 2.0]), 8000)))
        self.assertEqual(samples.tolist(), [-32767, 32767])
