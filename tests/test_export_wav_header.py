import unittest
import numpy as np
from src.audio_processor import export_audio

class ExportWavHeaderTests(unittest.TestCase):
    def test_export_has_riff_wave_header(self):
        payload = export_audio(np.zeros(16), 8000)
        self.assertEqual(payload[:4], b"RIFF")
        self.assertEqual(payload[8:12], b"WAVE")
