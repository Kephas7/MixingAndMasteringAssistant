import unittest
from src.mastering_model import get_mastering_recommendations

class MasteringGainTests(unittest.TestCase):
    def test_quiet_master_gets_positive_gain_guidance(self):
        result = get_mastering_recommendations({"integrated_lufs": -20, "true_peak_dbtp": -4, "lra": 10})
        self.assertEqual(result["gain_adjustment"], "+6.0 dB to reach -14 LUFS")
