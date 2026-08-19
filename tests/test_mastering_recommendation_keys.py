import unittest
from src.mastering_model import get_mastering_recommendations

class MasteringRecommendationKeyTests(unittest.TestCase):
    def test_recommendations_include_status_fields(self):
        result = get_mastering_recommendations({"integrated_lufs": -14, "true_peak_dbtp": -2, "lra": 10})
        self.assertEqual(result["loudness_status"], "good_loudness")
        self.assertEqual(result["true_peak_status"], "true_peak_ok")
        self.assertEqual(result["lra_status"], "lra_good")
