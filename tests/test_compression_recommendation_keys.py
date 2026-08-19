import unittest
from src.compression_model import get_compression_recommendations

class CompressionRecommendationKeyTests(unittest.TestCase):
    def test_recommendation_exposes_ui_fields(self):
        result = get_compression_recommendations({"crest_factor_db": 10, "dynamic_range": 12, "loudness_range": 5})
        self.assertTrue({"compression_ratio", "attack_time", "release_time"}.issubset(result))
