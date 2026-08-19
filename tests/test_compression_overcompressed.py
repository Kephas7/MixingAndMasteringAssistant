import unittest
from src.compression_model import get_compression_recommendations

class CompressionOvercompressedTests(unittest.TestCase):
    def test_low_dynamic_range_avoids_aggressive_ratio(self):
        result = get_compression_recommendations({"crest_factor_db": 4, "dynamic_range": 3, "loudness_range": 2})
        self.assertEqual(result["compression_ratio"], "1.5:1")
