import unittest

from src.grounding import combine_confidence, retrieval_is_sufficient


class GroundingTests(unittest.TestCase):
    def test_retrieval_threshold_rejects_weak_match(self) -> None:
        self.assertFalse(retrieval_is_sufficient(0.2, 0.35))
        self.assertTrue(retrieval_is_sufficient(0.35, 0.35))

    def test_confidence_is_bounded_by_weaker_signal(self) -> None:
        self.assertEqual(combine_confidence(0.91, 0.73), 0.73)
        self.assertEqual(combine_confidence(0.8, 1.2), 0.8)
        self.assertEqual(combine_confidence(-0.2, 0.9), 0.0)


if __name__ == "__main__":
    unittest.main()