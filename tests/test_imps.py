import unittest

from bridge_sim import calculate_imps


class ImpScoringTests(unittest.TestCase):
    def test_converts_standard_thresholds_with_ns_perspective(self) -> None:
        cases = (
            (0, 0),
            (19, 0),
            (20, 1),
            (49, 1),
            (50, 2),
            (899, 13),
            (900, 14),
            (3999, 23),
            (4000, 24),
            (-20, -1),
            (-900, -14),
            (-4000, -24),
        )
        for score_difference, expected_imps in cases:
            with self.subTest(score_difference=score_difference):
                self.assertEqual(calculate_imps(score_difference), expected_imps)

    def test_rejects_non_integer_score_differences(self) -> None:
        for score_difference in (True, 1.5, "20"):
            with self.subTest(score_difference=score_difference):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    calculate_imps(score_difference)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
