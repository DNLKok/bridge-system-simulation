import unittest

from bridge_sim import (
    Contract,
    Denomination,
    Seat,
    Vulnerability,
    calculate_score_ns,
)


class DuplicateScoringTests(unittest.TestCase):
    def test_made_contract_scores_include_game_and_notrump_points(self) -> None:
        self.assertEqual(
            calculate_score_ns(
                Contract(4, Denomination.SPADES, Seat.NORTH),
                Vulnerability.NONE,
                0,
            ),
            420,
        )
        self.assertEqual(
            calculate_score_ns(
                Contract(3, Denomination.NOTRUMP, Seat.SOUTH),
                Vulnerability.NONE,
                0,
            ),
            400,
        )
        self.assertEqual(
            calculate_score_ns(
                Contract(6, Denomination.NOTRUMP, Seat.SOUTH),
                Vulnerability.NONE,
                0,
            ),
            990,
        )
        self.assertEqual(
            calculate_score_ns(
                Contract(7, Denomination.NOTRUMP, Seat.SOUTH),
                Vulnerability.ALL,
                0,
            ),
            2220,
        )

    def test_doubles_redoubles_and_overtricks_are_scored(self) -> None:
        self.assertEqual(
            calculate_score_ns(
                Contract(4, Denomination.SPADES, Seat.NORTH, multiplier=2),
                Vulnerability.NORTH_SOUTH,
                0,
            ),
            790,
        )
        self.assertEqual(
            calculate_score_ns(
                Contract(4, Denomination.SPADES, Seat.NORTH, multiplier=4),
                Vulnerability.NONE,
                1,
            ),
            1080,
        )
        self.assertEqual(
            calculate_score_ns(
                Contract(1, Denomination.CLUBS, Seat.SOUTH, multiplier=4),
                Vulnerability.NONE,
                0,
            ),
            230,
        )

    def test_undertricks_use_vulnerability_and_multiplier(self) -> None:
        self.assertEqual(
            calculate_score_ns(
                Contract(3, Denomination.NOTRUMP, Seat.NORTH, multiplier=2),
                Vulnerability.NONE,
                -4,
            ),
            -800,
        )
        self.assertEqual(
            calculate_score_ns(
                Contract(3, Denomination.NOTRUMP, Seat.NORTH, multiplier=4),
                Vulnerability.NORTH_SOUTH,
                -3,
            ),
            -1600,
        )
        self.assertEqual(
            calculate_score_ns(
                Contract(3, Denomination.NOTRUMP, Seat.WEST, multiplier=4),
                Vulnerability.NONE,
                -1,
            ),
            200,
        )

    def test_east_west_contract_score_is_reversed_for_north_south(self) -> None:
        self.assertEqual(
            calculate_score_ns(
                Contract(4, Denomination.SPADES, Seat.EAST),
                Vulnerability.NONE,
                0,
            ),
            -420,
        )

    def test_rejects_impossible_trick_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "impossible"):
            calculate_score_ns(
                Contract(1, Denomination.NOTRUMP, Seat.NORTH),
                Vulnerability.NONE,
                -13,
            )


if __name__ == "__main__":
    unittest.main()
