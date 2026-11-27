import unittest

from bridge_sim import Denomination, Seat, Vulnerability
from bridge_sim.par import calculate_par_score_ns


_DDS_STRAIN_ORDER = (
    Denomination.SPADES,
    Denomination.HEARTS,
    Denomination.DIAMONDS,
    Denomination.CLUBS,
    Denomination.NOTRUMP,
)
_DDS_SEAT_ORDER = (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST)


def _trick_table(rows: tuple[tuple[int, int, int, int], ...]) -> dict:
    return {
        denomination: dict(zip(_DDS_SEAT_ORDER, row, strict=True))
        for denomination, row in zip(_DDS_STRAIN_ORDER, rows, strict=True)
    }


class ParScoreTests(unittest.TestCase):
    def test_matches_reference_par_scores_for_dealer_and_vulnerability(self) -> None:
        cases = (
            (
                _trick_table(
                    (
                        (5, 8, 5, 8),
                        (6, 6, 6, 6),
                        (5, 7, 5, 7),
                        (7, 5, 7, 5),
                        (6, 6, 6, 6),
                    )
                ),
                Seat.NORTH,
                Vulnerability.NONE,
                -110,
            ),
            (
                _trick_table(
                    (
                        (4, 9, 4, 9),
                        (10, 2, 10, 2),
                        (8, 3, 8, 3),
                        (6, 7, 6, 7),
                        (9, 3, 9, 3),
                    )
                ),
                Seat.EAST,
                Vulnerability.NORTH_SOUTH,
                100,
            ),
            (
                _trick_table(
                    (
                        (3, 10, 3, 10),
                        (9, 4, 9, 4),
                        (8, 4, 8, 4),
                        (3, 9, 3, 9),
                        (4, 8, 4, 8),
                    )
                ),
                Seat.NORTH,
                Vulnerability.NONE,
                -300,
            ),
        )

        for table, dealer, vulnerability, expected_score in cases:
            with self.subTest(dealer=dealer, vulnerability=vulnerability):
                self.assertEqual(
                    calculate_par_score_ns(table, dealer, vulnerability),
                    expected_score,
                )

    def test_returns_zero_when_no_side_can_make_a_contract(self) -> None:
        table = _trick_table(((6, 6, 6, 6),) * 5)

        self.assertEqual(
            calculate_par_score_ns(
                table,
                Seat.NORTH,
                Vulnerability.NONE,
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
