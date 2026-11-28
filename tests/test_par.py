import unittest

from bridge_sim import Contract, Denomination, ParResult, Seat, Vulnerability
from bridge_sim.par import calculate_par_result


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
                Contract(2, Denomination.SPADES, Seat.EAST),
                0,
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
                Contract(
                    4,
                    Denomination.SPADES,
                    Seat.EAST,
                    multiplier=2,
                ),
                -1,
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
                Contract(
                    5,
                    Denomination.HEARTS,
                    Seat.NORTH,
                    multiplier=2,
                ),
                -2,
                -300,
            ),
        )

        for (
            table,
            dealer,
            vulnerability,
            expected_contract,
            expected_delta,
            expected_score,
        ) in cases:
            with self.subTest(dealer=dealer, vulnerability=vulnerability):
                result = calculate_par_result(table, dealer, vulnerability)
                self.assertEqual(result.contract, expected_contract)
                self.assertEqual(result.trick_delta, expected_delta)
                self.assertEqual(result.score_ns, expected_score)

    def test_returns_zero_when_no_side_can_make_a_contract(self) -> None:
        table = _trick_table(((6, 6, 6, 6),) * 5)

        self.assertEqual(
            calculate_par_result(
                table,
                Seat.NORTH,
                Vulnerability.NONE,
            ),
            ParResult(None, None, 0),
        )

    def test_returns_all_making_contracts_tied_at_par_score(self) -> None:
        table = _trick_table(
            (
                (6, 6, 6, 6),
                (6, 6, 6, 6),
                (6, 6, 6, 6),
                (11, 6, 11, 6),
                (9, 6, 9, 6),
            )
        )

        result = calculate_par_result(table, Seat.NORTH, Vulnerability.NONE)

        self.assertEqual(result.contract, Contract(3, Denomination.NOTRUMP, Seat.NORTH))
        self.assertEqual(
            result.alternative_contracts,
            (Contract(5, Denomination.CLUBS, Seat.NORTH),),
        )
        self.assertEqual(
            result.contracts,
            (
                Contract(3, Denomination.NOTRUMP, Seat.NORTH),
                Contract(5, Denomination.CLUBS, Seat.NORTH),
            ),
        )
        self.assertEqual(result.score_ns, 400)

    def test_returns_tied_sacrifice_contracts(self) -> None:
        table = _trick_table(
            (
                (8, 4, 8, 5),
                (7, 8, 10, 10),
                (10, 6, 10, 5),
                (6, 4, 11, 9),
                (4, 6, 9, 10),
            )
        )

        result = calculate_par_result(table, Seat.NORTH, Vulnerability.NONE)

        self.assertEqual(
            result.contract,
            Contract(5, Denomination.HEARTS, Seat.WEST, multiplier=2),
        )
        self.assertEqual(
            result.alternative_contracts,
            (Contract(5, Denomination.NOTRUMP, Seat.WEST, multiplier=2),),
        )
        self.assertEqual(result.score_ns, 100)


if __name__ == "__main__":
    unittest.main()
