import unittest
from pathlib import Path

from bridge_sim import (
    AuctionEntry,
    Bid,
    Contract,
    Deal,
    Denomination,
    HandProfile,
    PassCall,
    Seat,
    load_system,
    run_auction,
)


SYSTEM_PATH = (
    Path(__file__).resolve().parents[1]
    / "systems"
    / "one-no-trump-15-17.toml"
)


def profile_deal() -> Deal:
    """Feature-only input for the auction tests; no card deal is created."""
    return Deal(
        {
            Seat.NORTH: HandProfile(8, (4, 3, 3, 3)),
            Seat.EAST: HandProfile(16, (3, 4, 3, 3)),
            Seat.SOUTH: HandProfile(9, (3, 3, 4, 3)),
            Seat.WEST: HandProfile(7, (3, 3, 3, 4)),
        }
    )


class AuctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = load_system(SYSTEM_PATH)

    def test_system_loads_one_no_trump_rule(self) -> None:
        self.assertEqual(self.system.system_id, "balanced-15-17-1nt")
        self.assertEqual(self.system.version, "1.0.0")
        self.assertEqual(self.system.min_hcp, 15)
        self.assertEqual(self.system.max_hcp, 17)

    def test_bids_one_no_trump_for_qualifying_hand_when_legal(self) -> None:
        hand = HandProfile(16, (5, 3, 3, 2))
        legal_calls = frozenset(
            {PassCall.PASS, Bid(1, Denomination.NOTRUMP)}
        )

        call = self.system.choose_call(hand, (), legal_calls)

        self.assertEqual(call, Bid(1, Denomination.NOTRUMP))

    def test_passes_for_nonmatching_or_illegal_one_no_trump(self) -> None:
        legal_calls = frozenset(
            {PassCall.PASS, Bid(1, Denomination.NOTRUMP)}
        )
        unbalanced_hand = HandProfile(16, (5, 4, 2, 2))
        too_few_points = HandProfile(14, (4, 3, 3, 3))

        self.assertEqual(
            self.system.choose_call(unbalanced_hand, (), legal_calls),
            PassCall.PASS,
        )
        self.assertEqual(
            self.system.choose_call(too_few_points, (), legal_calls),
            PassCall.PASS,
        )
        self.assertEqual(
            self.system.choose_call(
                HandProfile(16, (4, 3, 3, 3)),
                (),
                frozenset({PassCall.PASS}),
            ),
            PassCall.PASS,
        )

    def test_two_systems_generate_complete_auction_and_contract(self) -> None:
        result = run_auction(
            deal=profile_deal(),
            north_south_system=self.system,
            east_west_system=self.system,
            dealer=Seat.EAST,
        )

        self.assertEqual(
            result.entries,
            (
                AuctionEntry(Seat.EAST, Bid(1, Denomination.NOTRUMP)),
                AuctionEntry(Seat.SOUTH, PassCall.PASS),
                AuctionEntry(Seat.WEST, PassCall.PASS),
                AuctionEntry(Seat.NORTH, PassCall.PASS),
            ),
        )
        self.assertEqual(
            result.contract,
            Contract(1, Denomination.NOTRUMP, Seat.EAST),
        )
        self.assertEqual(str(result.contract), "1NT by E")
        self.assertEqual(
            result.transcript,
            "E: 1NT S: PASS W: PASS N: PASS",
        )

    def test_four_initial_passes_produce_passed_out_board(self) -> None:
        all_weak = Deal(
            {
                Seat.NORTH: HandProfile(10, (4, 3, 3, 3)),
                Seat.EAST: HandProfile(10, (3, 4, 3, 3)),
                Seat.SOUTH: HandProfile(10, (3, 3, 4, 3)),
                Seat.WEST: HandProfile(10, (3, 3, 3, 4)),
            }
        )

        result = run_auction(all_weak, self.system, self.system)

        self.assertTrue(result.passed_out)
        self.assertIsNone(result.contract)
        self.assertEqual(len(result.entries), 4)


if __name__ == "__main__":
    unittest.main()
