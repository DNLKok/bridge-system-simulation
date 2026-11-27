from collections.abc import Sequence
import unittest
from pathlib import Path

from bridge_sim import (
    AuctionEntry,
    Bid,
    Call,
    Contract,
    Deal,
    Denomination,
    DoubleCall,
    HandProfile,
    OneNoTrumpSystem,
    PassCall,
    RedoubleCall,
    Seat,
    load_system,
    run_auction,
)


SYSTEM_PATH = (
    Path(__file__).resolve().parents[1]
    / "systems"
    / "one-no-trump-15-17.toml"
)
PASSOUT_SYSTEM_PATH = (
    Path(__file__).resolve().parents[1]
    / "systems"
    / "passout-cheapest-call.toml"
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


class CallSequenceSystem:
    def __init__(self, seats: frozenset[Seat], calls: tuple[Call, ...]) -> None:
        self.seats = seats
        self.calls = calls

    def choose_call(
        self,
        hand: HandProfile,
        auction: Sequence[AuctionEntry],
        legal_calls: frozenset[Call],
    ) -> Call:
        del hand, legal_calls
        call_index = sum(entry.seat in self.seats for entry in auction)
        return self.calls[call_index]


class AuctionTests(unittest.TestCase):
    def setUp(self) -> None:
        system = load_system(SYSTEM_PATH)
        if not isinstance(system, OneNoTrumpSystem):
            raise AssertionError("Expected the configured 1NT system.")
        self.system = system

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

    def test_passout_system_loads_and_passes_outside_passout_seat(self) -> None:
        system = load_system(PASSOUT_SYSTEM_PATH)
        legal_calls = frozenset(
            {
                PassCall.PASS,
                DoubleCall.DOUBLE,
                Bid(1, Denomination.CLUBS),
            }
        )

        self.assertEqual(system.system_id, "passout-cheapest-call")
        self.assertEqual(
            system.choose_call(HandProfile(0, (4, 3, 3, 3)), (), legal_calls),
            PassCall.PASS,
        )

    def test_passout_system_opens_at_cheapest_legal_bid_after_three_passes(self) -> None:
        system = load_system(PASSOUT_SYSTEM_PATH)
        auction = tuple(
            AuctionEntry(seat, PassCall.PASS)
            for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH)
        )
        legal_calls = frozenset(
            {
                PassCall.PASS,
                Bid(1, Denomination.CLUBS),
                Bid(1, Denomination.DIAMONDS),
            }
        )

        self.assertEqual(
            system.choose_call(HandProfile(0, (4, 3, 3, 3)), auction, legal_calls),
            Bid(1, Denomination.CLUBS),
        )

    def test_passout_system_prefers_double_redouble_then_cheapest_bid(self) -> None:
        system = load_system(PASSOUT_SYSTEM_PATH)
        hand = HandProfile(0, (4, 3, 3, 3))

        double_auction = (
            AuctionEntry(Seat.NORTH, Bid(1, Denomination.CLUBS)),
            AuctionEntry(Seat.EAST, PassCall.PASS),
            AuctionEntry(Seat.SOUTH, PassCall.PASS),
        )
        self.assertEqual(
            system.choose_call(
                hand,
                double_auction,
                frozenset({PassCall.PASS, DoubleCall.DOUBLE}),
            ),
            DoubleCall.DOUBLE,
        )

        redouble_auction = (
            AuctionEntry(Seat.NORTH, Bid(1, Denomination.CLUBS)),
            AuctionEntry(Seat.EAST, DoubleCall.DOUBLE),
            AuctionEntry(Seat.SOUTH, PassCall.PASS),
            AuctionEntry(Seat.WEST, PassCall.PASS),
        )
        self.assertEqual(
            system.choose_call(
                hand,
                redouble_auction,
                frozenset({PassCall.PASS, RedoubleCall.REDOUBLE}),
            ),
            RedoubleCall.REDOUBLE,
        )

        bid_auction = redouble_auction + (
            AuctionEntry(Seat.NORTH, RedoubleCall.REDOUBLE),
            AuctionEntry(Seat.EAST, PassCall.PASS),
            AuctionEntry(Seat.SOUTH, PassCall.PASS),
        )
        self.assertEqual(
            system.choose_call(
                hand,
                bid_auction,
                frozenset(
                    {
                        PassCall.PASS,
                        Bid(1, Denomination.DIAMONDS),
                        Bid(1, Denomination.HEARTS),
                    }
                ),
            ),
            Bid(1, Denomination.DIAMONDS),
        )

    def test_passout_system_passes_when_no_non_pass_call_is_legal(self) -> None:
        system = load_system(PASSOUT_SYSTEM_PATH)
        auction = (
            AuctionEntry(Seat.NORTH, Bid(7, Denomination.NOTRUMP)),
            AuctionEntry(Seat.EAST, DoubleCall.DOUBLE),
            AuctionEntry(Seat.SOUTH, RedoubleCall.REDOUBLE),
            AuctionEntry(Seat.WEST, PassCall.PASS),
            AuctionEntry(Seat.NORTH, PassCall.PASS),
        )

        self.assertEqual(
            system.choose_call(
                HandProfile(0, (4, 3, 3, 3)),
                auction,
                frozenset({PassCall.PASS}),
            ),
            PassCall.PASS,
        )

    def test_auction_tracks_double_redouble_and_final_contract_multiplier(self) -> None:
        north_south_system = CallSequenceSystem(
            frozenset({Seat.NORTH, Seat.SOUTH}),
            (
                Bid(1, Denomination.CLUBS),
                RedoubleCall.REDOUBLE,
                PassCall.PASS,
            ),
        )
        east_west_system = CallSequenceSystem(
            frozenset({Seat.EAST, Seat.WEST}),
            (
                DoubleCall.DOUBLE,
                PassCall.PASS,
                PassCall.PASS,
            ),
        )

        result = run_auction(
            profile_deal(),
            north_south_system,
            east_west_system,
        )

        self.assertEqual(
            result.entries,
            (
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.CLUBS)),
                AuctionEntry(Seat.EAST, DoubleCall.DOUBLE),
                AuctionEntry(Seat.SOUTH, RedoubleCall.REDOUBLE),
                AuctionEntry(Seat.WEST, PassCall.PASS),
                AuctionEntry(Seat.NORTH, PassCall.PASS),
                AuctionEntry(Seat.EAST, PassCall.PASS),
            ),
        )
        self.assertEqual(
            result.contract,
            Contract(1, Denomination.CLUBS, Seat.NORTH, multiplier=4),
        )
        self.assertEqual(str(result.contract), "1CXX by N")

    def test_passout_system_drives_a_complete_auction(self) -> None:
        system = load_system(PASSOUT_SYSTEM_PATH)
        result = run_auction(
            profile_deal(),
            system,
            system,
        )

        self.assertEqual(result.entries[3].call, Bid(1, Denomination.CLUBS))
        self.assertIn(DoubleCall.DOUBLE, (entry.call for entry in result.entries))
        self.assertIn(
            RedoubleCall.REDOUBLE,
            (entry.call for entry in result.entries),
        )
        self.assertEqual(
            result.contract,
            Contract(7, Denomination.NOTRUMP, Seat.WEST, multiplier=4),
        )


if __name__ == "__main__":
    unittest.main()
