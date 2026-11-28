import unittest
from pathlib import Path

from bridge_sim.deals import Vulnerability
from bridge_sim.model import (
    AuctionEntry,
    Bid,
    Denomination,
    DoubleCall,
    HandProfile,
    PassCall,
    Seat,
)
from bridge_sim.systems import DecisionContext, load_module


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "systems" / "modules" / "stayman-jacoby.toml"
LEGAL_CALLS = frozenset(
    {
        PassCall.PASS,
        DoubleCall.DOUBLE,
        *(
            Bid(level, denomination)
            for level in range(1, 8)
            for denomination in Denomination
        ),
    }
)


def _context(
    hand: HandProfile,
    auction: tuple[AuctionEntry, ...] | None = None,
    seat: Seat = Seat.SOUTH,
) -> DecisionContext:
    if auction is None:
        auction = (
            AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
            AuctionEntry(Seat.EAST, PassCall.PASS),
        )
    dealer = auction[0].seat if auction else Seat.NORTH
    return DecisionContext(
        hand=hand,
        auction=auction,
        legal_calls=LEGAL_CALLS,
        seat=seat,
        dealer=dealer,
        vulnerability=Vulnerability.NONE,
    )


class StaymanJacobyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module(MODULE_PATH)

    def test_priority_calls_and_exact_suit_distributions(self) -> None:
        cases = (
            (9, (6, 2, 3, 2), Bid(4, Denomination.HEARTS)),
            (9, (2, 6, 3, 2), Bid(4, Denomination.DIAMONDS)),
            (9, (1, 3, 5, 4), Bid(3, Denomination.SPADES)),
            (9, (1, 3, 4, 5), Bid(3, Denomination.SPADES)),
            (9, (3, 1, 5, 4), Bid(3, Denomination.HEARTS)),
            (9, (3, 1, 4, 5), Bid(3, Denomination.HEARTS)),
            (8, (3, 3, 6, 1), Bid(3, Denomination.DIAMONDS)),
            (0, (3, 3, 6, 1), Bid(3, Denomination.CLUBS)),
            (9, (3, 3, 6, 1), Bid(3, Denomination.CLUBS)),
            (15, (3, 3, 6, 1), Bid(3, Denomination.CLUBS)),
            (9, (3, 3, 5, 2), Bid(3, Denomination.NOTRUMP)),
            (9, (4, 3, 3, 3), Bid(3, Denomination.NOTRUMP)),
            (8, (3, 3, 5, 2), Bid(2, Denomination.NOTRUMP)),
            (8, (4, 3, 3, 3), Bid(2, Denomination.NOTRUMP)),
            (7, (3, 3, 1, 6), Bid(2, Denomination.SPADES)),
            (8, (4, 4, 3, 2), Bid(2, Denomination.CLUBS)),
            (9, (4, 4, 3, 2), Bid(2, Denomination.CLUBS)),
            (15, (4, 4, 3, 2), Bid(2, Denomination.CLUBS)),
            (7, (5, 4, 2, 2), Bid(2, Denomination.CLUBS)),
            (7, (4, 5, 2, 2), Bid(2, Denomination.CLUBS)),
            (7, (5, 3, 3, 2), Bid(2, Denomination.HEARTS)),
            (7, (3, 5, 3, 2), Bid(2, Denomination.DIAMONDS)),
        )
        for hcp, shape, expected in cases:
            with self.subTest(hcp=hcp, shape=shape):
                decision = self.module.choose(
                    _context(HandProfile(hcp, shape))
                )
                if decision is None:
                    self.fail(f"No call selected for {hcp} HCP with shape {shape}.")
                self.assertEqual(decision.call, expected)

    def test_strength_tier_boundaries(self) -> None:
        for hcp in (0, 7, 9, 14, 15, 37):
            with self.subTest(hcp=hcp):
                decision = self.module.choose(
                    _context(HandProfile(hcp, (3, 3, 6, 1)))
                )
                if decision is None:
                    self.fail(f"No call selected for {hcp} HCP.")
                self.assertEqual(decision.call, Bid(3, Denomination.CLUBS))

    def test_only_applies_to_uncontested_partner_one_not_response(self) -> None:
        hand = HandProfile(9, (6, 2, 3, 2))
        invalid_auctions = (
            (
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.CLUBS)),
                AuctionEntry(Seat.EAST, PassCall.PASS),
            ),
            (
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
                AuctionEntry(Seat.EAST, Bid(2, Denomination.CLUBS)),
            ),
            (
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
                AuctionEntry(Seat.EAST, DoubleCall.DOUBLE),
            ),
        )
        for auction in invalid_auctions:
            with self.subTest(auction=auction):
                self.assertIsNone(self.module.choose(_context(hand, auction)))

        east_responder = _context(
            hand,
            (
                AuctionEntry(Seat.WEST, Bid(1, Denomination.NOTRUMP)),
                AuctionEntry(Seat.NORTH, PassCall.PASS),
            ),
            seat=Seat.EAST,
        )
        decision = self.module.choose(east_responder)
        if decision is None:
            self.fail("No call selected for the East-West partnership.")
        self.assertEqual(decision.call, Bid(4, Denomination.HEARTS))

    def test_dealer_constraints_include_named_strength_ranges(self) -> None:
        context = _context(HandProfile(12, (4, 4, 3, 2)))
        condition = self.module.dealer_call_condition(
            context,
            Bid(2, Denomination.CLUBS),
        )

        self.assertTrue(condition.selects_call)
        self.assertTrue(
            any(
                "(hcp(south) >= 9)" in expression
                and "(hcp(south) <= 14)" in expression
                for expression in condition.expressions
            )
        )


if __name__ == "__main__":
    unittest.main()
