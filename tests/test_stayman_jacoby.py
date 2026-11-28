import tempfile
import unittest
from pathlib import Path
from typing import Callable, Sequence
from unittest.mock import patch

from bridge_sim.deals import CardHand, PBNBoard, Vulnerability
from bridge_sim.model import (
    AuctionEntry,
    Bid,
    Call,
    Denomination,
    DoubleCall,
    HandProfile,
    PassCall,
    Seat,
)
from bridge_sim.auction import run_auction
from bridge_sim.model_assistant import CallRating
from bridge_sim.systems import (
    AuctionStateGenerator,
    AuctionStrength,
    CompositeBiddingSystem,
    DecisionContext,
    ModuleDecision,
    PrivateFitState,
    PublicFitState,
    SuitFitAssessment,
    SuitLengthRange,
    _context_with_public_signals,
    _derive_public_fit,
    load_module,
    load_system,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "systems" / "modules" / "stayman-jacoby.toml"
SYSTEM_PATH = ROOT / "systems" / "standard-stayman-jacoby.toml"
GAME_FORCE_MODULE_PATH = ROOT / "systems" / "modules" / "jump-to-game-in-GF.toml"
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


def _auction_context(
    hand: HandProfile,
    calls: tuple[Call, ...],
    seat: Seat,
    *,
    simulation_evaluator: Callable[[Sequence[Call]], Call] | None = None,
) -> DecisionContext:
    current_seat = Seat.NORTH
    auction: list[AuctionEntry] = []
    for call in calls:
        auction.append(AuctionEntry(current_seat, call))
        current_seat = current_seat.next
    return DecisionContext(
        hand=hand,
        auction=tuple(auction),
        legal_calls=LEGAL_CALLS,
        seat=seat,
        dealer=Seat.NORTH,
        vulnerability=Vulnerability.NONE,
        simulation_evaluator=simulation_evaluator,
    )


def _selected_call(decision: ModuleDecision | None) -> Call:
    if decision is None:
        raise AssertionError("Expected the module to select a call.")
    return decision.call


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
            (9, (5, 4, 2, 2), Bid(2, Denomination.CLUBS)),
            (9, (4, 5, 2, 2), Bid(2, Denomination.CLUBS)),
            (15, (5, 4, 2, 2), Bid(2, Denomination.CLUBS)),
            (15, (4, 5, 2, 2), Bid(2, Denomination.CLUBS)),
            (8, (5, 4, 2, 2), Bid(2, Denomination.HEARTS)),
            (8, (4, 5, 2, 2), Bid(2, Denomination.DIAMONDS)),
            (9, (5, 3, 3, 2), Bid(2, Denomination.HEARTS)),
            (9, (3, 5, 3, 2), Bid(2, Denomination.DIAMONDS)),
            (15, (5, 3, 3, 2), Bid(2, Denomination.HEARTS)),
            (15, (3, 5, 3, 2), Bid(2, Denomination.DIAMONDS)),
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

    def test_stayman_two_clubs_opener_rebids(self) -> None:
        cases = (
            (HandProfile(15, (3, 3, 4, 3)), Bid(2, Denomination.DIAMONDS)),
            (HandProfile(15, (3, 4, 3, 3)), Bid(2, Denomination.HEARTS)),
            (HandProfile(15, (4, 3, 3, 3)), Bid(2, Denomination.SPADES)),
            (HandProfile(15, (4, 4, 3, 2)), Bid(2, Denomination.HEARTS)),
        )
        auction = (
            Bid(1, Denomination.NOTRUMP),
            PassCall.PASS,
            Bid(2, Denomination.CLUBS),
            PassCall.PASS,
        )
        for hand, expected in cases:
            with self.subTest(hand=hand):
                decision = self.module.choose(
                    _auction_context(hand, auction, Seat.NORTH)
                )
                self.assertEqual(_selected_call(decision), expected)

    def test_stayman_two_clubs_responder_continuations_after_major_rebids(self) -> None:
        cases = (
            (Bid(2, Denomination.SPADES), HandProfile(7, (4, 3, 3, 3)), PassCall.PASS),
            (Bid(2, Denomination.SPADES), HandProfile(8, (3, 4, 3, 3)), Bid(2, Denomination.NOTRUMP)),
            (Bid(2, Denomination.SPADES), HandProfile(9, (2, 4, 2, 5)), Bid(3, Denomination.CLUBS)),
            (Bid(2, Denomination.SPADES), HandProfile(9, (2, 4, 5, 2)), Bid(3, Denomination.DIAMONDS)),
            (Bid(2, Denomination.SPADES), HandProfile(15, (4, 3, 3, 3)), Bid(3, Denomination.HEARTS)),
            (Bid(2, Denomination.SPADES), HandProfile(8, (4, 2, 4, 3)), Bid(3, Denomination.SPADES)),
            (Bid(2, Denomination.SPADES), HandProfile(9, (3, 4, 3, 3)), Bid(3, Denomination.NOTRUMP)),
            (Bid(2, Denomination.SPADES), HandProfile(15, (4, 4, 4, 1)), Bid(4, Denomination.CLUBS)),
            (Bid(2, Denomination.SPADES), HandProfile(15, (4, 4, 1, 4)), Bid(4, Denomination.DIAMONDS)),
            (Bid(2, Denomination.SPADES), HandProfile(15, (4, 1, 4, 4)), Bid(4, Denomination.HEARTS)),
            (Bid(2, Denomination.SPADES), HandProfile(9, (4, 3, 3, 3)), Bid(4, Denomination.SPADES)),
            (Bid(2, Denomination.HEARTS), HandProfile(7, (4, 3, 3, 3)), PassCall.PASS),
            (Bid(2, Denomination.HEARTS), HandProfile(8, (4, 3, 3, 3)), Bid(2, Denomination.SPADES)),
            (Bid(2, Denomination.HEARTS), HandProfile(9, (4, 2, 2, 5)), Bid(3, Denomination.CLUBS)),
            (Bid(2, Denomination.HEARTS), HandProfile(15, (4, 2, 5, 2)), Bid(3, Denomination.DIAMONDS)),
            (Bid(2, Denomination.HEARTS), HandProfile(15, (5, 4, 2, 2)), Bid(3, Denomination.SPADES)),
            (Bid(2, Denomination.HEARTS), HandProfile(8, (3, 4, 3, 3)), Bid(3, Denomination.HEARTS)),
            (Bid(2, Denomination.HEARTS), HandProfile(9, (4, 3, 3, 3)), Bid(3, Denomination.NOTRUMP)),
            (Bid(2, Denomination.HEARTS), HandProfile(15, (4, 4, 4, 1)), Bid(4, Denomination.CLUBS)),
            (Bid(2, Denomination.HEARTS), HandProfile(15, (4, 4, 1, 4)), Bid(4, Denomination.DIAMONDS)),
            (Bid(2, Denomination.HEARTS), HandProfile(9, (3, 4, 3, 3)), Bid(4, Denomination.HEARTS)),
        )
        for opener_call, hand, expected in cases:
            with self.subTest(opener_call=opener_call, hand=hand):
                calls = (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.CLUBS),
                    PassCall.PASS,
                    opener_call,
                    PassCall.PASS,
                )
                decision = self.module.choose(
                    _auction_context(hand, calls, Seat.SOUTH)
                )
                self.assertEqual(_selected_call(decision), expected)

    def test_stayman_two_clubs_responder_continuations_after_two_diamonds(self) -> None:
        weak_cases = (
            (HandProfile(7, (5, 5, 2, 1), (3, 3, 1, 0)), Bid(2, Denomination.HEARTS)),
            (HandProfile(7, (5, 5, 2, 1), (4, 3, 0, 0)), Bid(2, Denomination.SPADES)),
            (HandProfile(7, (5, 6, 1, 1), (0, 7, 0, 0)), Bid(2, Denomination.HEARTS)),
            (HandProfile(7, (6, 5, 1, 1), (0, 7, 0, 0)), Bid(2, Denomination.SPADES)),
        )
        continuation_cases = (
            (HandProfile(8, (4, 4, 3, 2)), Bid(2, Denomination.NOTRUMP)),
            (HandProfile(9, (3, 4, 1, 5)), Bid(3, Denomination.CLUBS)),
            (HandProfile(9, (4, 2, 2, 5)), Bid(3, Denomination.CLUBS)),
            (HandProfile(9, (3, 4, 5, 1)), Bid(3, Denomination.DIAMONDS)),
            (HandProfile(9, (4, 2, 5, 2)), Bid(3, Denomination.DIAMONDS)),
            (HandProfile(9, (5, 4, 2, 2)), Bid(3, Denomination.HEARTS)),
            (HandProfile(15, (4, 5, 2, 2)), Bid(3, Denomination.SPADES)),
            (HandProfile(9, (3, 4, 3, 3)), Bid(3, Denomination.NOTRUMP)),
        )
        auction = (
            Bid(1, Denomination.NOTRUMP),
            PassCall.PASS,
            Bid(2, Denomination.CLUBS),
            PassCall.PASS,
            Bid(2, Denomination.DIAMONDS),
            PassCall.PASS,
        )
        for hand, expected in (*weak_cases, *continuation_cases):
            with self.subTest(hand=hand):
                decision = self.module.choose(
                    _auction_context(hand, auction, Seat.SOUTH)
                )
                self.assertEqual(_selected_call(decision), expected)

    def test_stayman_two_clubs_generic_modules_resolve_gf_and_invitation(self) -> None:
        system = load_system(SYSTEM_PATH)
        game_force_context = _auction_context(
            HandProfile(15, (4, 3, 3, 3)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.CLUBS),
                PassCall.PASS,
                Bid(2, Denomination.SPADES),
                PassCall.PASS,
                Bid(3, Denomination.CLUBS),
                PassCall.PASS,
            ),
            Seat.NORTH,
            simulation_evaluator=lambda candidates: (
                Bid(5, Denomination.CLUBS)
                if Bid(5, Denomination.CLUBS) in candidates
                else candidates[0]
            ),
        )
        self.assertEqual(
            system.choose_call(game_force_context),
            Bid(5, Denomination.CLUBS),
        )

        slam_interest_context = _auction_context(
            HandProfile(15, (4, 3, 3, 3)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.CLUBS),
                PassCall.PASS,
                Bid(2, Denomination.SPADES),
                PassCall.PASS,
                Bid(3, Denomination.HEARTS),
                PassCall.PASS,
            ),
            Seat.NORTH,
        )
        self.assertEqual(
            system.choose_call(slam_interest_context),
            Bid(4, Denomination.SPADES),
        )

        invitation_context = _auction_context(
            HandProfile(15, (4, 3, 3, 3)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.CLUBS),
                PassCall.PASS,
                Bid(2, Denomination.SPADES),
                PassCall.PASS,
                Bid(3, Denomination.SPADES),
                PassCall.PASS,
            ),
            Seat.NORTH,
            simulation_evaluator=lambda candidates: (
                Bid(4, Denomination.SPADES)
                if Bid(4, Denomination.SPADES) in candidates
                else candidates[0]
            ),
        )
        self.assertEqual(
            system.choose_call(invitation_context),
            Bid(4, Denomination.SPADES),
        )

        no_major_fit_invitation = _auction_context(
            HandProfile(15, (3, 3, 4, 3)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.CLUBS),
                PassCall.PASS,
                Bid(2, Denomination.DIAMONDS),
                PassCall.PASS,
                Bid(2, Denomination.NOTRUMP),
                PassCall.PASS,
            ),
            Seat.NORTH,
            simulation_evaluator=lambda candidates: (
                Bid(3, Denomination.NOTRUMP)
                if Bid(3, Denomination.NOTRUMP) in candidates
                else candidates[0]
            ),
        )
        self.assertEqual(
            system.choose_call(no_major_fit_invitation),
            Bid(3, Denomination.NOTRUMP),
        )

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

    def test_continuations_are_limited_to_their_auction_stage(self) -> None:
        opener = self.module.choose(
            _auction_context(
                HandProfile(16, (4, 3, 3, 3)),
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(4, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
            )
        )
        self.assertEqual(_selected_call(opener), Bid(4, Denomination.SPADES))

        responder = self.module.choose(
            _auction_context(
                HandProfile(9, (6, 2, 3, 2)),
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(4, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(4, Denomination.SPADES),
                    PassCall.PASS,
                ),
                Seat.SOUTH,
            )
        )
        self.assertEqual(_selected_call(responder), PassCall.PASS)

    def test_public_game_force_and_fit_annotations_drive_game_bids(self) -> None:
        system = load_system(SYSTEM_PATH)
        cases = (
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(3, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
                Bid(3, Denomination.NOTRUMP),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(4, Denomination.DIAMONDS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
                Bid(4, Denomination.HEARTS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(4, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
                Bid(4, Denomination.SPADES),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(3, Denomination.CLUBS),
                    PassCall.PASS,
                    Bid(4, Denomination.CLUBS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
                Bid(5, Denomination.CLUBS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(3, Denomination.CLUBS),
                    PassCall.PASS,
                    Bid(3, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(3, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
                Bid(5, Denomination.DIAMONDS),
            ),
        )
        for calls, seat, expected in cases:
            with self.subTest(calls=calls):
                seen_candidates: list[Call] = []

                def evaluate(
                    candidates: Sequence[Call],
                    selected: Call = expected,
                ) -> Call:
                    seen_candidates.extend(candidates)
                    return selected

                decision = system.choose_call(
                    _auction_context(
                        HandProfile(9, (4, 3, 3, 3)),
                        calls,
                        seat,
                        simulation_evaluator=evaluate,
                    )
                )
                self.assertEqual(decision, expected)
                if expected in (
                    Bid(5, Denomination.CLUBS),
                    Bid(5, Denomination.DIAMONDS),
                ):
                    self.assertIn(Bid(3, Denomination.NOTRUMP), seen_candidates)
                    self.assertIn(expected, seen_candidates)

    def test_game_force_corrects_three_notrump_to_a_known_major_fit(self) -> None:
        system = load_system(SYSTEM_PATH)
        context = _auction_context(
            HandProfile(16, (4, 5, 2, 2)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.CLUBS),
                PassCall.PASS,
                Bid(2, Denomination.HEARTS),
                PassCall.PASS,
                Bid(3, Denomination.NOTRUMP),
                PassCall.PASS,
            ),
            Seat.NORTH,
        )

        self.assertEqual(
            system.choose_call(context),
            Bid(4, Denomination.SPADES),
        )

    def test_generic_game_force_prefers_major_fit_without_simulation(self) -> None:
        module = load_module(GAME_FORCE_MODULE_PATH)
        context = DecisionContext(
            hand=HandProfile(16, (4, 3, 3, 3)),
            auction=(),
            legal_calls=LEGAL_CALLS,
            seat=Seat.NORTH,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
            auction_strength=AuctionStrength.GAME_FORCING,
            fit_states={
                Denomination.HEARTS: SuitFitAssessment(
                    private=PrivateFitState.FIT
                ),
                Denomination.CLUBS: SuitFitAssessment(
                    private=PrivateFitState.FIT
                ),
            },
        )

        decision = module.choose(context)

        self.assertEqual(
            _selected_call(decision),
            Bid(4, Denomination.HEARTS),
        )

    def test_generic_game_force_simulates_minor_game_against_three_notrump(self) -> None:
        module = load_module(GAME_FORCE_MODULE_PATH)
        expected_candidates = (
            Bid(3, Denomination.NOTRUMP),
            Bid(5, Denomination.CLUBS),
        )
        for selected_call in expected_candidates:
            with self.subTest(selected_call=selected_call):
                seen_candidates: list[Call] = []

                def evaluate(candidates: Sequence[Call]) -> Call:
                    seen_candidates.extend(candidates)
                    return selected_call

                context = DecisionContext(
                    hand=HandProfile(16, (4, 3, 3, 3)),
                    auction=(),
                    legal_calls=LEGAL_CALLS,
                    seat=Seat.NORTH,
                    dealer=Seat.NORTH,
                    vulnerability=Vulnerability.NONE,
                    simulation_evaluator=evaluate,
                    auction_strength=AuctionStrength.GAME_FORCING,
                    fit_states={
                        Denomination.CLUBS: SuitFitAssessment(
                            private=PrivateFitState.FIT
                        ),
                    },
                )

                decision = module.choose(context)

                self.assertEqual(_selected_call(decision), selected_call)
                self.assertEqual(tuple(seen_candidates), expected_candidates)

    def test_generic_game_force_requires_simulation_for_minor_fit(self) -> None:
        module = load_module(GAME_FORCE_MODULE_PATH)
        context = DecisionContext(
            hand=HandProfile(16, (4, 3, 3, 3)),
            auction=(),
            legal_calls=LEGAL_CALLS,
            seat=Seat.NORTH,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
            auction_strength=AuctionStrength.GAME_FORCING,
            fit_states={
                Denomination.DIAMONDS: SuitFitAssessment(
                    private=PrivateFitState.FIT
                ),
            },
        )

        with self.assertRaisesRegex(RuntimeError, "requires a simulation evaluator"):
            module.choose(context)

    def test_auction_state_tracks_strength_and_public_private_fits(self) -> None:
        system = load_system(SYSTEM_PATH)
        after_heart_transfer = _auction_context(
            HandProfile(16, (4, 3, 3, 3)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.DIAMONDS),
                PassCall.PASS,
            ),
            Seat.NORTH,
        )
        state = _context_with_public_signals(after_heart_transfer, system.modules)
        self.assertEqual(
            state.auction_strength,
            AuctionStrength.GAME_MIGHT_STILL_BE_POSSIBLE,
        )
        self.assertEqual(
            state.fit_states[Denomination.HEARTS].public,
            PublicFitState.FIT_UNKNOWN,
        )
        self.assertEqual(
            state.fit_states[Denomination.HEARTS].private,
            PrivateFitState.FIT,
        )
        self.assertGreaterEqual(
            state.fit_states[Denomination.HEARTS].partner_length.minimum,
            5,
        )

        after_game_force_without_fit = _auction_context(
            HandProfile(16, (4, 2, 4, 3)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.DIAMONDS),
                PassCall.PASS,
                Bid(2, Denomination.HEARTS),
                PassCall.PASS,
                Bid(3, Denomination.NOTRUMP),
                PassCall.PASS,
            ),
            Seat.NORTH,
        )
        state = _context_with_public_signals(
            after_game_force_without_fit,
            system.modules,
        )
        self.assertEqual(state.auction_strength, AuctionStrength.GAME_FORCING)
        self.assertEqual(
            state.fit_states[Denomination.HEARTS].public,
            PublicFitState.FIT_UNKNOWN,
        )
        self.assertEqual(
            state.fit_states[Denomination.HEARTS].private,
            PrivateFitState.NO_FIT,
        )
        self.assertEqual(
            _derive_public_fit(
                SuitLengthRange(0, 2),
                SuitLengthRange(0, 5),
            ),
            PublicFitState.FIT_REJECTED,
        )

    def test_incremental_auction_state_refines_information_call_by_call(self) -> None:
        system = load_system(SYSTEM_PATH)
        default_strength = AuctionStrength.GAME_MIGHT_STILL_BE_POSSIBLE
        scenarios = (
            (
                "game-force",
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.CLUBS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(3, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                AuctionStrength.GAME_FORCING,
                True,
                False,
                Denomination.SPADES,
                PrivateFitState.FIT,
                None,
            ),
            (
                "transfer-fit",
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                default_strength,
                False,
                False,
                Denomination.HEARTS,
                None,
                PublicFitState.FIT_ESTABLISHED,
            ),
            (
                "invitation",
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.CLUBS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                ),
                default_strength,
                False,
                True,
                None,
                None,
                None,
            ),
            (
                "game-level-transfer",
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(4, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                AuctionStrength.GAME_FORCING,
                False,
                False,
                Denomination.SPADES,
                None,
                PublicFitState.FIT_ESTABLISHED,
            ),
        )
        hand = HandProfile(16, (4, 5, 2, 2))
        for (
            name,
            calls,
            expected_strength,
            expected_game_reached,
            expected_invitation,
            expected_suit,
            expected_private_fit,
            expected_public_fit,
        ) in scenarios:
            with self.subTest(scenario=name):
                generator = AuctionStateGenerator(system.modules)
                history: list[Call] = []
                seat = Seat.NORTH
                for call in calls:
                    before_call = _auction_context(
                        hand,
                        tuple(history),
                        seat,
                    )
                    generator.advance(
                        AuctionEntry(seat, call),
                        before_call,
                    )
                    history.append(call)
                    seat = seat.next
                    current_context = _auction_context(
                        hand,
                        tuple(history),
                        seat,
                    )
                    incremental = generator.context_for(current_context)
                    replayed = _context_with_public_signals(
                        current_context,
                        system.modules,
                    )
                    self.assertEqual(
                        incremental.auction_strength,
                        replayed.auction_strength,
                    )
                    self.assertEqual(incremental.fit_states, replayed.fit_states)
                    self.assertEqual(
                        incremental.game_reached,
                        replayed.game_reached,
                    )
                    self.assertEqual(
                        incremental.invitation,
                        replayed.invitation,
                    )

                incremental = generator.context_for(
                    _auction_context(hand, tuple(history), seat)
                )
                self.assertEqual(incremental.auction_strength, expected_strength)
                self.assertEqual(
                    incremental.game_reached,
                    expected_game_reached,
                )
                self.assertEqual(incremental.invitation, expected_invitation)
                if expected_suit is not None:
                    if expected_private_fit is not None:
                        self.assertEqual(
                            incremental.fit_states[expected_suit].private,
                            expected_private_fit,
                        )
                    if expected_public_fit is not None:
                        self.assertEqual(
                            incremental.fit_states[expected_suit].public,
                            expected_public_fit,
                        )

    def test_auction_strength_can_be_set_to_game_no_longer_possible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            module_path = Path(temporary_directory) / "game-no-longer-possible.toml"
            module_path.write_text(
                """\
schema_version = 1
kind = "module"
id = "test-game-strength"
version = "1.0.0"
name = "Test game strength"

[[public_signals]]
after_opening_calls = ["PASS"]
call = "2NT"
sets_auction_strength = "game_no_longer_possible"

[[rules]]
id = "placeholder"
call = "PASS"

[rules.when]
passout_seat = true
""",
                encoding="utf-8",
            )
            module = load_module(module_path)
            context = _auction_context(
                HandProfile(16, (4, 3, 3, 3)),
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                Seat.NORTH,
            )
            state = _context_with_public_signals(context, (module,))
            self.assertEqual(
                state.auction_strength,
                AuctionStrength.GAME_NO_LONGER_POSSIBLE,
            )

    def test_invitational_responses_simulate_only_known_game_options(self) -> None:
        system = load_system(SYSTEM_PATH)
        cases = (
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                HandProfile(16, (4, 3, 3, 3)),
                [PassCall.PASS, Bid(3, Denomination.NOTRUMP)],
                Bid(3, Denomination.NOTRUMP),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                HandProfile(16, (4, 3, 3, 3)),
                [PassCall.PASS, Bid(4, Denomination.HEARTS)],
                Bid(4, Denomination.HEARTS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(3, Denomination.DIAMONDS),
                    PassCall.PASS,
                ),
                HandProfile(16, (4, 3, 2, 4)),
                [
                    PassCall.PASS,
                    Bid(5, Denomination.DIAMONDS),
                    Bid(3, Denomination.NOTRUMP),
                ],
                Bid(5, Denomination.DIAMONDS),
            ),
        )
        for auction, hand, expected_candidates, selected_call in cases:
            with self.subTest(auction=auction):
                seen_candidates: list[Call] = []

                def evaluate(candidates: Sequence[Call]) -> Call:
                    seen_candidates.extend(candidates)
                    return selected_call

                decision = system.choose_call(
                    _auction_context(
                        hand,
                        auction,
                        Seat.NORTH,
                        simulation_evaluator=evaluate,
                    )
                )
                self.assertEqual(decision, selected_call)
                self.assertEqual(seen_candidates, expected_candidates)

    def test_special_five_four_invitation_keeps_its_custom_simulation(self) -> None:
        system = load_system(SYSTEM_PATH)
        seen_candidates: list[Call] = []

        def evaluate(candidates: Sequence[Call]) -> Call:
            seen_candidates.extend(candidates)
            return PassCall.PASS

        decision = system.choose_call(
            _auction_context(
                HandProfile(16, (4, 3, 3, 3)),
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                ),
                Seat.NORTH,
                simulation_evaluator=evaluate,
            )
        )
        self.assertEqual(decision, PassCall.PASS)
        self.assertEqual(
            seen_candidates,
            [
                PassCall.PASS,
                Bid(2, Denomination.NOTRUMP),
                Bid(3, Denomination.HEARTS),
                Bid(3, Denomination.NOTRUMP),
                Bid(4, Denomination.HEARTS),
                Bid(4, Denomination.SPADES),
            ],
        )

    def test_three_card_major_support_is_shown_over_game_force_minor_bids(self) -> None:
        system = load_system(SYSTEM_PATH)
        cases = (
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(3, Denomination.CLUBS),
                    PassCall.PASS,
                ),
                HandProfile(9, (4, 3, 3, 3)),
                Bid(3, Denomination.HEARTS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(3, Denomination.DIAMONDS),
                    PassCall.PASS,
                ),
                HandProfile(9, (4, 3, 3, 3)),
                Bid(3, Denomination.HEARTS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(3, Denomination.CLUBS),
                    PassCall.PASS,
                ),
                HandProfile(9, (4, 3, 3, 3)),
                Bid(3, Denomination.SPADES),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(3, Denomination.DIAMONDS),
                    PassCall.PASS,
                ),
                HandProfile(9, (4, 3, 3, 3)),
                Bid(3, Denomination.SPADES),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(3, Denomination.CLUBS),
                    PassCall.PASS,
                ),
                HandProfile(9, (2, 2, 4, 5)),
                Bid(5, Denomination.CLUBS),
            ),
        )
        for auction, hand, expected in cases:
            with self.subTest(auction=auction, hand=hand):
                decision = system.choose_call(
                    _auction_context(
                        hand,
                        auction,
                        Seat.NORTH,
                        simulation_evaluator=lambda candidates: expected,
                    )
                )
                self.assertEqual(decision, expected)

        after_fit_raise = _auction_context(
            HandProfile(9, (4, 3, 3, 3)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.DIAMONDS),
                PassCall.PASS,
                Bid(2, Denomination.HEARTS),
                PassCall.PASS,
                Bid(3, Denomination.CLUBS),
                PassCall.PASS,
                Bid(3, Denomination.HEARTS),
                PassCall.PASS,
            ),
            Seat.SOUTH,
        )
        self.assertEqual(
            system.choose_call(after_fit_raise),
            Bid(4, Denomination.HEARTS),
        )

        after_spade_fit_raise = _auction_context(
            HandProfile(9, (4, 3, 3, 3)),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.HEARTS),
                PassCall.PASS,
                Bid(2, Denomination.SPADES),
                PassCall.PASS,
                Bid(3, Denomination.DIAMONDS),
                PassCall.PASS,
                Bid(3, Denomination.SPADES),
                PassCall.PASS,
            ),
            Seat.SOUTH,
        )
        self.assertEqual(
            system.choose_call(after_spade_fit_raise),
            Bid(4, Denomination.SPADES),
        )

    def test_nt_signoffs_after_jacoby_transfers_are_simulated_with_fit(self) -> None:
        system = load_system(SYSTEM_PATH)
        cases = (
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                Bid(4, Denomination.HEARTS),
                [PassCall.PASS, Bid(4, Denomination.HEARTS)],
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(3, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                Bid(4, Denomination.HEARTS),
                [PassCall.PASS, Bid(4, Denomination.HEARTS)],
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                Bid(4, Denomination.SPADES),
                [PassCall.PASS, Bid(4, Denomination.SPADES)],
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(3, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                Bid(4, Denomination.SPADES),
                [PassCall.PASS, Bid(4, Denomination.SPADES)],
            ),
        )
        for auction, selected_call, expected_candidates in cases:
            with self.subTest(auction=auction):
                seen_candidates: list[Call] = []

                def evaluate(candidates: Sequence[Call]) -> Call:
                    seen_candidates.extend(candidates)
                    return selected_call

                decision = system.choose_call(
                    _auction_context(
                        HandProfile(16, (4, 3, 3, 3)),
                        auction,
                        Seat.NORTH,
                        simulation_evaluator=evaluate,
                    )
                )
                self.assertEqual(decision, selected_call)
                self.assertEqual(seen_candidates, expected_candidates)

        gf_context_base = _auction_context(
            HandProfile(16, (4, 3, 3, 3)),
            cases[0][0],
            Seat.NORTH,
        )
        gf_context = DecisionContext(
            hand=gf_context_base.hand,
            auction=gf_context_base.auction,
            legal_calls=LEGAL_CALLS,
            seat=Seat.NORTH,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
            auction_strength=AuctionStrength.GAME_FORCING,
        )
        self.assertEqual(system.choose_call(gf_context), Bid(4, Denomination.HEARTS))

    def test_three_notrump_over_major_transfers_allows_game_force_and_slam_interest(
        self,
    ) -> None:
        system = load_system(SYSTEM_PATH)
        cases = (
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                ),
                (5, 3, 2, 3),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                (3, 5, 2, 3),
            ),
        )
        for auction, shape in cases:
            for hcp in (14, 15):
                with self.subTest(auction=auction, hcp=hcp):
                    decision = system.choose_call(
                        _auction_context(
                            HandProfile(hcp, shape),
                            auction,
                            Seat.SOUTH,
                        )
                    )
                    self.assertEqual(
                        decision,
                        Bid(3, Denomination.NOTRUMP),
                    )

    def test_stayman_three_notrump_after_any_reply_accepts_slam_interest(self) -> None:
        system = load_system(SYSTEM_PATH)
        cases = (
            (Bid(2, Denomination.SPADES), (3, 4, 3, 3)),
            (Bid(2, Denomination.HEARTS), (4, 3, 3, 3)),
            (Bid(2, Denomination.DIAMONDS), (3, 4, 3, 3)),
            (Bid(2, Denomination.DIAMONDS), (4, 3, 3, 3)),
        )
        for opener_reply, shape in cases:
            auction = (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.CLUBS),
                PassCall.PASS,
                opener_reply,
                PassCall.PASS,
            )
            with self.subTest(opener_reply=opener_reply, shape=shape):
                decision = system.choose_call(
                    _auction_context(
                        HandProfile(15, shape),
                        auction,
                        Seat.SOUTH,
                    )
                )
                self.assertEqual(decision, Bid(3, Denomination.NOTRUMP))

    def test_game_force_persists_and_private_strength_is_not_a_signal(self) -> None:
        system = load_system(SYSTEM_PATH)
        already_at_game = _context(
            HandProfile(9, (4, 3, 3, 3)),
            (
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
                AuctionEntry(Seat.EAST, PassCall.PASS),
                AuctionEntry(Seat.SOUTH, Bid(3, Denomination.HEARTS)),
                AuctionEntry(Seat.WEST, PassCall.PASS),
                AuctionEntry(Seat.NORTH, Bid(3, Denomination.NOTRUMP)),
                AuctionEntry(Seat.EAST, PassCall.PASS),
            ),
            seat=Seat.SOUTH,
        )
        self.assertEqual(system.choose_call(already_at_game), PassCall.PASS)

        no_public_force = _context(
            HandProfile(14, (4, 3, 3, 3)),
            (
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
                AuctionEntry(Seat.EAST, PassCall.PASS),
                AuctionEntry(Seat.SOUTH, Bid(2, Denomination.CLUBS)),
                AuctionEntry(Seat.WEST, PassCall.PASS),
            ),
            seat=Seat.NORTH,
        )
        self.assertEqual(
            system.choose_call(no_public_force),
            Bid(2, Denomination.SPADES),
        )

        slam_interest = DecisionContext(
            hand=HandProfile(16, (4, 3, 3, 3)),
            auction=(
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
                AuctionEntry(Seat.EAST, PassCall.PASS),
                AuctionEntry(Seat.SOUTH, Bid(2, Denomination.HEARTS)),
                AuctionEntry(Seat.WEST, PassCall.PASS),
                AuctionEntry(Seat.NORTH, Bid(2, Denomination.SPADES)),
                AuctionEntry(Seat.EAST, PassCall.PASS),
                AuctionEntry(Seat.SOUTH, Bid(4, Denomination.CLUBS)),
                AuctionEntry(Seat.WEST, PassCall.PASS),
            ),
            legal_calls=frozenset(
                {
                    PassCall.PASS,
                    Bid(4, Denomination.DIAMONDS),
                    Bid(4, Denomination.HEARTS),
                    Bid(4, Denomination.SPADES),
                    Bid(5, Denomination.CLUBS),
                    Bid(5, Denomination.DIAMONDS),
                }
            ),
            seat=Seat.NORTH,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
        )
        self.assertEqual(system.choose_call(slam_interest), Bid(4, Denomination.SPADES))

    def test_opener_continuations_over_two_notrump(self) -> None:
        auction = (
            Bid(1, Denomination.NOTRUMP),
            PassCall.PASS,
            Bid(2, Denomination.NOTRUMP),
            PassCall.PASS,
        )
        for hcp, expected in (
            (15, PassCall.PASS),
            (16, Bid(3, Denomination.NOTRUMP)),
            (17, Bid(3, Denomination.NOTRUMP)),
        ):
            with self.subTest(hcp=hcp):
                decision = self.module.choose(
                    _auction_context(
                        HandProfile(hcp, (4, 3, 3, 3)),
                        auction,
                        Seat.NORTH,
                    )
                )
                self.assertEqual(_selected_call(decision), expected)

    def test_three_club_diamond_relay_responder_actions(self) -> None:
        auction = (
            Bid(1, Denomination.NOTRUMP),
            PassCall.PASS,
            Bid(3, Denomination.CLUBS),
            PassCall.PASS,
            Bid(3, Denomination.DIAMONDS),
            PassCall.PASS,
        )
        cases = (
            (HandProfile(7, (2, 2, 7, 2)), PassCall.PASS),
            (HandProfile(9, (1, 2, 8, 2)), Bid(3, Denomination.SPADES)),
            (HandProfile(9, (2, 1, 8, 2)), Bid(3, Denomination.HEARTS)),
            (HandProfile(9, (2, 2, 7, 2)), Bid(3, Denomination.NOTRUMP)),
            (HandProfile(15, (2, 2, 8, 1)), Bid(4, Denomination.CLUBS)),
            (HandProfile(15, (2, 2, 7, 2)), Bid(4, Denomination.DIAMONDS)),
        )
        for hand, expected in cases:
            with self.subTest(hand=hand):
                decision = self.module.choose(
                    _auction_context(hand, auction, Seat.SOUTH)
                )
                self.assertEqual(_selected_call(decision), expected)

    def test_two_spade_club_relay_and_responder_strength_actions(self) -> None:
        opener_auction = (
            Bid(1, Denomination.NOTRUMP),
            PassCall.PASS,
            Bid(2, Denomination.SPADES),
            PassCall.PASS,
        )
        opener_cases = (
            (
                HandProfile(15, (3, 3, 3, 4), (4, 4, 5, 2)),
                Bid(2, Denomination.NOTRUMP),
            ),
            (
                HandProfile(15, (3, 3, 3, 4), (5, 4, 5, 1)),
                Bid(3, Denomination.CLUBS),
            ),
        )
        for hand, expected in opener_cases:
            with self.subTest(hand=hand):
                decision = self.module.choose(
                    _auction_context(hand, opener_auction, Seat.NORTH)
                )
                self.assertEqual(_selected_call(decision), expected)

        responder_cases = (
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(3, Denomination.CLUBS),
                    PassCall.PASS,
                ),
                HandProfile(7, (3, 3, 3, 4)),
                PassCall.PASS,
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(3, Denomination.CLUBS),
                    PassCall.PASS,
                ),
                HandProfile(9, (2, 3, 1, 7)),
                Bid(3, Denomination.DIAMONDS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                HandProfile(7, (3, 3, 3, 4)),
                Bid(3, Denomination.CLUBS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                HandProfile(8, (3, 3, 3, 4)),
                Bid(3, Denomination.NOTRUMP),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                HandProfile(9, (2, 1, 3, 7)),
                Bid(3, Denomination.HEARTS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                HandProfile(9, (2, 1, 1, 9)),
                Bid(3, Denomination.DIAMONDS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.SPADES),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                HandProfile(15, (2, 2, 3, 6)),
                Bid(4, Denomination.CLUBS),
            ),
        )
        for auction, hand, expected in responder_cases:
            with self.subTest(auction=auction, hand=hand):
                decision = self.module.choose(
                    _auction_context(hand, auction, Seat.SOUTH)
                )
                self.assertEqual(_selected_call(decision), expected)

    def test_two_heart_and_two_diamond_continuations(self) -> None:
        opener_cases = (
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.HEARTS), PassCall.PASS),
                HandProfile(15, (4, 4, 3, 2)),
                Bid(3, Denomination.SPADES),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.HEARTS), PassCall.PASS),
                HandProfile(15, (4, 3, 3, 3)),
                Bid(2, Denomination.SPADES),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.HEARTS), PassCall.PASS),
                HandProfile(16, (4, 4, 3, 2)),
                Bid(2, Denomination.NOTRUMP),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.HEARTS), PassCall.PASS),
                HandProfile(15, (5, 3, 3, 2)),
                Bid(3, Denomination.SPADES),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.HEARTS), PassCall.PASS),
                HandProfile(16, (5, 3, 3, 2)),
                Bid(2, Denomination.NOTRUMP),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.HEARTS), PassCall.PASS),
                HandProfile(16, (4, 3, 3, 3)),
                Bid(2, Denomination.SPADES),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.HEARTS), PassCall.PASS),
                HandProfile(18, (4, 4, 3, 2)),
                Bid(2, Denomination.SPADES),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.DIAMONDS), PassCall.PASS),
                HandProfile(15, (3, 4, 4, 2)),
                Bid(3, Denomination.HEARTS),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.DIAMONDS), PassCall.PASS),
                HandProfile(15, (3, 5, 3, 2)),
                Bid(3, Denomination.HEARTS),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.DIAMONDS), PassCall.PASS),
                HandProfile(17, (2, 5, 3, 3)),
                Bid(2, Denomination.NOTRUMP),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.DIAMONDS), PassCall.PASS),
                HandProfile(15, (3, 4, 3, 3)),
                Bid(2, Denomination.HEARTS),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.DIAMONDS), PassCall.PASS),
                HandProfile(16, (3, 4, 3, 3)),
                Bid(2, Denomination.HEARTS),
            ),
            (
                (Bid(1, Denomination.NOTRUMP), PassCall.PASS,
                 Bid(2, Denomination.DIAMONDS), PassCall.PASS),
                HandProfile(17, (4, 4, 3, 2)),
                Bid(2, Denomination.NOTRUMP),
            ),
        )
        for auction, hand, expected in opener_cases:
            with self.subTest(auction=auction, hand=hand):
                decision = self.module.choose(
                    _auction_context(hand, auction, Seat.NORTH)
                )
                self.assertEqual(_selected_call(decision), expected)

        dealer_condition = self.module.dealer_call_condition(
            _auction_context(
                HandProfile(16, (2, 5, 3, 3)),
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
            ),
            Bid(2, Denomination.NOTRUMP),
        )
        self.assertTrue(dealer_condition.selects_call)
        self.assertIn(
            "!(shape(north, any 4333))",
            " ".join(dealer_condition.expressions),
        )

        continuation_cases = (
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                Seat.SOUTH,
                Bid(3, Denomination.HEARTS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(3, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
                Bid(3, Denomination.SPADES),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                ),
                Seat.SOUTH,
                Bid(3, Denomination.DIAMONDS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(3, Denomination.DIAMONDS),
                    PassCall.PASS,
                ),
                Seat.NORTH,
                Bid(3, Denomination.HEARTS),
            ),
        )
        for auction, seat, expected in continuation_cases:
            with self.subTest(auction=auction):
                decision = self.module.choose(
                    _auction_context(HandProfile(16, (4, 3, 3, 3)), auction, seat)
                )
                self.assertEqual(_selected_call(decision), expected)

    def test_spade_and_heart_game_choices_are_simulated_after_transfer(self) -> None:
        cases = (
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(3, Denomination.SPADES),
                    PassCall.PASS,
                ),
                Seat.SOUTH,
                Bid(4, Denomination.SPADES),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(3, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(3, Denomination.SPADES),
                    PassCall.PASS,
                ),
                Seat.SOUTH,
                Bid(4, Denomination.SPADES),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(3, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                Seat.SOUTH,
                Bid(4, Denomination.HEARTS),
            ),
            (
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(2, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(3, Denomination.DIAMONDS),
                    PassCall.PASS,
                    Bid(3, Denomination.HEARTS),
                    PassCall.PASS,
                ),
                Seat.SOUTH,
                Bid(4, Denomination.HEARTS),
            ),
        )
        for auction, seat, selected_call in cases:
            with self.subTest(auction=auction):
                seen_candidates: list[Call] = []

                def evaluate(candidates: Sequence[Call]) -> Call:
                    seen_candidates.extend(candidates)
                    return selected_call

                decision = self.module.choose(
                    _auction_context(
                        HandProfile(16, (4, 3, 3, 3)),
                        auction,
                        seat,
                        simulation_evaluator=evaluate,
                    )
                )
                self.assertEqual(_selected_call(decision), selected_call)
                self.assertEqual(seen_candidates, [PassCall.PASS, selected_call])

        seen_candidates = []

        def pass_three_spades(candidates: Sequence[Call]) -> Call:
            seen_candidates.extend(candidates)
            return PassCall.PASS

        pass_decision = self.module.choose(
            _auction_context(
                HandProfile(8, (5, 3, 3, 2)),
                (
                    Bid(1, Denomination.NOTRUMP),
                    PassCall.PASS,
                    Bid(2, Denomination.HEARTS),
                    PassCall.PASS,
                    Bid(3, Denomination.SPADES),
                    PassCall.PASS,
                ),
                Seat.SOUTH,
                simulation_evaluator=pass_three_spades,
            )
        )
        self.assertEqual(_selected_call(pass_decision), PassCall.PASS)
        self.assertEqual(
            seen_candidates,
            [PassCall.PASS, Bid(4, Denomination.SPADES)],
        )

    def test_responder_calls_after_two_spades_or_two_hearts_relay(self) -> None:
        spade_cases = (
            (HandProfile(8, (5, 3, 3, 2)), Bid(2, Denomination.NOTRUMP)),
            (HandProfile(9, (5, 2, 2, 4)), Bid(3, Denomination.CLUBS)),
            (HandProfile(9, (5, 2, 4, 2)), Bid(3, Denomination.DIAMONDS)),
            (HandProfile(9, (5, 5, 2, 1)), Bid(3, Denomination.HEARTS)),
            (HandProfile(8, (6, 3, 2, 2)), Bid(3, Denomination.SPADES)),
            (HandProfile(9, (5, 3, 3, 2)), Bid(3, Denomination.NOTRUMP)),
            (HandProfile(15, (6, 3, 3, 1)), Bid(4, Denomination.CLUBS)),
            (HandProfile(15, (6, 3, 1, 3)), Bid(4, Denomination.DIAMONDS)),
            (HandProfile(15, (6, 1, 3, 3)), Bid(4, Denomination.HEARTS)),
            (HandProfile(15, (6, 2, 3, 2)), Bid(4, Denomination.SPADES)),
        )
        spade_auction = (
            Bid(1, Denomination.NOTRUMP),
            PassCall.PASS,
            Bid(2, Denomination.HEARTS),
            PassCall.PASS,
            Bid(2, Denomination.SPADES),
            PassCall.PASS,
        )
        heart_cases = (
            (HandProfile(8, (3, 5, 3, 2)), Bid(2, Denomination.NOTRUMP)),
            (HandProfile(9, (2, 5, 2, 4)), Bid(3, Denomination.CLUBS)),
            (HandProfile(9, (2, 5, 4, 2)), Bid(3, Denomination.DIAMONDS)),
            (HandProfile(9, (5, 5, 1, 2)), Bid(3, Denomination.SPADES)),
            (HandProfile(8, (3, 6, 2, 2)), Bid(3, Denomination.HEARTS)),
            (HandProfile(9, (3, 5, 3, 2)), Bid(3, Denomination.NOTRUMP)),
            (HandProfile(15, (3, 6, 3, 1)), Bid(4, Denomination.CLUBS)),
            (HandProfile(15, (3, 6, 1, 3)), Bid(4, Denomination.DIAMONDS)),
            (HandProfile(15, (1, 6, 3, 3)), Bid(4, Denomination.SPADES)),
            (HandProfile(15, (2, 6, 3, 2)), Bid(4, Denomination.HEARTS)),
        )
        heart_auction = (
            Bid(1, Denomination.NOTRUMP),
            PassCall.PASS,
            Bid(2, Denomination.DIAMONDS),
            PassCall.PASS,
            Bid(2, Denomination.HEARTS),
            PassCall.PASS,
        )
        for auction, cases in (
            (spade_auction, spade_cases),
            (heart_auction, heart_cases),
        ):
            for hand, expected in cases:
                with self.subTest(auction=auction, hand=hand):
                    decision = self.module.choose(
                        _auction_context(hand, auction, Seat.SOUTH)
                    )
                    self.assertEqual(_selected_call(decision), expected)

    def test_invitational_five_hearts_four_spades_over_two_hearts(self) -> None:
        responder_auction = (
            Bid(1, Denomination.NOTRUMP),
            PassCall.PASS,
            Bid(2, Denomination.DIAMONDS),
            PassCall.PASS,
            Bid(2, Denomination.HEARTS),
            PassCall.PASS,
        )
        responder = self.module.choose(
            _auction_context(
                HandProfile(8, (4, 5, 2, 2)),
                responder_auction,
                Seat.SOUTH,
            )
        )
        self.assertEqual(_selected_call(responder), Bid(2, Denomination.SPADES))

        seen_candidates: list[Call] = []

        def evaluate(candidates: Sequence[Call]) -> Call:
            seen_candidates.extend(candidates)
            return Bid(4, Denomination.HEARTS)

        opener = self.module.choose(
            _auction_context(
                HandProfile(16, (4, 3, 3, 3)),
                (*responder_auction, Bid(2, Denomination.SPADES), PassCall.PASS),
                Seat.NORTH,
                simulation_evaluator=evaluate,
            )
        )
        self.assertEqual(_selected_call(opener), Bid(4, Denomination.HEARTS))
        self.assertEqual(
            seen_candidates,
            [
                PassCall.PASS,
                Bid(2, Denomination.NOTRUMP),
                Bid(3, Denomination.HEARTS),
                Bid(3, Denomination.NOTRUMP),
                Bid(4, Denomination.HEARTS),
                Bid(4, Denomination.SPADES),
            ],
        )

    def test_opener_passes_after_responder_selects_the_game(self) -> None:
        completed_auctions = (
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.HEARTS),
                PassCall.PASS,
                Bid(3, Denomination.SPADES),
                PassCall.PASS,
                Bid(4, Denomination.SPADES),
                PassCall.PASS,
            ),
            (
                Bid(1, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(2, Denomination.DIAMONDS),
                PassCall.PASS,
                Bid(2, Denomination.NOTRUMP),
                PassCall.PASS,
                Bid(3, Denomination.DIAMONDS),
                PassCall.PASS,
                Bid(3, Denomination.HEARTS),
                PassCall.PASS,
                Bid(4, Denomination.HEARTS),
                PassCall.PASS,
            ),
        )
        for auction in completed_auctions:
            with self.subTest(auction=auction):
                decision = self.module.choose(
                    _auction_context(
                        HandProfile(16, (4, 3, 3, 3)),
                        auction,
                        Seat.NORTH,
                    )
                )
                self.assertEqual(_selected_call(decision), PassCall.PASS)

    def test_simulation_rule_uses_configured_legal_candidates(self) -> None:
        cases = (
            (
                Bid(3, Denomination.DIAMONDS),
                [PassCall.PASS, Bid(3, Denomination.NOTRUMP)],
            ),
            (
                Bid(3, Denomination.SPADES),
                [
                    Bid(3, Denomination.NOTRUMP),
                    Bid(4, Denomination.HEARTS),
                    Bid(5, Denomination.CLUBS),
                    Bid(5, Denomination.DIAMONDS),
                ],
            ),
            (
                Bid(3, Denomination.HEARTS),
                [
                    Bid(3, Denomination.NOTRUMP),
                    Bid(4, Denomination.SPADES),
                    Bid(5, Denomination.CLUBS),
                    Bid(5, Denomination.DIAMONDS),
                ],
            )
        )
        for response, expected_candidates in cases:
            with self.subTest(response=response):
                seen_candidates: list[Call] = []

                def evaluate(candidates: Sequence[Call]) -> Call:
                    seen_candidates.extend(candidates)
                    return Bid(3, Denomination.NOTRUMP)

                decision = self.module.choose(
                    _auction_context(
                        HandProfile(16, (4, 3, 3, 3)),
                        (
                            Bid(1, Denomination.NOTRUMP),
                            PassCall.PASS,
                            response,
                            PassCall.PASS,
                        ),
                        Seat.NORTH,
                        simulation_evaluator=evaluate,
                    )
                )
                self.assertEqual(
                    _selected_call(decision),
                    Bid(3, Denomination.NOTRUMP),
                )
                self.assertEqual(seen_candidates, expected_candidates)

    def test_run_auction_executes_simulation_for_pbn_deals(self) -> None:
        system = CompositeBiddingSystem(
            "stayman-test",
            "1.0.0",
            "Stayman test",
            (
                load_module(ROOT / "systems/modules/standard-openings.toml"),
                self.module,
            ),
            PassCall.PASS,
        )
        board = PBNBoard(
            1,
            Seat.NORTH,
            Vulnerability.NONE,
            {
                Seat.NORTH: CardHand(("AKQ7", "A43", "K32", "432")),
                Seat.EAST: CardHand(("JT85", "JT9", "AQT", "987")),
                Seat.SOUTH: CardHand(("96", "KQ2", "J98765", "Q5")),
                Seat.WEST: CardHand(("432", "8765", "4", "AKJT6")),
            },
        )
        expected_call = Bid(3, Denomination.NOTRUMP)
        with patch(
            "bridge_sim.model_assistant.rate_candidate_calls",
            return_value=(
                CallRating(PassCall.PASS, 0.0),
                CallRating(expected_call, 1.0),
            ),
        ) as rate_calls:
            result = run_auction(
                board,
                system,
                system,
                simulation_n=1,
                simulation_seed=12345,
            )

        self.assertIn(AuctionEntry(Seat.NORTH, expected_call), result.entries)
        rate_calls.assert_called_once()
        self.assertEqual(rate_calls.call_args.kwargs["n"], 1)
        self.assertEqual(rate_calls.call_args.kwargs["seed"], 12345)
        self.assertEqual(
            rate_calls.call_args.kwargs["candidates"],
            (PassCall.PASS, expected_call),
        )


if __name__ == "__main__":
    unittest.main()
