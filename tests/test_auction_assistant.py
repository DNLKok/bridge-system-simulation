import tempfile
import unittest
from pathlib import Path

from bridge_sim.auction_assistant import generate_dealer_script
from bridge_sim.deals import CardHand, DealerProvider, Vulnerability
from bridge_sim.model import AuctionEntry, Bid, Denomination, PassCall, Seat


ROOT = Path(__file__).resolve().parents[1]
ONE_NT_SYSTEM = ROOT / "systems" / "one-no-trump-15-17.toml"
PASSOUT_SYSTEM = ROOT / "systems" / "passout-cheapest-call.toml"
PREEMPT_SYSTEM = ROOT / "systems" / "traditional-preempts.toml"
SIMPLE_OVERCALL_SYSTEM = ROOT / "systems" / "simple-overcalls.toml"
STANDARD_OPENINGS_SYSTEM = ROOT / "systems" / "standard-openings.toml"
HAND = CardHand(("AKQ", "JT9", "876", "5432"))


class AuctionAssistantTests(unittest.TestCase):
    def test_standard_opening_dealer_constraints_enforce_rule_of_20_and_suit_choice(
        self,
    ) -> None:
        script = generate_dealer_script(
            perspective=Seat.EAST,
            hand=HAND,
            auction=(AuctionEntry(Seat.NORTH, Bid(1, Denomination.SPADES)),),
            partner_system=STANDARD_OPENINGS_SYSTEM,
            opponent_system=STANDARD_OPENINGS_SYSTEM,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
            generate=1_000_000,
        )

        self.assertIn(
            "((hcp(north) + spades(north) + hearts(north)) >= 20)",
            script,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "standard-opening.dealer"
            path.write_text(script, encoding="utf-8")
            board = next(DealerProvider().generate(path))

        hand = board.hands[Seat.NORTH]
        spades, hearts, diamonds, clubs = hand.shape
        rule_of_20 = hand.hcp + sum(sorted(hand.shape, reverse=True)[:2])
        self.assertGreaterEqual(rule_of_20, 20)
        self.assertGreaterEqual(spades, 5)
        self.assertFalse(
            (hearts >= 5 and hearts > spades)
            or (diamonds >= 4 and diamonds > spades)
            or (clubs >= 2 and clubs > spades)
        )

    def test_generates_predeal_and_positive_rule_constraint_for_prior_bid(self) -> None:
        script = generate_dealer_script(
            perspective=Seat.EAST,
            hand=HAND,
            auction=(
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
            ),
            partner_system=PASSOUT_SYSTEM,
            opponent_system=ONE_NT_SYSTEM,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NORTH_SOUTH,
            produce=2,
        )

        self.assertIn("generate 10000000000", script)
        self.assertIn("produce 2", script)
        self.assertIn("vulnerable ns", script)
        self.assertIn("predeal east SAKQ, HJT9, D876, C5432", script)
        self.assertIn("(hcp(north) >= 15)", script)
        self.assertIn(
            "shape(north, any 4333 + any 4432 + any 5332)",
            script,
        )
        self.assertIn("action printpbn", script)

    def test_constrains_pass_by_negating_the_available_one_nt_rule(self) -> None:
        script = generate_dealer_script(
            perspective=Seat.EAST,
            hand=HAND,
            auction=(AuctionEntry(Seat.NORTH, PassCall.PASS),),
            partner_system=ONE_NT_SYSTEM,
            opponent_system=ONE_NT_SYSTEM,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
        )

        self.assertIn(
            "condition (!((hcp(north) >= 15)",
            script,
        )

    def test_translates_suit_preempt_constraints_to_dealer_syntax(self) -> None:
        script = generate_dealer_script(
            perspective=Seat.EAST,
            hand=HAND,
            auction=(AuctionEntry(Seat.NORTH, Bid(2, Denomination.DIAMONDS)),),
            partner_system=PASSOUT_SYSTEM,
            opponent_system=PREEMPT_SYSTEM,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
        )

        self.assertIn("(hcp(north) >= 4)", script)
        self.assertIn("(hcp(north) <= 10)", script)
        self.assertIn("(diamonds(north) >= 6)", script)
        self.assertIn("(diamonds(north) <= 6)", script)
        self.assertIn("(hcp(north, diamonds) >= 4)", script)
        self.assertIn("(spades(north) <= 3)", script)
        self.assertIn("(hearts(north) <= 3)", script)
        self.assertIn("(clubs(north) > diamonds(north))", script)

    def test_translates_and_executes_major_overcall_stopper_constraints(self) -> None:
        script = generate_dealer_script(
            perspective=Seat.SOUTH,
            hand=HAND,
            auction=(
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.HEARTS)),
                AuctionEntry(Seat.EAST, Bid(1, Denomination.NOTRUMP)),
            ),
            partner_system=PASSOUT_SYSTEM,
            opponent_system=SIMPLE_OVERCALL_SYSTEM,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
            unconstrained_entries=(0,),
            generate=1_000_000,
        )

        self.assertIn("hascard(east, AH)", script)
        self.assertIn("hascard(east, KH)", script)
        self.assertIn("(hearts(east) >= 3)", script)
        self.assertIn("(hearts(east) >= 4)", script)
        self.assertIn("(hearts(east) >= 5)", script)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simple-overcall.dealer"
            path.write_text(script, encoding="utf-8")
            board = next(DealerProvider().generate(path))

        hand = board.hands[Seat.EAST]
        hearts = hand.suits[1]
        stopped = (
            "A" in hearts
            or "K" in hearts
            or ("Q" in hearts and len(hearts) >= 3)
            or ("J" in hearts and len(hearts) >= 4)
            or len(hearts) >= 5
        )
        self.assertIn(hand.hcp, range(15, 19))
        self.assertIn(
            hand.profile.canonical_shape,
            {(4, 3, 3, 3), (4, 4, 3, 2), (5, 3, 3, 2)},
        )
        self.assertTrue(stopped)

    def test_dealer_constraints_preserve_longest_qualifying_suit_overcall(self) -> None:
        script = generate_dealer_script(
            perspective=Seat.SOUTH,
            hand=HAND,
            auction=(
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.HEARTS)),
                AuctionEntry(Seat.EAST, Bid(2, Denomination.CLUBS)),
            ),
            partner_system=PASSOUT_SYSTEM,
            opponent_system=SIMPLE_OVERCALL_SYSTEM,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
            unconstrained_entries=(0,),
            generate=1_000_000,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simple-suit-overcall.dealer"
            path.write_text(script, encoding="utf-8")
            board = next(DealerProvider().generate(path))

        hand = board.hands[Seat.EAST]
        spades, _, diamonds, clubs = hand.shape
        hcp = hand.hcp
        club_qualifies = (
            (hcp >= 12 and clubs >= 5)
            or (hcp >= 10 and clubs >= 6)
        )
        spade_qualifies = hcp >= 9 and spades >= 5
        diamond_qualifies = (
            (hcp >= 12 and diamonds >= 5)
            or (hcp >= 10 and diamonds >= 6)
        )
        self.assertTrue(club_qualifies)
        self.assertLessEqual(hcp, 17)
        self.assertFalse(spade_qualifies and spades >= clubs)
        self.assertFalse(diamond_qualifies and diamonds > clubs)

    def test_rejects_a_call_inconsistent_with_a_system(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            generate_dealer_script(
                perspective=Seat.EAST,
                hand=HAND,
                auction=(
                    AuctionEntry(
                        Seat.NORTH,
                        Bid(1, Denomination.NOTRUMP),
                    ),
                ),
                partner_system=PASSOUT_SYSTEM,
                opponent_system=PASSOUT_SYSTEM,
                dealer=Seat.NORTH,
                vulnerability=Vulnerability.NONE,
            )

    def test_requires_the_perspective_to_be_next_to_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "currently awaiting"):
            generate_dealer_script(
                perspective=Seat.NORTH,
                hand=HAND,
                auction=(),
                partner_system=ONE_NT_SYSTEM,
                opponent_system=ONE_NT_SYSTEM,
                dealer=Seat.EAST,
                vulnerability=Vulnerability.NONE,
            )

    def test_skips_policy_constraints_for_hypothetical_auction_entries(self) -> None:
        script = generate_dealer_script(
            perspective=Seat.SOUTH,
            hand=HAND,
            auction=(
                AuctionEntry(Seat.NORTH, Bid(1, Denomination.CLUBS)),
                AuctionEntry(Seat.EAST, PassCall.PASS),
            ),
            partner_system=PASSOUT_SYSTEM,
            opponent_system=PASSOUT_SYSTEM,
            dealer=Seat.NORTH,
            vulnerability=Vulnerability.NONE,
            unconstrained_entries=(0, 1),
        )

        self.assertIn("action printpbn", script)
        self.assertNotIn("condition ", script)

    def test_generated_script_runs_with_dealer_and_predeals_known_hand(self) -> None:
        auctions = (
            (
                Seat.EAST,
                (AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),),
                True,
            ),
            (
                Seat.EAST,
                (AuctionEntry(Seat.NORTH, PassCall.PASS),),
                False,
            ),
            (
                Seat.WEST,
                (
                    AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
                    AuctionEntry(Seat.EAST, PassCall.PASS),
                    AuctionEntry(Seat.SOUTH, PassCall.PASS),
                ),
                True,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            for index, (perspective, auction, expected_one_nt) in enumerate(auctions):
                with self.subTest(expected_one_nt=expected_one_nt):
                    script_path = Path(temporary_directory) / f"assistant-{index}.txt"
                    script_path.write_text(
                        generate_dealer_script(
                            perspective=perspective,
                            hand=HAND,
                            auction=auction,
                            partner_system=PASSOUT_SYSTEM,
                            opponent_system=ONE_NT_SYSTEM,
                            dealer=Seat.NORTH,
                            vulnerability=Vulnerability.NONE,
                        ),
                        encoding="utf-8",
                    )
                    boards = tuple(
                        DealerProvider().generate(script_path)
                    )

                self.assertEqual(len(boards), 1)
                self.assertEqual(boards[0].hands[perspective], HAND)
                north = boards[0].hand_for(Seat.NORTH)
                matches_one_nt = (
                    15 <= north.hcp <= 17
                    and north.canonical_shape
                    in {(4, 3, 3, 3), (4, 4, 3, 2), (5, 3, 3, 2)}
                )
                self.assertEqual(matches_one_nt, expected_one_nt)


if __name__ == "__main__":
    unittest.main()
