import tempfile
import unittest
from pathlib import Path

from bridge_sim.auction_assistant import generate_dealer_script
from bridge_sim.deals import CardHand, DealerProvider, Vulnerability
from bridge_sim.model import AuctionEntry, Bid, Denomination, PassCall, Seat


ROOT = Path(__file__).resolve().parents[1]
ONE_NT_SYSTEM = ROOT / "systems" / "one-no-trump-15-17.toml"
PASSOUT_SYSTEM = ROOT / "systems" / "passout-cheapest-call.toml"
HAND = CardHand(("AKQ", "JT9", "876", "5432"))


class AuctionAssistantTests(unittest.TestCase):
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
