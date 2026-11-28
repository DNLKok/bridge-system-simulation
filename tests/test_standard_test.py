import tempfile
import unittest
from pathlib import Path

from bridge_sim.deals import parse_pbn
from bridge_sim.model import (
    AuctionEntry,
    AuctionResult,
    Bid,
    Contract,
    Denomination,
    PassCall,
    Seat,
)
from bridge_sim.standard_test import (
    BoardAuction,
    _report_html,
    load_settings,
)
from bridge_sim.systems import load_system


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "simulation" / "standard-test.toml"


class StandardTestReportTests(unittest.TestCase):
    def test_settings_pin_both_sides_seed_count_and_basic_mode(self) -> None:
        settings = load_settings(CONFIG)

        self.assertEqual(settings.deal_count, 16)
        self.assertEqual(settings.seed, 20261128)
        self.assertFalse(settings.advanced_out_of_book)
        self.assertEqual(settings.north_south_system, ROOT / "systems/standard-test.toml")
        self.assertEqual(settings.east_west_system, ROOT / "systems/standard-test.toml")
        system = load_system(ROOT / "systems/standard-test.toml")
        self.assertEqual(
            tuple(module.module_id for module in system.modules),
            ("standard-openings", "traditional-preempts", "simple-overcalls"),
        )

    def test_settings_reject_advanced_out_of_book_logic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "standard-test.toml"
            config.write_text(
                CONFIG.read_text(encoding="utf-8").replace(
                    "advanced_out_of_book = false",
                    "advanced_out_of_book = true",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be disabled"):
                load_settings(config)

    def test_report_contains_full_hands_auction_and_simulation_settings(self) -> None:
        board, = parse_pbn(
            (
                '[Event "Test seed 20261128"]',
                '[Board "1"]',
                '[Dealer "N"]',
                '[Vulnerable "None"]',
                '[Deal "N:AKQJ.AKQ.AKQ.AKQ T98.JT98.JT9.JT9 '
                '765.765.8765.876 432.432.432.5432"]',
                "",
            )
        )
        auction = (
            AuctionEntry(Seat.NORTH, Bid(1, Denomination.CLUBS)),
            AuctionEntry(Seat.EAST, PassCall.PASS),
            AuctionEntry(Seat.SOUTH, Bid(1, Denomination.NOTRUMP)),
            AuctionEntry(Seat.WEST, PassCall.PASS),
            AuctionEntry(Seat.NORTH, PassCall.PASS),
            AuctionEntry(Seat.EAST, PassCall.PASS),
        )
        result = BoardAuction(
            board,
            AuctionResult(
                auction,
                Contract(1, Denomination.NOTRUMP, Seat.SOUTH),
            ),
        )
        settings = load_settings(CONFIG)
        html = _report_html(
            settings,
            load_system(settings.north_south_system),
            load_system(settings.east_west_system),
            (result,),
        )

        self.assertIn("Deals:</span> 16", html)
        self.assertIn("Dealer seed:</span> 20261128", html)
        self.assertIn("Advanced out-of-book logic:</span> Disabled", html)
        self.assertIn("Standard openings, preempts, and overcalls", html)
        self.assertIn("Simple overcalls and takeout doubles", html)
        self.assertIn("AKQJ", html)
        self.assertIn("JT98", html)
        self.assertIn("5432", html)
        self.assertIn("1NT", html)
        self.assertIn("Result: 1NT by S", html)


if __name__ == "__main__":
    unittest.main()
