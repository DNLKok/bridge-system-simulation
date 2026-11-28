import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from bridge_sim.deals import Vulnerability, parse_pbn
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
    _auction_headers,
    _auction_rows,
    _conditional_dealer_conditions,
    _conditional_dealer_script,
    _conditional_prefix_for_dealer,
    _report_html,
    _output_filename,
    _standard_rotation,
    load_settings,
)
from bridge_sim.systems import load_system


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "simulation" / "standard-test.toml"
CONDITIONAL_CONFIG = ROOT / "simulation" / "conditional-test.toml"


class StandardTestReportTests(unittest.TestCase):
    def test_settings_pin_both_sides_seed_count_and_basic_mode(self) -> None:
        with patch("bridge_sim.standard_test.time.time", return_value=1_795_000_123):
            settings = load_settings(CONFIG)

        self.assertEqual(settings.deal_count, 16)
        self.assertEqual(settings.seed, 1_795_000_123)
        self.assertEqual(settings.decision_simulation_n, 64)
        self.assertFalse(settings.advanced_out_of_book)
        self.assertEqual(
            settings.north_south_system,
            ROOT / "systems/standard-test.toml",
        )
        self.assertEqual(
            settings.east_west_system,
            ROOT / "systems/standard-test.toml",
        )
        self.assertEqual(settings.north_south_abbreviation, "STD")
        self.assertEqual(settings.east_west_abbreviation, "STD")
        self.assertEqual(settings.output_directory, ROOT / "reports")
        system = load_system(ROOT / "systems/standard-test.toml")
        self.assertEqual(
            tuple(module.module_id for module in system.modules),
            ("standard-openings", "traditional-preempts", "simple-overcalls"),
        )

    def test_unix_seed_changes_each_second_and_explicit_seed_is_supported(self) -> None:
        with patch(
            "bridge_sim.standard_test.time.time",
            side_effect=(1_795_000_123, 1_795_000_124),
        ):
            seeds = (load_settings(CONFIG).seed, load_settings(CONFIG).seed)
        self.assertEqual(seeds, (1_795_000_123, 1_795_000_124))

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "standard-test.toml"
            config.write_text(
                CONFIG.read_text(encoding="utf-8").replace(
                    'seed = "unix"',
                    "seed = 20261128",
                ).replace(
                    "decision_simulation_n = 64",
                    "decision_simulation_n = 3",
                ),
                encoding="utf-8",
            )
            settings = load_settings(config)
            self.assertEqual(settings.seed, 20261128)
            self.assertEqual(settings.decision_simulation_n, 3)

    def test_standard_board_rotation_repeats_after_sixteen_boards(self) -> None:
        expected_vulnerability = (
            Vulnerability.NONE,
            Vulnerability.NORTH_SOUTH,
            Vulnerability.EAST_WEST,
            Vulnerability.ALL,
            Vulnerability.NORTH_SOUTH,
            Vulnerability.EAST_WEST,
            Vulnerability.ALL,
            Vulnerability.NONE,
            Vulnerability.EAST_WEST,
            Vulnerability.ALL,
            Vulnerability.NONE,
            Vulnerability.NORTH_SOUTH,
            Vulnerability.ALL,
            Vulnerability.NONE,
            Vulnerability.NORTH_SOUTH,
            Vulnerability.EAST_WEST,
        )
        rotations = tuple(_standard_rotation(number) for number in range(1, 17))
        self.assertEqual(
            tuple(dealer for dealer, _ in rotations),
            (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST) * 4,
        )
        self.assertEqual(
            tuple(vulnerability for _, vulnerability in rotations),
            expected_vulnerability,
        )
        self.assertEqual(_standard_rotation(17), rotations[0])
        self.assertEqual(_standard_rotation(32), rotations[15])

    def test_auction_seat_headers_are_red_only_for_vulnerable_sides(self) -> None:
        north_south = _auction_headers(Vulnerability.NORTH_SOUTH)
        self.assertIn('<th class="vulnerable">North</th>', north_south)
        self.assertIn('<th class="vulnerable">South</th>', north_south)
        self.assertIn("<th>East</th>", north_south)
        self.assertIn("<th>West</th>", north_south)

        east_west = _auction_headers(Vulnerability.EAST_WEST)
        self.assertIn('<th class="vulnerable">East</th>', east_west)
        self.assertIn('<th class="vulnerable">West</th>', east_west)
        self.assertIn("<th>North</th>", east_west)
        self.assertIn("<th>South</th>", east_west)

        all_vulnerable = _auction_headers(Vulnerability.ALL)
        self.assertEqual(all_vulnerable.count('class="vulnerable"'), 4)
        none_vulnerable = _auction_headers(Vulnerability.NONE)
        self.assertNotIn('class="vulnerable"', none_vulnerable)

    def test_auction_rows_start_at_the_dealer_and_wrap_after_west(self) -> None:
        expected_rows = {
            Seat.NORTH: (
                ("PASS", "PASS", "PASS", "PASS"),
                ("PASS", "", "", ""),
            ),
            Seat.EAST: (
                ("", "PASS", "PASS", "PASS"),
                ("PASS", "PASS", "", ""),
            ),
            Seat.SOUTH: (
                ("", "", "PASS", "PASS"),
                ("PASS", "PASS", "PASS", ""),
            ),
            Seat.WEST: (
                ("", "", "", "PASS"),
                ("PASS", "PASS", "PASS", "PASS"),
            ),
        }
        for dealer, expected in expected_rows.items():
            entries = []
            seat = dealer
            for _ in range(5):
                entries.append(AuctionEntry(seat, PassCall.PASS))
                seat = seat.next
            auction = AuctionResult(tuple(entries), None)
            expected_html = "".join(
                "<tr>" + "".join(f"<td>{call}</td>" for call in row) + "</tr>"
                for row in expected
            )
            with self.subTest(dealer=dealer):
                self.assertEqual(_auction_rows(auction), expected_html)

    def test_filename_uses_timestamp_and_both_system_abbreviations(self) -> None:
        settings = load_settings(CONFIG)
        timestamp = datetime(2026, 11, 28, 10, 12, 42, tzinfo=timezone.utc)
        self.assertEqual(
            _output_filename(settings, timestamp),
            "20261128_101242_NS-STD_EW-STD.pdf",
        )

    def test_conditional_settings_load_systems_and_auction_prefix(self) -> None:
        settings = load_settings(CONDITIONAL_CONFIG)
        self.assertEqual(settings.condition_label, "SJ-2S")
        self.assertEqual(
            tuple(
                (entry.seat, str(entry.call))
                for entry in settings.auction_prefix
            ),
            (
                (Seat.NORTH, "1NT"),
                (Seat.EAST, "PASS"),
                (Seat.SOUTH, "2S"),
            ),
        )
        north_south = load_system(settings.north_south_system)
        self.assertEqual(
            tuple(module.module_id for module in north_south.modules),
            ("standard-openings", "jump-to-game-in-GF", "stayman-jacoby"),
        )
        self.assertEqual(load_system(settings.east_west_system).modules, ())
        timestamp = datetime(2026, 11, 28, 10, 12, 42, tzinfo=timezone.utc)
        self.assertEqual(
            _output_filename(settings, timestamp),
            "20261128_101242_NS-SJ_EW-PASS_COND-SJ-2S.pdf",
        )

    def test_conditional_prefix_adds_leading_passes_for_rotated_dealers(self) -> None:
        settings = load_settings(CONDITIONAL_CONFIG)
        prefix = _conditional_prefix_for_dealer(
            settings.auction_prefix,
            Seat.SOUTH,
        )
        self.assertEqual(
            tuple((entry.seat, str(entry.call)) for entry in prefix),
            (
                (Seat.SOUTH, "PASS"),
                (Seat.WEST, "PASS"),
                (Seat.NORTH, "1NT"),
                (Seat.EAST, "PASS"),
                (Seat.SOUTH, "2S"),
            ),
        )

    def test_conditional_settings_reject_nonconsecutive_seats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "conditional-test.toml"
            config.write_text(
                CONDITIONAL_CONFIG.read_text(encoding="utf-8").replace(
                    '"E:PASS"',
                    '"W:PASS"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "follow the previous seat"):
                load_settings(config)

    def test_conditional_dealer_script_combines_conditions_and_rotation(self) -> None:
        settings = load_settings(CONDITIONAL_CONFIG)
        prefix = _conditional_prefix_for_dealer(
            settings.auction_prefix,
            Seat.SOUTH,
        )
        conditions = _conditional_dealer_conditions(
            prefix,
            Seat.SOUTH,
            Vulnerability.NORTH_SOUTH,
            load_system(settings.north_south_system),
            load_system(settings.east_west_system),
        )
        script = _conditional_dealer_script(
            Seat.SOUTH,
            Vulnerability.NORTH_SOUTH,
            conditions,
        )
        self.assertIn("dealer south", script)
        self.assertIn("vulnerable ns", script)
        self.assertEqual(
            sum(line.startswith("condition ") for line in script.splitlines()),
            1,
        )
        self.assertIn("conditionalcondition1 =", script)
        self.assertIn(" && ", script)

    def test_conditional_conditions_defer_simulation_selected_calls(self) -> None:
        settings = load_settings(CONDITIONAL_CONFIG)
        prefix = (
            AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
            AuctionEntry(Seat.EAST, PassCall.PASS),
            AuctionEntry(Seat.SOUTH, Bid(3, Denomination.SPADES)),
            AuctionEntry(Seat.WEST, PassCall.PASS),
            AuctionEntry(Seat.NORTH, Bid(3, Denomination.NOTRUMP)),
        )
        conditions = _conditional_dealer_conditions(
            prefix,
            Seat.NORTH,
            Vulnerability.NONE,
            load_system(settings.north_south_system),
            load_system(settings.east_west_system),
        )
        self.assertTrue(conditions)

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
            datetime(2026, 11, 28, 10, 12, 42, tzinfo=timezone.utc),
        )

        self.assertIn("Deals:</span> 16", html)
        self.assertIn(f"Dealer seed:</span> {settings.seed}", html)
        self.assertIn(
            f"Decision simulation N:</span> {settings.decision_simulation_n}",
            " ".join(html.split()),
        )
        self.assertIn("Advanced out-of-book logic:</span> Disabled", html)
        self.assertIn("Standard 16-board rotation", html)
        self.assertIn("North-South (STD)", html)
        self.assertIn("Standard openings, preempts, and overcalls", html)
        self.assertIn("Simple overcalls and takeout doubles", html)
        self.assertIn("AKQJ", html)
        self.assertIn("JT98", html)
        self.assertIn("5432", html)
        self.assertIn("1NT", html)
        self.assertIn("Result: 1NT by S", html)

        conditional_settings = load_settings(CONDITIONAL_CONFIG)
        conditional_html = _report_html(
            conditional_settings,
            load_system(conditional_settings.north_south_system),
            load_system(conditional_settings.east_west_system),
            (result,),
            datetime(2026, 11, 28, 10, 12, 42, tzinfo=timezone.utc),
            dealer_seeds=(20261128,),
        )
        self.assertIn("Condition label:</span> SJ-2S", conditional_html)
        self.assertIn(
            "Required auction prefix:</span> N:1NT E:PASS S:2S",
            conditional_html,
        )
        self.assertIn(
            "Conditional test for auction prefix N:1NT E:PASS S:2S",
            conditional_html,
        )
        self.assertIn("Simulation seed:", conditional_html)
        self.assertIn("Dealer seeds (boards 1\u2013N):</span> 20261128", conditional_html)


if __name__ == "__main__":
    unittest.main()
