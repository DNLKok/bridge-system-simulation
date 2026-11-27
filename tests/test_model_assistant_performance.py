import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bridge_sim.deals import CardHand, DealerProvider, Vulnerability
from bridge_sim.dds import TablesPBN4Solver
from bridge_sim.model import Seat
from bridge_sim.model_assistant import run_model_assistant


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("RUN_ADVANCED_PERFORMANCE_TESTS") == "1",
    "Set RUN_ADVANCED_PERFORMANCE_TESTS=1 to run the real-binary N=16 test.",
)
class AdvancedModelAssistantPerformanceTests(unittest.TestCase):
    def test_empty_auction_with_fixed_hand_and_16_samples(self) -> None:
        dealer_executable = ROOT / "bin" / "dealer"
        solver = TablesPBN4Solver()
        for executable in (dealer_executable, solver.executable):
            if not executable.is_file() or not os.access(executable, os.X_OK):
                self.skipTest(f"Required executable is unavailable: {executable}")

        system_config = """\
schema_version = 2
id = "advanced-performance-test"
version = "1.0.0"
name = "Advanced performance test"
out_of_book = "when_no_rule_matches"

[rule]
id = "open-1nt"
call = "1NT"
min_hcp = 15
max_hcp = 17
balanced_shapes = ["4333", "4432", "5332"]
"""
        hand = CardHand(("AKQ", "JT9", "876", "5432"))
        with TemporaryDirectory() as directory:
            system_path = Path(directory) / "system.toml"
            system_path.write_text(system_config, encoding="utf-8")

            started = time.perf_counter()
            result = run_model_assistant(
                perspective=Seat.NORTH,
                hand=hand,
                auction=(),
                partner_system=system_path,
                opponent_system=system_path,
                dealer=Seat.NORTH,
                vulnerability=Vulnerability.NONE,
                n=16,
                seed=20261127,
                advanced=True,
                dealer_provider=DealerProvider(dealer_executable),
                dds_solver=solver,
            )
            elapsed = time.perf_counter() - started

        print(f"Advanced model assistant N=16 completed in {elapsed:.2f}s")
        self.assertTrue(result.out_of_book)
        self.assertEqual(len(result.boards), 16)
        self.assertEqual(len(result.ratings), 36)
        self.assertIn(
            result.recommended_call,
            {rating.call for rating in result.ratings},
        )
