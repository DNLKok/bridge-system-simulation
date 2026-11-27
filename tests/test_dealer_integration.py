import os
import shutil
import unittest
from pathlib import Path

from bridge_sim import (
    DealerProvider,
    Seat,
    Vulnerability,
    load_system,
    run_auction,
)


ROOT = Path(__file__).resolve().parents[1]
DEALER_SCRIPT = ROOT / "dealer" / "unrestricted-single-deal.txt"
SYSTEM_FILE = ROOT / "systems" / "one-no-trump-15-17.toml"
SEED = 42069


class DealerPipelineIntegrationTests(unittest.TestCase):
    def test_seeded_deal_runs_through_bidding_and_returns_auction(self) -> None:
        executable = os.environ.get("DEALER_EXECUTABLE") or shutil.which("dealer")
        if executable is None:
            self.skipTest(
                "Hans van Staveren dealer executable not found; set DEALER_EXECUTABLE."
            )

        board, = DealerProvider(executable).generate(DEALER_SCRIPT, seed=SEED)
        system = load_system(SYSTEM_FILE)

        auction = run_auction(
            board,
            north_south_system=system,
            east_west_system=system,
            dealer=board.dealer,
        )

        self.assertEqual(board.seed, SEED)
        self.assertEqual(board.dealer, Seat.NORTH)
        self.assertEqual(board.vulnerability, Vulnerability.NONE)
        self.assertTrue(auction.entries)
        self.assertTrue(auction.transcript)


if __name__ == "__main__":
    unittest.main()
