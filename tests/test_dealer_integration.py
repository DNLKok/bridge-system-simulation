import unittest
from pathlib import Path

from bridge_sim import (
    Contract,
    DealerProvider,
    Denomination,
    Seat,
    Vulnerability,
    load_system,
    run_auction,
)


ROOT = Path(__file__).resolve().parents[1]
DEALER_SCRIPT = ROOT / "dealer" / "unrestricted-single-deal.txt"
SOUTH_NT_SCRIPT = ROOT / "dealer" / "south-16hcp-4432.txt"
SYSTEM_FILE = ROOT / "systems" / "one-no-trump-15-17.toml"
PASSOUT_SYSTEM_FILE = ROOT / "systems" / "passout-cheapest-call.toml"
SEED = 42069
SOUTH_NT_SEED = 20261127


class DealerPipelineIntegrationTests(unittest.TestCase):
    def _provider(self) -> DealerProvider:
        return DealerProvider()

    def test_seeded_deal_runs_through_bidding_and_returns_auction(self) -> None:
        board, = self._provider().generate(DEALER_SCRIPT, seed=SEED)
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

    def test_unrestricted_deal_with_passout_systems_reaches_7ntxx(self) -> None:
        board, = self._provider().generate(DEALER_SCRIPT, seed=SEED)
        system = load_system(PASSOUT_SYSTEM_FILE)

        auction = run_auction(
            board,
            north_south_system=system,
            east_west_system=system,
            dealer=board.dealer,
        )

        self.assertEqual(board.seed, SEED)
        self.assertEqual(
            auction.contract,
            Contract(7, Denomination.NOTRUMP, Seat.WEST, multiplier=4),
        )
        self.assertEqual(len(auction.entries), 319)

    def test_south_16_hcp_4432_hand_contracts_1nt(self) -> None:
        board, = self._provider().generate(SOUTH_NT_SCRIPT, seed=SOUTH_NT_SEED)
        system = load_system(SYSTEM_FILE)

        self.assertEqual(board.seed, SOUTH_NT_SEED)
        self.assertEqual(board.hand_for(Seat.SOUTH).hcp, 16)
        self.assertEqual(
            board.hand_for(Seat.SOUTH).canonical_shape,
            (4, 4, 3, 2),
        )

        auction = run_auction(
            board,
            north_south_system=system,
            east_west_system=system,
            dealer=board.dealer,
        )

        self.assertEqual(
            auction.contract,
            Contract(1, Denomination.NOTRUMP, Seat.SOUTH),
        )


if __name__ == "__main__":
    unittest.main()
