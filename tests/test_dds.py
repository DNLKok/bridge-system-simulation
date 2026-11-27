import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from bridge_sim import (
    AuctionResult,
    Contract,
    Denomination,
    Seat,
    TablesPBN4Solver,
    load_system,
    parse_pbn,
    run_auction_batch,
)


ROOT = Path(__file__).resolve().parents[1]
PBN_SAMPLE = ROOT / "dealer" / "output2.txt"
PASSOUT_SYSTEM_FILE = ROOT / "systems" / "passout-cheapest-call.toml"


class TablesPBN4SolverTests(unittest.TestCase):
    def setUp(self) -> None:
        with PBN_SAMPLE.open(encoding="utf-8") as sample:
            self.boards = tuple(parse_pbn(sample))

    def test_evaluates_contracts_in_one_batch_and_skips_passed_out_boards(self) -> None:
        auctions = (
            AuctionResult((), Contract(3, Denomination.NOTRUMP, Seat.WEST)),
            AuctionResult((), Contract(4, Denomination.CLUBS, Seat.NORTH)),
            AuctionResult((), None),
        )
        output = "\n".join(
            (
                f'{self.boards[0].tags["Deal"]}:00000,00000,00000,00000',
                f'{self.boards[1].tags["Deal"]}:000A0,00000,00000,00000',
            )
        )
        solver = TablesPBN4Solver("mock-TablesPBN4")

        with patch("bridge_sim.dds.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=["mock-TablesPBN4"],
                returncode=0,
                stdout=output,
                stderr="",
            )
            evaluated = solver.evaluate(self.boards, auctions)

        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args.kwargs["input"].splitlines(),
            [self.boards[0].tags["Deal"], self.boards[1].tags["Deal"]],
        )
        self.assertEqual(
            tuple(result.trick_delta for result in evaluated),
            (-9, 0, None),
        )

    def test_reports_tablespbn4_process_failure(self) -> None:
        auction = AuctionResult(
            (),
            Contract(1, Denomination.NOTRUMP, Seat.NORTH),
        )
        solver = TablesPBN4Solver("mock-TablesPBN4")

        with patch("bridge_sim.dds.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=["mock-TablesPBN4"],
                returncode=1,
                stdout="",
                stderr="solver failed",
            )
            with self.assertRaisesRegex(RuntimeError, "solver failed"):
                solver.evaluate((self.boards[0],), (auction,))

    def test_runs_all_auctions_then_evaluates_with_local_tablespbn4(self) -> None:
        solver = TablesPBN4Solver()
        if not solver.executable.is_file() or not os.access(
            solver.executable, os.X_OK
        ):
            self.skipTest("Workspace TablesPBN4 executable is unavailable.")

        system = load_system(PASSOUT_SYSTEM_FILE)
        results = run_auction_batch(
            self.boards,
            system,
            system,
            dds_solver=solver,
        )

        self.assertEqual(len(results), len(self.boards))
        for result in results:
            self.assertIsNotNone(result.contract)
            self.assertIsNotNone(result.trick_delta)
            self.assertGreaterEqual(result.trick_delta, -13)
            self.assertLessEqual(result.trick_delta, 13)


if __name__ == "__main__":
    unittest.main()
