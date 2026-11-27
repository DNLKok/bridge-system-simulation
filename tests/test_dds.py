import os
import shutil
import subprocess
from statistics import mean
import unittest
from pathlib import Path
from unittest.mock import patch

from bridge_sim import (
    AuctionResult,
    Contract,
    DealerProvider,
    Denomination,
    Seat,
    TablesPBN4Solver,
    load_system,
    parse_pbn,
    run_auction_batch,
)


ROOT = Path(__file__).resolve().parents[1]
PBN_SAMPLE = ROOT / "dealer" / "output2.txt"
UNRESTRICTED_100_DEALS_SCRIPT = ROOT / "dealer" / "unrestricted-100-deals.txt"
PASSOUT_SYSTEM_FILE = ROOT / "systems" / "passout-cheapest-call.toml"
UNRESTRICTED_100_DEALS_SEED = 20261128


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
        self.assertEqual(
            tuple(result.score_ns for result in evaluated),
            (900, 130, None),
        )
        self.assertEqual(
            tuple(result.par_score_ns for result in evaluated),
            (0, 130, None),
        )
        self.assertEqual(
            tuple(result.par_contract for result in evaluated),
            (None, Contract(4, Denomination.CLUBS, Seat.NORTH), None),
        )
        self.assertEqual(
            tuple(
                None if result.par_result is None else result.par_result.trick_delta
                for result in evaluated
            ),
            (None, 0, None),
        )
        self.assertEqual(
            tuple(result.score_vs_par for result in evaluated),
            (900, 0, None),
        )
        self.assertEqual(
            tuple(result.imps_vs_par for result in evaluated),
            (14, 0, None),
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
            self.assertTrue(
                result.trick_delta is not None
                and -13 <= result.trick_delta <= 13
            )
            self.assertTrue(
                result.score_ns is not None
                and result.par_result is not None
                and result.score_vs_par
                == result.score_ns - result.par_score_ns
            )

    def test_100_unrestricted_passout_auctions_have_expected_dds_overview(self) -> None:
        dealer_executable = os.environ.get("DEALER_EXECUTABLE") or shutil.which(
            "dealer"
        )
        if dealer_executable is None:
            self.skipTest("Hans van Staveren dealer executable is unavailable.")

        solver = TablesPBN4Solver()
        if not solver.executable.is_file() or not os.access(
            solver.executable, os.X_OK
        ):
            self.skipTest("Workspace TablesPBN4 executable is unavailable.")

        boards = tuple(
            DealerProvider(dealer_executable).generate(
                UNRESTRICTED_100_DEALS_SCRIPT,
                seed=UNRESTRICTED_100_DEALS_SEED,
            )
        )
        self.assertEqual(len(boards), 100)
        self.assertTrue(
            all(board.seed == UNRESTRICTED_100_DEALS_SEED for board in boards)
        )

        system = load_system(PASSOUT_SYSTEM_FILE)
        results = run_auction_batch(
            boards,
            system,
            system,
            dds_solver=solver,
        )
        self.assertTrue(
            all(
                result.contract
                == Contract(7, Denomination.NOTRUMP, Seat.WEST, multiplier=4)
                for result in results
            )
        )

        deltas = tuple(result.trick_delta for result in results)
        self.assertTrue(all(delta is not None for delta in deltas))
        trick_deltas = tuple(delta for delta in deltas if delta is not None)
        self.assertTrue(all(delta <= 0 for delta in trick_deltas))
        self.assertTrue(
            all(
                result.score_ns is not None
                and result.par_result is not None
                and result.score_vs_par
                == result.score_ns - result.par_score_ns
                for result in results
            )
        )
        overview = (
            min(trick_deltas),
            max(trick_deltas),
            round(mean(trick_deltas), 2),
        )
        self.assertEqual(overview, (-13, 0, -6.95))
        self.assertEqual(trick_deltas.count(-13), 4)
        self.assertEqual(trick_deltas.count(0), 1)


if __name__ == "__main__":
    unittest.main()
