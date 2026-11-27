import unittest
from pathlib import Path
from typing import Iterator, Sequence

from bridge_sim.deals import PBNBoard, Vulnerability, parse_pbn
from bridge_sim.model import Denomination, Seat
from bridge_sim.auction_assistant import DEFAULT_GENERATE_LIMIT
from bridge_sim.model_assistant import (
    DEFAULT_N,
    ModelAssistantResult,
    run_model_assistant,
)


ROOT = Path(__file__).resolve().parents[1]
PBN_SAMPLE = ROOT / "dealer" / "output2.txt"
PASSOUT_SYSTEM = ROOT / "systems" / "passout-cheapest-call.toml"


class FakeDealerProvider:
    def __init__(self, boards: Sequence[PBNBoard]) -> None:
        self.boards = boards
        self.script: str | None = None
        self.seed: int | None = None

    def generate(
        self,
        script_path: str | Path,
        *,
        seed: int | None = None,
    ) -> Iterator[PBNBoard]:
        self.script = Path(script_path).read_text(encoding="utf-8")
        self.seed = seed
        yield from self.boards


class FakeDDS:
    def __init__(self, table_count: int | None = None) -> None:
        self.table_count = table_count
        self.boards: tuple[PBNBoard, ...] | None = None
        self.call_count = 0

    def solve_trick_tables(
        self,
        boards: Sequence[PBNBoard],
    ) -> tuple[dict[Denomination, dict[Seat, int]], ...]:
        self.call_count += 1
        self.boards = tuple(boards)
        count = (
            len(self.boards)
            if self.table_count is None
            else self.table_count
        )
        table = {
            denomination: {seat: 0 for seat in Seat}
            for denomination in Denomination
        }
        return (table,) * count


class ModelAssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        with PBN_SAMPLE.open(encoding="utf-8") as sample:
            self.board = next(parse_pbn(sample))
        self.hand = self.board.hands[self.board.dealer]

    def _run(
        self,
        provider: FakeDealerProvider,
        solver: FakeDDS,
        *,
        n: int = DEFAULT_N,
    ) -> ModelAssistantResult:
        return run_model_assistant(
            perspective=self.board.dealer,
            hand=self.hand,
            auction=(),
            partner_system=PASSOUT_SYSTEM,
            opponent_system=PASSOUT_SYSTEM,
            dealer=self.board.dealer,
            vulnerability=self.board.vulnerability,
            n=n,
            seed=20261127,
            dealer_provider=provider,
            dds_solver=solver,
        )

    def test_defaults_to_64_and_solves_all_boards_in_one_batch(self) -> None:
        provider = FakeDealerProvider((self.board,) * DEFAULT_N)
        solver = FakeDDS()

        result = self._run(provider, solver)

        self.assertIn(f"produce {DEFAULT_N}", provider.script or "")
        self.assertIn(
            f"generate {DEFAULT_GENERATE_LIMIT}",
            provider.script or "",
        )
        self.assertEqual(provider.seed, 20261127)
        self.assertEqual(len(result.boards), DEFAULT_N)
        self.assertEqual(len(result.trick_tables), DEFAULT_N)
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(solver.boards, result.boards)

    def test_uses_requested_sample_size_in_generated_script(self) -> None:
        provider = FakeDealerProvider((self.board,) * 3)
        solver = FakeDDS()

        result = self._run(provider, solver, n=3)

        self.assertIn("produce 3", provider.script or "")
        self.assertEqual(len(result.boards), 3)

    def test_fails_if_dealer_cannot_produce_requested_sample_size(self) -> None:
        provider = FakeDealerProvider((self.board,))
        solver = FakeDDS()

        with self.assertRaisesRegex(RuntimeError, "requested exactly 2"):
            self._run(provider, solver, n=2)

        self.assertIsNone(solver.boards)

    def test_fails_if_dds_returns_the_wrong_number_of_tables(self) -> None:
        provider = FakeDealerProvider((self.board,) * 2)
        solver = FakeDDS(table_count=1)

        with self.assertRaisesRegex(RuntimeError, "returned 1 trick tables"):
            self._run(provider, solver, n=2)


if __name__ == "__main__":
    unittest.main()
