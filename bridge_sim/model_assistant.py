from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .auction_assistant import (
    DEFAULT_GENERATE_LIMIT,
    _parse_auction,
    _parse_hand,
    generate_dealer_script,
)
from .dds import TablesPBN4Solver
from .deals import CardHand, DealerProvider, PBNBoard, Vulnerability
from .model import AuctionEntry, Denomination, Seat
from .par import TrickTable

DEFAULT_N = 64


class _BoardProvider(Protocol):
    def generate(
        self,
        script_path: str | Path,
        *,
        seed: int | None = None,
    ) -> Iterable[PBNBoard]:
        """Generate boards from a Dealer script."""
        ...


class _TrickTableSolver(Protocol):
    def solve_trick_tables(
        self,
        boards: Sequence[PBNBoard],
    ) -> tuple[dict[Denomination, dict[Seat, int]], ...]:
        """Solve every board in one batch."""
        ...


@dataclass(frozen=True)
class ModelAssistantResult:
    boards: tuple[PBNBoard, ...]
    trick_tables: tuple[TrickTable, ...]

    def __post_init__(self) -> None:
        if len(self.boards) != len(self.trick_tables):
            raise ValueError("There must be one DDS trick table for each board.")


def run_model_assistant(
    *,
    perspective: Seat,
    hand: CardHand,
    auction: Sequence[AuctionEntry],
    partner_system: str | Path,
    opponent_system: str | Path,
    dealer: Seat,
    vulnerability: Vulnerability,
    n: int = DEFAULT_N,
    seed: int | None = None,
    generate: int = DEFAULT_GENERATE_LIMIT,
    dealer_provider: _BoardProvider | None = None,
    dds_solver: _TrickTableSolver | None = None,
) -> ModelAssistantResult:
    """Generate N auction-consistent deals and solve their full DDS tables."""
    provider = dealer_provider if dealer_provider is not None else DealerProvider()
    solver = dds_solver if dds_solver is not None else TablesPBN4Solver()
    script = generate_dealer_script(
        perspective=perspective,
        hand=hand,
        auction=auction,
        partner_system=partner_system,
        opponent_system=opponent_system,
        dealer=dealer,
        vulnerability=vulnerability,
        produce=n,
        generate=generate,
    )

    with tempfile.TemporaryDirectory(prefix="bridge-model-assistant-") as directory:
        script_path = Path(directory) / "model-assistant.dealer"
        script_path.write_text(script, encoding="utf-8")
        boards = tuple(provider.generate(script_path, seed=seed))

    if len(boards) != n:
        raise RuntimeError(
            f"Dealer produced {len(boards)} boards; requested exactly {n}. "
            "Increase the generate limit or relax the auction constraints."
        )
    for board in boards:
        if board.dealer is not dealer:
            raise RuntimeError(
                f"Dealer returned board {board.board_number} with unexpected "
                f"dealer {board.dealer.value}."
            )
        if board.vulnerability is not vulnerability:
            raise RuntimeError(
                f"Dealer returned board {board.board_number} with unexpected "
                f"vulnerability {board.vulnerability.value}."
            )
        if board.hands[perspective] != hand:
            raise RuntimeError(
                f"Dealer did not preserve the predealt hand for "
                f"{perspective.value} on board {board.board_number}."
            )

    trick_tables = solver.solve_trick_tables(boards)
    if len(trick_tables) != n:
        raise RuntimeError(
            f"Double dummy solver returned {len(trick_tables)} trick tables "
            f"for {n} boards."
        )
    return ModelAssistantResult(boards, trick_tables)


def _result_as_json(result: ModelAssistantResult, seed: int | None) -> str:
    board_results = []
    for board, trick_table in zip(
        result.boards,
        result.trick_tables,
        strict=True,
    ):
        board_results.append(
            {
                "board": board.board_number,
                "dealer": board.dealer.value,
                "vulnerability": board.vulnerability.value,
                "deal": board.tags["Deal"],
                "tricks": {
                    seat.value: {
                        denomination.value: trick_table[denomination][seat]
                        for denomination in Denomination
                    }
                    for seat in Seat
                },
            }
        )
    return json.dumps(
        {
            "sample_size": len(result.boards),
            "seed": seed,
            "boards": board_results,
        },
        indent=2,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Monte Carlo model-assistant simulation and solve each "
            "generated deal with double-dummy analysis."
        )
    )
    parser.add_argument("--seat", required=True, choices=("N", "E", "S", "W"))
    parser.add_argument(
        "--hand",
        required=True,
        help="Known hand in PBN suit order, e.g. AKQ.JT9.87.65432.",
    )
    parser.add_argument("--dealer", required=True, choices=("N", "E", "S", "W"))
    parser.add_argument(
        "--vulnerability",
        required=True,
        choices=("None", "NS", "EW", "All"),
    )
    parser.add_argument(
        "--auction",
        default="",
        help="Calls so far in order, e.g. '1NT PASS PASS'; pass is P or PASS.",
    )
    parser.add_argument("--partner-system", required=True, type=Path)
    parser.add_argument("--opponent-system", required=True, type=Path)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--generate",
        type=int,
        default=DEFAULT_GENERATE_LIMIT,
        help=(
            "Maximum Dealer attempts before giving up "
            f"(default: {DEFAULT_GENERATE_LIMIT})."
        ),
    )
    parser.add_argument("--dealer-executable", type=Path)
    parser.add_argument("--dds-executable", type=Path)
    args = parser.parse_args(argv)

    try:
        dealer = Seat(args.dealer)
        provider = (
            DealerProvider(args.dealer_executable)
            if args.dealer_executable is not None
            else DealerProvider()
        )
        solver = (
            TablesPBN4Solver(args.dds_executable)
            if args.dds_executable is not None
            else TablesPBN4Solver()
        )
        result = run_model_assistant(
            perspective=Seat(args.seat),
            hand=_parse_hand(args.hand),
            auction=_parse_auction(args.auction, dealer),
            partner_system=args.partner_system,
            opponent_system=args.opponent_system,
            dealer=dealer,
            vulnerability=Vulnerability(args.vulnerability),
            n=args.n,
            seed=args.seed,
            generate=args.generate,
            dealer_provider=provider,
            dds_solver=solver,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(_result_as_json(result, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
