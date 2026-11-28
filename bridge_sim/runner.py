from __future__ import annotations

from collections.abc import Iterable

from .auction import DEFAULT_SIMULATION_N, run_auction
from .dds import TablesPBN4Solver
from .deals import PBNBoard
from .model import AuctionResult
from .systems import BiddingSystem


def run_auction_batch(
    boards: Iterable[PBNBoard],
    north_south_system: BiddingSystem,
    east_west_system: BiddingSystem,
    *,
    dds_solver: TablesPBN4Solver | None = None,
    simulation_n: int = DEFAULT_SIMULATION_N,
    simulation_seed: int | None = None,
) -> tuple[AuctionResult, ...]:
    """Complete every auction, then batch-evaluate contracts with DDS."""
    board_batch = tuple(boards)
    auctions = tuple(
        run_auction(
            board,
            north_south_system,
            east_west_system,
            dealer=board.dealer,
            vulnerability=board.vulnerability,
            simulation_n=simulation_n,
            simulation_seed=simulation_seed,
        )
        for board in board_batch
    )
    solver = dds_solver if dds_solver is not None else TablesPBN4Solver()
    return solver.evaluate(board_batch, auctions)
