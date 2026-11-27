from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, replace
from statistics import mean
from pathlib import Path
from typing import Protocol, Sequence

from . import auction as _auction
from .auction_assistant import (
    DEFAULT_GENERATE_LIMIT,
    _parse_auction,
    _parse_hand,
    generate_dealer_script,
)
from .dds import TablesPBN4Solver
from .deals import CardHand, DealerProvider, PBNBoard, Vulnerability
from .imps import calculate_imps
from .model import (
    AuctionEntry,
    Bid,
    Call,
    Contract,
    Denomination,
    DoubleCall,
    PassCall,
    RedoubleCall,
    Seat,
)
from .par import TrickTable, calculate_par_result
from .scoring import calculate_score_ns
from .systems import OutOfBookDecision, load_system

DEFAULT_N = 64


class _BoardProvider(Protocol):
    def generate(
        self,
        script_path: str | Path,
        *,
        seed: int,
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
    out_of_book: bool
    system_call: Call | None
    ratings: tuple[CallRating, ...]
    recommended_call: Call

    def __post_init__(self) -> None:
        if len(self.boards) != len(self.trick_tables):
            raise ValueError("There must be one DDS trick table for each board.")
        if not self.ratings:
            raise ValueError("At least one legal call must be rated.")


@dataclass(frozen=True)
class CallRating:
    call: Call
    expected_imps: float
    note: str | None = None


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
    seed: int,
    generate: int = DEFAULT_GENERATE_LIMIT,
    advanced: bool = False,
    dealer_provider: _BoardProvider | None = None,
    dds_solver: _TrickTableSolver | None = None,
    unconstrained_entries: Sequence[int] = (),
) -> ModelAssistantResult:
    """Rate legal calls, optionally using one partner-simulation level out of book."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(advanced, bool):
        raise TypeError("advanced must be a bool.")
    provider = dealer_provider if dealer_provider is not None else DealerProvider()
    solver = dds_solver if dds_solver is not None else TablesPBN4Solver()
    system = load_system(partner_system)
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
        unconstrained_entries=unconstrained_entries,
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

    legal_calls, standing_contract = _auction_state(
        auction,
        dealer,
        perspective,
    )
    system_decision = system.choose_call(
        hand.profile,
        auction,
        legal_calls,
    )
    out_of_book = system_decision is OutOfBookDecision.OUT_OF_BOOK
    if out_of_book:
        system_call = None
    else:
        if system_decision not in legal_calls:
            raise RuntimeError(
                f"The in-book system returned illegal call {system_decision!s}."
            )
        system_call = system_decision

    ratings = _rate_calls(
        boards,
        trick_tables,
        legal_calls,
        standing_contract,
        perspective,
    )
    if advanced and out_of_book:
        ratings = _rate_calls_with_partner_simulations(
            ratings,
            boards,
            legal_calls,
            perspective,
            auction,
            partner_system,
            opponent_system,
            dealer,
            vulnerability,
            n,
            seed,
            generate,
            provider,
            solver,
        )
    recommended_call = (
        max(ratings, key=lambda rating: rating.expected_imps).call
        if out_of_book
        else system_call
    )
    if recommended_call is None:
        raise RuntimeError("The model assistant could not select a call.")
    return ModelAssistantResult(
        boards,
        trick_tables,
        out_of_book,
        system_call,
        ratings,
        recommended_call,
    )


def _auction_state(
    auction: Sequence[AuctionEntry],
    dealer: Seat,
    perspective: Seat,
) -> tuple[frozenset[Call], Contract | None]:
    current_seat = dealer
    current_bid: Bid | None = None
    current_bidder: Seat | None = None
    multiplier = 1
    consecutive_passes = 0

    for entry in auction:
        if entry.seat is not current_seat:
            raise ValueError(
                f"Auction turn expected {current_seat.value}, got {entry.seat.value}."
            )
        legal_calls = _auction._legal_calls(
            current_bid,
            current_bidder,
            multiplier,
            current_seat,
        )
        if entry.call not in legal_calls:
            raise ValueError(
                f"Auction call {entry.call!s} is illegal for seat {entry.seat.value}."
            )
        if isinstance(entry.call, Bid):
            current_bid = entry.call
            current_bidder = current_seat
            multiplier = 1
            consecutive_passes = 0
        elif entry.call is DoubleCall.DOUBLE:
            multiplier = 2
            consecutive_passes = 0
        elif entry.call is RedoubleCall.REDOUBLE:
            multiplier = 4
            consecutive_passes = 0
        else:
            consecutive_passes += 1
        current_seat = current_seat.next

    if current_seat is not perspective:
        raise ValueError(
            f"The auction is currently awaiting {current_seat.value}, not "
            f"{perspective.value}."
        )
    if (current_bid is None and consecutive_passes == 4) or (
        current_bid is not None and consecutive_passes == 3
    ):
        raise ValueError("The auction is complete; provide an unfinished auction.")

    legal_calls = _auction._legal_calls(
        current_bid,
        current_bidder,
        multiplier,
        current_seat,
    )
    standing_contract = (
        None
        if current_bid is None
        else _auction._make_contract(current_bid, list(auction), multiplier)
    )
    return legal_calls, standing_contract


def _rate_calls(
    boards: Sequence[PBNBoard],
    trick_tables: Sequence[TrickTable],
    legal_calls: frozenset[Call],
    standing_contract: Contract | None,
    perspective: Seat,
) -> tuple[CallRating, ...]:
    call_scores: dict[Call, list[int]] = {
        call: [] for call in sorted(legal_calls, key=_call_order)
    }

    for board, trick_table in zip(boards, trick_tables, strict=True):
        par_score_ns = calculate_par_result(
            trick_table,
            board.dealer,
            board.vulnerability,
        ).score_ns
        for call, scores in call_scores.items():
            score_ns = _score_call_ns(
                call,
                trick_table,
                board.vulnerability,
                perspective,
                standing_contract,
            )
            imps_ns = calculate_imps(score_ns - par_score_ns)
            scores.append(
                imps_ns
                if perspective in (Seat.NORTH, Seat.SOUTH)
                else -imps_ns
            )

    return tuple(
        CallRating(
            call=call,
            expected_imps=mean(scores),
            note=(
                "TODO: refine pass evaluation; it currently keeps the standing "
                "contract or scores zero when no contract exists."
                if call is PassCall.PASS
                else None
            ),
        )
        for call, scores in call_scores.items()
    )


def _rate_calls_with_partner_simulations(
    basic_ratings: tuple[CallRating, ...],
    boards: Sequence[PBNBoard],
    legal_calls: frozenset[Call],
    perspective: Seat,
    auction: Sequence[AuctionEntry],
    partner_system: str | Path,
    opponent_system: str | Path,
    dealer: Seat,
    vulnerability: Vulnerability,
    n: int,
    seed: int,
    generate: int,
    provider: _BoardProvider,
    solver: _TrickTableSolver,
) -> tuple[CallRating, ...]:
    calls = tuple(
        sorted(
            (call for call in legal_calls if call is not PassCall.PASS),
            key=_call_order,
        )
    )
    if not calls:
        return basic_ratings

    scores: dict[Call, list[float]] = {call: [] for call in calls}
    partner = perspective.next.next
    hypothetical_auction = len(auction)
    for board_index, board in enumerate(boards):
        for call_index, call in enumerate(calls):
            partner_auction = (
                *auction,
                AuctionEntry(perspective, call),
                AuctionEntry(perspective.next, PassCall.PASS),
            )
            partner_result = run_model_assistant(
                perspective=partner,
                hand=board.hands[partner],
                auction=partner_auction,
                partner_system=partner_system,
                opponent_system=opponent_system,
                dealer=dealer,
                vulnerability=vulnerability,
                n=n,
                seed=seed + 1 + board_index * len(calls) + call_index,
                generate=generate,
                advanced=False,
                dealer_provider=provider,
                dds_solver=solver,
                unconstrained_entries=(
                    hypothetical_auction,
                    hypothetical_auction + 1,
                ),
            )
            partner_rating = next(
                (
                    rating
                    for rating in partner_result.ratings
                    if rating.call == partner_result.recommended_call
                ),
                None,
            )
            if partner_rating is None:
                raise RuntimeError(
                    "The partner model assistant did not rate its recommended call."
                )
            scores[call].append(partner_rating.expected_imps)

    basic_by_call = {rating.call: rating for rating in basic_ratings}
    return tuple(
        CallRating(call, mean(scores[call]), basic_by_call[call].note)
        if call in scores
        else basic_by_call[call]
        for call in sorted(legal_calls, key=_call_order)
    )


def _score_call_ns(
    call: Call,
    trick_table: TrickTable,
    vulnerability: Vulnerability,
    perspective: Seat,
    standing_contract: Contract | None,
) -> int:
    if call is PassCall.PASS:
        if standing_contract is None:
            return 0
        contract = standing_contract
        trick_delta = _contract_trick_delta(contract, trick_table)
        if trick_delta < 0 and contract.multiplier == 1:
            contract = replace(contract, multiplier=2)
    elif isinstance(call, Bid):
        declarer_tricks = trick_table[call.denomination][perspective]
        trick_delta = declarer_tricks - (call.level + 6)
        contract = Contract(
            call.level,
            call.denomination,
            perspective,
            multiplier=2 if trick_delta < 0 else 1,
        )
    elif call is DoubleCall.DOUBLE:
        if standing_contract is None:
            raise ValueError("A double can only be evaluated with a standing contract.")
        contract = replace(standing_contract, multiplier=2)
        trick_delta = _contract_trick_delta(contract, trick_table)
    elif call is RedoubleCall.REDOUBLE:
        if standing_contract is None:
            raise ValueError(
                "A redouble can only be evaluated with a standing contract."
            )
        contract = replace(standing_contract, multiplier=4)
        trick_delta = _contract_trick_delta(contract, trick_table)
    else:
        raise ValueError(f"Unsupported legal call: {call!s}.")

    return calculate_score_ns(contract, vulnerability, trick_delta)


def _contract_trick_delta(contract: Contract, trick_table: TrickTable) -> int:
    declarer_tricks = trick_table[contract.denomination][contract.declarer]
    return declarer_tricks - (contract.level + 6)


def _call_order(call: Call) -> tuple[int, int, int]:
    if call is PassCall.PASS:
        return 0, 0, 0
    if call is DoubleCall.DOUBLE:
        return 1, 0, 0
    if call is RedoubleCall.REDOUBLE:
        return 2, 0, 0
    if isinstance(call, Bid):
        level, denomination = call.order
        return 3, level, denomination
    raise ValueError(f"Unsupported legal call: {call!s}.")


def _result_as_json(result: ModelAssistantResult, seed: int) -> str:
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
            "system_status": "out_of_book" if result.out_of_book else "in_book",
            "system_call": (
                None if result.system_call is None else str(result.system_call)
            ),
            "recommended_call": str(result.recommended_call),
            "recommendation_source": (
                "model_assistant" if result.out_of_book else "bidding_system"
            ),
            "call_ratings": [
                {
                    "call": str(rating.call),
                    "expected_imps": rating.expected_imps,
                    "note": rating.note,
                }
                for rating in result.ratings
            ],
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
    parser.add_argument(
        "--advanced",
        action="store_true",
        help=(
            "When out of book, evaluate non-pass calls using one basic "
            "partner-assistant simulation level."
        ),
    )
    parser.add_argument("--seed", required=True, type=int)
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
            advanced=args.advanced,
            dealer_provider=provider,
            dds_solver=solver,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(_result_as_json(result, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
