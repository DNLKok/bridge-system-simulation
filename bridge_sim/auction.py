from __future__ import annotations

import time
from collections.abc import Sequence

from . import model as _model
from .deals import NoDealsGeneratedError, PBNBoard, Vulnerability
from .model import (
    AuctionEntry,
    AuctionDeal,
    Bid,
    Call,
    Contract,
    Denomination,
    DoubleCall,
    PassCall,
    RedoubleCall,
    Seat,
)
from .systems import (
    AuctionStateGenerator,
    BiddingSystem,
    CompositeBiddingSystem,
    DecisionContext,
    OutOfBookDecision,
)


DEFAULT_SIMULATION_N = 64


def run_auction(
    deal: AuctionDeal,
    north_south_system: BiddingSystem,
    east_west_system: BiddingSystem,
    dealer: Seat = Seat.NORTH,
    *,
    vulnerability: Vulnerability = Vulnerability.NONE,
    simulation_n: int = DEFAULT_SIMULATION_N,
    simulation_seed: int | None = None,
) -> _model.AuctionResult:
    """Run an auction from an existing deal's HCP/shape profiles.

    Systems receive the current bidder's profile, public auction, legal calls,
    seat, dealer, and vulnerability. The auction supports PASS, contract bids,
    doubles, and redoubles. Simulation-driven rules use ``simulation_n`` sampled
    deals and ``simulation_seed``, defaulting to Unix time in seconds.
    """
    if not isinstance(simulation_n, int) or isinstance(simulation_n, bool) or simulation_n < 1:
        raise ValueError("simulation_n must be a positive integer.")
    if simulation_seed is not None and (
        not isinstance(simulation_seed, int)
        or isinstance(simulation_seed, bool)
        or simulation_seed < 0
    ):
        raise ValueError("simulation_seed must be a non-negative integer.")
    resolved_simulation_seed = (
        int(time.time()) if simulation_seed is None else simulation_seed
    )
    entries: list[AuctionEntry] = []
    current_seat = dealer
    current_bid: Bid | None = None
    current_bidder: Seat | None = None
    multiplier = 1
    consecutive_passes = 0
    state_generators = {
        id(system): AuctionStateGenerator(system.modules)
        for system in (north_south_system, east_west_system)
        if isinstance(system, CompositeBiddingSystem)
    }

    while True:
        legal_calls = _legal_calls(
            current_bid,
            current_bidder,
            multiplier,
            current_seat,
        )
        system = (
            north_south_system
            if current_seat in (Seat.NORTH, Seat.SOUTH)
            else east_west_system
        )
        opponent_system = (
            east_west_system
            if current_seat in (Seat.NORTH, Seat.SOUTH)
            else north_south_system
        )

        def evaluate_candidates(candidates: Sequence[Call]) -> Call:
            if not isinstance(deal, PBNBoard):
                raise RuntimeError(
                    "Simulation-based bidding requires a PBNBoard with the "
                    "complete cards for each hand."
                )
            if not isinstance(system, CompositeBiddingSystem) or not isinstance(
                opponent_system, CompositeBiddingSystem
            ):
                raise RuntimeError(
                    "Simulation-based bidding requires modular composite systems."
                )
            from .model_assistant import (
                _highest_rated_call,
                _warn_no_compatible_deals,
                rate_candidate_calls,
            )

            auction = tuple(entries)
            unconstrained_entries = tuple(
                index
                for index, entry in enumerate(auction)
                if _partnership(entry.seat) != _partnership(current_seat)
            )
            try:
                ratings = rate_candidate_calls(
                    perspective=current_seat,
                    hand=deal.hands[current_seat],
                    auction=auction,
                    partner_system=system,
                    opponent_system=opponent_system,
                    dealer=dealer,
                    vulnerability=vulnerability,
                    candidates=candidates,
                    n=simulation_n,
                    seed=resolved_simulation_seed,
                    unconstrained_entries=unconstrained_entries,
                )
            except NoDealsGeneratedError:
                _warn_no_compatible_deals()
                return PassCall.PASS
            return _highest_rated_call(candidates, ratings)

        context = DecisionContext(
            hand=deal.hand_for(current_seat),
            auction=tuple(entries),
            legal_calls=legal_calls,
            seat=current_seat,
            dealer=dealer,
            vulnerability=vulnerability,
            simulation_evaluator=evaluate_candidates,
        )
        if isinstance(system, CompositeBiddingSystem):
            call = system.choose_call_with_state(
                context,
                state_generators[id(system)],
            )
        else:
            call = system.choose_call(context)
        if call is OutOfBookDecision.OUT_OF_BOOK:
            raise RuntimeError(
                f"System {type(system).__name__!r} is out of book for "
                f"{current_seat.value}; use the model assistant to choose a call."
            )
        if call not in legal_calls:
            raise ValueError(
                f"System {type(system).__name__!r} returned illegal call {call!s} "
                f"for seat {current_seat.value}."
            )

        entry = AuctionEntry(current_seat, call)
        for state_generator in state_generators.values():
            state_generator.advance(entry, context)
        entries.append(entry)
        if isinstance(call, Bid):
            current_bid = call
            current_bidder = current_seat
            multiplier = 1
            consecutive_passes = 0
        elif call is DoubleCall.DOUBLE:
            multiplier = 2
            consecutive_passes = 0
        elif call is RedoubleCall.REDOUBLE:
            multiplier = 4
            consecutive_passes = 0
        else:
            consecutive_passes += 1

        if current_bid is None and consecutive_passes == 4:
            return _model.AuctionResult(tuple(entries), None)
        if current_bid is not None and consecutive_passes == 3:
            return _model.AuctionResult(
                tuple(entries),
                _make_contract(current_bid, entries, multiplier),
            )

        current_seat = current_seat.next


def _legal_calls(
    current_bid: Bid | None,
    current_bidder: Seat | None,
    multiplier: int,
    current_seat: Seat,
) -> frozenset[Call]:
    calls: set[Call] = {PassCall.PASS}
    for level in range(1, 8):
        for denomination in Denomination:
            bid = Bid(level, denomination)
            if current_bid is None or bid.order > current_bid.order:
                calls.add(bid)
    if current_bid is not None and current_bidder is not None:
        if multiplier == 1 and _partnership(current_seat) != _partnership(current_bidder):
            calls.add(DoubleCall.DOUBLE)
        elif multiplier == 2 and _partnership(current_seat) == _partnership(current_bidder):
            calls.add(RedoubleCall.REDOUBLE)
    return frozenset(calls)


def _make_contract(
    final_bid: Bid,
    entries: list[AuctionEntry],
    multiplier: int,
) -> Contract:
    declaring_side = _partnership(_final_bidder(entries, final_bid))
    declarer = next(
        entry.seat
        for entry in entries
        if _partnership(entry.seat) == declaring_side
        and isinstance(entry.call, Bid)
        and entry.call.denomination == final_bid.denomination
    )
    return Contract(final_bid.level, final_bid.denomination, declarer, multiplier)


def _final_bidder(entries: list[AuctionEntry], final_bid: Bid) -> Seat:
    return next(
        entry.seat
        for entry in reversed(entries)
        if entry.call == final_bid
    )


def _partnership(seat: Seat) -> str:
    return "NS" if seat in (Seat.NORTH, Seat.SOUTH) else "EW"
