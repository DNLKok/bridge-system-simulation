from __future__ import annotations

from . import model as _model
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
from .systems import BiddingSystem, OutOfBookDecision


def run_auction(
    deal: AuctionDeal,
    north_south_system: BiddingSystem,
    east_west_system: BiddingSystem,
    dealer: Seat = Seat.NORTH,
) -> _model.AuctionResult:
    """Run an auction from an existing deal's HCP/shape profiles.

    Systems receive only the current bidder's profile, public auction, and
    legal calls. The auction supports PASS, contract bids, doubles, and
    redoubles.
    """
    entries: list[AuctionEntry] = []
    current_seat = dealer
    current_bid: Bid | None = None
    current_bidder: Seat | None = None
    multiplier = 1
    consecutive_passes = 0

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
        call = system.choose_call(
            hand=deal.hand_for(current_seat),
            auction=tuple(entries),
            legal_calls=legal_calls,
        )
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

        entries.append(AuctionEntry(current_seat, call))
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
