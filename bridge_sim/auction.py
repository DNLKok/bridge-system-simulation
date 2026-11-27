from __future__ import annotations

from .model import (
    AuctionEntry,
    AuctionDeal,
    AuctionResult,
    Bid,
    Call,
    Contract,
    Denomination,
    PassCall,
    Seat,
)
from .systems import BiddingSystem


def run_auction(
    deal: AuctionDeal,
    north_south_system: BiddingSystem,
    east_west_system: BiddingSystem,
    dealer: Seat = Seat.NORTH,
) -> AuctionResult:
    """Run an auction from an existing deal's HCP/shape profiles.

    Systems receive only the current bidder's profile, public auction, and
    legal calls. This implementation supports PASS and contract bids; doubles
    and redoubles are not part of the current system format.
    """
    entries: list[AuctionEntry] = []
    current_seat = dealer
    current_bid: Bid | None = None
    consecutive_passes = 0

    while True:
        legal_calls = _legal_calls(current_bid)
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
        if call not in legal_calls:
            raise ValueError(
                f"System {type(system).__name__!r} returned illegal call {call!s} "
                f"for seat {current_seat.value}."
            )

        entries.append(AuctionEntry(current_seat, call))
        if isinstance(call, Bid):
            current_bid = call
            consecutive_passes = 0
        else:
            consecutive_passes += 1

        if current_bid is None and consecutive_passes == 4:
            return AuctionResult(tuple(entries), None)
        if current_bid is not None and consecutive_passes == 3:
            return AuctionResult(
                tuple(entries),
                _make_contract(current_bid, entries),
            )

        current_seat = current_seat.next


def _legal_calls(current_bid: Bid | None) -> frozenset[Call]:
    calls: set[Call] = {PassCall.PASS}
    for level in range(1, 8):
        for denomination in Denomination:
            bid = Bid(level, denomination)
            if current_bid is None or bid.order > current_bid.order:
                calls.add(bid)
    return frozenset(calls)


def _make_contract(final_bid: Bid, entries: list[AuctionEntry]) -> Contract:
    declaring_side = _partnership(_final_bidder(entries, final_bid))
    declarer = next(
        entry.seat
        for entry in entries
        if _partnership(entry.seat) == declaring_side
        and isinstance(entry.call, Bid)
        and entry.call.denomination == final_bid.denomination
    )
    return Contract(final_bid.level, final_bid.denomination, declarer)


def _final_bidder(entries: list[AuctionEntry], final_bid: Bid) -> Seat:
    return next(
        entry.seat
        for entry in reversed(entries)
        if entry.call == final_bid
    )


def _partnership(seat: Seat) -> str:
    return "NS" if seat in (Seat.NORTH, Seat.SOUTH) else "EW"
