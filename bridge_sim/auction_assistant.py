from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import auction as _auction
from .deals import CardHand, Vulnerability
from .model import (
    AuctionEntry,
    Bid,
    Call,
    Denomination,
    DoubleCall,
    PassCall,
    RedoubleCall,
    Seat,
)
from .systems import BiddingSystem, OneNoTrumpSystem, PassoutSystem, load_system

_SEAT_NAMES = {
    Seat.NORTH: "north",
    Seat.EAST: "east",
    Seat.SOUTH: "south",
    Seat.WEST: "west",
}
_VULNERABILITY_NAMES = {
    Vulnerability.NONE: "none",
    Vulnerability.NORTH_SOUTH: "ns",
    Vulnerability.EAST_WEST: "ew",
    Vulnerability.ALL: "all",
}
_PASSOUT_HAND = CardHand(("AKQJ", "AKQ", "AKQ", "AKQ"))
DEFAULT_GENERATE_LIMIT = 10_000_000_000


def generate_dealer_script(
    *,
    perspective: Seat,
    hand: CardHand,
    auction: Sequence[AuctionEntry],
    partner_system: str | Path | BiddingSystem,
    opponent_system: str | Path | BiddingSystem,
    dealer: Seat,
    vulnerability: Vulnerability,
    produce: int = 1,
    generate: int = DEFAULT_GENERATE_LIMIT,
    unconstrained_entries: Sequence[int] = (),
) -> str:
    """Generate a Dealer script, optionally skipping hypothetical call constraints."""
    if not isinstance(perspective, Seat):
        raise TypeError("perspective must be a Seat.")
    if not isinstance(hand, CardHand):
        raise TypeError("hand must be a CardHand.")
    if not isinstance(dealer, Seat):
        raise TypeError("dealer must be a Seat.")
    if not isinstance(vulnerability, Vulnerability):
        raise TypeError("vulnerability must be a Vulnerability.")
    _validate_count(produce, "produce")
    _validate_count(generate, "generate")
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(auction)
        for index in unconstrained_entries
    ):
        raise ValueError(
            "unconstrained_entries must contain valid auction entry indices."
        )

    partner_policy = _load_policy(partner_system)
    opponent_policy = _load_policy(opponent_system)
    constraints = _auction_conditions(
        perspective,
        auction,
        dealer,
        partner_policy,
        opponent_policy,
        frozenset(unconstrained_entries),
    )

    predeal_cards = ", ".join(
        f"{suit}{cards}"
        for suit, cards in zip(("S", "H", "D", "C"), hand.suits, strict=True)
        if cards
    )
    lines = [
        f"generate {generate}",
        f"produce {produce}",
        f"vulnerable {_VULNERABILITY_NAMES[vulnerability]}",
        f"dealer {_SEAT_NAMES[dealer]}",
        f"predeal {_SEAT_NAMES[perspective]} {predeal_cards}",
    ]
    if constraints:
        lines.extend(
            (
                "",
                "// Dealer constraints translated from the TOML rules for these calls.",
            )
        )
        lines.extend(
            f"// {call_label}: {condition}"
            for call_label, condition in constraints
        )
        lines.append(
            "condition "
            + " && ".join(
                f"({condition})" for _, condition in constraints
            )
        )
    lines.extend(("", "action printpbn", ""))
    return "\n".join(lines)


def _auction_conditions(
    perspective: Seat,
    auction: Sequence[AuctionEntry],
    dealer: Seat,
    partner_system: BiddingSystem,
    opponent_system: BiddingSystem,
    unconstrained_entries: frozenset[int],
) -> list[tuple[str, str]]:
    conditions: list[tuple[str, str]] = []
    current_seat = dealer
    current_bid: Bid | None = None
    current_bidder: Seat | None = None
    multiplier = 1
    consecutive_passes = 0
    previous_entries: list[AuctionEntry] = []

    for entry_index, entry in enumerate(auction):
        if not isinstance(entry, AuctionEntry):
            raise TypeError("auction must contain AuctionEntry values.")
        if not isinstance(entry.seat, Seat):
            raise TypeError("Auction entry seats must be Seat values.")
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
        if consecutive_passes == 3 and current_bid is not None:
            raise ValueError("The auction already ended before its final call.")
        if consecutive_passes == 4 and current_bid is None:
            raise ValueError("The auction already ended before its final call.")

        if (
            entry.seat is not perspective
            and entry_index not in unconstrained_entries
        ):
            policy = (
                partner_system
                if _same_partnership(entry.seat, perspective)
                else opponent_system
            )
            condition = _call_condition(
                entry,
                previous_entries,
                legal_calls,
                policy,
            )
            if condition is not None:
                conditions.append(
                    (f"{_SEAT_NAMES[entry.seat]} {entry.call!s}", condition)
                )

        previous_entries.append(entry)
        if isinstance(entry.call, Bid):
            current_bid = entry.call
            current_bidder = entry.seat
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

    if consecutive_passes == 3 and current_bid is not None:
        raise ValueError("The auction is complete; provide an unfinished auction.")
    if consecutive_passes == 4 and current_bid is None:
        raise ValueError("The auction is complete; provide an unfinished auction.")
    if current_seat is not perspective:
        raise ValueError(
            f"The auction is currently awaiting {current_seat.value}, not "
            f"{perspective.value}."
        )
    return conditions


def _call_condition(
    entry: AuctionEntry,
    previous_entries: Sequence[AuctionEntry],
    legal_calls: frozenset[Call],
    policy: BiddingSystem,
) -> str | None:
    if isinstance(policy, PassoutSystem):
        expected_call = policy.choose_call(
            _PASSOUT_HAND,
            previous_entries,
            legal_calls,
        )
        if entry.call != expected_call:
            raise ValueError(
                f"Call {entry.call!s} by {entry.seat.value} is inconsistent "
                f"with the {policy.name!r} system."
            )
        return None

    if not isinstance(policy, OneNoTrumpSystem):
        raise TypeError(
            f"Cannot derive Dealer constraints for system {type(policy).__name__}."
        )

    rule_call = Bid(1, Denomination.NOTRUMP)
    if entry.call == rule_call:
        return _one_nt_condition(policy, entry.seat)
    if entry.call is PassCall.PASS:
        if rule_call in legal_calls:
            return f"!({_one_nt_condition(policy, entry.seat)})"
        return None
    if policy.out_of_book_when_unmatched:
        return None
    raise ValueError(
        f"Call {entry.call!s} by {entry.seat.value} cannot be made by "
        f"the {policy.name!r} system."
    )


def _one_nt_condition(system: OneNoTrumpSystem, seat: Seat) -> str:
    seat_name = _SEAT_NAMES[seat]
    shape_options = " + ".join(
        f"any {''.join(str(length) for length in shape)}"
        for shape in sorted(system.balanced_shapes)
    )
    return (
        f"(hcp({seat_name}) >= {system.min_hcp}) && "
        f"(hcp({seat_name}) <= {system.max_hcp}) && "
        f"shape({seat_name}, {shape_options})"
    )


def _load_policy(
    policy: str | Path | BiddingSystem,
) -> BiddingSystem:
    if isinstance(policy, (str, Path)):
        return load_system(policy)
    if isinstance(policy, (OneNoTrumpSystem, PassoutSystem)):
        return policy
    raise TypeError("System must be a supported policy or a TOML file path.")


def _same_partnership(left: Seat, right: Seat) -> bool:
    return (left in (Seat.NORTH, Seat.SOUTH)) == (
        right in (Seat.NORTH, Seat.SOUTH)
    )


def _validate_count(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _parse_call(token: str) -> Call:
    normalized = token.strip().upper()
    if normalized in ("P", "PASS"):
        return PassCall.PASS
    if normalized in ("X", "DOUBLE"):
        return DoubleCall.DOUBLE
    if normalized in ("XX", "REDOUBLE"):
        return RedoubleCall.REDOUBLE
    if len(normalized) < 2 or not normalized[0].isdigit():
        raise ValueError(f"Invalid auction call: {token!r}.")
    try:
        return Bid(int(normalized[0]), Denomination(normalized[1:]))
    except ValueError as error:
        raise ValueError(f"Invalid auction call: {token!r}.") from error


def _parse_hand(value: str) -> CardHand:
    suits = tuple(suit.strip().upper() for suit in value.split("."))
    if len(suits) != 4:
        raise ValueError("Hand must use PBN suit order: spades.hearts.diamonds.clubs.")
    return CardHand((suits[0], suits[1], suits[2], suits[3]))


def _parse_auction(value: str, dealer: Seat) -> tuple[AuctionEntry, ...]:
    tokens = value.replace(",", " ").split()
    entries: list[AuctionEntry] = []
    seat = dealer
    for token in tokens:
        entries.append(AuctionEntry(seat, _parse_call(token)))
        seat = seat.next
    return tuple(entries)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Dealer script constrained by a partial auction."
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
    parser.add_argument("--produce", type=int, default=1)
    parser.add_argument(
        "--generate",
        type=int,
        default=DEFAULT_GENERATE_LIMIT,
        help=(
            "Maximum Dealer attempts before giving up "
            f"(default: {DEFAULT_GENERATE_LIMIT})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the script to this path instead of standard output.",
    )
    args = parser.parse_args(argv)

    try:
        dealer = Seat(args.dealer)
        script = generate_dealer_script(
            perspective=Seat(args.seat),
            hand=_parse_hand(args.hand),
            auction=_parse_auction(args.auction, dealer),
            partner_system=args.partner_system,
            opponent_system=args.opponent_system,
            dealer=dealer,
            vulnerability=Vulnerability(args.vulnerability),
            produce=args.produce,
            generate=args.generate,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    if args.output is None:
        print(script, end="")
    else:
        args.output.write_text(script, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
