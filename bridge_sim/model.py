from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, TypeAlias


class Seat(str, Enum):
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"

    @property
    def next(self) -> Seat:
        order = (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST)
        return order[(order.index(self) + 1) % len(order)]


class Denomination(str, Enum):
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"
    NOTRUMP = "NT"


_DENOMINATION_ORDER = {
    denomination: index
    for index, denomination in enumerate(Denomination)
}


@dataclass(frozen=True)
class Bid:
    level: int
    denomination: Denomination

    def __post_init__(self) -> None:
        if not 1 <= self.level <= 7:
            raise ValueError("A bridge bid level must be between 1 and 7.")

    def __str__(self) -> str:
        return f"{self.level}{self.denomination.value}"

    @property
    def order(self) -> tuple[int, int]:
        return self.level, _DENOMINATION_ORDER[self.denomination]


class PassCall(str, Enum):
    PASS = "PASS"

    def __str__(self) -> str:
        return self.value


Call: TypeAlias = Bid | PassCall


@dataclass(frozen=True)
class HandProfile:
    """Auction-relevant facts from an existing hand; contains no card data."""

    hcp: int
    shape: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hcp, int)
            or isinstance(self.hcp, bool)
            or not 0 <= self.hcp <= 37
        ):
            raise ValueError("HCP must be between 0 and 37.")
        if not isinstance(self.shape, tuple) or len(self.shape) != 4:
            raise ValueError("A hand shape must contain four suit lengths.")
        if any(
            not isinstance(length, int)
            or isinstance(length, bool)
            or not 0 <= length <= 13
            for length in self.shape
        ):
            raise ValueError("Each suit length must be an integer between 0 and 13.")
        if sum(self.shape) != 13:
            raise ValueError("The four suit lengths in a hand must total 13.")

    @property
    def canonical_shape(self) -> tuple[int, int, int, int]:
        return tuple(sorted(self.shape, reverse=True))


@dataclass(frozen=True)
class Deal:
    """Profiles for an already supplied deal, keyed by compass seat.

    This auction-only representation deliberately does not create, store, or
    validate cards. An adapter for an existing card-based deal can provide each
    hand's HCP and suit lengths here.
    """

    hands: Mapping[Seat, HandProfile]

    def __post_init__(self) -> None:
        copied_hands = dict(self.hands)
        if set(copied_hands) != set(Seat):
            raise ValueError("A deal must provide one hand profile for each seat.")
        if any(not isinstance(hand, HandProfile) for hand in copied_hands.values()):
            raise TypeError("Each seat must map to a HandProfile.")
        object.__setattr__(self, "hands", MappingProxyType(copied_hands))

    def hand_for(self, seat: Seat) -> HandProfile:
        return self.hands[seat]


class AuctionDeal(Protocol):
    def hand_for(self, seat: Seat) -> HandProfile:
        """Return the auction-relevant profile for a seat."""


@dataclass(frozen=True)
class AuctionEntry:
    seat: Seat
    call: Call

    def __str__(self) -> str:
        return f"{self.seat.value}: {self.call}"


@dataclass(frozen=True)
class Contract:
    level: int
    denomination: Denomination
    declarer: Seat

    def __str__(self) -> str:
        return f"{self.level}{self.denomination.value} by {self.declarer.value}"


@dataclass(frozen=True)
class AuctionResult:
    entries: tuple[AuctionEntry, ...]
    contract: Contract | None

    @property
    def passed_out(self) -> bool:
        return self.contract is None

    @property
    def transcript(self) -> str:
        return " ".join(str(entry) for entry in self.entries)
