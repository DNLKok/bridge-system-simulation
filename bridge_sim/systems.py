from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .model import (
    AuctionEntry,
    Bid,
    Call,
    Denomination,
    HandProfile,
    PassCall,
)


class BiddingSystem(Protocol):
    def choose_call(
        self,
        hand: HandProfile,
        auction: Sequence[AuctionEntry],
        legal_calls: frozenset[Call],
    ) -> Call:
        """Choose a call using this hand and the public auction only."""


@dataclass(frozen=True)
class OneNoTrumpSystem:
    system_id: str
    version: str
    name: str
    min_hcp: int
    max_hcp: int
    balanced_shapes: frozenset[tuple[int, int, int, int]]
    default_call: PassCall = PassCall.PASS

    def choose_call(
        self,
        hand: HandProfile,
        auction: Sequence[AuctionEntry],
        legal_calls: frozenset[Call],
    ) -> Call:
        del auction  # This system's only rule depends on the hand and legality.
        preferred_call = Bid(1, Denomination.NOTRUMP)
        if (
            self.min_hcp <= hand.hcp <= self.max_hcp
            and hand.canonical_shape in self.balanced_shapes
            and preferred_call in legal_calls
        ):
            return preferred_call
        return self.default_call


def load_system(path: str | Path) -> OneNoTrumpSystem:
    """Load and validate the supported 1NT-or-pass TOML system format."""
    system_path = Path(path)
    with system_path.open("rb") as system_file:
        config = tomllib.load(system_file)

    _require_keys(
        config,
        {"schema_version", "id", "version", "name", "default_call", "rule"},
        f"system file {system_path}",
    )
    if (
        not isinstance(config["schema_version"], int)
        or isinstance(config["schema_version"], bool)
        or config["schema_version"] != 1
    ):
        raise ValueError("Only system schema_version 1 is supported.")
    if not isinstance(config["id"], str) or not config["id"].strip():
        raise ValueError("System id must be a non-empty string.")
    if not isinstance(config["version"], str) or not config["version"].strip():
        raise ValueError("System version must be a non-empty string.")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("System name must be a non-empty string.")
    if config["default_call"] != PassCall.PASS.value:
        raise ValueError("The supported system's default_call must be PASS.")

    rule = config["rule"]
    if not isinstance(rule, dict):
        raise ValueError("System rule must be a TOML table.")
    _require_keys(
        rule,
        {"id", "call", "min_hcp", "max_hcp", "balanced_shapes"},
        "system rule",
    )
    if not isinstance(rule["id"], str) or not rule["id"].strip():
        raise ValueError("System rule id must be a non-empty string.")
    if rule["call"] != "1NT":
        raise ValueError("The supported system rule call must be 1NT.")

    min_hcp = rule["min_hcp"]
    max_hcp = rule["max_hcp"]
    if (
        not isinstance(min_hcp, int)
        or isinstance(min_hcp, bool)
        or not isinstance(max_hcp, int)
        or isinstance(max_hcp, bool)
        or not 0 <= min_hcp <= max_hcp <= 37
    ):
        raise ValueError("System HCP bounds must satisfy 0 <= min_hcp <= max_hcp <= 37.")

    raw_shapes = rule["balanced_shapes"]
    if not isinstance(raw_shapes, list) or not raw_shapes:
        raise ValueError("balanced_shapes must be a non-empty TOML array.")
    balanced_shapes = frozenset(_parse_shape(shape) for shape in raw_shapes)

    return OneNoTrumpSystem(
        system_id=config["id"],
        version=config["version"],
        name=config["name"],
        min_hcp=min_hcp,
        max_hcp=max_hcp,
        balanced_shapes=balanced_shapes,
    )


def _parse_shape(value: object) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, str)
        or len(value) != 4
        or not value.isascii()
        or not value.isdigit()
    ):
        raise ValueError("Each balanced shape must be a four-digit string such as '4333'.")
    shape = tuple(int(length) for length in value)
    if tuple(sorted(shape, reverse=True)) != shape or sum(shape) != 13:
        raise ValueError(f"Balanced shape {value!r} must be descending and total 13.")
    return shape[0], shape[1], shape[2], shape[3]


def _require_keys(config: dict[str, object], expected: set[str], context: str) -> None:
    actual = set(config)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown keys: {', '.join(sorted(unknown))}")
        raise ValueError(f"Invalid {context} ({'; '.join(details)}).")
