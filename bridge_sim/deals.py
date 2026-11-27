from __future__ import annotations

import json
import re
import subprocess
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Thread
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping

from .model import HandProfile, Seat

_PBN_TAG = re.compile(r'^\[(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+"(?P<value>(?:\\.|[^"\\])*)"\]$')
_SEED = re.compile(r"\bseed\s+(\d+)\b", re.IGNORECASE)
_RANKS = frozenset("AKQJT98765432")
_HCP = {"A": 4, "K": 3, "Q": 2, "J": 1}
_PBN_SUITS = ("S", "H", "D", "C")
_EXPECTED_DECK = {
    (suit, rank)
    for suit in _PBN_SUITS
    for rank in _RANKS
}
_DEFAULT_DEALER_EXECUTABLE = (
    Path(__file__).resolve().parents[1] / "bin" / "dealer"
)


class Vulnerability(str, Enum):
    NONE = "None"
    NORTH_SOUTH = "NS"
    EAST_WEST = "EW"
    ALL = "All"


@dataclass(frozen=True)
class CardHand:
    """A PBN hand, with suit strings ordered spades, hearts, diamonds, clubs."""

    suits: tuple[str, str, str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.suits, tuple) or len(self.suits) != 4:
            raise ValueError("A card hand must contain four PBN suit strings.")
        normalized: list[str] = []
        for suit_cards in self.suits:
            if not isinstance(suit_cards, str):
                raise TypeError("PBN suit strings must be text.")
            if suit_cards == "-":
                suit_cards = ""
            if any(rank not in _RANKS for rank in suit_cards):
                raise ValueError(f"Invalid rank in PBN suit {suit_cards!r}.")
            if len(set(suit_cards)) != len(suit_cards):
                raise ValueError(f"Duplicate rank in PBN suit {suit_cards!r}.")
            normalized.append(suit_cards)
        if sum(map(len, normalized)) != 13:
            raise ValueError("A bridge hand must contain exactly 13 cards.")
        object.__setattr__(self, "suits", tuple(normalized))

    @property
    def hcp(self) -> int:
        return sum(_HCP.get(rank, 0) for suit in self.suits for rank in suit)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            len(self.suits[0]),
            len(self.suits[1]),
            len(self.suits[2]),
            len(self.suits[3]),
        )

    @property
    def profile(self) -> HandProfile:
        suit_hcp = (
            sum(_HCP.get(rank, 0) for rank in self.suits[0]),
            sum(_HCP.get(rank, 0) for rank in self.suits[1]),
            sum(_HCP.get(rank, 0) for rank in self.suits[2]),
            sum(_HCP.get(rank, 0) for rank in self.suits[3]),
        )
        return HandProfile(self.hcp, self.shape, suit_hcp)


@dataclass(frozen=True)
class PBNBoard:
    board_number: int
    dealer: Seat
    vulnerability: Vulnerability
    hands: Mapping[Seat, CardHand]
    seed: int | None = None
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.board_number < 1:
            raise ValueError("PBN board numbers must be positive.")
        copied_hands = dict(self.hands)
        if set(copied_hands) != set(Seat):
            raise ValueError("A PBN board must contain a hand for each seat.")
        if any(not isinstance(hand, CardHand) for hand in copied_hands.values()):
            raise TypeError("Each PBN seat must map to a CardHand.")
        _validate_deck(copied_hands)
        object.__setattr__(self, "hands", MappingProxyType(copied_hands))
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))

    def hand_for(self, seat: Seat) -> HandProfile:
        return self.hands[seat].profile


def parse_pbn(lines: Iterable[str]) -> Iterator[PBNBoard]:
    """Parse PBN board records from dealer output, ignoring its text summary."""
    tags: dict[str, str] = {}
    line_number = 0
    first_content = True
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped and first_content:
            first_content = False
            if len(stripped) > 1 and stripped[0].isalpha() and stripped[1] == "[":
                stripped = stripped[1:]
        if not stripped:
            if tags:
                yield _board_from_tags(tags, line_number)
                tags = {}
            continue

        match = _PBN_TAG.fullmatch(stripped)
        if match is None:
            if "Board" in tags and "Deal" in tags:
                yield _board_from_tags(tags, line_number)
                tags = {}
            continue

        name = match.group("name")
        if name == "Event" and "Board" in tags:
            yield _board_from_tags(tags, line_number)
            tags = {}
        try:
            value = json.loads(f'"{match.group("value")}"')
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid quoted PBN tag value on line {line_number}."
            ) from error
        if not isinstance(value, str):
            raise ValueError(f"PBN tag {name!r} must have a string value.")
        tags[name] = value

    if tags:
        yield _board_from_tags(tags, line_number)


def _board_from_tags(tags: Mapping[str, str], line_number: int) -> PBNBoard:
    required = {"Board", "Dealer", "Vulnerable", "Deal"}
    missing = required - tags.keys()
    if missing:
        raise ValueError(
            f"Incomplete PBN board near line {line_number}; missing "
            f"{', '.join(sorted(missing))}."
        )
    try:
        board_number = int(tags["Board"])
        dealer = Seat(tags["Dealer"])
        vulnerability = Vulnerability(tags["Vulnerable"])
    except ValueError as error:
        raise ValueError(
            f"Invalid board number, dealer, or vulnerability near line {line_number}."
        ) from error

    hands = _parse_pbn_deal(tags["Deal"], line_number)
    seed_match = _SEED.search(tags.get("Event", ""))
    seed = int(seed_match.group(1)) if seed_match else None
    return PBNBoard(
        board_number=board_number,
        dealer=dealer,
        vulnerability=vulnerability,
        hands=hands,
        seed=seed,
        tags=tags,
    )


def _parse_pbn_deal(value: str, line_number: int) -> dict[Seat, CardHand]:
    start_seat_text, separator, hand_data = value.partition(":")
    if not separator:
        raise ValueError(f"PBN Deal tag near line {line_number} has no starting seat.")
    try:
        current_seat = Seat(start_seat_text)
    except ValueError as error:
        raise ValueError(
            f"PBN Deal tag near line {line_number} has an invalid starting seat."
        ) from error

    hand_records = hand_data.split()
    if len(hand_records) != 4:
        raise ValueError(
            f"PBN Deal tag near line {line_number} must contain four hands."
        )

    hands: dict[Seat, CardHand] = {}
    for hand_record in hand_records:
        suits = hand_record.split(".")
        if len(suits) != 4:
            raise ValueError(
                f"PBN hand near line {line_number} must contain four suits."
            )
        hands[current_seat] = CardHand(
            (suits[0], suits[1], suits[2], suits[3])
        )
        current_seat = current_seat.next
    _validate_deck(hands)
    return hands


def _validate_deck(hands: Mapping[Seat, CardHand]) -> None:
    cards = [
        (suit, rank)
        for hand in hands.values()
        for suit, ranks in zip(_PBN_SUITS, hand.suits)
        for rank in ranks
    ]
    if len(cards) != 52 or set(cards) != _EXPECTED_DECK:
        raise ValueError("A deal must contain each of the 52 distinct cards exactly once.")


class DealerProvider:
    """Run Hans van Staveren's dealer executable and stream its PBN output."""

    def __init__(
        self,
        executable: str | Path = _DEFAULT_DEALER_EXECUTABLE,
    ) -> None:
        self.executable = str(executable)

    def generate(
        self,
        script_path: str | Path,
        *,
        seed: int | None = None,
    ) -> Iterator[PBNBoard]:
        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"Dealer script not found: {script}")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ValueError("Dealer seed must be a non-negative integer.")

        command = [self.executable]
        if seed is not None:
            command.extend(["-s", str(seed)])
        command.append(str(script))

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise RuntimeError("Could not capture dealer output.")

        output_tail: deque[str] = deque(maxlen=20)
        error_tail: deque[str] = deque(maxlen=20)
        board_count = 0

        def capture_output() -> Iterator[str]:
            for output_line in process.stdout:
                output_tail.append(output_line.rstrip())
                yield output_line

        def capture_errors() -> None:
            for error_line in process.stderr:
                error_tail.append(error_line.rstrip())

        error_reader = Thread(target=capture_errors, daemon=True)
        error_reader.start()
        try:
            for board in parse_pbn(capture_output()):
                board_count += 1
                yield board

            return_code = process.wait()
            error_reader.join()
            if return_code != 0:
                diagnostic = (
                    "stderr:\n"
                    + "\n".join(error_tail)
                    + "\nstdout:\n"
                    + "\n".join(output_tail)
                )
                raise RuntimeError(
                    f"dealer exited with status {return_code}.\n{diagnostic}"
                )
            if board_count == 0:
                raise RuntimeError(
                    "dealer produced no PBN boards; use an action that emits "
                    "complete PBN deals, such as 'action printpbn'."
                )
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            error_reader.join()
            process.stdout.close()
            process.stderr.close()
