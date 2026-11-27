from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import TypeAlias

from .deals import Vulnerability
from .model import Contract, Denomination, Seat
from .scoring import calculate_score_ns

TrickTable: TypeAlias = Mapping[Denomination, Mapping[Seat, int]]

_BID_ORDER = (
    Denomination.CLUBS,
    Denomination.DIAMONDS,
    Denomination.HEARTS,
    Denomination.SPADES,
    Denomination.NOTRUMP,
)
_DDS_TABLE_STRAIN_INDEX = {
    Denomination.SPADES: 0,
    Denomination.HEARTS: 1,
    Denomination.DIAMONDS: 2,
    Denomination.CLUBS: 3,
    Denomination.NOTRUMP: 4,
}
_SIDE_SEATS = ((Seat.NORTH, Seat.SOUTH), (Seat.EAST, Seat.WEST))
_SEAT_ORDER = (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST)


@dataclass(frozen=True)
class _Candidate:
    denomination: Denomination
    contract_number: int
    tricks: int
    score: int


def calculate_par_score_ns(
    trick_table: TrickTable,
    dealer: Seat,
    vulnerability: Vulnerability,
) -> int:
    """Calculate the dealer-specific double-dummy par score for North-South.

    The table maps each denomination to the tricks available to each seat.
    """
    _validate_table(trick_table)
    if not isinstance(dealer, Seat):
        raise TypeError("dealer must be a Seat.")
    if not isinstance(vulnerability, Vulnerability):
        raise TypeError("vulnerability must be a Vulnerability.")

    candidates = tuple(
        _make_side_candidates(trick_table, side, vulnerability)
        for side in range(2)
    )
    highest_numbers = tuple(
        max(
            (candidate.contract_number for candidate in side if candidate.tricks >= 7),
            default=0,
        )
        for side in candidates
    )
    if highest_numbers == (0, 0):
        return 0

    if highest_numbers[0] != highest_numbers[1]:
        primacy = 0 if highest_numbers[0] > highest_numbers[1] else 1
    else:
        top_denomination = _BID_ORDER[(highest_numbers[0] - 1) % 5]
        top_tricks = trick_table[top_denomination]
        primacy = next(
            _side_for_seat(_SEAT_ORDER[( _SEAT_ORDER.index(dealer) + offset) % 4])
            for offset in range(4)
            if top_tricks[
                _SEAT_ORDER[(_SEAT_ORDER.index(dealer) + offset) % 4]
            ]
            == max(top_tricks[seat] for seat in Seat)
        )

    opposite_side = 1 - primacy
    best_making = max(
        candidates[primacy],
        key=lambda candidate: (
            candidate.score,
            -candidate.contract_number,
        ),
    )
    candidate_contracts = sorted(
        (
            candidate
            for candidate in candidates[primacy]
            if candidate.tricks >= 7
            and candidate.contract_number >= best_making.contract_number
        ),
        key=lambda candidate: candidate.contract_number,
        reverse=True,
    )
    opponent_by_denomination = {
        candidate.denomination: candidate for candidate in candidates[opposite_side]
    }

    best_plus = 0
    best_down = 0
    sacrifice_found = False
    for candidate in candidate_contracts:
        down = _best_sacrifice_down(
            trick_table,
            candidate,
            opponent_by_denomination,
            primacy,
            dealer,
        )
        down_limit = _affordable_sacrifice(
            candidate.score,
            opposite_side,
            vulnerability,
        )
        if down <= down_limit:
            best_down = max(best_down, down)
            sacrifice_found = True
        else:
            best_plus = max(best_plus, candidate.score)

    sacrifice_score_ns = _sacrifice_score_ns(
        best_down,
        opposite_side,
        vulnerability,
    )
    if not sacrifice_found or best_plus > abs(sacrifice_score_ns):
        return best_plus if primacy == 0 else -best_plus
    return sacrifice_score_ns


def _validate_table(trick_table: TrickTable) -> None:
    if set(trick_table) != set(Denomination):
        raise ValueError("A trick table must contain all five denominations.")
    for denomination in Denomination:
        tricks_by_seat = trick_table[denomination]
        if set(tricks_by_seat) != set(Seat):
            raise ValueError(
                f"The {denomination.value} row must contain all four seats."
            )
        if any(
            not isinstance(tricks, int)
            or isinstance(tricks, bool)
            or not 0 <= tricks <= 13
            for tricks in tricks_by_seat.values()
        ):
            raise ValueError("Every trick count must be an integer from 0 to 13.")


def _make_side_candidates(
    trick_table: TrickTable,
    side: int,
    vulnerability: Vulnerability,
) -> tuple[_Candidate, ...]:
    candidates: list[_Candidate] = []
    for denomination_index, denomination in enumerate(_BID_ORDER):
        tricks = max(
            trick_table[denomination][seat]
            for seat in _SIDE_SEATS[side]
        )
        contract_number = 5 * (tricks - 7) + denomination_index + 1
        score = 0
        if tricks >= 7:
            declarer = next(
                seat
                for seat in _SIDE_SEATS[side]
                if trick_table[denomination][seat] == tricks
            )
            score_ns = calculate_score_ns(
                Contract(tricks - 6, denomination, declarer),
                vulnerability,
                0,
            )
            score = score_ns if side == 0 else -score_ns
        candidates.append(
            _Candidate(denomination, contract_number, tricks, score)
        )
    return tuple(candidates)


def _best_sacrifice_down(
    trick_table: TrickTable,
    candidate: _Candidate,
    opponent_by_denomination: Mapping[Denomination, _Candidate],
    declaring_side: int,
    dealer: Seat,
) -> int:
    best_down = 9999
    for denomination in _BID_ORDER:
        opponent_contract = opponent_by_denomination[denomination]
        if denomination == candidate.denomination:
            down = _same_denomination_sacrifice_down(
                trick_table,
                candidate,
                declaring_side,
                dealer,
            )
        else:
            distance = candidate.contract_number - opponent_contract.contract_number
            down = ceil(distance / 5)
        if opponent_contract.contract_number + 5 * down <= 35:
            best_down = min(best_down, down)
    return best_down


def _same_denomination_sacrifice_down(
    trick_table: TrickTable,
    candidate: _Candidate,
    declaring_side: int,
    dealer: Seat,
) -> int:
    current_contract_tricks = (candidate.contract_number + 34) // 5
    increment_required = False
    possible_downs: list[int] = []
    dealer_index = _SEAT_ORDER.index(dealer)
    for offset in range(4):
        seat = _SEAT_ORDER[(dealer_index + offset) % 4]
        difference = (
            current_contract_tricks
            - trick_table[candidate.denomination][seat]
        )
        if _side_for_seat(seat) == declaring_side:
            if difference == 0:
                increment_required = True
        else:
            possible_downs.append(difference + int(increment_required))
    return min(possible_downs)


def _affordable_sacrifice(
    making_score: int,
    sacrificing_side: int,
    vulnerability: Vulnerability,
) -> int:
    sacrifice_declarer = _SIDE_SEATS[sacrificing_side][0]
    max_down = 0
    doubled_seven_clubs = Contract(
        7,
        Denomination.CLUBS,
        sacrifice_declarer,
        multiplier=2,
    )
    for down in range(1, 14):
        penalty = abs(
            calculate_score_ns(
                doubled_seven_clubs,
                vulnerability,
                -down,
            )
        )
        if penalty <= making_score:
            max_down = down
    return max_down


def _sacrifice_score_ns(
    undertricks: int,
    sacrificing_side: int,
    vulnerability: Vulnerability,
) -> int:
    if undertricks <= 0 or undertricks > 13:
        return 0
    sacrifice_declarer = _SIDE_SEATS[sacrificing_side][0]
    return calculate_score_ns(
        Contract(
            7,
            Denomination.CLUBS,
            sacrifice_declarer,
            multiplier=2,
        ),
        vulnerability,
        -undertricks,
    )


def _side_for_seat(seat: Seat) -> int:
    return 0 if seat in (Seat.NORTH, Seat.SOUTH) else 1
