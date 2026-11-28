from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import TypeAlias

from .deals import Vulnerability
from .model import Contract, Denomination, ParResult, Seat
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
    declarer: Seat


@dataclass(frozen=True)
class _SacrificeOption:
    contract: Contract
    trick_delta: int
    undertricks: int


def calculate_par_result(
    trick_table: TrickTable,
    dealer: Seat,
    vulnerability: Vulnerability,
) -> ParResult:
    """Return a dealer-specific double-dummy par result and all tied contracts.

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
        return ParResult(None, None, 0)

    if highest_numbers[0] != highest_numbers[1]:
        primacy = 0 if highest_numbers[0] > highest_numbers[1] else 1
    else:
        top_denomination = _BID_ORDER[(highest_numbers[0] - 1) % 5]
        top_tricks = trick_table[top_denomination]
        dealer_index = _SEAT_ORDER.index(dealer)
        top_trick_count = max(top_tricks.values())
        primacy = next(
            _side_for_seat(_SEAT_ORDER[(dealer_index + offset) % 4])
            for offset in range(4)
            if top_tricks[_SEAT_ORDER[(dealer_index + offset) % 4]]
            == top_trick_count
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
    best_making_contract: _Candidate | None = None
    best_sacrifice: _SacrificeOption | None = None
    best_plus_contracts: list[_Candidate] = []
    best_sacrifice_options: list[_SacrificeOption] = []
    for candidate in candidate_contracts:
        sacrifices = _best_sacrifices(
            trick_table,
            candidate,
            opponent_by_denomination,
            primacy,
            dealer,
        )
        sacrifice = sacrifices[0] if sacrifices else None
        down = 9999 if sacrifice is None else sacrifice.undertricks
        down_limit = _affordable_sacrifice(
            candidate.score,
            opposite_side,
            vulnerability,
        )
        if down <= down_limit:
            sacrifice_found = True
            if down > best_down:
                best_down = down
                best_sacrifice = sacrifice
                best_sacrifice_options = list(sacrifices)
            elif down == best_down:
                best_sacrifice_options.extend(sacrifices)
        else:
            if candidate.score > best_plus:
                best_plus = candidate.score
                best_making_contract = candidate
                best_plus_contracts = [candidate]
            elif candidate.score == best_plus:
                best_plus_contracts.append(candidate)
                if (
                    best_making_contract is not None
                    and candidate.contract_number
                    < best_making_contract.contract_number
                ):
                    best_making_contract = candidate

    sacrifice_score_ns = _sacrifice_score_ns(
        best_down,
        opposite_side,
        vulnerability,
    )
    if not sacrifice_found or best_plus > abs(sacrifice_score_ns):
        if best_plus == 0 or best_making_contract is None:
            return ParResult(None, None, 0)
        contract = Contract(
            best_making_contract.tricks - 6,
            best_making_contract.denomination,
            best_making_contract.declarer,
        )
        score_ns = best_plus if primacy == 0 else -best_plus
        alternative_contracts = tuple(
            Contract(candidate.tricks - 6, candidate.denomination, candidate.declarer)
            for candidate in sorted(
                best_plus_contracts,
                key=lambda option: option.contract_number,
            )
            if candidate is not best_making_contract
        )
        return _scored_par_result(
            contract,
            0,
            vulnerability,
            score_ns,
            alternative_contracts,
        )
    if best_sacrifice is None:
        return ParResult(None, None, 0)
    return _scored_par_result(
        best_sacrifice.contract,
        best_sacrifice.trick_delta,
        vulnerability,
        sacrifice_score_ns,
        tuple(
            dict.fromkeys(
                option.contract
                for option in best_sacrifice_options
                if option.contract != best_sacrifice.contract
            )
        ),
    )


def calculate_par_score_ns(
    trick_table: TrickTable,
    dealer: Seat,
    vulnerability: Vulnerability,
) -> int:
    """Return only the North-South par score."""
    return calculate_par_result(trick_table, dealer, vulnerability).score_ns


def _scored_par_result(
    contract: Contract,
    trick_delta: int,
    vulnerability: Vulnerability,
    expected_score_ns: int,
    alternative_contracts: tuple[Contract, ...] = (),
) -> ParResult:
    score_ns = calculate_score_ns(contract, vulnerability, trick_delta)
    if score_ns != expected_score_ns:
        raise RuntimeError(
            "Calculated par contract does not produce the calculated par score."
        )
    return ParResult(contract, trick_delta, score_ns, alternative_contracts)


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
        declarer = max(
            _SIDE_SEATS[side],
            key=lambda seat: trick_table[denomination][seat],
        )
        tricks = trick_table[denomination][declarer]
        contract_number = 5 * (tricks - 7) + denomination_index + 1
        score = 0
        if tricks >= 7:
            score_ns = calculate_score_ns(
                Contract(tricks - 6, denomination, declarer),
                vulnerability,
                0,
            )
            score = score_ns if side == 0 else -score_ns
        candidates.append(
            _Candidate(denomination, contract_number, tricks, score, declarer)
        )
    return tuple(candidates)


def _best_sacrifices(
    trick_table: TrickTable,
    candidate: _Candidate,
    opponent_by_denomination: Mapping[Denomination, _Candidate],
    declaring_side: int,
    dealer: Seat,
) -> tuple[_SacrificeOption, ...]:
    best: list[_SacrificeOption] = []
    for denomination in _BID_ORDER:
        opponent_contract = opponent_by_denomination[denomination]
        if denomination == candidate.denomination:
            options = _same_denomination_sacrifices(
                trick_table, candidate, declaring_side, dealer
            )
        else:
            distance = candidate.contract_number - opponent_contract.contract_number
            down = ceil(distance / 5)
            options = ()
            if (
                1 <= down <= 13
                and opponent_contract.contract_number + 5 * down <= 35
            ):
                options = (
                    _SacrificeOption(
                        Contract(
                            opponent_contract.tricks - 6 + down,
                            denomination,
                            opponent_contract.declarer,
                            multiplier=2,
                        ),
                        -down,
                        down,
                    ),
                )
        for option in options:
            if opponent_contract.contract_number + 5 * option.undertricks > 35:
                continue
            if not best or option.undertricks < best[0].undertricks:
                best = [option]
            elif option.undertricks == best[0].undertricks:
                best.append(option)
    return tuple(best)


def _same_denomination_sacrifices(
    trick_table: TrickTable,
    candidate: _Candidate,
    declaring_side: int,
    dealer: Seat,
) -> tuple[_SacrificeOption, ...]:
    current_contract_tricks = (candidate.contract_number + 34) // 5
    increment_required = False
    options: list[_SacrificeOption] = []
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
            down = difference + int(increment_required)
            level = current_contract_tricks - 6 + int(increment_required)
            if 1 <= level <= 7 and 1 <= down <= 13:
                options.append(
                    _SacrificeOption(
                        Contract(
                            level,
                            candidate.denomination,
                            seat,
                            multiplier=2,
                        ),
                        -down,
                        down,
                    )
                )
    return tuple(options)


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
