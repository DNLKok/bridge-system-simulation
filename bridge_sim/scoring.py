from __future__ import annotations

from .deals import Vulnerability
from .model import Contract, Denomination, Seat


def calculate_score_ns(
    contract: Contract,
    vulnerability: Vulnerability,
    trick_delta: int,
) -> int:
    """Return duplicate score from North-South's perspective.

    ``trick_delta`` is declarer tricks minus the contract target, so negative
    values are undertricks and positive values are overtricks.
    """
    if not isinstance(contract, Contract):
        raise TypeError("contract must be a Contract.")
    if not isinstance(vulnerability, Vulnerability):
        raise TypeError("vulnerability must be a Vulnerability.")
    if (
        not isinstance(trick_delta, int)
        or isinstance(trick_delta, bool)
        or not -13 <= trick_delta <= 13
    ):
        raise ValueError("trick_delta must be an integer between -13 and 13.")
    if not 1 <= contract.level <= 7:
        raise ValueError("Contract level must be between 1 and 7.")
    if not isinstance(contract.denomination, Denomination):
        raise TypeError("Contract denomination must be a Denomination.")
    if not isinstance(contract.declarer, Seat):
        raise TypeError("Contract declarer must be a Seat.")

    tricks_taken = contract.level + 6 + trick_delta
    if not 0 <= tricks_taken <= 13:
        raise ValueError("trick_delta is impossible for this contract.")

    vulnerable = _is_vulnerable(contract.declarer, vulnerability)
    if trick_delta < 0:
        declaring_side_score = -_undertrick_penalty(
            -trick_delta,
            vulnerable,
            contract.multiplier,
        )
    else:
        contract_points = _contract_points(contract)
        declaring_side_score = contract_points
        declaring_side_score += _overtrick_score(
            trick_delta,
            contract,
            vulnerable,
        )
        declaring_side_score += (
            (500 if vulnerable else 300)
            if contract_points >= 100
            else 50
        )
        if contract.level == 6:
            declaring_side_score += 750 if vulnerable else 500
        elif contract.level == 7:
            declaring_side_score += 1500 if vulnerable else 1000
        if contract.multiplier == 2:
            declaring_side_score += 50
        elif contract.multiplier == 4:
            declaring_side_score += 100

    if contract.declarer in (Seat.NORTH, Seat.SOUTH):
        return declaring_side_score
    return -declaring_side_score


def _contract_points(contract: Contract) -> int:
    if contract.denomination in (Denomination.CLUBS, Denomination.DIAMONDS):
        points_per_trick = 20
        points = contract.level * points_per_trick
    elif contract.denomination in (Denomination.HEARTS, Denomination.SPADES):
        points_per_trick = 30
        points = contract.level * points_per_trick
    else:
        points = 40 + (contract.level - 1) * 30
    return points * contract.multiplier


def _overtrick_score(
    overtricks: int,
    contract: Contract,
    vulnerable: bool,
) -> int:
    if contract.multiplier == 1:
        if contract.denomination in (
            Denomination.CLUBS,
            Denomination.DIAMONDS,
        ):
            points_per_trick = 20
        else:
            points_per_trick = 30
        return overtricks * points_per_trick

    points_per_trick = 200 if vulnerable else 100
    if contract.multiplier == 4:
        points_per_trick *= 2
    return overtricks * points_per_trick


def _undertrick_penalty(
    undertricks: int,
    vulnerable: bool,
    multiplier: int,
) -> int:
    if multiplier == 1:
        return undertricks * (100 if vulnerable else 50)

    penalty = 0
    for trick_number in range(1, undertricks + 1):
        if vulnerable:
            penalty += 200 if trick_number == 1 else 300
        elif trick_number == 1:
            penalty += 100
        elif trick_number <= 3:
            penalty += 200
        else:
            penalty += 300
    if multiplier == 4:
        penalty *= 2
    return penalty


def _is_vulnerable(declarer: Seat, vulnerability: Vulnerability) -> bool:
    if vulnerability is Vulnerability.ALL:
        return True
    if vulnerability is Vulnerability.NORTH_SOUTH:
        return declarer in (Seat.NORTH, Seat.SOUTH)
    if vulnerability is Vulnerability.EAST_WEST:
        return declarer in (Seat.EAST, Seat.WEST)
    return False
