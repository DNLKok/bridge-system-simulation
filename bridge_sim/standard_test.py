from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .auction import run_auction
from .deals import DealerProvider, PBNBoard, Vulnerability
from .dds import TablesPBN4Solver
from .model import (
    AuctionEntry,
    AuctionResult,
    Bid,
    Call,
    Contract,
    Denomination,
    DoubleCall,
    HandProfile,
    PassCall,
    RedoubleCall,
    Seat,
)
from .model_assistant import _auction_state
from .par import calculate_par_result
from .systems import (
    CompositeBiddingSystem,
    DecisionContext,
    OutOfBookDecision,
    _matches_module_opening,
    load_system,
)

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "simulation" / "standard-test.toml"
)
_SEAT_COLUMNS = {
    Seat.NORTH: 0,
    Seat.EAST: 1,
    Seat.SOUTH: 2,
    Seat.WEST: 3,
}
_SUIT_ROWS = (
    ("S", "&spades;", "black"),
    ("H", "&hearts;", "#b00020"),
    ("D", "&diams;", "#b00020"),
    ("C", "&clubs;", "black"),
)
_SUIT_INDICES = {
    Denomination.SPADES: 0,
    Denomination.HEARTS: 1,
    Denomination.DIAMONDS: 2,
    Denomination.CLUBS: 3,
}
_GAME_LEVELS = {
    Denomination.NOTRUMP: 3,
    Denomination.HEARTS: 4,
    Denomination.SPADES: 4,
    Denomination.CLUBS: 5,
    Denomination.DIAMONDS: 5,
}
_RANK_ORDER = "AKQJT98765432"
_DEALER_CYCLE = (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST)
_CONDITIONAL_MAX_ATTEMPTS = 256
_VULNERABILITY_CYCLE = (
    Vulnerability.NONE,
    Vulnerability.NORTH_SOUTH,
    Vulnerability.EAST_WEST,
    Vulnerability.ALL,
    Vulnerability.NORTH_SOUTH,
    Vulnerability.EAST_WEST,
    Vulnerability.ALL,
    Vulnerability.NONE,
    Vulnerability.EAST_WEST,
    Vulnerability.ALL,
    Vulnerability.NONE,
    Vulnerability.NORTH_SOUTH,
    Vulnerability.ALL,
    Vulnerability.NONE,
    Vulnerability.NORTH_SOUTH,
    Vulnerability.EAST_WEST,
)


@dataclass(frozen=True)
class StandardTestSettings:
    config_path: Path
    deal_count: int
    seed: int
    decision_simulation_n: int
    advanced_out_of_book: bool
    north_south_system: Path
    north_south_abbreviation: str
    east_west_system: Path
    east_west_abbreviation: str
    output_directory: Path
    condition_label: str | None = None
    auction_prefix: tuple[AuctionEntry, ...] = ()


@dataclass(frozen=True)
class BoardAuction:
    board: PBNBoard
    auction: AuctionResult


_TrickTable = Mapping[Denomination, Mapping[Seat, int]]
_TrickTablesByBoard = Mapping[int, _TrickTable]


@dataclass(frozen=True)
class _ReportStatistic:
    label: str
    occurrences: int
    reference_label: str
    reference_count: int

    @property
    def percentage(self) -> float:
        if self.reference_count == 0:
            return 0.0
        return self.occurrences * 100 / self.reference_count


@dataclass(frozen=True)
class _ContractOutcome:
    game_bid: bool
    made_double_dummy: bool
    declaring_hcp: int
    missed_major_fit: bool
    short_trump_fit: bool
    double_dummy_game_available: bool


def load_settings(path: str | Path = _DEFAULT_CONFIG) -> StandardTestSettings:
    config_path = Path(path).resolve()
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    if set(config) not in (
        {"schema_version", "simulation", "systems"},
        {"schema_version", "simulation", "systems", "condition"},
    ):
        raise ValueError("Standard-test settings have missing or unknown sections.")
    if config["schema_version"] != 1:
        raise ValueError("Standard-test settings must use schema_version 1.")

    simulation = config["simulation"]
    systems = config["systems"]
    condition = config.get("condition")
    if not isinstance(simulation, dict) or set(simulation) != {
        "deal_count",
        "seed",
        "decision_simulation_n",
        "advanced_out_of_book",
        "output_directory",
    }:
        raise ValueError("The simulation settings have missing or unknown keys.")
    if not isinstance(systems, dict) or set(systems) != {
        "north_south",
        "north_south_abbreviation",
        "east_west",
        "east_west_abbreviation",
    }:
        raise ValueError(
            "The systems section must define both system paths and abbreviations."
        )
    condition_label: str | None = None
    auction_prefix: tuple[AuctionEntry, ...] = ()
    if condition is not None:
        if not isinstance(condition, dict) or set(condition) != {
            "label",
            "auction",
        }:
            raise ValueError(
                "The condition section must define exactly label and auction."
            )
        condition_label = _system_abbreviation(
            condition["label"],
            "condition.label",
        )
        raw_auction = condition["auction"]
        if (
            not isinstance(raw_auction, list)
            or not raw_auction
            or any(not isinstance(entry, str) for entry in raw_auction)
        ):
            raise ValueError("condition.auction must be a non-empty array of calls.")
        auction_prefix = tuple(
            _parse_auction_entry(entry, index)
            for index, entry in enumerate(raw_auction, start=1)
        )
        if any(
            current.seat is not previous.seat.next
            for previous, current in zip(auction_prefix, auction_prefix[1:])
        ):
            raise ValueError(
                "Each condition.auction seat must follow the previous seat "
                "clockwise."
            )

    deal_count = _integer_setting(simulation, "deal_count", minimum=1)
    decision_simulation_n = _integer_setting(
        simulation,
        "decision_simulation_n",
        minimum=1,
    )
    if simulation["seed"] == "unix":
        seed = int(time.time())
    else:
        seed = _integer_setting(simulation, "seed", minimum=0)
    advanced = simulation["advanced_out_of_book"]
    if not isinstance(advanced, bool):
        raise ValueError("advanced_out_of_book must be a boolean.")
    if advanced:
        raise ValueError(
            "The standard test runs its bidding systems directly; "
            "advanced out-of-book logic must be disabled."
        )

    def relative_path(value: object, key: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty path.")
        resolved = Path(value)
        return (config_path.parent / resolved).resolve()

    return StandardTestSettings(
        config_path=config_path,
        deal_count=deal_count,
        seed=seed,
        decision_simulation_n=decision_simulation_n,
        advanced_out_of_book=advanced,
        north_south_system=relative_path(systems["north_south"], "north_south"),
        north_south_abbreviation=_system_abbreviation(
            systems["north_south_abbreviation"],
            "north_south_abbreviation",
        ),
        east_west_system=relative_path(systems["east_west"], "east_west"),
        east_west_abbreviation=_system_abbreviation(
            systems["east_west_abbreviation"],
            "east_west_abbreviation",
        ),
        output_directory=relative_path(
            simulation["output_directory"],
            "output_directory",
        ),
        condition_label=condition_label,
        auction_prefix=auction_prefix,
    )


def run_standard_test(
    config_path: str | Path = _DEFAULT_CONFIG,
    *,
    output: str | Path | None = None,
) -> Path:
    return _run_standard_test(load_settings(config_path), output=output)


def _run_standard_test(
    settings: StandardTestSettings,
    *,
    output: str | Path | None = None,
) -> Path:
    generated_at = datetime.now().astimezone()
    north_south = load_system(settings.north_south_system)
    east_west = load_system(settings.east_west_system)
    output_path = (
        Path(output).resolve()
        if output is not None
        else settings.output_directory
        / _output_filename(settings, generated_at)
    )

    if settings.auction_prefix:
        results, dealer_seeds = _run_conditional_auctions(
            settings,
            north_south,
            east_west,
        )
        trick_tables = _calculate_report_trick_tables(results)
        statistics = _calculate_report_statistics(
            results,
            trick_tables=trick_tables,
        )
        document = _report_html(
            settings,
            north_south,
            east_west,
            results,
            generated_at,
            statistics=statistics,
            trick_tables=trick_tables,
            dealer_seeds=dealer_seeds,
        )
        _render_pdf(document, output_path)
        return output_path

    with tempfile.TemporaryDirectory(prefix="bridge-standard-test-") as directory:
        dealer_script = Path(directory) / "unrestricted.dealer"
        dealer_script.write_text(
            _dealer_script(settings),
            encoding="utf-8",
        )
        generated_boards = tuple(
            DealerProvider().generate(dealer_script, seed=settings.seed)
        )

    if len(generated_boards) != settings.deal_count:
        raise RuntimeError(
            f"Dealer produced {len(generated_boards)} boards; "
            f"expected {settings.deal_count}."
        )
    if any(board.seed != settings.seed for board in generated_boards):
        raise RuntimeError("Dealer output did not retain the requested seed.")
    boards = tuple(_apply_standard_rotation(board) for board in generated_boards)

    results = tuple(
        BoardAuction(
            board,
            run_auction(
                board,
                north_south,
                east_west,
                dealer=board.dealer,
                vulnerability=board.vulnerability,
                simulation_n=settings.decision_simulation_n,
                simulation_seed=settings.seed,
            ),
        )
        for board in boards
    )
    trick_tables = _calculate_report_trick_tables(results)
    statistics = _calculate_report_statistics(
        results,
        trick_tables=trick_tables,
    )
    document = _report_html(
        settings,
        north_south,
        east_west,
        results,
        generated_at,
        statistics=statistics,
        trick_tables=trick_tables,
    )
    _render_pdf(document, output_path)
    return output_path


def _run_conditional_auctions(
    settings: StandardTestSettings,
    north_south: CompositeBiddingSystem,
    east_west: CompositeBiddingSystem,
) -> tuple[tuple[BoardAuction, ...], tuple[int, ...]]:
    if not settings.auction_prefix:
        raise ValueError("A conditional test requires a non-empty auction prefix.")

    provider = DealerProvider()
    results: list[BoardAuction] = []
    dealer_seeds: list[int] = []
    seen_deals: set[tuple[tuple[str, ...], ...]] = set()
    for board_number in range(1, settings.deal_count + 1):
        dealer, vulnerability = _standard_rotation(board_number)
        expected_prefix = _conditional_prefix_for_dealer(
            settings.auction_prefix,
            dealer,
        )
        conditions = _conditional_dealer_conditions(
            expected_prefix,
            dealer,
            vulnerability,
            north_south,
            east_west,
        )
        accepted = False
        for attempt in range(_CONDITIONAL_MAX_ATTEMPTS):
            generation_seed = (
                settings.seed
                + (board_number - 1) * _CONDITIONAL_MAX_ATTEMPTS
                + attempt
            )
            script = _conditional_dealer_script(
                dealer,
                vulnerability,
                conditions,
            )
            with tempfile.TemporaryDirectory(
                prefix="bridge-conditional-test-"
            ) as directory:
                dealer_script = Path(directory) / "condition.dealer"
                dealer_script.write_text(script, encoding="utf-8")
                generated = tuple(
                    provider.generate(dealer_script, seed=generation_seed)
                )
            if not generated:
                raise RuntimeError(
                    f"Dealer produced no deals for board {board_number}."
                )
            if len(generated) != 1:
                raise RuntimeError(
                    f"Dealer produced {len(generated)} deals; expected one."
                )
            candidate = generated[0]
            if candidate.seed != generation_seed:
                raise RuntimeError(
                    "Dealer output did not retain the requested conditional-test seed."
                )

            tags = dict(candidate.tags)
            tags["Board"] = str(board_number)
            candidate = _apply_standard_rotation(
                replace(
                    candidate,
                    board_number=board_number,
                    tags=tags,
                )
            )
            fingerprint = tuple(
                candidate.hands[seat].suits
                for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST)
            )
            if fingerprint in seen_deals:
                continue

            prefix_matches = _matches_prefix_without_simulation(
                candidate,
                expected_prefix,
                north_south,
                east_west,
            )
            if prefix_matches is False:
                continue
            auction = run_auction(
                candidate,
                north_south,
                east_west,
                dealer=dealer,
                vulnerability=vulnerability,
                simulation_n=settings.decision_simulation_n,
                simulation_seed=settings.seed,
            )
            if tuple(auction.entries[: len(expected_prefix)]) != expected_prefix:
                continue

            seen_deals.add(fingerprint)
            results.append(BoardAuction(candidate, auction))
            dealer_seeds.append(generation_seed)
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                f"Could not generate board {board_number} matching the "
                f"configured auction prefix after "
                f"{_CONDITIONAL_MAX_ATTEMPTS} attempts."
            )

    return tuple(results), tuple(dealer_seeds)


def _conditional_prefix_for_dealer(
    configured_prefix: Sequence[AuctionEntry],
    dealer: Seat,
) -> tuple[AuctionEntry, ...]:
    prefix: list[AuctionEntry] = []
    seat = dealer
    while seat is not configured_prefix[0].seat:
        prefix.append(AuctionEntry(seat, PassCall.PASS))
        seat = seat.next
    prefix.extend(configured_prefix)

    auction: list[AuctionEntry] = []
    for entry in prefix:
        legal_calls, _ = _auction_state(auction, dealer, entry.seat)
        if entry.call not in legal_calls:
            raise ValueError(
                f"Configured call {entry.call!s} is illegal for "
                f"{entry.seat.value} after the generated passes."
            )
        auction.append(entry)
    return tuple(prefix)


def _conditional_dealer_conditions(
    prefix: Sequence[AuctionEntry],
    dealer: Seat,
    vulnerability: Vulnerability,
    north_south: CompositeBiddingSystem,
    east_west: CompositeBiddingSystem,
) -> tuple[str, ...]:
    conditions: list[str] = []
    auction: list[AuctionEntry] = []
    hand = HandProfile(0, (4, 3, 3, 3))
    for entry in prefix:
        legal_calls, _ = _auction_state(auction, dealer, entry.seat)
        system = (
            north_south
            if entry.seat in (Seat.NORTH, Seat.SOUTH)
            else east_west
        )
        context = DecisionContext(
            hand=hand,
            auction=tuple(auction),
            legal_calls=legal_calls,
            seat=entry.seat,
            dealer=dealer,
            vulnerability=vulnerability,
        )
        try:
            decision = system.dealer_call_condition(context, entry.call)
        except ValueError:
            if not _has_simulation_candidate(system, context, entry.call):
                raise
            decision = None
        if decision is not None:
            conditions.extend(decision.expressions)
        auction.append(entry)
    return tuple(conditions)


def _has_simulation_candidate(
    system: CompositeBiddingSystem,
    context: DecisionContext,
    call: Call,
) -> bool:
    return any(
        _matches_module_opening(context, module)
        and rule.evaluate_calls
        and call in rule.preferred_calls
        and rule.condition.matches_public_context(context)
        for module in system.modules
        for rule in module.rules
    )


def _conditional_dealer_script(
    dealer: Seat,
    vulnerability: Vulnerability,
    conditions: Sequence[str],
) -> str:
    vulnerability_name = {
        Vulnerability.NONE: "none",
        Vulnerability.NORTH_SOUTH: "ns",
        Vulnerability.EAST_WEST: "ew",
        Vulnerability.ALL: "all",
    }[vulnerability]
    lines = [
        "generate 10000000000",
        "produce 1",
        f"dealer {dealer.name.lower()}",
        f"vulnerable {vulnerability_name}",
    ]
    if conditions:
        condition_names = []
        for index, condition in enumerate(conditions, start=1):
            name = f"conditionalcondition{index}"
            lines.append(f"{name} = ({condition})")
            condition_names.append(name)
        lines.append("condition " + " && ".join(condition_names))
    lines.extend(("action printpbn", ""))
    return "\n".join(lines)


def _matches_prefix_without_simulation(
    board: PBNBoard,
    prefix: Sequence[AuctionEntry],
    north_south: CompositeBiddingSystem,
    east_west: CompositeBiddingSystem,
) -> bool | None:
    auction: list[AuctionEntry] = []
    for entry in prefix:
        legal_calls, _ = _auction_state(auction, board.dealer, entry.seat)
        system = (
            north_south
            if entry.seat in (Seat.NORTH, Seat.SOUTH)
            else east_west
        )
        try:
            call = system.choose_call(
                DecisionContext(
                    hand=board.hand_for(entry.seat),
                    auction=tuple(auction),
                    legal_calls=legal_calls,
                    seat=entry.seat,
                    dealer=board.dealer,
                    vulnerability=board.vulnerability,
                )
            )
        except RuntimeError as error:
            if "requires a simulation evaluator" in str(error):
                return None
            raise
        if call is OutOfBookDecision.OUT_OF_BOOK:
            return False
        if call != entry.call:
            return False
        auction.append(AuctionEntry(entry.seat, call))
    return True


def _integer_setting(
    settings: dict[str, object],
    key: str,
    *,
    minimum: int,
) -> int:
    value = settings[key]
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(f"{key} must be an integer greater than or equal to {minimum}.")
    return value


def _system_abbreviation(value: object, key: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,15}", value) is None
    ):
        raise ValueError(
            f"{key} must be 1-16 letters, numbers, underscores, or hyphens."
        )
    return value.upper()


def _parse_auction_entry(value: str, index: int) -> AuctionEntry:
    seat_token, separator, call_token = value.partition(":")
    if not separator or not seat_token.strip() or not call_token.strip():
        raise ValueError(
            f"condition.auction entry {index} must use SEAT:CALL format."
        )
    seat_names = {
        "N": Seat.NORTH,
        "NORTH": Seat.NORTH,
        "E": Seat.EAST,
        "EAST": Seat.EAST,
        "S": Seat.SOUTH,
        "SOUTH": Seat.SOUTH,
        "W": Seat.WEST,
        "WEST": Seat.WEST,
    }
    seat = seat_names.get(seat_token.strip().upper())
    if seat is None:
        raise ValueError(f"Invalid seat in condition.auction entry {index}.")

    normalized_call = call_token.strip().upper()
    if normalized_call in ("P", "PASS"):
        call = PassCall.PASS
    elif normalized_call in ("X", "DOUBLE"):
        call = DoubleCall.DOUBLE
    elif normalized_call in ("XX", "REDOUBLE"):
        call = RedoubleCall.REDOUBLE
    else:
        if len(normalized_call) < 2 or normalized_call[0] not in "1234567":
            raise ValueError(f"Invalid call in condition.auction entry {index}.")
        try:
            call = Bid(
                int(normalized_call[0]),
                Denomination(normalized_call[1:]),
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid call in condition.auction entry {index}."
            ) from error
    return AuctionEntry(seat, call)


def _output_filename(
    settings: StandardTestSettings,
    generated_at: datetime,
) -> str:
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{timestamp}_NS-{settings.north_south_abbreviation}_"
        f"EW-{settings.east_west_abbreviation}.pdf"
    )
    if settings.condition_label is not None:
        filename = filename.removesuffix(".pdf")
        filename += f"_COND-{settings.condition_label}.pdf"
    return filename


def _calculate_report_statistics(
    results: Sequence[BoardAuction],
    *,
    trick_tables: _TrickTablesByBoard | None = None,
) -> tuple[_ReportStatistic, ...]:
    if trick_tables is None:
        trick_tables = _calculate_report_trick_tables(results)
    return _build_report_statistics(results, trick_tables)


def _calculate_report_trick_tables(
    results: Sequence[BoardAuction],
) -> dict[int, dict[Denomination, dict[Seat, int]]]:
    trick_tables = TablesPBN4Solver().solve_trick_tables(
        tuple(result.board for result in results)
    )
    return {
        result.board.board_number: trick_table
        for result, trick_table in zip(
            results,
            trick_tables,
            strict=True,
        )
    }


def _build_report_statistics(
    results: Sequence[BoardAuction],
    trick_tables: _TrickTablesByBoard,
) -> tuple[_ReportStatistic, ...]:
    outcomes: list[_ContractOutcome] = []
    for result in results:
        contract = result.auction.contract
        if contract is None:
            continue

        trick_table = trick_tables[result.board.board_number]
        declaring_side = (contract.declarer, contract.declarer.next.next)
        declaring_hcp = sum(
            result.board.hand_for(seat).hcp for seat in declaring_side
        )
        combined_lengths = {
            denomination: sum(
                result.board.hand_for(seat).shape[suit_index]
                for seat in declaring_side
            )
            for denomination, suit_index in _SUIT_INDICES.items()
        }
        major_fits = {
            denomination
            for denomination in (Denomination.SPADES, Denomination.HEARTS)
            if combined_lengths[denomination] >= 8
        }
        double_dummy_game_available = any(
            max(trick_table[denomination][seat] for seat in declaring_side)
            >= game_level + 6
            for denomination, game_level in _GAME_LEVELS.items()
        )
        outcomes.append(
            _ContractOutcome(
                game_bid=_is_game_or_higher(contract),
                made_double_dummy=(
                    trick_table[contract.denomination][contract.declarer]
                    >= contract.level + 6
                ),
                declaring_hcp=declaring_hcp,
                missed_major_fit=(
                    bool(major_fits)
                    and contract.denomination not in major_fits
                ),
                short_trump_fit=(
                    contract.denomination is not Denomination.NOTRUMP
                    and combined_lengths[contract.denomination] <= 6
                ),
                double_dummy_game_available=double_dummy_game_available,
            )
        )

    deal_count = len(results)
    contract_count = len(outcomes)
    game_bid_count = sum(outcome.game_bid for outcome in outcomes)
    partscore_count = sum(not outcome.game_bid for outcome in outcomes)
    high_hcp_outcomes = tuple(
        outcome for outcome in outcomes if outcome.declaring_hcp >= 26
    )
    low_hcp_outcomes = tuple(
        outcome for outcome in outcomes if outcome.declaring_hcp <= 21
    )
    passed_out_count = deal_count - contract_count

    return (
        _ReportStatistic(
            "Game or higher bid by declarer",
            game_bid_count,
            "deals",
            deal_count,
        ),
        _ReportStatistic(
            "Game or higher bid and made double dummy",
            sum(
                outcome.game_bid and outcome.made_double_dummy
                for outcome in outcomes
            ),
            "game-or-higher contracts",
            game_bid_count,
        ),
        _ReportStatistic(
            "Declarer stopped in partscore",
            partscore_count,
            "deals",
            deal_count,
        ),
        _ReportStatistic(
            "Game available double dummy after stopping in partscore",
            sum(
                not outcome.game_bid
                and outcome.double_dummy_game_available
                for outcome in outcomes
            ),
            "partscore contracts",
            partscore_count,
        ),
        _ReportStatistic(
            "North-South declared",
            sum(
                result.auction.contract is not None
                and result.auction.contract.declarer
                in (Seat.NORTH, Seat.SOUTH)
                for result in results
            ),
            "deals",
            deal_count,
        ),
        _ReportStatistic(
            "Declared outside an available 8+ major fit",
            sum(outcome.missed_major_fit for outcome in outcomes),
            "deals",
            deal_count,
        ),
        _ReportStatistic(
            "Played in a trump fit of 6 cards or fewer (Burn's Law)",
            sum(outcome.short_trump_fit for outcome in outcomes),
            "deals",
            deal_count,
        ),
        _ReportStatistic(
            "Declaring side had 26+ HCP but did not bid game",
            sum(not outcome.game_bid for outcome in high_hcp_outcomes),
            "contracts with a 26+ HCP declaring side",
            len(high_hcp_outcomes),
        ),
        _ReportStatistic(
            "Declaring side had 21 or fewer HCP and bid game",
            sum(outcome.game_bid for outcome in low_hcp_outcomes),
            "contracts with a 21-or-fewer-HCP declaring side",
            len(low_hcp_outcomes),
        ),
        _ReportStatistic(
            "Bid contract made double dummy",
            sum(outcome.made_double_dummy for outcome in outcomes),
            "bid contracts",
            contract_count,
        ),
        _ReportStatistic(
            "Passed out",
            passed_out_count,
            "deals",
            deal_count,
        ),
    )


def _is_game_or_higher(contract: Contract) -> bool:
    return contract.level >= _GAME_LEVELS[contract.denomination]


def _standard_rotation(board_number: int) -> tuple[Seat, Vulnerability]:
    if (
        not isinstance(board_number, int)
        or isinstance(board_number, bool)
        or board_number < 1
    ):
        raise ValueError("Board numbers must be positive integers.")
    index = (board_number - 1) % len(_VULNERABILITY_CYCLE)
    return (
        _DEALER_CYCLE[(board_number - 1) % len(_DEALER_CYCLE)],
        _VULNERABILITY_CYCLE[index],
    )


def _apply_standard_rotation(board: PBNBoard) -> PBNBoard:
    dealer, vulnerability = _standard_rotation(board.board_number)
    tags = dict(board.tags)
    tags["Dealer"] = dealer.value
    tags["Vulnerable"] = vulnerability.value
    return replace(
        board,
        dealer=dealer,
        vulnerability=vulnerability,
        tags=tags,
    )


def _dealer_script(settings: StandardTestSettings) -> str:
    # Dealer generates the cards; standard board metadata is applied afterward.
    return "\n".join(
        (
            f"generate {settings.deal_count}",
            f"produce {settings.deal_count}",
            "dealer north",
            "vulnerable none",
            "action printpbn",
            "",
        )
    )


def _report_html(
    settings: StandardTestSettings,
    north_south: CompositeBiddingSystem,
    east_west: CompositeBiddingSystem,
    results: Sequence[BoardAuction],
    generated_at: datetime,
    *,
    statistics: Sequence[_ReportStatistic],
    trick_tables: _TrickTablesByBoard,
    dealer_seeds: Sequence[int] | None = None,
) -> str:
    system_details = (
        _system_html(
            "North-South",
            settings.north_south_abbreviation,
            north_south,
        ),
        _system_html(
            "East-West",
            settings.east_west_abbreviation,
            east_west,
        ),
    )
    boards = "\n".join(
        _board_html(
            result,
            first=index == 0,
            trick_table=trick_tables[result.board.board_number],
        )
        for index, result in enumerate(results)
    )
    statistic_rows = "\n".join(
        "<tr>"
        f'<th scope="row">{html.escape(statistic.label)}</th>'
        f'<td class="number">{statistic.occurrences}</td>'
        f'<td>{statistic.reference_count} '
        f"{html.escape(statistic.reference_label)}</td>"
        f'<td class="rate">{statistic.percentage:.1f}%</td>'
        "</tr>"
        for statistic in statistics
    )
    condition_details = ""
    seed_label = "Dealer seed"
    if settings.auction_prefix:
        seed_label = "Simulation seed"
        prefix_text = " ".join(
            f"{entry.seat.value}:{entry.call}" for entry in settings.auction_prefix
        )
        subtitle = f"Conditional test for auction prefix {prefix_text}"
        condition_details = (
            '<p class="setting"><span class="label">Condition label:</span> '
            f"{html.escape(settings.condition_label or '')}</p>\n"
            '<p class="setting"><span class="label">Required auction prefix:</span> '
            f"{html.escape(prefix_text)}</p>\n"
        )
        if dealer_seeds is not None:
            seed_text = ", ".join(str(seed) for seed in dealer_seeds)
            condition_details += (
                '<p class="setting"><span class="label">Dealer seeds '
                "(boards 1\u2013N):</span> "
                f"{html.escape(seed_text)}</p>\n"
            )
    else:
        subtitle = "Randomly generated deals and deterministic system auctions"
    return f"""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bridge standard test</title>
<style>
@page {{ size: A4; margin: 12mm; }}
* {{ box-sizing: border-box; }}
body {{ color: #17212b; font: 10pt/1.3 Arial, sans-serif; margin: 0; }}
h1 {{ color: #123b5d; font-size: 20pt; margin: 0 0 2mm; }}
h2 {{ color: #123b5d; font-size: 16pt; margin: 4mm 0 2mm; }}
h3 {{ font-size: 10pt; margin: 0 0 1mm; text-align: center; }}
.report-header {{ border-bottom: 1px solid #8199aa; break-after: page; margin-bottom: 4mm; padding-bottom: 3mm; page-break-after: always; }}
.subtitle {{ color: #536978; margin: 0 0 3mm; }}
.metadata {{ display: grid; gap: 2mm 7mm; grid-template-columns: 1fr 1fr; }}
.setting {{ margin: 0; }}
.label {{ color: #536978; font-weight: bold; }}
.statistics {{ margin-top: 3mm; }}
.statistics h2 {{ margin-top: 0; }}
.statistics-table {{ border-collapse: collapse; margin-top: 2mm; width: 100%; }}
.statistics-table th, .statistics-table td {{ border: 1px solid #aebdc7; padding: 1mm 1.5mm; text-align: left; }}
.statistics-table thead th {{ background: #edf3f6; }}
.statistics-table tbody tr {{ break-inside: avoid; page-break-inside: avoid; }}
.statistics-table .number, .statistics-table .rate {{ text-align: right; white-space: nowrap; }}
.board {{ break-before: page; page-break-before: always; }}
.board.first {{ break-before: auto; page-break-before: auto; }}
.board-meta {{ color: #536978; display: flex; gap: 7mm; margin-bottom: 3mm; }}
.deal-layout {{ display: grid; grid-template-columns: 1fr 1fr 1fr; grid-template-rows: auto auto auto; gap: 2mm; }}
.hand {{ border: 1px solid #c5d0d8; border-radius: 2mm; padding: 2mm; }}
.hand.north {{ grid-area: 1 / 2; }}
.hand.west {{ grid-area: 2 / 1; }}
.hand.east {{ grid-area: 2 / 3; }}
.hand.south {{ grid-area: 3 / 2; }}
.suit-row {{ display: grid; grid-template-columns: 6mm 1fr; font-family: "Liberation Mono", monospace; font-size: 8.5pt; white-space: nowrap; }}
.double-dummy {{ margin-top: 3mm; max-width: 125mm; }}
.double-dummy table, .auction table {{ border-collapse: collapse; width: 100%; }}
.double-dummy th, .double-dummy td, .auction th, .auction td {{ border: 1px solid #aebdc7; padding: 1.2mm 3mm; text-align: center; }}
.double-dummy thead th, .auction th {{ background: #edf3f6; }}
.double-dummy tbody th {{ text-align: left; }}
.par-summary {{ margin: 2mm 0 0; max-width: 125mm; }}
.auction {{ margin-top: 3mm; max-width: 125mm; }}
.auction th.vulnerable {{ background: #b00020; color: #fff; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
.contract {{ font-weight: bold; margin-top: 2mm; }}
.small {{ color: #536978; font-size: 8.5pt; }}
</style>
</head>
<body>
<header class="report-header">
  <h1>Bridge Standard Test</h1>
  <p class="subtitle">{html.escape(subtitle)}</p>
  <div class="metadata">
    <p class="setting"><span class="label">Deals:</span> {settings.deal_count}</p>
    <p class="setting"><span class="label">{seed_label}:</span> {settings.seed}</p>
    <p class="setting"><span class="label">Decision simulation N:</span>
      {settings.decision_simulation_n}
    </p>
    <p class="setting"><span class="label">Advanced out-of-book logic:</span> Disabled</p>
    <p class="setting"><span class="label">Dealer / vulnerability:</span>
      Standard 16-board rotation
    </p>
    <p class="setting"><span class="label">Generated:</span>
      {html.escape(generated_at.strftime("%Y-%m-%d %H:%M:%S %Z"))}
    </p>
    {condition_details}
    {system_details[0]}
    {system_details[1]}
  </div>
  <section class="statistics">
    <h2>Outcome statistics</h2>
    <p class="small">Percentages use the reference set shown. Passed-out deals are included in deal-based rates; declaring-side HCP rates use contracts only. A zero denominator gives 0%.</p>
    <table class="statistics-table">
      <thead>
        <tr><th>Statistic</th><th>Occurrences</th><th>Reference set</th><th>Percentage</th></tr>
      </thead>
      <tbody>{statistic_rows}</tbody>
    </table>
  </section>
</header>
{boards}
</body>
</html>
"""


def _system_html(
    label: str,
    abbreviation: str,
    system: CompositeBiddingSystem,
) -> str:
    modules = " &rarr; ".join(
        html.escape(module.name) for module in system.modules
    )
    return (
        f'<p class="setting"><span class="label">{html.escape(label)} '
        f"({html.escape(abbreviation)}):</span> "
        f"{html.escape(system.name)}<br><span class=\"small\">{modules}</span></p>"
    )


def _board_html(
    result: BoardAuction,
    *,
    first: bool,
    trick_table: _TrickTable,
) -> str:
    board = result.board
    dealer = html.escape(board.dealer.name.title())
    vulnerability = html.escape(board.vulnerability.value)
    contract = (
        "Passed out"
        if result.auction.contract is None
        else html.escape(str(result.auction.contract))
    )
    first_class = " first" if first else ""
    hands = "\n".join(
        _hand_html(board, seat, seat.name.lower())
        for seat in (Seat.NORTH, Seat.WEST, Seat.EAST, Seat.SOUTH)
    )
    par_result = calculate_par_result(
        trick_table,
        board.dealer,
        board.vulnerability,
    )
    par_contracts = "; ".join(
        html.escape(str(contract)) for contract in par_result.contracts
    ) or "Passed out"
    rows = _auction_rows(result.auction)
    headers = _auction_headers(board.vulnerability)
    return f"""\
<article class="board{first_class}">
  <h2>Board {board.board_number}</h2>
  <div class="board-meta">
    <span>Dealer: {dealer}</span>
    <span>Vulnerability: {vulnerability}</span>
  </div>
  <div class="deal-layout">{hands}</div>
  {_double_dummy_table_html(trick_table)}
  <p class="par-summary"><strong>PAR score (NS):</strong>
    {par_result.score_ns:+d}
    &nbsp; <strong>Contract(s):</strong> {par_contracts}
  </p>
  <section class="auction">
    <h3>Auction</h3>
    <table>
      <thead>{headers}</thead>
      <tbody>{rows}</tbody>
    </table>
    <p class="contract">Result: {contract}</p>
  </section>
</article>
"""


def _double_dummy_table_html(trick_table: _TrickTable) -> str:
    strains = (
        (Denomination.NOTRUMP, "NT"),
        (Denomination.SPADES, "♠"),
        (Denomination.HEARTS, "♥"),
        (Denomination.DIAMONDS, "♦"),
        (Denomination.CLUBS, "♣"),
    )
    headers = "".join(
        f'<th scope="col">{seat.value}</th>'
        for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST)
    )
    rows = "".join(
        "<tr>"
        f'<th scope="row">{strain}</th>'
        + "".join(
            f"<td>{trick_table[denomination][seat]}</td>"
            for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST)
        )
        + "</tr>"
        for denomination, strain in strains
    )
    return f"""\
<section class="double-dummy">
  <h3>Double-dummy tricks</h3>
  <table>
    <thead><tr><th scope="col">Strain</th>{headers}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</section>
"""


def _hand_html(board: PBNBoard, seat: Seat, position: str) -> str:
    hand = board.hands[seat]
    rows = []
    for suit, symbol, color in _SUIT_ROWS:
        ranks = "".join(
            rank for rank in _RANK_ORDER if rank in hand.suits["SHDC".index(suit)]
        )
        ranks = ranks or "&mdash;"
        rows.append(
            f'<div class="suit-row"><span style="color:{color}">{symbol}</span>'
            f"<span>{ranks}</span></div>"
        )
    return (
        f'<section class="hand {position}"><h3>{seat.name.title()}</h3>'
        + "".join(rows)
        + "</section>"
    )


def _auction_rows(auction: AuctionResult) -> str:
    if not auction.entries:
        return ""
    first_column = _SEAT_COLUMNS[auction.entries[0].seat]
    rows: list[list[str]] = []
    for call_index, entry in enumerate(auction.entries):
        row_index = (first_column + call_index) // 4
        while len(rows) <= row_index:
            rows.append([""] * 4)
        rows[row_index][_SEAT_COLUMNS[entry.seat]] = html.escape(str(entry.call))
    return "".join(
        "<tr>" + "".join(f"<td>{call}</td>" for call in calls) + "</tr>"
        for calls in rows
    )


def _auction_headers(vulnerability: Vulnerability) -> str:
    headers: list[str] = []
    for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH, Seat.WEST):
        vulnerable = (
            vulnerability is Vulnerability.ALL
            or (
                vulnerability is Vulnerability.NORTH_SOUTH
                and seat in (Seat.NORTH, Seat.SOUTH)
            )
            or (
                vulnerability is Vulnerability.EAST_WEST
                and seat in (Seat.EAST, Seat.WEST)
            )
        )
        class_name = ' class="vulnerable"' if vulnerable else ""
        headers.append(f"<th{class_name}>{seat.name.title()}</th>")
    return "<tr>" + "".join(headers) + "</tr>"


def _render_pdf(document: str, output: Path) -> None:
    browser = next(
        (
            executable
            for name in ("chromium", "chromium-browser", "google-chrome")
            if (executable := shutil.which(name)) is not None
        ),
        None,
    )
    if browser is None:
        raise FileNotFoundError(
            "PDF generation requires Chromium; install it or add it to PATH."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".bridge-report-",
        dir=output.parent,
    ) as directory:
        temp_dir = Path(directory)
        html_path = temp_dir / "report.html"
        pdf_path = temp_dir / "report.pdf"
        html_path.write_text(document, encoding="utf-8")
        completed = subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--user-data-dir={temp_dir / 'browser-profile'}",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Chromium failed to render the PDF "
                f"(exit {completed.returncode}):\n{completed.stderr[-4000:]}"
            )
        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            raise RuntimeError("Chromium completed without producing a PDF.")
        pdf_path.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a standard bridge deal-and-auction PDF."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="TOML settings file (default: simulation/standard-test.toml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="override the PDF output path from the settings file",
    )
    arguments = parser.parse_args(argv)
    settings = load_settings(arguments.config)
    output = _run_standard_test(settings, output=arguments.output)
    print(
        f"Generated {settings.deal_count} boards with seed {settings.seed}: "
        f"{output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
