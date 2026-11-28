from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .auction import run_auction
from .deals import DealerProvider, PBNBoard, Vulnerability
from .model import (
    AuctionEntry,
    AuctionResult,
    Bid,
    Call,
    Denomination,
    DoubleCall,
    HandProfile,
    PassCall,
    RedoubleCall,
    Seat,
)
from .model_assistant import _auction_state
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
        document = _report_html(
            settings,
            north_south,
            east_west,
            results,
            generated_at,
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
    document = _report_html(
        settings,
        north_south,
        east_west,
        results,
        generated_at,
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
        _board_html(result, first=index == 0)
        for index, result in enumerate(results)
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
.report-header {{ border-bottom: 1px solid #8199aa; margin-bottom: 4mm; padding-bottom: 3mm; }}
.subtitle {{ color: #536978; margin: 0 0 3mm; }}
.metadata {{ display: grid; gap: 2mm 7mm; grid-template-columns: 1fr 1fr; }}
.setting {{ margin: 0; }}
.label {{ color: #536978; font-weight: bold; }}
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
.auction {{ margin-top: 5mm; max-width: 125mm; }}
.auction table {{ border-collapse: collapse; width: 100%; }}
.auction th, .auction td {{ border: 1px solid #aebdc7; padding: 1.2mm 3mm; text-align: center; }}
.auction th {{ background: #edf3f6; }}
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


def _board_html(result: BoardAuction, *, first: bool) -> str:
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
