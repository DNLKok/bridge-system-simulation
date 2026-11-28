from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .auction import run_auction
from .deals import DealerProvider, PBNBoard, Vulnerability
from .model import AuctionResult, Seat
from .systems import CompositeBiddingSystem, load_system

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


@dataclass(frozen=True)
class StandardTestSettings:
    config_path: Path
    deal_count: int
    seed: int
    advanced_out_of_book: bool
    dealer: Seat
    vulnerability: Vulnerability
    north_south_system: Path
    east_west_system: Path
    output: Path


@dataclass(frozen=True)
class BoardAuction:
    board: PBNBoard
    auction: AuctionResult


def load_settings(path: str | Path = _DEFAULT_CONFIG) -> StandardTestSettings:
    config_path = Path(path).resolve()
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    if set(config) != {"schema_version", "simulation", "systems"}:
        raise ValueError("Standard-test settings have missing or unknown sections.")
    if config["schema_version"] != 1:
        raise ValueError("Standard-test settings must use schema_version 1.")

    simulation = config["simulation"]
    systems = config["systems"]
    if not isinstance(simulation, dict) or set(simulation) != {
        "deal_count",
        "seed",
        "advanced_out_of_book",
        "dealer",
        "vulnerability",
        "output",
    }:
        raise ValueError("The simulation settings have missing or unknown keys.")
    if not isinstance(systems, dict) or set(systems) != {
        "north_south",
        "east_west",
    }:
        raise ValueError("The systems section must define north_south and east_west.")

    deal_count = _integer_setting(simulation, "deal_count", minimum=1)
    seed = _integer_setting(simulation, "seed", minimum=0)
    advanced = simulation["advanced_out_of_book"]
    if not isinstance(advanced, bool):
        raise ValueError("advanced_out_of_book must be a boolean.")
    if advanced:
        raise ValueError(
            "The standard test runs its bidding systems directly; "
            "advanced out-of-book logic must be disabled."
        )

    dealer_name = simulation["dealer"]
    dealers = {
        seat.name.lower(): seat
        for seat in Seat
    }
    if not isinstance(dealer_name, str) or dealer_name.lower() not in dealers:
        raise ValueError("dealer must be north, east, south, or west.")
    vulnerability_name = simulation["vulnerability"]
    vulnerabilities = {
        "none": Vulnerability.NONE,
        "ns": Vulnerability.NORTH_SOUTH,
        "ew": Vulnerability.EAST_WEST,
        "all": Vulnerability.ALL,
    }
    if (
        not isinstance(vulnerability_name, str)
        or vulnerability_name.lower() not in vulnerabilities
    ):
        raise ValueError("vulnerability must be none, ns, ew, or all.")

    def relative_path(value: object, key: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty path.")
        resolved = Path(value)
        return (config_path.parent / resolved).resolve()

    return StandardTestSettings(
        config_path=config_path,
        deal_count=deal_count,
        seed=seed,
        advanced_out_of_book=advanced,
        dealer=dealers[dealer_name.lower()],
        vulnerability=vulnerabilities[vulnerability_name.lower()],
        north_south_system=relative_path(systems["north_south"], "north_south"),
        east_west_system=relative_path(systems["east_west"], "east_west"),
        output=relative_path(simulation["output"], "output"),
    )


def run_standard_test(
    config_path: str | Path = _DEFAULT_CONFIG,
    *,
    output: str | Path | None = None,
) -> Path:
    settings = load_settings(config_path)
    north_south = load_system(settings.north_south_system)
    east_west = load_system(settings.east_west_system)
    output_path = Path(output).resolve() if output is not None else settings.output

    with tempfile.TemporaryDirectory(prefix="bridge-standard-test-") as directory:
        dealer_script = Path(directory) / "unrestricted.dealer"
        dealer_script.write_text(
            _dealer_script(settings),
            encoding="utf-8",
        )
        boards = tuple(
            DealerProvider().generate(dealer_script, seed=settings.seed)
        )

    if len(boards) != settings.deal_count:
        raise RuntimeError(
            f"Dealer produced {len(boards)} boards; "
            f"expected {settings.deal_count}."
        )
    if any(board.seed != settings.seed for board in boards):
        raise RuntimeError("Dealer output did not retain the requested seed.")

    results = tuple(
        BoardAuction(
            board,
            run_auction(
                board,
                north_south,
                east_west,
                dealer=board.dealer,
                vulnerability=board.vulnerability,
            ),
        )
        for board in boards
    )
    document = _report_html(settings, north_south, east_west, results)
    _render_pdf(document, output_path)
    return output_path


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


def _dealer_script(settings: StandardTestSettings) -> str:
    return "\n".join(
        (
            f"generate {settings.deal_count}",
            f"produce {settings.deal_count}",
            f"dealer {settings.dealer.name.lower()}",
            f"vulnerable {settings.vulnerability.value.lower()}",
            "action printpbn",
            "",
        )
    )


def _report_html(
    settings: StandardTestSettings,
    north_south: CompositeBiddingSystem,
    east_west: CompositeBiddingSystem,
    results: Sequence[BoardAuction],
) -> str:
    system_details = (
        _system_html("North-South", north_south),
        _system_html("East-West", east_west),
    )
    boards = "\n".join(
        _board_html(result, first=index == 0)
        for index, result in enumerate(results)
    )
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
.contract {{ font-weight: bold; margin-top: 2mm; }}
.small {{ color: #536978; font-size: 8.5pt; }}
</style>
</head>
<body>
<header class="report-header">
  <h1>Bridge Standard Test</h1>
  <p class="subtitle">Randomly generated deals and deterministic system auctions</p>
  <div class="metadata">
    <p class="setting"><span class="label">Deals:</span> {settings.deal_count}</p>
    <p class="setting"><span class="label">Dealer seed:</span> {settings.seed}</p>
    <p class="setting"><span class="label">Advanced out-of-book logic:</span> Disabled</p>
    <p class="setting"><span class="label">Dealer / vulnerability:</span>
      {html.escape(settings.dealer.name.title())} / {html.escape(settings.vulnerability.value)}
    </p>
    {system_details[0]}
    {system_details[1]}
  </div>
</header>
{boards}
</body>
</html>
"""


def _system_html(label: str, system: CompositeBiddingSystem) -> str:
    modules = " &rarr; ".join(
        html.escape(module.name) for module in system.modules
    )
    return (
        f'<p class="setting"><span class="label">{html.escape(label)}:</span> '
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
      <thead><tr><th>North</th><th>East</th><th>South</th><th>West</th></tr></thead>
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
    rows: list[str] = []
    for start in range(0, len(auction.entries), 4):
        calls = [""] * 4
        for entry in auction.entries[start : start + 4]:
            calls[_SEAT_COLUMNS[entry.seat]] = html.escape(str(entry.call))
        rows.append(
            "<tr>" + "".join(f"<td>{call}</td>" for call in calls) + "</tr>"
        )
    return "".join(rows)


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
    output = run_standard_test(arguments.config, output=arguments.output)
    settings = load_settings(arguments.config)
    print(
        f"Generated {settings.deal_count} boards with seed {settings.seed}: "
        f"{output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
