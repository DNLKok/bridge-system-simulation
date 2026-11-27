from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .deals import PBNBoard
from .model import AuctionResult, Denomination, Seat

_TRICKS_BY_CODE = "0123456789ABCD"
_SEAT_INDEX = {
    Seat.NORTH: 0,
    Seat.EAST: 1,
    Seat.SOUTH: 2,
    Seat.WEST: 3,
}
_STRAIN_INDEX = {
    Denomination.SPADES: 0,
    Denomination.HEARTS: 1,
    Denomination.DIAMONDS: 2,
    Denomination.CLUBS: 3,
    Denomination.NOTRUMP: 4,
}


class TablesPBN4Solver:
    """Batch-evaluate contracts with this workspace's TablesPBN4 executable."""

    def __init__(self, executable: str | Path | None = None) -> None:
        self.executable = (
            Path(executable)
            if executable is not None
            else Path(__file__).resolve().parents[1]
            / "Dealer-Version-2--main"
            / "DDS_Demo"
            / "SolveTables"
            / "TablesPBN4"
        )

    def evaluate(
        self,
        boards: Sequence[PBNBoard],
        auctions: Sequence[AuctionResult],
    ) -> tuple[AuctionResult, ...]:
        """Attach signed over/undertricks after all auctions have completed."""
        if len(boards) != len(auctions):
            raise ValueError("There must be one auction result for each board.")

        deal_lines: list[str] = []
        contract_indices: list[int] = []
        for index, (board, auction) in enumerate(zip(boards, auctions, strict=True)):
            if auction.contract is None:
                continue
            deal_line = board.tags.get("Deal")
            if deal_line is None:
                raise ValueError(
                    f"Board {board.board_number} has a contract but no PBN Deal tag."
                )
            if len(deal_line) != 69 or "\n" in deal_line or "\r" in deal_line:
                raise ValueError(
                    f"Board {board.board_number} does not have a one-line "
                    "69-character PBN Deal value for TablesPBN4."
                )
            deal_lines.append(deal_line)
            contract_indices.append(index)

        if not deal_lines:
            return tuple(replace(auction, trick_delta=None) for auction in auctions)

        completed = subprocess.run(
            [str(self.executable)],
            input="\n".join(deal_lines) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"TablesPBN4 exited with status {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )

        output_lines = [line for line in completed.stdout.splitlines() if line]
        if len(output_lines) != len(deal_lines):
            raise RuntimeError(
                f"TablesPBN4 returned {len(output_lines)} result lines for "
                f"{len(deal_lines)} input deals. stderr: {completed.stderr.strip()}"
            )

        evaluated = list(auctions)
        for index, expected_deal, output_line in zip(
            contract_indices,
            deal_lines,
            output_lines,
            strict=True,
        ):
            output_deal, separator, table = output_line.rpartition(":")
            if not separator or output_deal != expected_deal:
                raise RuntimeError(
                    "TablesPBN4 output did not match the corresponding input deal."
                )
            table_groups = table.split(",")
            if len(table_groups) != 4 or any(len(group) != 5 for group in table_groups):
                raise RuntimeError("TablesPBN4 returned a malformed trick table.")

            contract = auctions[index].contract
            if contract is None:
                raise RuntimeError("Internal error: contract mapping was lost.")
            encoded_tricks = table_groups[_SEAT_INDEX[contract.declarer]][
                _STRAIN_INDEX[contract.denomination]
            ]
            tricks = _decode_tricks(encoded_tricks)
            trick_delta = tricks - (contract.level + 6)
            if not -13 <= trick_delta <= 13:
                raise RuntimeError(
                    f"TablesPBN4 returned an invalid trick delta: {trick_delta}."
                )
            evaluated[index] = replace(auctions[index], trick_delta=trick_delta)

        return tuple(evaluated)


def _decode_tricks(value: str) -> int:
    try:
        return _TRICKS_BY_CODE.index(value)
    except ValueError as error:
        raise RuntimeError(
            f"TablesPBN4 returned an invalid trick-count code: {value!r}."
        ) from error
