import unittest
from pathlib import Path

from bridge_sim import (
    HandProfile,
    Seat,
    Vulnerability,
    load_system,
    parse_pbn,
    run_auction,
)


ROOT = Path(__file__).resolve().parents[1]
PBN_SAMPLE = ROOT / "dealer" / "output2.txt"
SYSTEM_FILE = ROOT / "systems" / "one-no-trump-15-17.toml"


class PBNTests(unittest.TestCase):
    def test_parse_multi_board_dealer_output(self) -> None:
        with PBN_SAMPLE.open(encoding="utf-8") as sample:
            boards = list(parse_pbn(sample))

        self.assertEqual(len(boards), 3)
        self.assertEqual([board.board_number for board in boards], [1, 2, 3])
        self.assertTrue(all(board.seed == 42069 for board in boards))
        self.assertTrue(all(board.dealer is Seat.WEST for board in boards))
        self.assertTrue(
            all(board.vulnerability is Vulnerability.ALL for board in boards)
        )
        self.assertEqual(
            boards[0].hand_for(Seat.NORTH),
            HandProfile(5, (3, 4, 6, 0)),
        )
        self.assertEqual(
            boards[0].hands[Seat.NORTH].suits,
            ("Q54", "9762", "QJT972", ""),
        )

    def test_parse_dealer_progress_marker_before_first_pbn_tag(self) -> None:
        pbn_lines = PBN_SAMPLE.read_text(encoding="utf-8").splitlines()
        for marker in ("F", "O"):
            lines = [f"{marker}{pbn_lines[0]}", *pbn_lines[1:]]
            with self.subTest(marker=marker):
                boards = list(parse_pbn(lines))

                self.assertEqual(len(boards), 3)
                self.assertTrue(all(board.seed == 42069 for board in boards))

    def test_pbn_board_can_be_used_directly_for_auction(self) -> None:
        with PBN_SAMPLE.open(encoding="utf-8") as sample:
            board = next(parse_pbn(sample))
        system = load_system(SYSTEM_FILE)

        result = run_auction(board, system, system, dealer=board.dealer)

        self.assertTrue(result.passed_out)
        self.assertEqual(len(result.entries), 4)


if __name__ == "__main__":
    unittest.main()
