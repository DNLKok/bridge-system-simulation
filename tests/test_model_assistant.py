import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Sequence

from bridge_sim.deals import PBNBoard, Vulnerability, parse_pbn
from bridge_sim.model import (
    AuctionEntry,
    Bid,
    Contract,
    Denomination,
    DoubleCall,
    PassCall,
    RedoubleCall,
    Seat,
)
from bridge_sim.auction_assistant import DEFAULT_GENERATE_LIMIT
from bridge_sim.model_assistant import (
    DEFAULT_N,
    ModelAssistantResult,
    _advanced_candidate_calls,
    _result_as_json,
    _score_call_ns,
    rate_candidate_calls,
    run_model_assistant,
)


ROOT = Path(__file__).resolve().parents[1]
PBN_SAMPLE = ROOT / "dealer" / "output2.txt"
PASSOUT_SYSTEM = ROOT / "systems" / "passout-cheapest-call.toml"


class FakeDealerProvider:
    def __init__(self, boards: Sequence[PBNBoard]) -> None:
        self.boards = boards
        self.script: str | None = None
        self.seed: int | None = None
        self.calls: list[tuple[str, int | None]] = []

    def generate(
        self,
        script_path: str | Path,
        *,
        seed: int | None = None,
    ) -> Iterator[PBNBoard]:
        self.script = Path(script_path).read_text(encoding="utf-8")
        self.seed = seed
        self.calls.append((self.script, seed))
        requested = next(
            int(line.split()[1])
            for line in self.script.splitlines()
            if line.startswith("produce ")
        )
        yield from self.boards[:requested]


class FakeDDS:
    def __init__(
        self,
        table_count: int | None = None,
        table: dict[Denomination, dict[Seat, int]] | None = None,
    ) -> None:
        self.table_count = table_count
        self.table = table
        self.boards: tuple[PBNBoard, ...] | None = None
        self.call_count = 0

    def solve_trick_tables(
        self,
        boards: Sequence[PBNBoard],
    ) -> tuple[dict[Denomination, dict[Seat, int]], ...]:
        self.call_count += 1
        self.boards = tuple(boards)
        count = (
            len(self.boards)
            if self.table_count is None
            else self.table_count
        )
        table = self.table or {
            denomination: {seat: 0 for seat in Seat}
            for denomination in Denomination
        }
        return (table,) * count


class ModelAssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        with PBN_SAMPLE.open(encoding="utf-8") as sample:
            self.board = next(parse_pbn(sample))
        self.hand = self.board.hands[self.board.dealer]

    def _run(
        self,
        provider: FakeDealerProvider,
        solver: FakeDDS,
        *,
        n: int = DEFAULT_N,
        advanced: bool = False,
    ) -> ModelAssistantResult:
        return run_model_assistant(
            perspective=self.board.dealer,
            hand=self.hand,
            auction=(),
            partner_system=PASSOUT_SYSTEM,
            opponent_system=PASSOUT_SYSTEM,
            dealer=self.board.dealer,
            vulnerability=self.board.vulnerability,
            n=n,
            seed=20261127,
            advanced=advanced,
            dealer_provider=provider,
            dds_solver=solver,
        )

    def test_defaults_to_64_and_solves_all_boards_in_one_batch(self) -> None:
        provider = FakeDealerProvider((self.board,) * DEFAULT_N)
        solver = FakeDDS()

        result = self._run(provider, solver)

        self.assertIn(f"produce {DEFAULT_N}", provider.script or "")
        self.assertIn(
            f"generate {DEFAULT_GENERATE_LIMIT}",
            provider.script or "",
        )
        self.assertEqual(provider.seed, 20261127)
        self.assertEqual(len(result.boards), DEFAULT_N)
        self.assertEqual(len(result.trick_tables), DEFAULT_N)
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(solver.boards, result.boards)
        self.assertFalse(result.out_of_book)
        self.assertEqual(result.system_call, PassCall.PASS)
        self.assertEqual(result.recommended_call, PassCall.PASS)

    def test_no_generated_deals_warns_and_recommends_pass(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "incompatible"):
            result = self._run(FakeDealerProvider(()), FakeDDS(), n=1)

        self.assertEqual(result.boards, ())
        self.assertEqual(result.system_call, PassCall.PASS)
        self.assertEqual(result.recommended_call, PassCall.PASS)
        self.assertEqual(tuple(rating.call for rating in result.ratings), (PassCall.PASS,))

    def test_advanced_flag_keeps_basic_ratings_when_in_book(self) -> None:
        provider = FakeDealerProvider((self.board,))
        result = self._run(provider, FakeDDS(), n=1, advanced=True)

        self.assertFalse(result.out_of_book)
        self.assertEqual(len(provider.calls), 1)

    def test_advanced_candidates_include_cheapest_strains_and_game_bids(self) -> None:
        legal_calls = frozenset(
            {
                PassCall.PASS,
                *(
                    Bid(level, denomination)
                    for level in range(1, 8)
                    for denomination in Denomination
                ),
            }
        )

        self.assertEqual(
            _advanced_candidate_calls(legal_calls),
            (
                Bid(1, Denomination.CLUBS),
                Bid(1, Denomination.DIAMONDS),
                Bid(1, Denomination.HEARTS),
                Bid(1, Denomination.SPADES),
                Bid(1, Denomination.NOTRUMP),
                Bid(3, Denomination.NOTRUMP),
                Bid(4, Denomination.HEARTS),
                Bid(4, Denomination.SPADES),
                Bid(5, Denomination.CLUBS),
                Bid(5, Denomination.DIAMONDS),
            ),
        )

    def test_advanced_candidates_include_legal_doubles_and_deduplicate_overlap(
        self,
    ) -> None:
        game_bids = (
            Bid(3, Denomination.NOTRUMP),
            Bid(4, Denomination.HEARTS),
            Bid(4, Denomination.SPADES),
            Bid(5, Denomination.CLUBS),
            Bid(5, Denomination.DIAMONDS),
        )
        legal_calls = frozenset(
            {PassCall.PASS, DoubleCall.DOUBLE, RedoubleCall.REDOUBLE, *game_bids}
        )

        self.assertEqual(
            _advanced_candidate_calls(legal_calls),
            (
                DoubleCall.DOUBLE,
                RedoubleCall.REDOUBLE,
                *game_bids,
            ),
        )

    def test_uses_requested_sample_size_in_generated_script(self) -> None:
        provider = FakeDealerProvider((self.board,) * 3)
        solver = FakeDDS()

        result = self._run(provider, solver, n=3)

        self.assertIn("produce 3", provider.script or "")
        self.assertEqual(len(result.boards), 3)

    def test_rates_only_configured_candidate_calls(self) -> None:
        candidates = (
            PassCall.PASS,
            Bid(1, Denomination.CLUBS),
        )
        provider = FakeDealerProvider((self.board,))

        ratings = rate_candidate_calls(
            perspective=self.board.dealer,
            hand=self.hand,
            auction=(),
            partner_system=PASSOUT_SYSTEM,
            opponent_system=PASSOUT_SYSTEM,
            dealer=self.board.dealer,
            vulnerability=self.board.vulnerability,
            candidates=candidates,
            n=1,
            seed=20261127,
            dealer_provider=provider,
            dds_solver=FakeDDS(),
        )

        self.assertEqual(tuple(rating.call for rating in ratings), candidates)
        self.assertIn("produce 1", provider.script or "")

    def test_advanced_mode_uses_partner_basic_rating_for_non_pass_calls(self) -> None:
        system_config = """\
schema_version = 2
id = "test-one-no-trump"
version = "1.0.0"
name = "Test 1NT system"
out_of_book = "when_no_rule_matches"

[rule]
id = "open-1nt"
call = "1NT"
min_hcp = 7
max_hcp = 7
balanced_shapes = ["4333"]
"""
        table = {
            denomination: {seat: 0 for seat in Seat}
            for denomination in Denomination
        }
        table[Denomination.NOTRUMP][Seat.EAST] = 6
        seed = 20261127

        with TemporaryDirectory() as directory:
            system_path = Path(directory) / "system.toml"
            system_path.write_text(system_config, encoding="utf-8")
            provider = FakeDealerProvider((self.board,) * 2)
            solver = FakeDDS(table=table)
            result = run_model_assistant(
                perspective=Seat.WEST,
                hand=self.board.hands[Seat.WEST],
                auction=(),
                partner_system=system_path,
                opponent_system=system_path,
                dealer=Seat.WEST,
                vulnerability=self.board.vulnerability,
                n=2,
                seed=seed,
                advanced=True,
                dealer_provider=provider,
                dds_solver=solver,
            )
            basic_result = run_model_assistant(
                perspective=Seat.WEST,
                hand=self.board.hands[Seat.WEST],
                auction=(),
                partner_system=system_path,
                opponent_system=system_path,
                dealer=Seat.WEST,
                vulnerability=self.board.vulnerability,
                n=2,
                seed=seed,
                dealer_provider=FakeDealerProvider((self.board,) * 2),
                dds_solver=FakeDDS(table=table),
            )
            partner_result = run_model_assistant(
                perspective=Seat.EAST,
                hand=self.board.hands[Seat.EAST],
                auction=(
                    AuctionEntry(Seat.WEST, Bid(1, Denomination.CLUBS)),
                    AuctionEntry(Seat.NORTH, PassCall.PASS),
                ),
                partner_system=system_path,
                opponent_system=system_path,
                dealer=Seat.WEST,
                vulnerability=self.board.vulnerability,
                n=2,
                seed=seed + 1,
                dealer_provider=FakeDealerProvider((self.board,) * 2),
                dds_solver=FakeDDS(table=table),
                unconstrained_entries=(0, 1),
            )

        self.assertTrue(result.out_of_book)
        self.assertEqual(
            partner_result.recommended_call,
            Bid(1, Denomination.NOTRUMP),
        )
        advanced_rating = next(
            rating
            for rating in result.ratings
            if rating.call == Bid(1, Denomination.CLUBS)
        )
        partner_rating = next(
            rating
            for rating in partner_result.ratings
            if rating.call == partner_result.recommended_call
        )
        self.assertEqual(
            advanced_rating.expected_imps,
            partner_rating.expected_imps,
        )
        self.assertLess(partner_rating.expected_imps, 0)
        self.assertNotEqual(
            advanced_rating.expected_imps,
            next(
                rating.expected_imps
                for rating in basic_result.ratings
                if rating.call == Bid(1, Denomination.CLUBS)
            ),
        )
        self.assertEqual(
            next(
                rating.expected_imps
                for rating in result.ratings
                if rating.call is PassCall.PASS
            ),
            next(
                rating.expected_imps
                for rating in basic_result.ratings
                if rating.call is PassCall.PASS
            ),
        )
        self.assertEqual(
            next(
                rating.expected_imps
                for rating in result.ratings
                if rating.call == Bid(2, Denomination.CLUBS)
            ),
            next(
                rating.expected_imps
                for rating in basic_result.ratings
                if rating.call == Bid(2, Denomination.CLUBS)
            ),
        )
        candidate_count = len(
            _advanced_candidate_calls(
                frozenset(rating.call for rating in result.ratings)
            )
        )
        self.assertEqual(
            len(provider.calls),
            2 + candidate_count,
        )
        self.assertEqual(
            len({call_seed for _, call_seed in provider.calls}),
            2 + candidate_count,
        )
        self.assertEqual(solver.call_count, 1 + candidate_count)
        self.assertIn("produce 2", provider.calls[0][0])
        self.assertIn("produce 1", provider.calls[1][0])

    def test_fails_if_dealer_cannot_produce_requested_sample_size(self) -> None:
        provider = FakeDealerProvider((self.board,))
        solver = FakeDDS()

        with self.assertRaisesRegex(RuntimeError, "requested exactly 2"):
            self._run(provider, solver, n=2)

        self.assertIsNone(solver.boards)

    def test_fails_if_dds_returns_the_wrong_number_of_tables(self) -> None:
        provider = FakeDealerProvider((self.board,) * 2)
        solver = FakeDDS(table_count=1)

        with self.assertRaisesRegex(RuntimeError, "returned 1 trick tables"):
            self._run(provider, solver, n=2)

    def test_out_of_book_recommendation_uses_ratings_and_fixed_seed(self) -> None:
        perspective = next(
            seat
            for seat in Seat
            if not (
                15 <= self.board.hands[seat].hcp <= 17
                and self.board.hands[seat].profile.canonical_shape
                in {(4, 3, 3, 3), (4, 4, 3, 2), (5, 3, 3, 2)}
            )
        )
        board = PBNBoard(
            self.board.board_number,
            perspective,
            self.board.vulnerability,
            self.board.hands,
            self.board.seed,
            self.board.tags,
        )
        table = {
            denomination: {
                seat: 13 if seat is perspective else 0
                for seat in Seat
            }
            for denomination in Denomination
        }
        provider = FakeDealerProvider((board,) * 2)
        solver = FakeDDS(table=table)
        system_config = """\
schema_version = 2
id = "partial-one-no-trump"
version = "1.0.0"
name = "Partial 1NT system"
out_of_book = "when_no_rule_matches"

[rule]
id = "open-1nt"
call = "1NT"
min_hcp = 15
max_hcp = 17
balanced_shapes = ["4333", "4432", "5332"]
"""

        with TemporaryDirectory() as directory:
            system_path = Path(directory) / "system.toml"
            system_path.write_text(system_config, encoding="utf-8")
            result = run_model_assistant(
                perspective=perspective,
                hand=board.hands[perspective],
                auction=(),
                partner_system=system_path,
                opponent_system=system_path,
                dealer=perspective,
                vulnerability=board.vulnerability,
                n=2,
                seed=20261127,
                dealer_provider=provider,
                dds_solver=solver,
            )

        self.assertTrue(result.out_of_book)
        self.assertIsNone(result.system_call)
        self.assertEqual(provider.seed, 20261127)
        self.assertEqual(
            result.recommended_call,
            Bid(7, Denomination.HEARTS),
        )
        self.assertEqual(
            result.recommended_call,
            max(result.ratings, key=lambda rating: rating.expected_imps).call,
        )
        pass_rating = next(
            rating for rating in result.ratings if rating.call is PassCall.PASS
        )
        self.assertIn("TODO", pass_rating.note or "")
        payload = json.loads(_result_as_json(result, 20261127))
        self.assertEqual(payload["system_status"], "out_of_book")
        self.assertEqual(payload["recommended_call"], "7H")
        self.assertEqual(payload["sample_size"], 2)
        self.assertEqual(len(payload["call_ratings"]), len(result.ratings))

    def test_down_contract_is_assumed_doubled_and_pass_keeps_that_score(self) -> None:
        table = {
            denomination: {seat: 0 for seat in Seat}
            for denomination in Denomination
        }
        table[Denomination.CLUBS][Seat.NORTH] = 5
        contract = Contract(1, Denomination.CLUBS, Seat.NORTH)

        self.assertEqual(
            _score_call_ns(
                Bid(1, Denomination.CLUBS),
                table,
                Vulnerability.NONE,
                Seat.NORTH,
                None,
            ),
            -300,
        )
        self.assertEqual(
            _score_call_ns(
                PassCall.PASS,
                table,
                Vulnerability.NONE,
                Seat.EAST,
                contract,
            ),
            -300,
        )
        self.assertEqual(
            _score_call_ns(
                DoubleCall.DOUBLE,
                table,
                Vulnerability.NONE,
                Seat.EAST,
                contract,
            ),
            -300,
        )
        self.assertEqual(
            _score_call_ns(
                RedoubleCall.REDOUBLE,
                table,
                Vulnerability.NONE,
                Seat.EAST,
                Contract(1, Denomination.CLUBS, Seat.NORTH, multiplier=2),
            ),
            -600,
        )
        self.assertEqual(
            _score_call_ns(
                PassCall.PASS,
                table,
                Vulnerability.NONE,
                Seat.NORTH,
                None,
            ),
            0,
        )
        making_table = {
            denomination: {seat: 0 for seat in Seat}
            for denomination in Denomination
        }
        making_table[Denomination.CLUBS][Seat.NORTH] = 8
        self.assertEqual(
            _score_call_ns(
                Bid(1, Denomination.CLUBS),
                making_table,
                Vulnerability.NONE,
                Seat.NORTH,
                None,
            ),
            90,
        )

    def test_candidate_contract_uses_partners_first_bid_as_declarer(self) -> None:
        table = {
            denomination: {seat: 0 for seat in Seat}
            for denomination in Denomination
        }
        table[Denomination.HEARTS][Seat.NORTH] = 5
        table[Denomination.HEARTS][Seat.SOUTH] = 10
        auction = (
            AuctionEntry(Seat.NORTH, Bid(1, Denomination.NOTRUMP)),
            AuctionEntry(Seat.EAST, PassCall.PASS),
            AuctionEntry(Seat.SOUTH, Bid(3, Denomination.HEARTS)),
            AuctionEntry(Seat.WEST, PassCall.PASS),
        )

        score = _score_call_ns(
            Bid(4, Denomination.HEARTS),
            table,
            Vulnerability.NONE,
            Seat.NORTH,
            None,
            auction,
        )

        self.assertEqual(score, 420)


if __name__ == "__main__":
    unittest.main()
