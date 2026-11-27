import tempfile
import unittest
from pathlib import Path

from bridge_sim import (
    AuctionEntry,
    Bid,
    Call,
    DecisionContext,
    Denomination,
    HandProfile,
    OutOfBookDecision,
    PassCall,
    Seat,
    Vulnerability,
    load_system,
)


ROOT = Path(__file__).resolve().parents[1]
ONE_NT_MODULE = ROOT / "systems" / "modules" / "open-1nt-balanced-15-17.toml"
PASSOUT_SYSTEM = ROOT / "systems" / "passout-cheapest-call.toml"
PREEMPT_SYSTEM = ROOT / "systems" / "traditional-preempts.toml"


def context(
    hand: HandProfile,
    auction: tuple[AuctionEntry, ...],
    legal_calls: frozenset[Call],
    seat: Seat,
) -> DecisionContext:
    return DecisionContext(
        hand=hand,
        auction=auction,
        legal_calls=legal_calls,
        seat=seat,
        dealer=Seat.NORTH,
        vulnerability=Vulnerability.NONE,
    )


def _write_test_system(directory: Path) -> Path:
    module_templates = (
        (
            "strong.toml",
            "strong-openings",
            "strong-opening",
            "1C",
            20,
            37,
        ),
        (
            "medium.toml",
            "medium-openings",
            "medium-opening",
            "1D",
            10,
            19,
        ),
    )
    for filename, module_id, rule_id, call, minimum, maximum in module_templates:
        module = f"""\
schema_version = 1
kind = "module"
id = "{module_id}"
version = "1.0.0"
name = "{module_id}"

[[rules]]
id = "{rule_id}"
call = "{call}"

[rules.when]
min_hcp = {minimum}
max_hcp = {maximum}
"""
        (directory / filename).write_text(module, encoding="utf-8")

    manifest = """\
schema_version = 1
kind = "system"
id = "test-system"
version = "1.0.0"
name = "Module composition test"
on_no_match = "PASS"

[[modules]]
id = "strong-openings"
version = "1.0.0"
path = "strong.toml"

[[modules]]
id = "medium-openings"
version = "1.0.0"
path = "medium.toml"
"""
    system_path = directory / "system.toml"
    system_path.write_text(manifest, encoding="utf-8")
    return system_path


def _preempt_legal_calls() -> frozenset[Call]:
    calls: set[Call] = {PassCall.PASS}
    for level, denominations in (
        (2, (Denomination.DIAMONDS, Denomination.HEARTS, Denomination.SPADES)),
        (
            3,
            (
                Denomination.CLUBS,
                Denomination.DIAMONDS,
                Denomination.HEARTS,
                Denomination.SPADES,
            ),
        ),
        (
            4,
            (
                Denomination.CLUBS,
                Denomination.DIAMONDS,
                Denomination.HEARTS,
                Denomination.SPADES,
            ),
        ),
        (5, (Denomination.CLUBS, Denomination.DIAMONDS)),
    ):
        calls.update(Bid(level, denomination) for denomination in denominations)
    return frozenset(calls)


class BiddingModuleTests(unittest.TestCase):
    def test_composed_system_uses_ordered_modules_then_explicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            system_path = _write_test_system(Path(directory))
            system = load_system(system_path)

            self.assertEqual(
                tuple(module.module_id for module in system.modules),
                ("strong-openings", "medium-openings"),
            )
            legal_calls = frozenset(
                {
                    PassCall.PASS,
                    Bid(1, Denomination.CLUBS),
                    Bid(1, Denomination.DIAMONDS),
                }
            )
            self.assertEqual(
                system.choose_call(
                    context(
                        HandProfile(22, (4, 3, 3, 3)),
                        (),
                        legal_calls,
                        Seat.NORTH,
                    )
                ),
                Bid(1, Denomination.CLUBS),
            )
            self.assertEqual(
                system.choose_call(
                    context(
                        HandProfile(14, (4, 3, 3, 3)),
                        (),
                        legal_calls,
                        Seat.NORTH,
                    )
                ),
                Bid(1, Denomination.DIAMONDS),
            )
            self.assertEqual(
                system.choose_call(
                    context(
                        HandProfile(5, (4, 3, 3, 3)),
                        (),
                        legal_calls,
                        Seat.NORTH,
                    )
                ),
                PassCall.PASS,
            )

    def test_passout_module_fires_after_three_passes(self) -> None:
        system = load_system(PASSOUT_SYSTEM)
        auction = tuple(
            AuctionEntry(seat, PassCall.PASS)
            for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH)
        )

        call = system.choose_call(
            context(
                HandProfile(10, (4, 3, 3, 3)),
                auction,
                frozenset(
                    {
                        PassCall.PASS,
                        Bid(1, Denomination.CLUBS),
                        Bid(1, Denomination.DIAMONDS),
                    }
                ),
                Seat.WEST,
            )
        )

        self.assertEqual(call, Bid(1, Denomination.CLUBS))

    def test_try_calls_follow_the_module_file_order(self) -> None:
        module = """\
schema_version = 1
kind = "module"
id = "test-preference"
version = "1.0.0"
name = "Preference order test"

[[rules]]
id = "prefer-diamonds"
try_calls = ["1D", "1C", "PASS"]

[rules.when]
passout_seat = true
"""
        manifest = """\
schema_version = 1
kind = "system"
id = "test-preference-system"
version = "1.0.0"
name = "Preference order test"
on_no_match = "PASS"

[[modules]]
id = "test-preference"
version = "1.0.0"
path = "module.toml"
"""
        auction = tuple(
            AuctionEntry(seat, PassCall.PASS)
            for seat in (Seat.NORTH, Seat.EAST, Seat.SOUTH)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "module.toml").write_text(module, encoding="utf-8")
            system_path = path / "system.toml"
            system_path.write_text(manifest, encoding="utf-8")
            system = load_system(system_path)

            call = system.choose_call(
                context(
                    HandProfile(10, (4, 3, 3, 3)),
                    auction,
                    frozenset(
                        {
                            PassCall.PASS,
                            Bid(1, Denomination.CLUBS),
                            Bid(1, Denomination.DIAMONDS),
                        }
                    ),
                    Seat.WEST,
                )
            )

        self.assertEqual(call, Bid(1, Denomination.DIAMONDS))

    def test_system_rejects_a_pinned_module_version_mismatch(self) -> None:
        module_text = ONE_NT_MODULE.read_text(encoding="utf-8")
        manifest = """\
schema_version = 1
kind = "system"
id = "test-system"
version = "1.0.0"
name = "Version pin test"
on_no_match = "PASS"

[[modules]]
id = "opening-1nt-balanced-15-17"
version = "9.0.0"
path = "module.toml"
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "module.toml").write_text(module_text, encoding="utf-8")
            system_path = path / "system.toml"
            system_path.write_text(manifest, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "pins .* at 9.0.0"):
                load_system(system_path)

    def test_modular_manifest_can_explicitly_return_out_of_book(self) -> None:
        manifest = """\
schema_version = 1
kind = "system"
id = "out-of-book-test"
version = "1.0.0"
name = "Out-of-book test"
on_no_match = "OUT_OF_BOOK"
modules = []
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system.toml"
            path.write_text(manifest, encoding="utf-8")
            system = load_system(path)

        self.assertIs(
            system.choose_call(
                context(
                    HandProfile(10, (4, 3, 3, 3)),
                    (),
                    frozenset({PassCall.PASS}),
                    Seat.NORTH,
                )
            ),
            OutOfBookDecision.OUT_OF_BOOK,
        )

    def test_traditional_preempts_apply_hcp_suit_hcp_and_length_boundaries(self) -> None:
        system = load_system(PREEMPT_SYSTEM)
        legal_calls = _preempt_legal_calls()

        for points, suit_hcp, expected in (
            (4, (4, 0, 0, 0), Bid(2, Denomination.SPADES)),
            (10, (4, 3, 2, 1), Bid(2, Denomination.SPADES)),
            (3, (3, 0, 0, 0), PassCall.PASS),
            (11, (4, 4, 2, 1), PassCall.PASS),
            (4, (3, 1, 0, 0), PassCall.PASS),
        ):
            with self.subTest(points=points, suit_hcp=suit_hcp):
                shape = (6, 3, 2, 2)
                profile = HandProfile(points, shape, suit_hcp)
                self.assertEqual(
                    system.choose_call(context(profile, (), legal_calls, Seat.NORTH)),
                    expected,
                )

    def test_preempt_length_maps_to_supported_levels(self) -> None:
        system = load_system(PREEMPT_SYSTEM)
        legal_calls = _preempt_legal_calls()
        cases = (
            ((6, 3, 2, 2), (4, 0, 0, 0), Bid(2, Denomination.SPADES)),
            ((7, 3, 2, 1), (4, 0, 0, 0), Bid(3, Denomination.SPADES)),
            ((8, 3, 1, 1), (4, 0, 0, 0), Bid(4, Denomination.SPADES)),
            ((1, 1, 2, 9), (0, 0, 0, 4), Bid(5, Denomination.CLUBS)),
        )
        for shape, suit_hcp, expected in cases:
            with self.subTest(shape=shape):
                profile = HandProfile(sum(suit_hcp), shape, suit_hcp)
                self.assertEqual(
                    system.choose_call(context(profile, (), legal_calls, Seat.NORTH)),
                    expected,
                )

    def test_preempts_choose_longest_qualifying_suit_then_suit_tie_order(self) -> None:
        system = load_system(PREEMPT_SYSTEM)
        legal_calls = _preempt_legal_calls()
        cases = (
            ((0, 0, 6, 7), (0, 0, 4, 0), Bid(2, Denomination.DIAMONDS)),
            ((0, 0, 7, 6), (0, 0, 4, 4), Bid(3, Denomination.DIAMONDS)),
            ((0, 1, 6, 6), (0, 0, 4, 4), Bid(2, Denomination.DIAMONDS)),
        )
        for shape, suit_hcp, expected in cases:
            with self.subTest(shape=shape):
                profile = HandProfile(sum(suit_hcp), shape, suit_hcp)
                self.assertEqual(
                    system.choose_call(context(profile, (), legal_calls, Seat.NORTH)),
                    expected,
                )

    def test_suit_tie_order_is_selected_by_module_toml(self) -> None:
        suit_order = '["clubs", "diamonds", "hearts", "spades"]'
        conditions = (
            ("diamonds", "2D"),
            ("clubs", "2C"),
        )
        rules = "\n".join(
            f"""\
[[rules]]
id = "open-{suit}"
call = "{call}"

[rules.when]
min_hcp = 4
max_hcp = 10
suit = "{suit}"
min_suit_length = 6
min_suit_hcp = 4
max_outside_major_length = 3
longest_qualifying_suit_min_length = 6
suit_tie_order = {suit_order}
"""
            for suit, call in conditions
        )
        module = f"""\
schema_version = 1
kind = "module"
id = "tie-order"
version = "1.0.0"
name = "Configurable suit tie order"

{rules}"""
        manifest = """\
schema_version = 1
kind = "system"
id = "tie-order-system"
version = "1.0.0"
name = "Configurable suit tie order"
on_no_match = "PASS"

[[modules]]
id = "tie-order"
version = "1.0.0"
path = "module.toml"
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "module.toml").write_text(module, encoding="utf-8")
            system_path = path / "system.toml"
            system_path.write_text(manifest, encoding="utf-8")
            system = load_system(system_path)

        profile = HandProfile(8, (0, 1, 6, 6), (0, 0, 4, 4))
        self.assertEqual(
            system.choose_call(
                context(
                    profile,
                    (),
                    frozenset(
                        {
                            PassCall.PASS,
                            Bid(2, Denomination.CLUBS),
                            Bid(2, Denomination.DIAMONDS),
                        }
                    ),
                    Seat.NORTH,
                )
            ),
            Bid(2, Denomination.CLUBS),
        )

    def test_preempts_reject_outside_major_and_any_prior_contract_bid(self) -> None:
        system = load_system(PREEMPT_SYSTEM)
        legal_calls = _preempt_legal_calls()
        outside_major = HandProfile(4, (4, 0, 6, 3), (0, 0, 4, 0))
        self.assertIs(
            system.choose_call(context(outside_major, (), legal_calls, Seat.NORTH)),
            PassCall.PASS,
        )

        eligible = HandProfile(4, (6, 3, 2, 2), (4, 0, 0, 0))
        after_pass = (AuctionEntry(Seat.NORTH, PassCall.PASS),)
        self.assertEqual(
            system.choose_call(
                context(eligible, after_pass, legal_calls, Seat.EAST)
            ),
            Bid(2, Denomination.SPADES),
        )
        after_contract = (
            AuctionEntry(Seat.NORTH, Bid(1, Denomination.CLUBS)),
            AuctionEntry(Seat.EAST, PassCall.PASS),
        )
        self.assertIs(
            system.choose_call(
                context(eligible, after_contract, legal_calls, Seat.SOUTH)
            ),
            PassCall.PASS,
        )

    def test_preempt_rule_requires_per_suit_hcp_in_hand_profile(self) -> None:
        system = load_system(PREEMPT_SYSTEM)
        profile_without_suit_hcp = HandProfile(8, (6, 3, 2, 2))
        with self.assertRaisesRegex(ValueError, "per-suit HCP"):
            system.choose_call(
                context(
                    profile_without_suit_hcp,
                    (),
                    _preempt_legal_calls(),
                    Seat.NORTH,
                )
            )


if __name__ == "__main__":
    unittest.main()
