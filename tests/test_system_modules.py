import tempfile
import unittest
from pathlib import Path

from bridge_sim import (
    AuctionEntry,
    Bid,
    Call,
    DecisionContext,
    Denomination,
    DoubleCall,
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
SIMPLE_OVERCALL_SYSTEM = ROOT / "systems" / "simple-overcalls.toml"
STANDARD_OPENINGS_SYSTEM = ROOT / "systems" / "standard-openings.toml"


def _standard_opening_legal_calls() -> frozenset[Call]:
    return frozenset(
        {
            PassCall.PASS,
            *(Bid(1, suit) for suit in Denomination),
            Bid(2, Denomination.CLUBS),
            Bid(2, Denomination.NOTRUMP),
        }
    )


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


def _simple_overcall_context(
    hand: HandProfile,
    opening: Bid,
) -> DecisionContext:
    auction = (AuctionEntry(Seat.NORTH, opening),)
    legal_calls = {
        PassCall.PASS,
        DoubleCall.DOUBLE,
        *(
            Bid(level, denomination)
            for level in range(1, 8)
            for denomination in Denomination
            if Bid(level, denomination).order > opening.order
        ),
    }
    return context(hand, auction, frozenset(legal_calls), Seat.EAST)


class BiddingModuleTests(unittest.TestCase):
    def test_standard_openings_apply_rule_of_20_and_call_priorities(self) -> None:
        system = load_system(STANDARD_OPENINGS_SYSTEM)
        legal_calls = _standard_opening_legal_calls()
        cases = (
            (HandProfile(12, (4, 3, 3, 3)), PassCall.PASS),
            (HandProfile(12, (4, 4, 3, 2)), Bid(1, Denomination.CLUBS)),
            (HandProfile(12, (5, 3, 3, 2)), Bid(1, Denomination.CLUBS)),
            (HandProfile(15, (4, 3, 3, 3)), Bid(1, Denomination.NOTRUMP)),
            (HandProfile(17, (5, 3, 3, 2)), Bid(1, Denomination.NOTRUMP)),
            (HandProfile(18, (4, 4, 3, 2)), Bid(1, Denomination.CLUBS)),
            (HandProfile(18, (5, 3, 3, 2)), Bid(1, Denomination.SPADES)),
            (HandProfile(20, (4, 4, 3, 2)), Bid(2, Denomination.NOTRUMP)),
            (HandProfile(22, (4, 3, 3, 3)), Bid(2, Denomination.CLUBS)),
            (HandProfile(9, (5, 6, 1, 1)), Bid(1, Denomination.HEARTS)),
            (HandProfile(10, (5, 5, 2, 1)), Bid(1, Denomination.SPADES)),
            (HandProfile(12, (1, 4, 4, 4)), Bid(1, Denomination.DIAMONDS)),
            (HandProfile(12, (4, 4, 4, 1)), Bid(1, Denomination.DIAMONDS)),
            (HandProfile(12, (4, 1, 4, 4)), Bid(1, Denomination.CLUBS)),
            (HandProfile(12, (4, 4, 1, 4)), Bid(1, Denomination.CLUBS)),
            (HandProfile(11, (1, 4, 4, 4)), PassCall.PASS),
            (HandProfile(22, (1, 4, 4, 4)), Bid(2, Denomination.CLUBS)),
            (HandProfile(18, (2, 1, 3, 7)), Bid(1, Denomination.CLUBS)),
        )
        for hand, expected in cases:
            with self.subTest(hand=hand):
                self.assertEqual(
                    system.choose_call(
                        context(hand, (), legal_calls, Seat.NORTH)
                    ),
                    expected,
                )

    def test_standard_openings_do_not_reopen_after_a_contract_bid(self) -> None:
        system = load_system(STANDARD_OPENINGS_SYSTEM)
        auction = (AuctionEntry(Seat.NORTH, Bid(1, Denomination.DIAMONDS)),)
        self.assertIs(
            system.choose_call(
                context(
                    HandProfile(22, (4, 3, 3, 3)),
                    auction,
                    _standard_opening_legal_calls(),
                    Seat.EAST,
                )
            ),
            PassCall.PASS,
        )

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

    def test_simple_overcalls_use_each_level_and_longest_qualifying_suit(self) -> None:
        system = load_system(SIMPLE_OVERCALL_SYSTEM)
        cases = (
            (
                HandProfile(9, (5, 3, 3, 2)),
                Bid(1, Denomination.CLUBS),
                Bid(1, Denomination.SPADES),
            ),
            (
                HandProfile(12, (5, 3, 3, 2)),
                Bid(2, Denomination.CLUBS),
                Bid(2, Denomination.SPADES),
            ),
            (
                HandProfile(10, (6, 3, 2, 2)),
                Bid(2, Denomination.CLUBS),
                Bid(2, Denomination.SPADES),
            ),
            (
                HandProfile(13, (2, 4, 2, 5)),
                Bid(2, Denomination.SPADES),
                Bid(3, Denomination.CLUBS),
            ),
            (
                HandProfile(11, (3, 2, 2, 6)),
                Bid(2, Denomination.SPADES),
                Bid(3, Denomination.CLUBS),
            ),
            (
                HandProfile(11, (5, 1, 1, 6)),
                Bid(1, Denomination.HEARTS),
                Bid(2, Denomination.CLUBS),
            ),
            (
                HandProfile(9, (5, 5, 2, 1)),
                Bid(1, Denomination.CLUBS),
                Bid(1, Denomination.SPADES),
            ),
        )
        for hand, opening, expected in cases:
            with self.subTest(opening=opening, shape=hand.shape, hcp=hand.hcp):
                self.assertEqual(
                    system.choose_call(_simple_overcall_context(hand, opening)),
                    expected,
                )

    def test_simple_overcalls_enforce_strength_caps_and_tier_boundaries(self) -> None:
        system = load_system(SIMPLE_OVERCALL_SYSTEM)
        cases = (
            (HandProfile(11, (5, 3, 3, 2)), Bid(2, Denomination.CLUBS)),
            (HandProfile(9, (6, 3, 2, 2)), Bid(2, Denomination.CLUBS)),
            (HandProfile(10, (5, 4, 2, 2)), Bid(2, Denomination.SPADES)),
            (HandProfile(18, (6, 3, 2, 2)), Bid(1, Denomination.HEARTS)),
        )
        for hand, opening in cases:
            with self.subTest(opening=opening, shape=hand.shape, hcp=hand.hcp):
                self.assertIs(
                    system.choose_call(_simple_overcall_context(hand, opening)),
                    PassCall.PASS,
                )

    def test_balanced_overcalls_use_opening_level_stoppers_and_strength(self) -> None:
        system = load_system(SIMPLE_OVERCALL_SYSTEM)
        balanced_minor = HandProfile(15, (5, 3, 3, 2))
        balanced_major_no_stop_with_long_suit = HandProfile(
            15,
            (5, 3, 3, 2),
            (7, 1, 7, 0),
            ("AK", "J", "AK", ""),
        )
        heart_stopper = HandProfile(
            15,
            (4, 3, 3, 3),
            (7, 3, 4, 1),
            ("AK", "QJ", "A", "J"),
        )
        no_heart_stopper = HandProfile(
            15,
            (4, 3, 3, 3),
            (7, 1, 7, 0),
            ("AK", "J", "AK", ""),
        )
        two_level_jack_stopper = HandProfile(
            16,
            (4, 4, 3, 2),
            (7, 1, 7, 1),
            ("AK", "J", "AK", "J"),
        )
        no_two_level_stopper = HandProfile(
            16,
            (4, 4, 3, 2),
            (7, 0, 7, 2),
            ("AK", "", "AK", "Q"),
        )
        strong_balanced = HandProfile(19, (4, 3, 3, 3))

        expectations = (
            (balanced_minor, Bid(1, Denomination.CLUBS), Bid(1, Denomination.NOTRUMP)),
            (heart_stopper, Bid(1, Denomination.HEARTS), Bid(1, Denomination.NOTRUMP)),
            (no_heart_stopper, Bid(1, Denomination.HEARTS), DoubleCall.DOUBLE),
            (
                balanced_major_no_stop_with_long_suit,
                Bid(1, Denomination.HEARTS),
                DoubleCall.DOUBLE,
            ),
            (
                two_level_jack_stopper,
                Bid(2, Denomination.HEARTS),
                Bid(2, Denomination.NOTRUMP),
            ),
            (no_two_level_stopper, Bid(2, Denomination.HEARTS), DoubleCall.DOUBLE),
            (strong_balanced, Bid(1, Denomination.SPADES), DoubleCall.DOUBLE),
        )
        for hand, opening, expected in expectations:
            with self.subTest(opening=opening, hcp=hand.hcp):
                self.assertEqual(
                    system.choose_call(_simple_overcall_context(hand, opening)),
                    expected,
                )

    def test_major_stopper_rules_require_honor_profile(self) -> None:
        system = load_system(SIMPLE_OVERCALL_SYSTEM)
        with self.assertRaisesRegex(ValueError, "suit honors"):
            system.choose_call(
                _simple_overcall_context(
                    HandProfile(15, (4, 3, 3, 3)),
                    Bid(1, Denomination.HEARTS),
                )
            )

    def test_simple_takeout_doubles_apply_the_minor_and_major_shapes(self) -> None:
        system = load_system(SIMPLE_OVERCALL_SYSTEM)
        cases = (
            (
                HandProfile(12, (4, 3, 3, 3)),
                Bid(1, Denomination.CLUBS),
                DoubleCall.DOUBLE,
            ),
            (
                HandProfile(12, (5, 3, 3, 2)),
                Bid(1, Denomination.CLUBS),
                Bid(1, Denomination.SPADES),
            ),
            (
                HandProfile(12, (4, 3, 3, 3)),
                Bid(1, Denomination.DIAMONDS),
                DoubleCall.DOUBLE,
            ),
            (
                HandProfile(10, (4, 3, 1, 5)),
                Bid(2, Denomination.DIAMONDS),
                DoubleCall.DOUBLE,
            ),
            (
                HandProfile(12, (4, 2, 4, 3)),
                Bid(1, Denomination.HEARTS),
                DoubleCall.DOUBLE,
            ),
            (
                HandProfile(10, (4, 1, 4, 4)),
                Bid(2, Denomination.HEARTS),
                DoubleCall.DOUBLE,
            ),
        )
        for hand, opening, expected in cases:
            with self.subTest(opening=opening, shape=hand.shape, hcp=hand.hcp):
                self.assertEqual(
                    system.choose_call(_simple_overcall_context(hand, opening)),
                    expected,
                )

    def test_major_takeout_double_requires_exactly_four_in_other_major(self) -> None:
        system = load_system(SIMPLE_OVERCALL_SYSTEM)
        hand = HandProfile(10, (5, 2, 3, 3))
        self.assertIs(
            system.choose_call(
                _simple_overcall_context(hand, Bid(2, Denomination.HEARTS))
            ),
            PassCall.PASS,
        )

    def test_simple_overcalls_only_apply_to_the_immediate_opponent_opening(self) -> None:
        system = load_system(SIMPLE_OVERCALL_SYSTEM)
        hand = HandProfile(12, (4, 3, 3, 3))
        wrong_auction = (
            AuctionEntry(Seat.NORTH, Bid(1, Denomination.CLUBS)),
            AuctionEntry(Seat.EAST, PassCall.PASS),
            AuctionEntry(Seat.SOUTH, Bid(1, Denomination.HEARTS)),
        )
        decision = system.choose_call(
            context(
                hand,
                wrong_auction,
                frozenset({PassCall.PASS, DoubleCall.DOUBLE}),
                Seat.WEST,
            )
        )
        self.assertIs(decision, PassCall.PASS)

    def test_higher_priority_module_can_handle_artificial_eligible_opening(self) -> None:
        special_module = """\
schema_version = 1
kind = "module"
id = "special-2c-response"
version = "1.0.0"
name = "Special 2C response"

[[rules]]
id = "special-response"
call = "2D"

[rules.when]
min_hcp = 0
"""
        manifest = f"""\
schema_version = 1
kind = "system"
id = "combined-overcalls"
version = "1.0.0"
name = "Special response before simple overcalls"
on_no_match = "PASS"

[[modules]]
id = "special-2c-response"
version = "1.0.0"
path = "special.toml"

[[modules]]
id = "simple-overcalls"
version = "1.0.0"
path = "{ROOT / 'systems/modules/simple-overcalls.toml'}"
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "special.toml").write_text(special_module, encoding="utf-8")
            system_path = path / "system.toml"
            system_path.write_text(manifest, encoding="utf-8")
            system = load_system(system_path)

        self.assertEqual(
            system.choose_call(
                _simple_overcall_context(
                    HandProfile(10, (6, 3, 2, 2)),
                    Bid(2, Denomination.CLUBS),
                )
            ),
            Bid(2, Denomination.DIAMONDS),
        )


if __name__ == "__main__":
    unittest.main()
