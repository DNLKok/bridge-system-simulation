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


if __name__ == "__main__":
    unittest.main()
