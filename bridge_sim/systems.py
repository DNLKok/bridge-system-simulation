from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence

from .deals import Vulnerability
from .model import (
    AuctionEntry,
    Bid,
    Call,
    DoubleCall,
    Denomination,
    HandProfile,
    PassCall,
    RedoubleCall,
    Seat,
)


class OutOfBookDecision(str, Enum):
    OUT_OF_BOOK = "OUT_OF_BOOK"


@dataclass(frozen=True)
class DecisionContext:
    hand: HandProfile
    auction: tuple[AuctionEntry, ...]
    legal_calls: frozenset[Call]
    seat: Seat
    dealer: Seat
    vulnerability: Vulnerability


@dataclass(frozen=True)
class ModuleDecision:
    call: Call
    module_id: str
    rule_id: str
    explanation: str | None = None


@dataclass(frozen=True)
class DealerCallCondition:
    expressions: tuple[str, ...] = ()
    selects_call: bool = False
    impossible: bool = False


@dataclass(frozen=True)
class RuleCondition:
    min_hcp: int | None = None
    max_hcp: int | None = None
    shapes: frozenset[tuple[int, int, int, int]] | None = None
    passout_seat: bool | None = None
    no_contract_bid: bool | None = None
    suit: Denomination | None = None
    min_suit_length: int | None = None
    max_suit_length: int | None = None
    min_suit_hcp: int | None = None
    max_outside_major_length: int | None = None
    longest_qualifying_suit_min_length: int | None = None
    suit_tie_order: tuple[Denomination, ...] | None = None

    def matches(self, context: DecisionContext) -> bool:
        if not self.matches_public_context(context):
            return False
        if self.min_hcp is not None and context.hand.hcp < self.min_hcp:
            return False
        if self.max_hcp is not None and context.hand.hcp > self.max_hcp:
            return False
        if self.shapes is not None and context.hand.canonical_shape not in self.shapes:
            return False
        if self.suit is not None:
            suit_index = _SUIT_ORDER.index(self.suit)
            suit_length = context.hand.shape[suit_index]
            suit_hcp = context.hand.suit_hcp
            if self.min_suit_length is not None and suit_length < self.min_suit_length:
                return False
            if self.max_suit_length is not None and suit_length > self.max_suit_length:
                return False
            if self.min_suit_hcp is not None:
                if suit_hcp is None:
                    raise ValueError(
                        "This bidding rule requires per-suit HCP, which is "
                        "unavailable in this hand profile."
                    )
                if suit_hcp[suit_index] < self.min_suit_hcp:
                    return False
            if self.max_outside_major_length is not None:
                if any(
                    context.hand.shape[index] > self.max_outside_major_length
                    for index, major in ((0, Denomination.SPADES), (1, Denomination.HEARTS))
                    if major is not self.suit
                ):
                    return False
            if self.longest_qualifying_suit_min_length is not None:
                if suit_hcp is None:
                    raise ValueError(
                        "This bidding rule requires per-suit HCP, which is "
                        "unavailable in this hand profile."
                    )
                if self.suit_tie_order is None:
                    raise RuntimeError(
                        "A longest-suit condition must define its tie order."
                    )
                candidate_length = suit_length
                candidate_priority = self.suit_tie_order.index(self.suit)
                for index, other_suit in enumerate(_SUIT_ORDER):
                    if other_suit is self.suit:
                        continue
                    other_length = context.hand.shape[index]
                    precedes_candidate = (
                        self.suit_tie_order.index(other_suit) < candidate_priority
                    )
                    if (
                        other_length >= self.longest_qualifying_suit_min_length
                        and suit_hcp[index] >= (self.min_suit_hcp or 0)
                        and other_length >= candidate_length
                        and (other_length > candidate_length or precedes_candidate)
                        and _outside_majors_qualify(
                            context.hand.shape,
                            other_suit,
                            self.max_outside_major_length,
                        )
                    ):
                        return False
        return True

    def matches_public_context(self, context: DecisionContext) -> bool:
        if self.passout_seat is not None and (
            _is_passout_seat(context.auction) is not self.passout_seat
        ):
            return False
        if self.no_contract_bid is not None:
            has_contract_bid = any(isinstance(entry.call, Bid) for entry in context.auction)
            if has_contract_bid is self.no_contract_bid:
                return False
        return True

    def dealer_expression(self, seat: Seat) -> str | None:
        seat_name = _SEAT_NAMES[seat]
        expressions: list[str] = []
        if self.min_hcp is not None:
            expressions.append(f"(hcp({seat_name}) >= {self.min_hcp})")
        if self.max_hcp is not None:
            expressions.append(f"(hcp({seat_name}) <= {self.max_hcp})")
        if self.shapes is not None:
            shape_options = " + ".join(
                f"any {''.join(str(length) for length in shape)}"
                for shape in sorted(self.shapes)
            )
            expressions.append(f"shape({seat_name}, {shape_options})")
        if self.suit is not None:
            suit_name = _SUIT_NAMES[self.suit]
            if self.min_suit_length is not None:
                expressions.append(
                    f"({suit_name}({seat_name}) >= {self.min_suit_length})"
                )
            if self.max_suit_length is not None:
                expressions.append(
                    f"({suit_name}({seat_name}) <= {self.max_suit_length})"
                )
            if self.min_suit_hcp is not None:
                expressions.append(
                    f"(hcp({seat_name}, {suit_name}) >= {self.min_suit_hcp})"
                )
            if self.max_outside_major_length is not None:
                expressions.extend(
                    f"({major_name}({seat_name}) <= "
                    f"{self.max_outside_major_length})"
                    for major, major_name in (
                        (Denomination.SPADES, "spades"),
                        (Denomination.HEARTS, "hearts"),
                    )
                    if major is not self.suit
                )
            if self.longest_qualifying_suit_min_length is not None:
                if self.suit_tie_order is None:
                    raise RuntimeError(
                        "A longest-suit condition must define its tie order."
                    )
                candidate_priority = self.suit_tie_order.index(self.suit)
                for other_suit in _SUIT_ORDER:
                    if other_suit is self.suit:
                        continue
                    other_suit_name = _SUIT_NAMES[other_suit]
                    other_qualifiers = [
                        f"({other_suit_name}({seat_name}) >= "
                        f"{self.longest_qualifying_suit_min_length})",
                    ]
                    if self.min_suit_hcp is not None:
                        other_qualifiers.append(
                            f"(hcp({seat_name}, {other_suit_name}) >= "
                            f"{self.min_suit_hcp})"
                        )
                    if self.max_outside_major_length is not None:
                        other_qualifiers.extend(
                            f"({major_name}({seat_name}) <= "
                            f"{self.max_outside_major_length})"
                            for major, major_name in (
                                (Denomination.SPADES, "spades"),
                                (Denomination.HEARTS, "hearts"),
                            )
                            if major is not other_suit
                        )
                    candidate_length = f"{_SUIT_NAMES[self.suit]}({seat_name})"
                    other_length = f"{other_suit_name}({seat_name})"
                    comparison = (
                        f"({other_length} >= {candidate_length})"
                        if self.suit_tie_order.index(other_suit) < candidate_priority
                        else f"({other_length} > {candidate_length})"
                    )
                    other_qualifiers.append(comparison)
                    expressions.append(
                        "!(" + " && ".join(other_qualifiers) + ")"
                    )
        if not expressions:
            return None
        return " && ".join(expressions)


@dataclass(frozen=True)
class BiddingRule:
    rule_id: str
    condition: RuleCondition
    preferred_calls: tuple[Call, ...]
    explanation: str | None = None

    def decide(self, context: DecisionContext, module_id: str) -> ModuleDecision | None:
        if not self.condition.matches(context):
            return None
        call = self._selected_call(context)
        if call is None:
            return None
        return ModuleDecision(call, module_id, self.rule_id, self.explanation)

    def dealer_call_condition(
        self,
        context: DecisionContext,
        observed_call: Call,
    ) -> DealerCallCondition:
        if not self.condition.matches_public_context(context):
            return DealerCallCondition()
        expected_call = self._selected_call(context)
        if expected_call is None:
            return DealerCallCondition()

        expression = self.condition.dealer_expression(context.seat)
        if expected_call == observed_call:
            expressions = (expression,) if expression is not None else ()
            return DealerCallCondition(expressions, selects_call=True)
        if expression is None:
            return DealerCallCondition(impossible=True)
        return DealerCallCondition((f"!({expression})",))

    def _selected_call(self, context: DecisionContext) -> Call | None:
        return next(
            (
                call
                for call in self.preferred_calls
                if call in context.legal_calls
            ),
            None,
        )


@dataclass(frozen=True)
class BiddingModule:
    module_id: str
    version: str
    name: str
    rules: tuple[BiddingRule, ...]

    def choose(self, context: DecisionContext) -> ModuleDecision | None:
        for rule in self.rules:
            decision = rule.decide(context, self.module_id)
            if decision is not None:
                return decision
        return None

    def dealer_call_condition(
        self,
        context: DecisionContext,
        observed_call: Call,
    ) -> DealerCallCondition:
        expressions: list[str] = []
        for rule in self.rules:
            condition = rule.dealer_call_condition(context, observed_call)
            expressions.extend(condition.expressions)
            if condition.impossible:
                return condition
            if condition.selects_call:
                return DealerCallCondition(tuple(expressions), selects_call=True)
        return DealerCallCondition(tuple(expressions))


class BiddingSystem(Protocol):
    def choose_call(
        self,
        context: DecisionContext,
    ) -> Call | OutOfBookDecision:
        """Choose a legal call using the current decision context."""
        ...


@dataclass(frozen=True)
class CompositeBiddingSystem:
    system_id: str
    version: str
    name: str
    modules: tuple[BiddingModule, ...]
    fallback: Call | OutOfBookDecision

    def __post_init__(self) -> None:
        module_ids = [module.module_id for module in self.modules]
        if len(set(module_ids)) != len(module_ids):
            raise ValueError("A system cannot include a module more than once.")

    def choose_call(
        self,
        context: DecisionContext,
    ) -> Call | OutOfBookDecision:
        for module in self.modules:
            decision = module.choose(context)
            if decision is None:
                continue
            if decision.call not in context.legal_calls:
                raise RuntimeError(
                    f"Module {module.module_id!r} selected illegal call "
                    f"{decision.call!s}."
                )
            return decision.call
        if self.fallback is OutOfBookDecision.OUT_OF_BOOK:
            return self.fallback
        if self.fallback not in context.legal_calls:
            raise RuntimeError(
                f"System fallback {self.fallback!s} is not legal in this auction."
            )
        return self.fallback

    def dealer_call_condition(
        self,
        context: DecisionContext,
        observed_call: Call,
    ) -> DealerCallCondition:
        return _dealer_call_condition(
            self.modules,
            self.fallback,
            context,
            observed_call,
            self.name,
        )


def load_system(path: str | Path) -> CompositeBiddingSystem:
    """Load a modular manifest or translate a legacy system into generic rules."""
    system_path = Path(path)
    with system_path.open("rb") as system_file:
        config = tomllib.load(system_file)

    kind = config.get("kind")
    if kind == "system":
        return _load_composite_system(system_path, config)
    if kind is not None:
        raise ValueError(f"Unsupported system kind: {kind!r}.")
    return _load_legacy_system(system_path, config)


def load_module(path: str | Path) -> BiddingModule:
    """Load and validate a data-only TOML module containing ordered rules."""
    module_path = Path(path)
    with module_path.open("rb") as module_file:
        config = tomllib.load(module_file)

    if config.get("schema_version") != 1 or config.get("kind") != "module":
        raise ValueError(f"Unsupported module schema in {module_path}.")
    _require_keys(
        config,
        {"schema_version", "kind", "id", "version", "name", "rules"},
        f"module file {module_path}",
    )
    module_id, version, name = _validate_metadata(config)
    raw_rules = config["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("A module must define at least one rule.")
    rules = tuple(_parse_rule(rule, index) for index, rule in enumerate(raw_rules))
    rule_ids = [rule.rule_id for rule in rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError(f"Module {module_id!r} defines duplicate rule IDs.")
    return BiddingModule(module_id, version, name, rules)


def _load_composite_system(
    system_path: Path,
    config: dict[str, object],
) -> CompositeBiddingSystem:
    expected = {
        "schema_version",
        "kind",
        "id",
        "version",
        "name",
        "on_no_match",
        "modules",
    }
    _require_keys(config, expected, f"system file {system_path}")
    if config["schema_version"] != 1:
        raise ValueError("Modular systems must use schema_version 1.")
    system_id, version, name = _validate_metadata(config)
    raw_modules = config["modules"]
    if not isinstance(raw_modules, list):
        raise ValueError("System modules must be an array of tables.")

    modules: list[BiddingModule] = []
    seen_ids: set[str] = set()
    for index, reference in enumerate(raw_modules):
        if not isinstance(reference, dict):
            raise ValueError(f"System module reference {index} must be a table.")
        _require_keys(
            reference,
            {"id", "version", "path"},
            f"system module reference {index}",
        )
        module_id = reference["id"]
        module_version = reference["version"]
        path_text = reference["path"]
        if (
            not isinstance(module_id, str)
            or not module_id.strip()
            or not isinstance(module_version, str)
            or not module_version.strip()
            or not isinstance(path_text, str)
            or not path_text.strip()
        ):
            raise ValueError(f"System module reference {index} is invalid.")
        if module_id in seen_ids:
            raise ValueError(f"System includes module {module_id!r} more than once.")
        module_path = Path(path_text)
        if not module_path.is_absolute():
            module_path = system_path.parent / module_path
        module = load_module(module_path)
        if module.module_id != module_id:
            raise ValueError(
                f"System expected module {module_id!r}, but {module_path} "
                f"declares {module.module_id!r}."
            )
        if module.version != module_version:
            raise ValueError(
                f"System pins {module_id!r} at {module_version}, but {module_path} "
                f"contains version {module.version}."
            )
        modules.append(module)
        seen_ids.add(module_id)

    return CompositeBiddingSystem(
        system_id=system_id,
        version=version,
        name=name,
        modules=tuple(modules),
        fallback=_parse_fallback(config["on_no_match"]),
    )


def _load_legacy_system(
    system_path: Path,
    config: dict[str, object],
) -> CompositeBiddingSystem:
    schema_version = config.get("schema_version")
    out_of_book_mode = _out_of_book_mode(config)
    common_keys = {"schema_version", "id", "version", "name"}
    if schema_version == 2:
        common_keys.add("out_of_book")

    strategy = config.get("strategy")
    if strategy is not None:
        raise ValueError(f"Unsupported legacy system strategy: {strategy!r}.")

    expected = common_keys | {"rule"}
    if out_of_book_mode == "never":
        expected.add("default_call")
    _require_keys(config, expected, f"system file {system_path}")
    system_id, version, name = _validate_metadata(config)
    default_call = (
        config.get("default_call")
        if out_of_book_mode == "never"
        else None
    )
    if out_of_book_mode == "never" and default_call != PassCall.PASS.value:
        raise ValueError("The supported system's default_call must be PASS.")

    raw_rule = config["rule"]
    if not isinstance(raw_rule, dict):
        raise ValueError("System rule must be a TOML table.")
    _require_keys(
        raw_rule,
        {"id", "call", "min_hcp", "max_hcp", "balanced_shapes"},
        "system rule",
    )
    rule_id = raw_rule["id"]
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise ValueError("System rule id must be a non-empty string.")
    min_hcp, max_hcp = _parse_hcp_range(raw_rule["min_hcp"], raw_rule["max_hcp"])
    raw_shapes = raw_rule["balanced_shapes"]
    if not isinstance(raw_shapes, list) or not raw_shapes:
        raise ValueError("balanced_shapes must be a non-empty TOML array.")
    rule = BiddingRule(
        rule_id=rule_id,
        condition=RuleCondition(
            min_hcp=min_hcp,
            max_hcp=max_hcp,
            shapes=frozenset(_parse_shape(shape) for shape in raw_shapes),
        ),
        preferred_calls=(_parse_call(raw_rule["call"]),),
    )
    fallback = (
        OutOfBookDecision.OUT_OF_BOOK
        if out_of_book_mode == "when_no_rule_matches"
        else PassCall.PASS
    )
    return _legacy_system(system_id, version, name, (rule,), fallback)


def _legacy_system(
    system_id: str,
    version: str,
    name: str,
    rules: tuple[BiddingRule, ...],
    fallback: Call | OutOfBookDecision,
) -> CompositeBiddingSystem:
    module = BiddingModule(
        module_id=system_id,
        version=version,
        name=name,
        rules=rules,
    )
    return CompositeBiddingSystem(
        system_id=system_id,
        version=version,
        name=name,
        modules=(module,),
        fallback=fallback,
    )


def _parse_rule(value: object, index: int) -> BiddingRule:
    if not isinstance(value, dict):
        raise ValueError(f"Module rule {index} must be a TOML table.")
    required = {"id", "when"}
    allowed = required | {"call", "try_calls", "explanation"}
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown keys: {', '.join(sorted(unknown))}")
        raise ValueError(f"Invalid module rule {index} ({'; '.join(details)}).")
    if ("call" in value) == ("try_calls" in value):
        raise ValueError(
            f"Module rule {index} must define exactly one of call or try_calls."
        )

    rule_id = value["id"]
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise ValueError(f"Module rule {index} id must be a non-empty string.")
    raw_condition = value["when"]
    condition = _parse_condition(raw_condition, index)
    if "call" in value:
        preferred_calls = (_parse_call(value["call"]),)
    else:
        raw_calls = value["try_calls"]
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError(f"Module rule {index} try_calls must be a non-empty array.")
        preferred_calls = tuple(_parse_call(call) for call in raw_calls)
        if len(set(preferred_calls)) != len(preferred_calls):
            raise ValueError(f"Module rule {index} has duplicate try_calls.")
    explanation = value.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise ValueError(f"Module rule {index} explanation must be a string.")
    return BiddingRule(rule_id, condition, preferred_calls, explanation)


def _parse_condition(value: object, index: int) -> RuleCondition:
    if not isinstance(value, dict):
        raise ValueError(f"Module rule {index} when must be a TOML table.")
    allowed = {
        "min_hcp",
        "max_hcp",
        "shapes",
        "passout_seat",
        "no_contract_bid",
        "suit",
        "min_suit_length",
        "max_suit_length",
        "min_suit_hcp",
        "max_outside_major_length",
        "longest_qualifying_suit_min_length",
        "suit_tie_order",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"Unknown conditions in module rule {index}: {', '.join(sorted(unknown))}."
        )
    if not value:
        raise ValueError(f"Module rule {index} must define at least one condition.")

    min_hcp: int | None = None
    max_hcp: int | None = None
    if "min_hcp" in value or "max_hcp" in value:
        min_hcp, max_hcp = _parse_hcp_range(
            value.get("min_hcp"),
            value.get("max_hcp"),
        )

    shapes: frozenset[tuple[int, int, int, int]] | None = None
    if "shapes" in value:
        raw_shapes = value["shapes"]
        if not isinstance(raw_shapes, list) or not raw_shapes:
            raise ValueError(f"Module rule {index} shapes must be a non-empty array.")
        shapes = frozenset(_parse_shape(shape) for shape in raw_shapes)

    passout_seat = value.get("passout_seat")
    if passout_seat is not None and not isinstance(passout_seat, bool):
        raise ValueError(f"Module rule {index} passout_seat must be a boolean.")
    no_contract_bid = value.get("no_contract_bid")
    if no_contract_bid is not None and not isinstance(no_contract_bid, bool):
        raise ValueError(f"Module rule {index} no_contract_bid must be a boolean.")
    raw_suit = value.get("suit")
    suit: Denomination | None = None
    if raw_suit is not None:
        if not isinstance(raw_suit, str):
            raise ValueError(f"Module rule {index} suit must be a suit name.")
        suit = _parse_suit(raw_suit, index)
    suit_length_values = {
        "min_suit_length": _parse_integer_condition(value, "min_suit_length", 0, 13),
        "max_suit_length": _parse_integer_condition(value, "max_suit_length", 0, 13),
        "min_suit_hcp": _parse_integer_condition(value, "min_suit_hcp", 0, 10),
        "max_outside_major_length": _parse_integer_condition(
            value, "max_outside_major_length", 0, 13
        ),
        "longest_qualifying_suit_min_length": _parse_integer_condition(
            value, "longest_qualifying_suit_min_length", 0, 13
        ),
    }
    suit_tie_order: tuple[Denomination, ...] | None = None
    if "suit_tie_order" in value:
        raw_tie_order = value["suit_tie_order"]
        if not isinstance(raw_tie_order, list) or len(raw_tie_order) != 4:
            raise ValueError(
                f"Module rule {index} suit_tie_order must list all four suits."
            )
        suit_tie_order = tuple(
            _parse_suit(suit_name, index) for suit_name in raw_tie_order
        )
        if len(set(suit_tie_order)) != 4:
            raise ValueError(
                f"Module rule {index} suit_tie_order must not repeat a suit."
            )
    if any(item is not None for item in suit_length_values.values()) and suit is None:
        raise ValueError(f"Module rule {index} suit conditions require a suit.")
    if (
        suit_length_values["min_suit_length"] is not None
        and suit_length_values["max_suit_length"] is not None
        and suit_length_values["min_suit_length"]
        > suit_length_values["max_suit_length"]
    ):
        raise ValueError("min_suit_length cannot be greater than max_suit_length.")
    if (
        suit_length_values["longest_qualifying_suit_min_length"] is not None
        and (
            suit_length_values["min_suit_hcp"] is None
            or suit_length_values["max_outside_major_length"] is None
            or suit_tie_order is None
        )
    ):
        raise ValueError(
            "longest_qualifying_suit_min_length requires min_suit_hcp and "
            "max_outside_major_length and suit_tie_order."
        )
    if (
        suit_tie_order is not None
        and suit_length_values["longest_qualifying_suit_min_length"] is None
    ):
        raise ValueError("suit_tie_order requires longest_qualifying_suit_min_length.")
    return RuleCondition(
        min_hcp=min_hcp,
        max_hcp=max_hcp,
        shapes=shapes,
        passout_seat=passout_seat,
        no_contract_bid=no_contract_bid,
        suit=suit,
        suit_tie_order=suit_tie_order,
        **suit_length_values,
    )


def _parse_integer_condition(
    conditions: dict[str, object],
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    value = conditions.get(name)
    if value is None:
        return None
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def _parse_suit(value: str, index: int) -> Denomination:
    suits = {
        "clubs": Denomination.CLUBS,
        "diamonds": Denomination.DIAMONDS,
        "hearts": Denomination.HEARTS,
        "spades": Denomination.SPADES,
    }
    try:
        return suits[value.lower()]
    except KeyError as error:
        raise ValueError(
            f"Module rule {index} suit must be clubs, diamonds, hearts, or spades."
        ) from error


def _outside_majors_qualify(
    shape: tuple[int, int, int, int],
    suit: Denomination,
    maximum_length: int | None,
) -> bool:
    if maximum_length is None:
        return True
    return all(
        shape[index] <= maximum_length
        for index, major in ((0, Denomination.SPADES), (1, Denomination.HEARTS))
        if major is not suit
    )


def _parse_hcp_range(
    raw_min: object,
    raw_max: object,
) -> tuple[int | None, int | None]:
    if raw_min is None:
        min_hcp = None
    elif (
        isinstance(raw_min, int)
        and not isinstance(raw_min, bool)
        and 0 <= raw_min <= 37
    ):
        min_hcp = raw_min
    else:
        raise ValueError("min_hcp must be an integer from 0 to 37.")

    if raw_max is None:
        max_hcp = None
    elif (
        isinstance(raw_max, int)
        and not isinstance(raw_max, bool)
        and 0 <= raw_max <= 37
    ):
        max_hcp = raw_max
    else:
        raise ValueError("max_hcp must be an integer from 0 to 37.")

    if min_hcp is not None and max_hcp is not None and min_hcp > max_hcp:
        raise ValueError("min_hcp cannot be greater than max_hcp.")
    return min_hcp, max_hcp


def _dealer_call_condition(
    modules: Sequence[BiddingModule],
    fallback: Call | OutOfBookDecision,
    context: DecisionContext,
    observed_call: Call,
    system_name: str,
) -> DealerCallCondition:
    expressions: list[str] = []
    for module in modules:
        condition = module.dealer_call_condition(context, observed_call)
        expressions.extend(condition.expressions)
        if condition.impossible:
            raise ValueError(
                f"Call {observed_call!s} is inconsistent with the "
                f"{system_name!r} system."
            )
        if condition.selects_call:
            return DealerCallCondition(tuple(expressions), selects_call=True)

    if fallback is OutOfBookDecision.OUT_OF_BOOK or fallback == observed_call:
        return DealerCallCondition(tuple(expressions))
    raise ValueError(
        f"Call {observed_call!s} is inconsistent with the {system_name!r} system."
    )


def _parse_fallback(value: object) -> Call | OutOfBookDecision:
    if isinstance(value, str) and value.upper() == OutOfBookDecision.OUT_OF_BOOK.value:
        return OutOfBookDecision.OUT_OF_BOOK
    return _parse_call(value)


def _parse_call(value: object) -> Call:
    if not isinstance(value, str):
        raise ValueError(f"Invalid auction call: {value!r}.")
    normalized = value.strip().upper()
    if normalized in ("P", "PASS"):
        return PassCall.PASS
    if normalized in ("X", "DOUBLE"):
        return DoubleCall.DOUBLE
    if normalized in ("XX", "REDOUBLE"):
        return RedoubleCall.REDOUBLE
    if len(normalized) < 2 or normalized[0] not in "1234567":
        raise ValueError(f"Invalid auction call: {value!r}.")
    try:
        return Bid(int(normalized[0]), Denomination(normalized[1:]))
    except ValueError as error:
        raise ValueError(f"Invalid auction call: {value!r}.") from error


def _out_of_book_mode(config: dict[str, object]) -> str:
    if config.get("schema_version") == 1:
        return "never"
    mode = config.get("out_of_book")
    if mode not in ("never", "when_no_rule_matches"):
        raise ValueError(
            "System out_of_book must be 'never' or 'when_no_rule_matches'."
        )
    return mode


def _validate_metadata(config: dict[str, object]) -> tuple[str, str, str]:
    if (
        not isinstance(config["schema_version"], int)
        or isinstance(config["schema_version"], bool)
        or config["schema_version"] not in (1, 2)
    ):
        raise ValueError("Only schema_version 1 or 2 is supported.")
    system_id = config.get("id")
    version = config.get("version")
    name = config.get("name")
    if (
        not isinstance(system_id, str)
        or not system_id.strip()
        or not isinstance(version, str)
        or not version.strip()
        or not isinstance(name, str)
        or not name.strip()
    ):
        raise ValueError("System/module id, version, and name must be non-empty strings.")
    return system_id, version, name


def _is_passout_seat(auction: Sequence[AuctionEntry]) -> bool:
    trailing_passes = 0
    for entry in reversed(auction):
        if entry.call is not PassCall.PASS:
            break
        trailing_passes += 1
    has_contract_bid = any(isinstance(entry.call, Bid) for entry in auction)
    return trailing_passes == (2 if has_contract_bid else 3)


_SEAT_NAMES = {
    Seat.NORTH: "north",
    Seat.EAST: "east",
    Seat.SOUTH: "south",
    Seat.WEST: "west",
}
_SUIT_ORDER = (
    Denomination.SPADES,
    Denomination.HEARTS,
    Denomination.DIAMONDS,
    Denomination.CLUBS,
)
_SUIT_NAMES = {
    Denomination.SPADES: "spades",
    Denomination.HEARTS: "hearts",
    Denomination.DIAMONDS: "diamonds",
    Denomination.CLUBS: "clubs",
}


def _parse_shape(value: object) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, str)
        or len(value) != 4
        or not value.isascii()
        or not value.isdigit()
    ):
        raise ValueError("Each shape must be a four-digit string such as '4333'.")
    shape = tuple(int(length) for length in value)
    if tuple(sorted(shape, reverse=True)) != shape or sum(shape) != 13:
        raise ValueError(f"Shape {value!r} must be descending and total 13.")
    return shape[0], shape[1], shape[2], shape[3]


def _require_keys(config: dict[str, object], expected: set[str], context: str) -> None:
    actual = set(config)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown keys: {', '.join(sorted(unknown))}")
        raise ValueError(f"Invalid {context} ({'; '.join(details)}).")
