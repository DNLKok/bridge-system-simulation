from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
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
    hcp_ranges: tuple[tuple[int | None, int | None], ...] = ()
    min_rule_of_20: int | None = None
    shapes: frozenset[tuple[int, int, int, int]] | None = None
    passout_seat: bool | None = None
    no_contract_bid: bool | None = None
    opponent_opening_calls: frozenset[Bid] | None = None
    suit: Denomination | None = None
    min_suit_length: int | None = None
    max_suit_length: int | None = None
    min_suit_hcp: int | None = None
    max_outside_major_length: int | None = None
    longest_qualifying_suit_min_length: int | None = None
    suit_tie_order: tuple[Denomination, ...] | None = None
    suit_lengths: tuple[
        tuple[Denomination, int | None, int | None], ...
    ] = ()
    suit_opening_min_lengths: tuple[tuple[Denomination, int], ...] = ()
    stopper_suit: Denomination | None = None
    has_stopper: bool | None = None
    nonjump_new_suit_overcall: bool = False
    suit_overcall_tiers: tuple[tuple[int, int, int], ...] = ()

    def matches(self, context: DecisionContext) -> bool:
        if not self.matches_public_context(context):
            return False
        if self.min_hcp is not None and context.hand.hcp < self.min_hcp:
            return False
        if self.max_hcp is not None and context.hand.hcp > self.max_hcp:
            return False
        if self.hcp_ranges and not any(
            (minimum is None or context.hand.hcp >= minimum)
            and (maximum is None or context.hand.hcp <= maximum)
            for minimum, maximum in self.hcp_ranges
        ):
            return False
        if self.min_rule_of_20 is not None:
            top_two_suits = sorted(context.hand.shape, reverse=True)[:2]
            if context.hand.hcp + sum(top_two_suits) < self.min_rule_of_20:
                return False
        if self.shapes is not None and context.hand.canonical_shape not in self.shapes:
            return False
        for suit, minimum, maximum in self.suit_lengths:
            length = context.hand.shape[_SUIT_ORDER.index(suit)]
            if minimum is not None and length < minimum:
                return False
            if maximum is not None and length > maximum:
                return False
        if self.stopper_suit is not None and self.has_stopper is not None:
            if context.hand.suit_honors is None:
                raise ValueError(
                    "This bidding rule requires suit honors, which are unavailable "
                    "in this hand profile."
                )
            suit_index = _SUIT_ORDER.index(self.stopper_suit)
            if (
                _has_stopper(
                    context.hand.suit_honors[suit_index],
                    context.hand.shape[suit_index],
                )
                is not self.has_stopper
            ):
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
        if not _matches_opponent_opening(context, self.opponent_opening_calls):
            return False
        return True

    def dealer_expression(self, seat: Seat) -> str | None:
        seat_name = _SEAT_NAMES[seat]
        expressions: list[str] = []
        if self.min_hcp is not None:
            expressions.append(f"(hcp({seat_name}) >= {self.min_hcp})")
        if self.max_hcp is not None:
            expressions.append(f"(hcp({seat_name}) <= {self.max_hcp})")
        if self.hcp_ranges:
            range_expressions: list[str] = []
            for minimum, maximum in self.hcp_ranges:
                bounds = []
                if minimum is not None:
                    bounds.append(f"(hcp({seat_name}) >= {minimum})")
                if maximum is not None:
                    bounds.append(f"(hcp({seat_name}) <= {maximum})")
                range_expressions.append("(" + " && ".join(bounds) + ")")
            expressions.append("(" + " || ".join(range_expressions) + ")")
        if self.min_rule_of_20 is not None:
            expressions.append(
                "("
                + " || ".join(
                    f"((hcp({seat_name}) + {_SUIT_NAMES[first]}({seat_name}) + "
                    f"{_SUIT_NAMES[second]}({seat_name})) >= "
                    f"{self.min_rule_of_20})"
                    for first, second in combinations(_SUIT_ORDER, 2)
                )
                + ")"
            )
        if self.shapes is not None:
            shape_options = " + ".join(
                f"any {''.join(str(length) for length in shape)}"
                for shape in sorted(self.shapes)
            )
            expressions.append(f"shape({seat_name}, {shape_options})")
        for suit, minimum, maximum in self.suit_lengths:
            suit_name = _SUIT_NAMES[suit]
            if minimum is not None:
                expressions.append(f"({suit_name}({seat_name}) >= {minimum})")
            if maximum is not None:
                expressions.append(f"({suit_name}({seat_name}) <= {maximum})")
        if self.stopper_suit is not None and self.has_stopper is not None:
            stopper = _dealer_stopper_expression(seat_name, self.stopper_suit)
            expressions.append(
                stopper if self.has_stopper else f"!({stopper})"
            )
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
        if self.condition.suit_overcall_tiers:
            return self._dealer_suit_overcall_condition(context, observed_call)
        if self.condition.suit_opening_min_lengths:
            return self._dealer_suit_opening_condition(context, observed_call)
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
        if self.condition.suit_opening_min_lengths:
            tie_order = self.condition.suit_tie_order
            if tie_order is None:
                raise RuntimeError("Suit opening choices must define a tie order.")
            minimum_lengths = dict(self.condition.suit_opening_min_lengths)
            candidates = [
                call
                for call in self.preferred_calls
                if isinstance(call, Bid)
                and call in context.legal_calls
                and call.denomination in minimum_lengths
                and context.hand.shape[_SUIT_ORDER.index(call.denomination)]
                >= minimum_lengths[call.denomination]
            ]
            return min(
                candidates,
                key=lambda call: (
                    -context.hand.shape[_SUIT_ORDER.index(call.denomination)],
                    tie_order.index(call.denomination),
                ),
                default=None,
            )
        if self.condition.suit_overcall_tiers:
            tie_order = self.condition.suit_tie_order
            if tie_order is None:
                raise RuntimeError("Suit overcall tiers must define a tie order.")
            candidates = [
                call
                for call in self.preferred_calls
                if isinstance(call, Bid)
                and call in context.legal_calls
                and _is_nonjump_new_suit_overcall(context, call)
                and _suit_overcall_qualifies(context, call, self.condition)
            ]
            return min(
                candidates,
                key=lambda call: (
                    -context.hand.shape[_SUIT_ORDER.index(call.denomination)],
                    tie_order.index(call.denomination),
                ),
                default=None,
            )
        return next(
            (
                call
                for call in self.preferred_calls
                if call in context.legal_calls
                and _call_matches_rule(context, call, self.condition)
            ),
            None,
        )

    def _dealer_suit_overcall_condition(
        self,
        context: DecisionContext,
        observed_call: Call,
    ) -> DealerCallCondition:
        tie_order = self.condition.suit_tie_order
        if tie_order is None:
            raise RuntimeError("Suit overcall tiers must define a tie order.")
        candidates = tuple(
            call
            for call in self.preferred_calls
            if isinstance(call, Bid)
            and call in context.legal_calls
            and any(
                level == call.level
                for level, _, _ in self.condition.suit_overcall_tiers
            )
            and _is_nonjump_new_suit_overcall(context, call)
        )
        qualifiers = {
            call: _suit_overcall_dealer_expression(context.seat, call, self.condition)
            for call in candidates
        }
        common = self.condition.dealer_expression(context.seat)
        if observed_call not in qualifiers:
            all_qualifiers = " || ".join(
                f"({expression})" for expression in qualifiers.values()
            )
            if not all_qualifiers:
                return DealerCallCondition()
            active = (
                f"({common}) && ({all_qualifiers})"
                if common is not None
                else f"({all_qualifiers})"
            )
            return DealerCallCondition((f"!({active})",))

        observed_suit = observed_call.denomination
        observed_length = (
            f"{_SUIT_NAMES[observed_suit]}({_SEAT_NAMES[context.seat]})"
        )
        higher_priority: list[str] = []
        for call, expression in qualifiers.items():
            if call == observed_call:
                continue
            other_length = (
                f"{_SUIT_NAMES[call.denomination]}({_SEAT_NAMES[context.seat]})"
            )
            is_preferred = (
                call.denomination is not observed_suit
                and tie_order.index(call.denomination)
                < tie_order.index(observed_suit)
            )
            comparison = (
                f"({other_length} >= {observed_length})"
                if is_preferred
                else f"({other_length} > {observed_length})"
            )
            higher_priority.append(f"(({expression}) && {comparison})")
        selection = f"({qualifiers[observed_call]})"
        if higher_priority:
            selection += " && !(" + " || ".join(higher_priority) + ")"
        if common is not None:
            selection = f"({common}) && ({selection})"
        return DealerCallCondition((selection,), selects_call=True)

    def _dealer_suit_opening_condition(
        self,
        context: DecisionContext,
        observed_call: Call,
    ) -> DealerCallCondition:
        tie_order = self.condition.suit_tie_order
        if tie_order is None:
            raise RuntimeError("Suit opening choices must define a tie order.")
        minimum_lengths = dict(self.condition.suit_opening_min_lengths)
        candidates = tuple(
            call
            for call in self.preferred_calls
            if isinstance(call, Bid)
            and call in context.legal_calls
            and call.denomination in minimum_lengths
        )
        seat_name = _SEAT_NAMES[context.seat]
        qualifiers = {
            call: (
                f"({_SUIT_NAMES[call.denomination]}({seat_name}) >= "
                f"{minimum_lengths[call.denomination]})"
            )
            for call in candidates
        }
        common = self.condition.dealer_expression(context.seat)
        if observed_call not in qualifiers:
            all_qualifiers = " || ".join(
                f"({expression})" for expression in qualifiers.values()
            )
            if not all_qualifiers:
                return DealerCallCondition()
            active = (
                f"({common}) && ({all_qualifiers})"
                if common is not None
                else f"({all_qualifiers})"
            )
            return DealerCallCondition((f"!({active})",))

        observed_suit = observed_call.denomination
        observed_length = f"{_SUIT_NAMES[observed_suit]}({seat_name})"
        higher_priority: list[str] = []
        for call, expression in qualifiers.items():
            if call == observed_call:
                continue
            other_suit = call.denomination
            other_length = f"{_SUIT_NAMES[other_suit]}({seat_name})"
            comparison = (
                f"({other_length} >= {observed_length})"
                if tie_order.index(other_suit) < tie_order.index(observed_suit)
                else f"({other_length} > {observed_length})"
            )
            higher_priority.append(f"(({expression}) && {comparison})")
        selection = f"({qualifiers[observed_call]})"
        if higher_priority:
            selection += " && !(" + " || ".join(higher_priority) + ")"
        if common is not None:
            selection = f"({common}) && ({selection})"
        return DealerCallCondition((selection,), selects_call=True)


@dataclass(frozen=True)
class BiddingModule:
    module_id: str
    version: str
    name: str
    rules: tuple[BiddingRule, ...]
    opponent_opening_calls: frozenset[Bid] | None = None
    partner_opening_calls: frozenset[Bid] | None = None
    uncontested_partner_opening: bool = False

    def choose(self, context: DecisionContext) -> ModuleDecision | None:
        if not _matches_module_opening(context, self):
            return None
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
        if not _matches_module_opening(context, self):
            return DealerCallCondition()
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
    module_keys = {"schema_version", "kind", "id", "version", "name", "rules"}
    optional_keys = {
        "opponent_opening_calls",
        "partner_opening_calls",
        "uncontested_partner_opening",
        "strength_ranges",
    }
    expected_keys = module_keys | (optional_keys & set(config))
    _require_keys(config, expected_keys, f"module file {module_path}")
    module_id, version, name = _validate_metadata(config)
    opening_calls = (
        _parse_opponent_opening_calls(config["opponent_opening_calls"])
        if "opponent_opening_calls" in config
        else None
    )
    partner_opening_calls = (
        _parse_partner_opening_calls(config["partner_opening_calls"])
        if "partner_opening_calls" in config
        else None
    )
    uncontested_partner_opening = config.get("uncontested_partner_opening", False)
    if not isinstance(uncontested_partner_opening, bool):
        raise ValueError("uncontested_partner_opening must be a boolean.")
    if uncontested_partner_opening and partner_opening_calls is None:
        raise ValueError(
            "uncontested_partner_opening requires partner_opening_calls."
        )
    strength_ranges = _parse_strength_ranges(config.get("strength_ranges"))
    raw_rules = config["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("A module must define at least one rule.")
    rules = tuple(
        _parse_rule(rule, index, strength_ranges)
        for index, rule in enumerate(raw_rules)
    )
    rule_ids = [rule.rule_id for rule in rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError(f"Module {module_id!r} defines duplicate rule IDs.")
    return BiddingModule(
        module_id,
        version,
        name,
        rules,
        opening_calls,
        partner_opening_calls,
        uncontested_partner_opening,
    )


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


def _parse_rule(
    value: object,
    index: int,
    strength_ranges: dict[str, tuple[int | None, int | None]],
) -> BiddingRule:
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
    condition = _parse_condition(raw_condition, index, strength_ranges)
    if "call" in value:
        preferred_calls = (_parse_call(value["call"]),)
    else:
        raw_calls = value["try_calls"]
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError(f"Module rule {index} try_calls must be a non-empty array.")
        preferred_calls = tuple(_parse_call(call) for call in raw_calls)
        if len(set(preferred_calls)) != len(preferred_calls):
            raise ValueError(f"Module rule {index} has duplicate try_calls.")
    if condition.suit_opening_min_lengths:
        opening_suits = {
            call.denomination
            for call in preferred_calls
            if isinstance(call, Bid) and call.level == 1
        }
        configured_suits = {
            suit for suit, _ in condition.suit_opening_min_lengths
        }
        if (
            len(opening_suits) != len(preferred_calls)
            or opening_suits != configured_suits
        ):
            raise ValueError(
                f"Module rule {index} suit opening lengths must match its "
                "level-one suit try_calls."
            )
    explanation = value.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise ValueError(f"Module rule {index} explanation must be a string.")
    return BiddingRule(rule_id, condition, preferred_calls, explanation)


def _parse_condition(
    value: object,
    index: int,
    strength_ranges: dict[str, tuple[int | None, int | None]],
) -> RuleCondition:
    if not isinstance(value, dict):
        raise ValueError(f"Module rule {index} when must be a TOML table.")
    allowed = {
        "min_hcp",
        "max_hcp",
        "strengths",
        "min_rule_of_20",
        "shapes",
        "passout_seat",
        "no_contract_bid",
        "opponent_opening_calls",
        "suit",
        "min_suit_length",
        "max_suit_length",
        "min_suit_hcp",
        "max_outside_major_length",
        "longest_qualifying_suit_min_length",
        "suit_tie_order",
        "suit_lengths",
        "suit_opening_min_lengths",
        "stopper_suit",
        "has_stopper",
        "nonjump_new_suit_overcall",
        "suit_overcall_tiers",
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
    hcp_ranges = _parse_strength_references(
        value.get("strengths"),
        strength_ranges,
        index,
    )
    min_rule_of_20 = _parse_integer_condition(
        value,
        "min_rule_of_20",
        0,
        50,
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
    opponent_opening_calls = (
        _parse_opponent_opening_calls(value["opponent_opening_calls"])
        if "opponent_opening_calls" in value
        else None
    )
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
    suit_lengths = _parse_suit_lengths(value.get("suit_lengths"), index)
    suit_opening_min_lengths = _parse_suit_opening_min_lengths(
        value.get("suit_opening_min_lengths"),
        index,
    )
    raw_stopper_suit = value.get("stopper_suit")
    stopper_suit: Denomination | None = None
    if raw_stopper_suit is not None:
        if not isinstance(raw_stopper_suit, str):
            raise ValueError(f"Module rule {index} stopper_suit must be a suit name.")
        stopper_suit = _parse_suit(raw_stopper_suit, index)
    has_stopper = value.get("has_stopper")
    if has_stopper is not None and not isinstance(has_stopper, bool):
        raise ValueError(f"Module rule {index} has_stopper must be a boolean.")
    if (stopper_suit is None) != (has_stopper is None):
        raise ValueError(
            f"Module rule {index} must define stopper_suit and has_stopper together."
        )
    nonjump_new_suit_overcall = value.get("nonjump_new_suit_overcall", False)
    if not isinstance(nonjump_new_suit_overcall, bool):
        raise ValueError(
            f"Module rule {index} nonjump_new_suit_overcall must be a boolean."
        )
    suit_overcall_tiers = _parse_suit_overcall_tiers(
        value.get("suit_overcall_tiers"),
        index,
    )
    if suit_overcall_tiers and not nonjump_new_suit_overcall:
        raise ValueError(
            f"Module rule {index} suit_overcall_tiers require "
            "nonjump_new_suit_overcall = true."
        )
    if suit_overcall_tiers and suit_tie_order is None:
        raise ValueError(
            f"Module rule {index} suit_overcall_tiers require suit_tie_order."
        )
    if suit_opening_min_lengths and suit_tie_order is None:
        raise ValueError(
            f"Module rule {index} suit_opening_min_lengths require suit_tie_order."
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
        and not suit_overcall_tiers
        and not suit_opening_min_lengths
    ):
        raise ValueError(
            "suit_tie_order requires a longest-suit or suit-overcall condition."
        )
    return RuleCondition(
        min_hcp=min_hcp,
        max_hcp=max_hcp,
        hcp_ranges=hcp_ranges,
        min_rule_of_20=min_rule_of_20,
        shapes=shapes,
        passout_seat=passout_seat,
        no_contract_bid=no_contract_bid,
        opponent_opening_calls=opponent_opening_calls,
        suit=suit,
        suit_tie_order=suit_tie_order,
        suit_lengths=suit_lengths,
        suit_opening_min_lengths=suit_opening_min_lengths,
        stopper_suit=stopper_suit,
        has_stopper=has_stopper,
        nonjump_new_suit_overcall=nonjump_new_suit_overcall,
        suit_overcall_tiers=suit_overcall_tiers,
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


def _parse_suit_lengths(
    value: object,
    index: int,
) -> tuple[tuple[Denomination, int | None, int | None], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Module rule {index} suit_lengths must be a non-empty table.")
    lengths: list[tuple[Denomination, int | None, int | None]] = []
    for suit_name, bounds in value.items():
        if not isinstance(suit_name, str) or not isinstance(bounds, dict):
            raise ValueError(
                f"Module rule {index} suit_lengths must map suit names to bounds."
            )
        suit = _parse_suit(suit_name, index)
        if set(bounds) - {"min", "max"} or not bounds:
            raise ValueError(
                f"Module rule {index} suit length bounds must use min and/or max."
            )
        minimum = _parse_integer_condition(bounds, "min", 0, 13)
        maximum = _parse_integer_condition(bounds, "max", 0, 13)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"Module rule {index} suit length minimum cannot exceed maximum."
            )
        lengths.append((suit, minimum, maximum))
    if len({suit for suit, _, _ in lengths}) != len(lengths):
        raise ValueError(f"Module rule {index} suit_lengths repeats a suit.")
    return tuple(lengths)


def _parse_suit_opening_min_lengths(
    value: object,
    index: int,
) -> tuple[tuple[Denomination, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"Module rule {index} suit_opening_min_lengths must be a non-empty table."
        )
    lengths: list[tuple[Denomination, int]] = []
    for suit_name, minimum in value.items():
        if not isinstance(suit_name, str):
            raise ValueError(
                f"Module rule {index} suit opening keys must be suit names."
            )
        suit = _parse_suit(suit_name, index)
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or not 1 <= minimum <= 13
        ):
            raise ValueError(
                f"Module rule {index} suit opening minimum lengths must be "
                "integers from 1 to 13."
            )
        lengths.append((suit, minimum))
    return tuple(lengths)


def _parse_suit_overcall_tiers(
    value: object,
    index: int,
) -> tuple[tuple[int, int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"Module rule {index} suit_overcall_tiers must be a non-empty array."
        )
    tiers: list[tuple[int, int, int]] = []
    for tier in value:
        if not isinstance(tier, dict) or set(tier) != {
            "level",
            "min_hcp",
            "min_suit_length",
        }:
            raise ValueError(
                f"Module rule {index} suit overcall tiers must define level, "
                "min_hcp, and min_suit_length."
            )
        level = _parse_integer_condition(tier, "level", 1, 7)
        minimum_hcp = _parse_integer_condition(tier, "min_hcp", 0, 37)
        minimum_length = _parse_integer_condition(tier, "min_suit_length", 0, 13)
        if level is None or minimum_hcp is None or minimum_length is None:
            raise ValueError(f"Module rule {index} has an incomplete suit overcall tier.")
        tiers.append((level, minimum_hcp, minimum_length))
    return tuple(tiers)


def _parse_opponent_opening_calls(value: object) -> frozenset[Bid]:
    if not isinstance(value, list) or not value:
        raise ValueError("opponent_opening_calls must be a non-empty array.")
    calls = tuple(_parse_call(call) for call in value)
    if any(not isinstance(call, Bid) for call in calls):
        raise ValueError(
            "opponent_opening_calls must contain suit bids at the 1- or 2-level."
        )
    opening_calls = tuple(call for call in calls if isinstance(call, Bid))
    if any(
        call.level > 2 or call.denomination is Denomination.NOTRUMP
        for call in opening_calls
    ):
        raise ValueError(
            "opponent_opening_calls must contain suit bids at the 1- or 2-level."
        )
    if len(set(calls)) != len(calls):
        raise ValueError("opponent_opening_calls must not repeat a bid.")
    return frozenset(opening_calls)


def _call_matches_rule(
    context: DecisionContext,
    call: Call,
    condition: RuleCondition,
) -> bool:
    if not condition.nonjump_new_suit_overcall:
        return True
    if (
        not isinstance(call, Bid)
        or not _is_nonjump_new_suit_overcall(context, call)
        or (condition.suit is not None and call.denomination is not condition.suit)
    ):
        return False
    return True


def _is_nonjump_new_suit_overcall(
    context: DecisionContext,
    call: Bid,
) -> bool:
    opening = _opponent_opening(context)
    if (
        opening is None
        or call.denomination is Denomination.NOTRUMP
        or call.denomination is opening.denomination
    ):
        return False
    cheapest_in_suit = min(
        (
            legal_call
            for legal_call in context.legal_calls
            if isinstance(legal_call, Bid)
            and legal_call.denomination is call.denomination
        ),
        key=lambda legal_call: legal_call.order,
        default=None,
    )
    return cheapest_in_suit == call


def _suit_overcall_qualifies(
    context: DecisionContext,
    call: Bid,
    condition: RuleCondition,
) -> bool:
    suit_length = context.hand.shape[_SUIT_ORDER.index(call.denomination)]
    return any(
        call.level == level
        and context.hand.hcp >= minimum_hcp
        and suit_length >= minimum_length
        for level, minimum_hcp, minimum_length in condition.suit_overcall_tiers
    )


def _suit_overcall_dealer_expression(
    seat: Seat,
    call: Bid,
    condition: RuleCondition,
) -> str:
    seat_name = _SEAT_NAMES[seat]
    suit_name = _SUIT_NAMES[call.denomination]
    alternatives = (
        f"((hcp({seat_name}) >= {minimum_hcp}) && "
        f"({suit_name}({seat_name}) >= {minimum_length}))"
        for level, minimum_hcp, minimum_length in condition.suit_overcall_tiers
        if level == call.level
    )
    return "(" + " || ".join(alternatives) + ")"


def _parse_strength_references(
    value: object,
    strength_ranges: dict[str, tuple[int | None, int | None]],
    index: int,
) -> tuple[tuple[int | None, int | None], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise ValueError(f"Module rule {index} strengths must be a non-empty array.")
    if any(
        not isinstance(name, str) or name not in strength_ranges
        for name in value
    ):
        raise ValueError(
            f"Module rule {index} strengths must reference ranges declared "
            "in strength_ranges."
        )
    if len(set(value)) != len(value):
        raise ValueError(f"Module rule {index} strengths must not repeat a range.")
    return tuple(strength_ranges[name] for name in value)


def _parse_strength_ranges(
    value: object,
) -> dict[str, tuple[int | None, int | None]]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not value:
        raise ValueError("strength_ranges must be a non-empty table.")
    ranges: dict[str, tuple[int | None, int | None]] = {}
    for name, bounds in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Strength range names must be non-empty strings.")
        if (
            not isinstance(bounds, dict)
            or not bounds
            or set(bounds) - {"min_hcp", "max_hcp"}
        ):
            raise ValueError(
                f"Strength range {name!r} must define min_hcp and/or max_hcp."
            )
        hcp_range = _parse_hcp_range(
            bounds.get("min_hcp"),
            bounds.get("max_hcp"),
        )
        if hcp_range == (None, None):
            raise ValueError(
                f"Strength range {name!r} must have at least one HCP bound."
            )
        ranges[name] = hcp_range
    return ranges


def _parse_partner_opening_calls(value: object) -> frozenset[Bid]:
    if not isinstance(value, list) or not value:
        raise ValueError("partner_opening_calls must be a non-empty array.")
    calls = tuple(_parse_call(call) for call in value)
    if any(not isinstance(call, Bid) for call in calls):
        raise ValueError("partner_opening_calls must contain contract bids.")
    opening_calls = tuple(call for call in calls if isinstance(call, Bid))
    if len(set(opening_calls)) != len(opening_calls):
        raise ValueError("partner_opening_calls must not repeat a bid.")
    return frozenset(opening_calls)


def _matches_module_opening(
    context: DecisionContext,
    module: BiddingModule,
) -> bool:
    if not _matches_opponent_opening(context, module.opponent_opening_calls):
        return False
    return _matches_partner_opening(
        context,
        module.partner_opening_calls,
        module.uncontested_partner_opening,
    )


def _matches_partner_opening(
    context: DecisionContext,
    opening_calls: frozenset[Bid] | None,
    require_uncontested: bool,
) -> bool:
    if opening_calls is None:
        return True
    opening_index = next(
        (
            index
            for index, entry in enumerate(context.auction)
            if entry.call is not PassCall.PASS
        ),
        None,
    )
    if opening_index is None:
        return False
    opening_entry = context.auction[opening_index]
    if (
        not isinstance(opening_entry.call, Bid)
        or opening_entry.call not in opening_calls
        or opening_entry.seat.next.next is not context.seat
    ):
        return False
    if require_uncontested and any(
        entry.call is not PassCall.PASS
        for entry in context.auction[opening_index + 1 :]
    ):
        return False
    return True


def _matches_opponent_opening(
    context: DecisionContext,
    opening_calls: frozenset[Bid] | None,
) -> bool:
    if opening_calls is None:
        return True
    opening = _opponent_opening(context)
    return opening is not None and opening in opening_calls


def _opponent_opening(context: DecisionContext) -> Bid | None:
    if not context.auction:
        return None
    last_entry = context.auction[-1]
    last_call = last_entry.call
    if (
        not isinstance(last_call, Bid)
        or last_entry.seat.next is not context.seat
        or _same_partnership(last_entry.seat, context.seat)
    ):
        return None
    contract_bids = [
        entry for entry in context.auction if isinstance(entry.call, Bid)
    ]
    if len(contract_bids) != 1 or contract_bids[0] != last_entry:
        return None
    return last_call


def _same_partnership(left: Seat, right: Seat) -> bool:
    return (left in (Seat.NORTH, Seat.SOUTH)) == (
        right in (Seat.NORTH, Seat.SOUTH)
    )


def _has_stopper(honors: str, suit_length: int) -> bool:
    return (
        "A" in honors
        or ("K" in honors and suit_length >= 2)
        or ("Q" in honors and suit_length >= 3)
        or ("J" in honors and suit_length >= 4)
        or suit_length >= 5
    )


def _dealer_stopper_expression(seat_name: str, suit: Denomination) -> str:
    suit_name = _SUIT_NAMES[suit]
    suit_code = _SUIT_CARD_CODES[suit]
    return "(" + " || ".join(
        (
            f"hascard({seat_name}, A{suit_code})",
            f"hascard({seat_name}, K{suit_code})",
            f"(({suit_name}({seat_name}) >= 3) && "
            f"hascard({seat_name}, Q{suit_code}))",
            f"(({suit_name}({seat_name}) >= 4) && "
            f"hascard({seat_name}, J{suit_code}))",
            f"({suit_name}({seat_name}) >= 5)",
        )
    ) + ")"


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
_SUIT_CARD_CODES = {
    Denomination.SPADES: "S",
    Denomination.HEARTS: "H",
    Denomination.DIAMONDS: "D",
    Denomination.CLUBS: "C",
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
