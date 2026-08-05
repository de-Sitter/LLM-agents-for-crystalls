"""Chemistry helpers used by the structure generator.

The generator keeps LLM-facing decisions at a high level: choose a prototype,
choose chemically plausible substitutions, then let pymatgen build the final
periodic structure.  This module provides the small oxidation-state and radius
tables needed for that first reliable baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Mapping, Sequence

from pymatgen.core.periodic_table import Element


@dataclass(frozen=True)
class Ion:
    """Element plus an intended oxidation state and approximate ionic radius."""

    element: str
    oxidation_state: int
    radius: float


@dataclass(frozen=True)
class Strategy:
    """Generation preferences, optionally supplied from a strategy JSON file."""

    focus_templates: frozenset[str] = frozenset()
    avoid_elements: frozenset[str] = frozenset()
    boost_elements: frozenset[str] = frozenset()
    preferred_anions: frozenset[str] = frozenset()
    template_target_counts: dict[str, int] = field(default_factory=dict)
    template_max_counts: dict[str, int] = field(default_factory=dict)
    fallback_template_order: tuple[str, ...] = tuple()
    template_preferred_anions: dict[str, frozenset[str]] = field(default_factory=dict)
    template_avoid_elements: dict[str, frozenset[str]] = field(default_factory=dict)
    template_boost_elements: dict[str, frozenset[str]] = field(default_factory=dict)
    template_role_avoid_elements: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    template_role_boost_elements: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)


CATION_POOLS: dict[int, tuple[str, ...]] = {
    1: ("Li", "Na", "K", "Rb", "Cs", "Cu", "Ag", "Tl"),
    2: (
        "Be",
        "Mg",
        "Ca",
        "Sr",
        "Ba",
        "Zn",
        "Cd",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Sn",
        "Pb",
        "Eu",
    ),
    3: (
        "Al",
        "Ga",
        "In",
        "Sc",
        "Y",
        "La",
        "Bi",
        "Tl",
        "Fe",
        "Cr",
        "V",
        "Mn",
        "Co",
        "Ni",
    ),
    4: ("Ti", "Zr", "Hf", "Si", "Ge", "Sn", "Ce", "Mn", "V", "Mo", "W", "Pb"),
    5: ("V", "Nb", "Ta", "P", "As", "Sb", "Bi"),
    6: ("Mo", "W", "Cr", "S", "Se", "Te"),
}

ANION_POOLS: dict[int, tuple[str, ...]] = {
    -1: ("F", "Cl", "Br", "I"),
    -2: ("O", "S", "Se", "Te"),
    -3: ("N", "P"),
}

DEFAULT_ANION_WEIGHTS: dict[str, float] = {
    "O": 3.2,
    "F": 2.6,
    "S": 2.1,
    "Cl": 1.4,
    "N": 1.3,
    "Se": 0.85,
    "Br": 0.75,
    "I": 0.60,
    "Te": 0.45,
    "P": 0.45,
}


def load_strategy(raw: Mapping[str, object] | None) -> Strategy:
    """Convert user JSON preferences to an immutable Strategy."""

    if not raw:
        return Strategy()

    def as_set(name: str) -> frozenset[str]:
        value = raw.get(name, [])
        if isinstance(value, str):
            return frozenset([value])
        if isinstance(value, Sequence):
            return frozenset(str(item) for item in value)
        return frozenset()

    def as_int_map(*names: str) -> dict[str, int]:
        for name in names:
            value = raw.get(name)
            if not isinstance(value, Mapping):
                continue
            result: dict[str, int] = {}
            for key, item in value.items():
                try:
                    count = int(item)
                except (TypeError, ValueError):
                    continue
                if count > 0:
                    result[str(key)] = count
            return result
        return {}

    def as_str_set(value: object) -> frozenset[str]:
        if isinstance(value, str):
            return frozenset([value])
        if isinstance(value, Sequence):
            return frozenset(str(item) for item in value)
        return frozenset()

    def as_str_tuple(value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Sequence):
            return tuple(str(item) for item in value)
        return tuple()

    template_preferred_anions: dict[str, frozenset[str]] = {}
    template_avoid_elements: dict[str, frozenset[str]] = {}
    template_boost_elements: dict[str, frozenset[str]] = {}
    template_role_avoid_elements: dict[str, dict[str, frozenset[str]]] = {}
    template_role_boost_elements: dict[str, dict[str, frozenset[str]]] = {}
    template_rules = raw.get("template_rules", {})
    if isinstance(template_rules, Mapping):
        for template, rule in template_rules.items():
            if not isinstance(rule, Mapping):
                continue
            template_name = str(template)
            anions = as_str_set(rule.get("preferred_anions", rule.get("prefer_anions", [])))
            if anions:
                template_preferred_anions[template_name] = anions

            avoids = as_str_set(rule.get("avoid_elements", rule.get("avoid_cations", [])))
            if avoids:
                template_avoid_elements[template_name] = avoids

            boosts = as_str_set(rule.get("boost_elements", rule.get("prefer_cations", [])))
            if boosts:
                template_boost_elements[template_name] = boosts

            role_avoids_raw = rule.get("role_avoid_elements", {})
            role_avoids: dict[str, frozenset[str]] = {}
            if isinstance(role_avoids_raw, Mapping):
                for role, elements in role_avoids_raw.items():
                    values = as_str_set(elements)
                    if values:
                        role_avoids[str(role)] = values
            if role_avoids:
                template_role_avoid_elements[template_name] = role_avoids

            role_boosts_raw = rule.get("role_boost_elements", {})
            role_boosts: dict[str, frozenset[str]] = {}
            if isinstance(role_boosts_raw, Mapping):
                for role, elements in role_boosts_raw.items():
                    values = as_str_set(elements)
                    if values:
                        role_boosts[str(role)] = values
            if role_boosts:
                template_role_boost_elements[template_name] = role_boosts

    return Strategy(
        focus_templates=as_set("focus_templates"),
        avoid_elements=as_set("avoid_elements"),
        boost_elements=as_set("boost_elements"),
        preferred_anions=as_set("preferred_anions"),
        template_target_counts=as_int_map("template_target_counts", "template_quotas"),
        template_max_counts=as_int_map("template_max_counts", "template_caps"),
        fallback_template_order=as_str_tuple(raw.get("fallback_template_order", [])),
        template_preferred_anions=template_preferred_anions,
        template_avoid_elements=template_avoid_elements,
        template_boost_elements=template_boost_elements,
        template_role_avoid_elements=template_role_avoid_elements,
        template_role_boost_elements=template_role_boost_elements,
    )


def ionic_radius(element: str, oxidation_state: int) -> float:
    """Return an approximate ionic radius in angstroms.

    pymatgen has Shannon-style radii for many common oxidation states.  When a
    specific state is missing, fall back to atomic radius with a conservative
    ionic scaling so generation can proceed instead of failing on one element.
    """

    el = Element(element)
    if oxidation_state in el.ionic_radii:
        radius = el.ionic_radii[oxidation_state]
        if radius:
            return float(radius)

    atomic_radius = getattr(el, "atomic_radius", None)
    if atomic_radius:
        base = float(atomic_radius)
    else:
        base = 1.35

    if oxidation_state > 0:
        return max(0.35, base * 0.68)
    return max(0.8, base * 1.25)


def make_ion(element: str, oxidation_state: int) -> Ion:
    return Ion(element=element, oxidation_state=oxidation_state, radius=ionic_radius(element, oxidation_state))


def weighted_choice(items: Sequence[str], rng: random.Random, strategy: Strategy, *, anion: bool = False) -> str:
    """Choose an element while honoring avoid/boost/preferred-anion settings."""

    candidates = [item for item in items if item not in strategy.avoid_elements]
    if not candidates:
        candidates = list(items)

    weights: list[float] = []
    for item in candidates:
        weight = 1.0
        if anion:
            weight *= DEFAULT_ANION_WEIGHTS.get(item, 1.0)
        if item in strategy.boost_elements:
            weight *= 2.5
        if anion and strategy.preferred_anions and item in strategy.preferred_anions:
            weight *= 3.0
        weights.append(weight)

    return rng.choices(candidates, weights=weights, k=1)[0]


def choose_ion(
    oxidation_state: int,
    rng: random.Random,
    strategy: Strategy,
    *,
    excluded: frozenset[str] = frozenset(),
) -> Ion:
    pool = ANION_POOLS[oxidation_state] if oxidation_state < 0 else CATION_POOLS[oxidation_state]
    local_strategy = Strategy(
        focus_templates=strategy.focus_templates,
        avoid_elements=strategy.avoid_elements | excluded,
        boost_elements=strategy.boost_elements,
        preferred_anions=strategy.preferred_anions,
        template_target_counts=strategy.template_target_counts,
        template_max_counts=strategy.template_max_counts,
        fallback_template_order=strategy.fallback_template_order,
        template_preferred_anions=strategy.template_preferred_anions,
        template_avoid_elements=strategy.template_avoid_elements,
        template_boost_elements=strategy.template_boost_elements,
        template_role_avoid_elements=strategy.template_role_avoid_elements,
        template_role_boost_elements=strategy.template_role_boost_elements,
    )
    element = weighted_choice(pool, rng, local_strategy, anion=oxidation_state < 0)
    return make_ion(element, oxidation_state)


def template_role_strategy(strategy: Strategy, template: str, role: str) -> Strategy:
    template_avoids = set(strategy.template_avoid_elements.get(template, frozenset()))
    role_avoids = strategy.template_role_avoid_elements.get(template, {})
    template_avoids.update(role_avoids.get("*", frozenset()))
    template_avoids.update(role_avoids.get(role, frozenset()))

    template_boosts = set(strategy.template_boost_elements.get(template, frozenset()))
    role_boosts = strategy.template_role_boost_elements.get(template, {})
    template_boosts.update(role_boosts.get("*", frozenset()))
    template_boosts.update(role_boosts.get(role, frozenset()))
    preferred_anions = strategy.template_preferred_anions.get(template, strategy.preferred_anions)
    return Strategy(
        focus_templates=strategy.focus_templates,
        avoid_elements=strategy.avoid_elements | frozenset(template_avoids),
        boost_elements=strategy.boost_elements | frozenset(template_boosts),
        preferred_anions=preferred_anions,
        template_target_counts=strategy.template_target_counts,
        template_max_counts=strategy.template_max_counts,
        fallback_template_order=strategy.fallback_template_order,
        template_preferred_anions=strategy.template_preferred_anions,
        template_avoid_elements=strategy.template_avoid_elements,
        template_boost_elements=strategy.template_boost_elements,
        template_role_avoid_elements=strategy.template_role_avoid_elements,
        template_role_boost_elements=strategy.template_role_boost_elements,
    )


def choose_role_ion(
    template: str,
    role: str,
    oxidation_state: int,
    rng: random.Random,
    strategy: Strategy,
    *,
    excluded: frozenset[str] = frozenset(),
) -> Ion:
    return choose_ion(
        oxidation_state,
        rng,
        template_role_strategy(strategy, template, role),
        excluded=excluded,
    )


def sample_ab(rng: random.Random, strategy: Strategy, template: str) -> dict[str, Ion]:
    ox_a, ox_x = rng.choice(((1, -1), (2, -2), (3, -3)))
    a = choose_role_ion(template, "A", ox_a, rng, strategy)
    x = choose_role_ion(template, "X", ox_x, rng, strategy, excluded=frozenset([a.element]))
    return {"A": a, "X": x}


def sample_ax2(rng: random.Random, strategy: Strategy, template: str) -> dict[str, Ion]:
    ox_a, ox_x = rng.choice(((4, -2), (2, -1), (6, -3)))
    a = choose_role_ion(template, "A", ox_a, rng, strategy)
    x = choose_role_ion(template, "X", ox_x, rng, strategy, excluded=frozenset([a.element]))
    return {"A": a, "X": x}


def sample_a2x(rng: random.Random, strategy: Strategy, template: str) -> dict[str, Ion]:
    a = choose_role_ion(template, "A", 1, rng, strategy)
    x = choose_role_ion(template, "X", -2, rng, strategy, excluded=frozenset([a.element]))
    return {"A": a, "X": x}


def sample_a2x3(rng: random.Random, strategy: Strategy, template: str) -> dict[str, Ion]:
    a = choose_role_ion(template, "A", 3, rng, strategy)
    x = choose_role_ion(template, "X", -2, rng, strategy, excluded=frozenset([a.element]))
    return {"A": a, "X": x}


def sample_perovskite(rng: random.Random, strategy: Strategy, template: str) -> dict[str, Ion]:
    ox_x, pairs = rng.choice(
        (
            (-2, ((1, 5), (2, 4), (3, 3))),
            (-1, ((1, 2),)),
            (-3, ((3, 6),)),
        )
    )
    ox_a, ox_b = rng.choice(pairs)
    a = choose_role_ion(template, "A", ox_a, rng, strategy)
    b = choose_role_ion(template, "B", ox_b, rng, strategy, excluded=frozenset([a.element]))
    x = choose_role_ion(template, "X", ox_x, rng, strategy, excluded=frozenset([a.element, b.element]))
    return {"A": a, "B": b, "X": x}


def sample_double_perovskite(rng: random.Random, strategy: Strategy, template: str) -> dict[str, Ion]:
    options = (
        (-2, 2, ((3, 5), (4, 4), (2, 6))),
        (-2, 1, ((4, 6), (5, 5))),
        (-2, 3, ((3, 3), (2, 4))),
        (-1, 1, ((2, 2), (1, 3))),
    )
    ox_x, ox_a, b_pairs = rng.choice(options)
    ox_b1, ox_b2 = rng.choice(b_pairs)
    a = choose_role_ion(template, "A", ox_a, rng, strategy)
    b1 = choose_role_ion(template, "B", ox_b1, rng, strategy, excluded=frozenset([a.element]))
    b2 = choose_role_ion(template, "B2", ox_b2, rng, strategy, excluded=frozenset([a.element, b1.element]))
    x = choose_role_ion(template, "X", ox_x, rng, strategy, excluded=frozenset([a.element, b1.element, b2.element]))
    return {"A": a, "B": b1, "B2": b2, "X": x}


def sample_spinel(rng: random.Random, strategy: Strategy, template: str) -> dict[str, Ion]:
    if rng.random() < 0.85:
        ox_a, ox_b, ox_x = 2, 3, -2
    else:
        ox_a, ox_b, ox_x = 4, 2, -2
    a = choose_role_ion(template, "A", ox_a, rng, strategy)
    b = choose_role_ion(template, "B", ox_b, rng, strategy, excluded=frozenset([a.element]))
    x = choose_role_ion(template, "X", ox_x, rng, strategy, excluded=frozenset([a.element, b.element]))
    return {"A": a, "B": b, "X": x}


def sample_delafossite(rng: random.Random, strategy: Strategy, template: str) -> dict[str, Ion]:
    a = choose_role_ion(template, "A", 1, rng, strategy)
    b = choose_role_ion(template, "B", 3, rng, strategy, excluded=frozenset([a.element]))
    x = choose_role_ion(template, "X", -2, rng, strategy, excluded=frozenset([a.element, b.element]))
    return {"A": a, "B": b, "X": x}


def oxidation_charge(ions: Mapping[str, Ion], counts: Mapping[str, int]) -> int:
    return sum(ions[role].oxidation_state * count for role, count in counts.items())


def tolerance_factor(a: Ion, b: Ion, x: Ion) -> float:
    return (a.radius + x.radius) / (math.sqrt(2.0) * (b.radius + x.radius))


SAMPLERS = {
    "rocksalt": sample_ab,
    "cesium_chloride": sample_ab,
    "zincblende": sample_ab,
    "wurtzite": sample_ab,
    "fluorite": sample_ax2,
    "rutile": sample_ax2,
    "antifluorite": sample_a2x,
    "perovskite": sample_perovskite,
    "double_perovskite": sample_double_perovskite,
    "spinel": sample_spinel,
    "corundum": sample_a2x3,
    "delafossite": sample_delafossite,
}
