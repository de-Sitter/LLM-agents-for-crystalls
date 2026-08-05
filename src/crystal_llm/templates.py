"""Crystal prototype builders.

Every builder returns a valid pymatgen Structure from a small role-to-Ion map.
The coordinates are conventional prototype positions; lattice constants are
estimated from ionic radii and then jittered slightly to avoid exact clones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Mapping

from pymatgen.core import Lattice, Structure

from crystal_llm.chemistry import Ion, tolerance_factor


@dataclass
class Candidate:
    structure: Structure
    template: str
    metadata: dict[str, float | str] = field(default_factory=dict)
    score: float = 0.0


Builder = Callable[[Mapping[str, Ion], random.Random], Candidate]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def jitter(rng: random.Random, width: float = 0.025) -> float:
    return 1.0 + rng.uniform(-width, width)


def _structure(lattice: Lattice, species: list[str], coords: list[list[float]]) -> Structure:
    return Structure(lattice, species, coords, coords_are_cartesian=False, to_unit_cell=True)


def _pair_radius(ions: Mapping[str, Ion], role_a: str = "A", role_x: str = "X") -> float:
    return ions[role_a].radius + ions[role_x].radius


def build_rocksalt(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, x_ion = ions["A"], ions["X"]
    a = clamp(2.04 * _pair_radius(ions) * jitter(rng), 3.0, 8.5)
    fcc = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    octa = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    structure = _structure(Lattice.cubic(a), [a_ion.element] * 4 + [x_ion.element] * 4, fcc + octa)
    return Candidate(structure, "rocksalt")


def build_cesium_chloride(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, x_ion = ions["A"], ions["X"]
    a = clamp(2.0 * _pair_radius(ions) / math.sqrt(3.0) * jitter(rng), 2.7, 7.0)
    structure = _structure(Lattice.cubic(a), [a_ion.element, x_ion.element], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return Candidate(structure, "cesium_chloride")


def build_zincblende(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, x_ion = ions["A"], ions["X"]
    a = clamp(4.0 * _pair_radius(ions) / math.sqrt(3.0) * jitter(rng), 3.2, 8.5)
    fcc = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    tetra = [[0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]]
    structure = _structure(Lattice.cubic(a), [a_ion.element] * 4 + [x_ion.element] * 4, fcc + tetra)
    return Candidate(structure, "zincblende")


def build_wurtzite(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, x_ion = ions["A"], ions["X"]
    a = clamp(2.0 * _pair_radius(ions) * jitter(rng), 2.8, 6.5)
    c = a * math.sqrt(8.0 / 3.0) * jitter(rng, 0.015)
    species = [a_ion.element, a_ion.element, x_ion.element, x_ion.element]
    coords = [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.375], [2 / 3, 1 / 3, 0.875]]
    structure = _structure(Lattice.hexagonal(a, c), species, coords)
    return Candidate(structure, "wurtzite")


def build_fluorite(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, x_ion = ions["A"], ions["X"]
    a = clamp(4.0 * _pair_radius(ions) / math.sqrt(3.0) * jitter(rng), 4.1, 9.5)
    fcc = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    tetra = [
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.75],
        [0.25, 0.75, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.25],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
        [0.75, 0.75, 0.75],
    ]
    structure = _structure(Lattice.cubic(a), [a_ion.element] * 4 + [x_ion.element] * 8, fcc + tetra)
    return Candidate(structure, "fluorite")


def build_antifluorite(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, x_ion = ions["A"], ions["X"]
    a = clamp(4.0 * _pair_radius(ions) / math.sqrt(3.0) * jitter(rng), 4.0, 9.5)
    fcc = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    tetra = [
        [0.25, 0.25, 0.25],
        [0.25, 0.25, 0.75],
        [0.25, 0.75, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.25],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
        [0.75, 0.75, 0.75],
    ]
    structure = _structure(Lattice.cubic(a), [x_ion.element] * 4 + [a_ion.element] * 8, fcc + tetra)
    return Candidate(structure, "antifluorite")


def build_perovskite(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, b_ion, x_ion = ions["A"], ions["B"], ions["X"]
    bx = b_ion.radius + x_ion.radius
    ax = a_ion.radius + x_ion.radius
    a = clamp(max(2.02 * bx, math.sqrt(2.0) * ax) * jitter(rng), 3.2, 8.5)
    species = [a_ion.element, b_ion.element, x_ion.element, x_ion.element, x_ion.element]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    t = tolerance_factor(a_ion, b_ion, x_ion)
    structure = _structure(Lattice.cubic(a), species, coords)
    return Candidate(structure, "perovskite", metadata={"tolerance_factor": t})


def build_double_perovskite(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, b1_ion, b2_ion, x_ion = ions["A"], ions["B"], ions["B2"], ions["X"]
    bx = (b1_ion.radius + b2_ion.radius) / 2.0 + x_ion.radius
    ax = a_ion.radius + x_ion.radius
    a = clamp(max(2.02 * bx, math.sqrt(2.0) * ax) * jitter(rng), 3.3, 8.8)
    lattice = Lattice.from_parameters(2 * a, a, a, 90, 90, 90)
    species = [
        a_ion.element,
        a_ion.element,
        b1_ion.element,
        b2_ion.element,
        x_ion.element,
        x_ion.element,
        x_ion.element,
        x_ion.element,
        x_ion.element,
        x_ion.element,
    ]
    coords = [
        [0, 0, 0],
        [0.5, 0, 0],
        [0.25, 0.5, 0.5],
        [0.75, 0.5, 0.5],
        [0.25, 0.5, 0],
        [0.25, 0, 0.5],
        [0, 0.5, 0.5],
        [0.75, 0.5, 0],
        [0.75, 0, 0.5],
        [0.5, 0.5, 0.5],
    ]
    t = tolerance_factor(a_ion, Ion("Bavg", 0, (b1_ion.radius + b2_ion.radius) / 2.0), x_ion)
    structure = _structure(lattice, species, coords)
    return Candidate(structure, "double_perovskite", metadata={"tolerance_factor": t})


def build_spinel(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, b_ion, x_ion = ions["A"], ions["B"], ions["X"]
    a = clamp(2.0 * (a_ion.radius + b_ion.radius + 2.0 * x_ion.radius) * jitter(rng), 7.0, 12.5)
    structure = Structure.from_spacegroup(
        "Fd-3m",
        Lattice.cubic(a),
        [a_ion.element, b_ion.element, x_ion.element],
        [[0, 0, 0], [0.625, 0.625, 0.625], [0.385, 0.385, 0.385]],
    )
    return Candidate(structure, "spinel")


def build_rutile(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, x_ion = ions["A"], ions["X"]
    a = clamp(2.25 * _pair_radius(ions) * jitter(rng), 3.5, 7.5)
    c = clamp(0.64 * a * jitter(rng, 0.015), 2.2, 5.2)
    structure = Structure.from_spacegroup(
        "P4_2/mnm",
        Lattice.tetragonal(a, c),
        [a_ion.element, x_ion.element],
        [[0, 0, 0], [0.305, 0.305, 0]],
    )
    return Candidate(structure, "rutile")


def build_corundum(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, x_ion = ions["A"], ions["X"]
    a = clamp(2.35 * _pair_radius(ions) * jitter(rng), 4.0, 7.2)
    c = clamp(2.73 * a * jitter(rng, 0.015), 10.0, 20.0)
    structure = Structure.from_spacegroup(
        "R-3c",
        Lattice.hexagonal(a, c),
        [a_ion.element, x_ion.element],
        [[0, 0, 0.352], [0.306, 0, 0.25]],
    )
    return Candidate(structure, "corundum")


def build_delafossite(ions: Mapping[str, Ion], rng: random.Random) -> Candidate:
    a_ion, b_ion, x_ion = ions["A"], ions["B"], ions["X"]
    a = clamp(1.78 * (b_ion.radius + x_ion.radius) * jitter(rng), 2.8, 5.2)
    c = clamp(6.0 * (a_ion.radius + x_ion.radius) * jitter(rng), 12.0, 22.0)
    structure = Structure.from_spacegroup(
        "R-3m",
        Lattice.hexagonal(a, c),
        [a_ion.element, b_ion.element, x_ion.element],
        [[0, 0, 0], [0, 0, 0.5], [0, 0, 0.11]],
    )
    return Candidate(structure, "delafossite")


BUILDERS: dict[str, Builder] = {
    "rocksalt": build_rocksalt,
    "cesium_chloride": build_cesium_chloride,
    "zincblende": build_zincblende,
    "wurtzite": build_wurtzite,
    "fluorite": build_fluorite,
    "rutile": build_rutile,
    "antifluorite": build_antifluorite,
    "perovskite": build_perovskite,
    "double_perovskite": build_double_perovskite,
    "spinel": build_spinel,
    "corundum": build_corundum,
    "delafossite": build_delafossite,
}


TEMPLATE_WEIGHTS: dict[str, float] = {
    "perovskite": 1.30,
    "double_perovskite": 1.20,
    "spinel": 1.18,
    "rocksalt": 1.05,
    "zincblende": 0.95,
    "wurtzite": 0.90,
    "fluorite": 1.05,
    "rutile": 1.00,
    "antifluorite": 0.95,
    "cesium_chloride": 0.85,
    "corundum": 0.95,
    "delafossite": 0.90,
}
