"""Local sanity checker for generated pymatgen JSON structures."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Sequence

from pymatgen.core import Structure

from crystal_llm.filters import reduced_formula, validate_structure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated pymatgen Structure JSON.")
    parser.add_argument("--input", default="input.json", help="Generated JSON file.")
    parser.add_argument("--max-sites", type=int, default=80, help="Maximum allowed sites per structure.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.input)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("input JSON must be a list of pymatgen Structure dictionaries")

    formulas: set[str] = set()
    site_counts: Counter = Counter()
    bad: list[dict[str, object]] = []

    for index, item in enumerate(data):
        structure = Structure.from_dict(item)
        formulas.add(reduced_formula(structure))
        site_counts[len(structure)] += 1
        validation = validate_structure(structure, max_sites=args.max_sites)
        if not validation.ok:
            bad.append(
                {
                    "index": index,
                    "formula": reduced_formula(structure),
                    "reasons": validation.reasons,
                }
            )

    summary = {
        "count": len(data),
        "unique_reduced_formulas": len(formulas),
        "bad_count": len(bad),
        "site_counts": dict(sorted(site_counts.items())),
        "bad_examples": bad[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
