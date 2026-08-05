"""Convert evaluator training CSV files to pymatgen Structure JSON lists.

The downloaded evaluator bundle includes CSV files whose ``structure`` column
contains Python-dict strings.  The official evaluator's CSV loader expects JSON,
POSCAR, or CIF strings, so this helper writes a strict JSON list that the
evaluator can load directly via its JSON path.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Sequence

from pymatgen.core import Structure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert training CSV structures to pymatgen JSON.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--structure-column", default="structure", help="CSV column containing structure data.")
    return parser.parse_args(argv)


def parse_structure_cell(value: str) -> Structure:
    text = value.strip()
    if text.startswith("{"):
        try:
            return Structure.from_dict(json.loads(text))
        except Exception:
            return Structure.from_dict(ast.literal_eval(text))
    return Structure.from_str(text, fmt="auto")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    structures: list[Structure] = []
    skipped = 0

    with Path(args.input).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if args.structure_column not in (reader.fieldnames or []):
            raise ValueError(f"column {args.structure_column!r} not found in {args.input}")

        for row in reader:
            value = row.get(args.structure_column) or ""
            if not value:
                skipped += 1
                continue
            try:
                structures.append(parse_structure_cell(value))
            except Exception:
                skipped += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([structure.as_dict() for structure in structures], handle, ensure_ascii=False, indent=2)

    print(f"wrote {len(structures)} structures to {output_path}")
    if skipped:
        print(f"skipped {skipped} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
