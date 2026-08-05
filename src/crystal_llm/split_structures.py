"""Split generated Structure JSON into shard files for Slurm array evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a pymatgen Structure JSON list into shards.")
    parser.add_argument("--input", default="input.json", help="Input JSON list.")
    parser.add_argument("--out-dir", required=True, help="Directory for shard JSON files.")
    parser.add_argument("--shards", type=int, default=8, help="Number of shards.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.shards <= 0:
        raise ValueError("--shards must be positive")

    with Path(args.input).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("input must be a JSON list")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: list[int] = []
    for shard in range(args.shards):
        shard_data = data[shard:: args.shards]
        path = out_dir / f"input_shard_{shard:03d}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(shard_data, handle, ensure_ascii=False, indent=2)
        counts.append(len(shard_data))
        print(f"{path}: {len(shard_data)}")

    manifest = {
        "input": str(Path(args.input).resolve()),
        "shards": args.shards,
        "total": len(data),
        "counts": counts,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
