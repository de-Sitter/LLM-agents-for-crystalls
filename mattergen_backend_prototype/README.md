# MatterGen Backend Prototype

This directory is an isolated prototype for using MatterGen as a structure
generation backend behind the current LLM agents.

The production project currently evaluates `input.json` files containing a list
of `pymatgen.core.Structure.as_dict()` objects. MatterGen writes generated
crystals as CIF/xyz artifacts, so the adapter here performs the bridge:

1. read a structured request from Z/W or a controller,
2. optionally launch `mattergen-generate`,
3. load generated CIF/extxyz structures,
4. apply project filters and duplicate checks,
5. write evaluator-ready `input.json`, `selected_records.json`, and
   `generation_report.json`.

## Why This Is Separate

MatterGen should not replace the existing template generator immediately. The
template generator is cheap, interpretable, and already works for some basins.
MatterGen is for cases where fixed prototypes are too restrictive or where X/Y
needs to test a topology beyond the current template library.

## Dry Run

Use dry-run first. It validates the request and writes the exact MatterGen CLI
command without requiring MatterGen to be installed.

```bash
PYTHONPATH=src python mattergen_backend_prototype/mattergen_adapter.py \
  --request mattergen_backend_prototype/examples/ce_h_energy_above_hull_request.json \
  --work-dir mattergen_backend_prototype/work/ce_h_dry_run \
  --dry-run
```

## Run MatterGen

After installing MatterGen and its checkpoints in an appropriate GPU
environment:

```bash
PYTHONPATH=src python mattergen_backend_prototype/mattergen_adapter.py \
  --request mattergen_backend_prototype/examples/ce_h_energy_above_hull_request.json \
  --work-dir mattergen_backend_prototype/work/ce_h_run \
  --strict-count
```

The adapter will invoke `mattergen-generate` and then write:

- `input.json`: evaluator input structures.
- `selected_records.json`: formula/backend metadata for each accepted structure.
- `generation_report.json`: counts, accepted formulas, and rejection reasons.
- `mattergen_command.json`: command used for reproducibility.
- `request.normalized.json`: normalized request visible to future agents.

If local checkpoints were downloaded with Git LFS, include `model_path` in the
request. The adapter will then pass `--model_path=/path/to/checkpoint_dir`
instead of `--pretrained-name=...`, which avoids a Hugging Face Hub lookup.

## Convert Existing MatterGen Output

If MatterGen was run elsewhere, pass the output directory or files:

```bash
PYTHONPATH=src python mattergen_backend_prototype/mattergen_adapter.py \
  --request mattergen_backend_prototype/examples/ce_h_energy_above_hull_request.json \
  --work-dir mattergen_backend_prototype/work/ce_h_convert \
  --from-existing path/to/mattergen/results
```

Supported inputs are directories containing `.cif`, `.xyz`, `.extxyz`, or `.zip`
files; standalone files with those suffixes are also supported.

## LLM Contract

See `prompts/llm_mattergen_contract.md`. In short:

- X/Y choose the scientific route.
- Z emits a `mattergen_request` instead of manual coordinates.
- W audits the request.
- MatterGen generates structures.
- The existing evaluator remains the source of truth for SUN.

## First Integration Target

The first production integration should add a new candidate source:

```json
{
  "source": "ml_generator",
  "backend": "mattergen",
  "count": 16,
  "mattergen_request": { "...": "..." }
}
```

The controller should materialize that source by calling this adapter, then
freeze accepted structures as `structure_dicts` before evaluation.
