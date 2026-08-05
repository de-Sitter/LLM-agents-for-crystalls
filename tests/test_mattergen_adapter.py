from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from pymatgen.core import Lattice, Structure

_ADAPTER_PATH = Path(__file__).resolve().parents[1] / "mattergen_backend_prototype" / "mattergen_adapter.py"
_SPEC = importlib.util.spec_from_file_location("mattergen_adapter_under_test", _ADAPTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mattergen_adapter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mattergen_adapter
_SPEC.loader.exec_module(mattergen_adapter)


def test_conditional_request_defaults_guidance_and_compact_condition_cli() -> None:
    request = mattergen_adapter.normalize_request(
        {
            "backend": "mattergen",
            "request_id": "ce_h",
            "model_path": "/tmp/fake_mattergen_model",
            "properties_to_condition_on": {"chemical_system": "Ce-H", "energy_above_hull": 0.0},
            "filters": {"chemical_system": ["Ce", "H"], "max_sites": 20},
            "target_count": 4,
            "batch_size": 8,
            "num_batches": 1,
        }
    )

    command = mattergen_adapter.build_mattergen_command(request, Path("/tmp/out"), "/tmp/mattergen-generate")
    condition_arg = next(item for item in command if item.startswith("--properties_to_condition_on="))

    assert request.diffusion_guidance_factor == mattergen_adapter.DEFAULT_DIFFUSION_GUIDANCE_FACTOR
    assert f"--diffusion_guidance_factor={mattergen_adapter.DEFAULT_DIFFUSION_GUIDANCE_FACTOR}" in command
    assert condition_arg == '--properties_to_condition_on={"chemical_system":"Ce-H","energy_above_hull":0.0}'


def test_result_directory_prefers_single_canonical_mattergen_output(tmp_path, monkeypatch) -> None:
    results_dir = tmp_path / "mattergen_results"
    results_dir.mkdir()
    extxyz = results_dir / "generated_crystals.extxyz"
    extxyz.write_text("", encoding="utf-8")
    (results_dir / "generated_crystals_cif.zip").write_bytes(b"not-read")

    sentinel = object()
    monkeypatch.setattr(mattergen_adapter, "read_extxyz", lambda path: [sentinel])

    assert mattergen_adapter.read_structures_from_path(results_dir) == [sentinel]


def test_mattergen_filter_uses_relaxed_volume_window_for_generated_heavy_cells() -> None:
    request = mattergen_adapter.normalize_request(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Eu-Si", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Eu", "Si"],
                "target_reduced_formula": "Eu2Si",
                "max_sites": 20,
            },
            "target_count": 1,
        }
    )
    structure = Structure(
        Lattice.cubic(4.75),
        ["Eu", "Eu", "Si"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.5]],
    )

    accepted, rejects, examples = mattergen_adapter.filter_structures([structure], request)

    assert len(accepted) == 1
    assert not rejects
    assert examples == []


def test_target_reduced_formula_is_soft_preference_by_default() -> None:
    request = mattergen_adapter.normalize_request(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Yb-Si", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Yb", "Si"],
                "target_reduced_formula": "Yb2Si",
                "max_sites": 20,
            },
            "target_count": 1,
        }
    )
    non_target = Structure(
        Lattice.cubic(4.2),
        ["Yb", "Si"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    target = Structure(
        Lattice.cubic(4.8),
        ["Yb", "Yb", "Si"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.5]],
    )

    accepted, rejects, examples = mattergen_adapter.filter_structures([non_target, target], request)

    assert [mattergen_adapter.reduced_formula(structure) for structure in accepted] == ["Yb2Si"]
    assert not rejects
    assert examples == []


def test_target_reduced_formula_can_be_required_explicitly() -> None:
    request = mattergen_adapter.normalize_request(
        {
            "backend": "mattergen",
            "properties_to_condition_on": {"chemical_system": "Yb-Si", "energy_above_hull": 0.0},
            "filters": {
                "chemical_system": ["Yb", "Si"],
                "target_reduced_formula": "Yb2Si",
                "require_target_reduced_formula": True,
                "max_sites": 20,
            },
            "target_count": 1,
        }
    )
    non_target = Structure(
        Lattice.cubic(4.2),
        ["Yb", "Si"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    accepted, rejects, examples = mattergen_adapter.filter_structures([non_target], request)

    assert accepted == []
    assert rejects["not_target_reduced_formula"] == 1
    assert examples[0]["reason"] == "not_target_reduced_formula"
