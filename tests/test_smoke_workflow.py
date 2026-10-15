from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_module():
    path = Path(__file__).parents[1] / "scripts" / "smoke_workflow.py"
    spec = importlib.util.spec_from_file_location("smoke_workflow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_annotations_create_two_learnable_tiers() -> None:
    smoke = _load_smoke_module()

    assert smoke._synthetic_pass("Banking77", "mock-fast") is True
    assert smoke._synthetic_pass("MASSIVE", "mock-balanced") is True
    assert smoke._synthetic_pass("FinQA", "mock-fast") is False
    assert smoke._synthetic_pass("FinQA", "mock-balanced") is False
    assert smoke._synthetic_pass("FinQA", "mock-strong") is True
