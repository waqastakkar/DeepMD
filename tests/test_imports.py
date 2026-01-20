import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = [
    "cli",
    "config",
    "core",
    "io",
    "learn",
    "stages",
    "validate",
]


def test_imports_smoke():
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                raise
            pytest.skip(
                f"Skipping import of {module_name} due to missing dependency: {exc.name}"
            )
