"""Shared loader for the shard-builder runner module (used by multiple test files).

Pytest imports every test module at collection time; loading the runner twice
under the same module name would leave two distinct module objects in
``sys.modules`` history and break pickling of the builder's worker function
(ProcessPoolExecutor resolves the function by reference at task time). A
single cached load under one name keeps the reference stable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_NAME = "build_odin_shards"


def get_builder():
    module = sys.modules.get(_NAME)
    if module is not None:
        return module
    path = Path(__file__).resolve().parents[1] / "runners" / "build_odin_shards.py"
    spec = importlib.util.spec_from_file_location(_NAME, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[_NAME] = module
    return module
