"""Shared pytest helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root_path() -> Path:
    return Path(__file__).resolve().parents[1]
