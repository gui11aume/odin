"""Line-oriented text dataset for map-style dataloaders."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import torch


class TextLineDataset(torch.utils.data.Dataset[Any]):
    """Loads non-empty trimmed lines from a UTF-8 text or ``.gz`` file."""

    def __init__(self, path: str | Path):
        """Read and cache corpus lines."""
        raw_path = Path(path)
        if not raw_path.is_file():
            raise FileNotFoundError(str(raw_path))

        opener = gzip.open if raw_path.suffix == ".gz" else open
        with opener(raw_path, "rt", encoding="utf-8") as handle:
            self.lines: list[str] = [line.strip() for line in handle if line.strip()]

        if not self.lines:
            raise ValueError(f"Dataset is empty after filtering blank lines: {raw_path}")

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index) -> Any:
        return self.lines[index]
