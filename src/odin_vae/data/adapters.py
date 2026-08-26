"""Adapters between the webdataset stream and the collator.

A shard sample is one cluster: ``{"__key__": str, "__url__": str,
"json": bytes}`` where ``json`` holds ``{"tags": [...], "cells": [...]}``.
The adapter parses and validates the record so that the collator receives
plain Python data that can be batched directly (no intermediate lookup store).
"""

from __future__ import annotations

import json

import pydantic

from .grandwds import GrandWebDataset


class ClusterRecord(pydantic.BaseModel):
    """One cluster: parallel lists of script tags and name surfaces."""

    model_config = pydantic.ConfigDict(extra="forbid")

    tags: list[str] = pydantic.Field(min_length=1)
    cells: list[str] = pydantic.Field(min_length=1)


class ClusterSampleAdapter(GrandWebDataset):
    """GrandWebDataset yielding parsed cluster records."""

    def __iter__(self):
        for item in super().__iter__():
            try:
                record = ClusterRecord.model_validate(json.loads(item["json"]))
            except (json.JSONDecodeError, pydantic.ValidationError) as exc:
                raise ValueError(f"Malformed cluster record {item['__key__']!r} in {item['__url__']!r}: {exc}") from exc
            if len(record.tags) != len(record.cells):
                raise ValueError(f"Cluster {item['__key__']!r}: {len(record.tags)} tags for {len(record.cells)} cells.")
            if any(not cell.strip() for cell in record.cells):
                raise ValueError(f"Cluster {item['__key__']!r}: empty cell value.")
            yield {"key": item["__key__"], "tags": record.tags, "cells": record.cells}
