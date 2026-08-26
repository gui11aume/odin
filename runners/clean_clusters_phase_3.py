"""Phase 3: clean the cluster corpus — drop structurally malformed lines.

Reads a phase-1/2 cluster corpus (tab-separated ``tag{value}`` cells, one
cluster per line) and writes back only the structurally valid lines.

A line is valid when:
  * it is non-empty,
  * every tab-separated cell is exactly ``tag{value}`` with a known tag
    (la/cy/gk/ab/cn/jp/kr/dv/hb/th/gg/am) and a non-empty value,
  * at least one ``la`` cell is present.

This drops, in particular, clusters split across physical lines by a
literal newline embedded in a cell value (an LLM-output artifact of the
phase-1/2 pipeline): the line with the unterminated cell and its
continuation fragment(s) both fail validation and are removed.

Also reports (and optionally drops, ``--drop-incomplete``) structurally
valid lines that miss one or more of the twelve expected script tags.

Usage:
    python runners/clean_clusters_phase_3.py INPUT [-o OUTPUT] [--drop-incomplete]
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

log = logging.getLogger(__name__)

TAGS = frozenset({"la", "cy", "gk", "ab", "cn", "jp", "kr", "dv", "hb", "th", "gg", "am"})

# Drop reasons.
REASON_EMPTY = "empty_line"
REASON_BAD_CELL = "malformed_cell"
REASON_EMPTY_VALUE = "empty_value"
REASON_NO_LA = "no_la_cell"
REASON_INCOMPLETE = "missing_tags"


def validate_line(line: str) -> tuple[bool, str, frozenset[str]]:
    """Validate one physical line of the cluster corpus.

    Returns ``(is_valid, drop_reason, tags_present)``. ``drop_reason`` is
    empty when valid; ``tags_present`` is the set of tags found in the
    cells (best-effort: parsed up to the first invalid cell).
    """
    line = line.rstrip("\r\n")
    if not line.strip():
        return False, REASON_EMPTY, frozenset()

    tags: set[str] = set()
    for cell in line.split("\t"):
        if len(cell) < 4 or cell[2] != "{" or not cell.endswith("}"):
            return False, REASON_BAD_CELL, frozenset(tags)
        tag = cell[:2]
        if tag not in TAGS:
            return False, REASON_BAD_CELL, frozenset(tags)
        if cell[3:-1] == "":
            return False, REASON_EMPTY_VALUE, frozenset(tags)
        tags.add(tag)

    if "la" not in tags:
        return False, REASON_NO_LA, frozenset(tags)
    return True, "", frozenset(tags)


def iter_lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line


def open_out(path: Path):
    """Text-mode writer; gzip output is deterministic (mtime=0)."""
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.GzipFile(path, mode="wb", mtime=0), encoding="utf-8")
    return open(path, "wt", encoding="utf-8")


def clean_file(
    input_path: Path,
    output_path: Path,
    *,
    drop_incomplete: bool = False,
    max_dump: int = 5,
) -> dict:
    """Stream ``input_path`` line by line, writing valid lines to ``output_path``.

    Returns a stats dict: total/kept/dropped counts per reason, tag-completeness
    counts, and the number of dumped drop samples.
    """
    stats: dict = {
        "total": 0,
        "kept": 0,
        "dropped": 0,
        "by_reason": Counter(),
        "complete_clusters": 0,
        "incomplete_clusters": 0,
        "dumped": 0,
    }
    dumps: list[str] = []

    with open_out(output_path) as out:
        for line in iter_lines(input_path):
            stats["total"] += 1
            ok, reason, tags = validate_line(line)
            if ok and drop_incomplete and tags != TAGS:
                ok, reason = False, REASON_INCOMPLETE
            if not ok:
                stats["dropped"] += 1
                stats["by_reason"][reason] += 1
                if reason != REASON_EMPTY and len(dumps) < max_dump:
                    dumps.append(line.rstrip("\r\n")[:200].replace("\t", "|"))
                continue

            stats["kept"] += 1
            if tags == TAGS:
                stats["complete_clusters"] += 1
            else:
                stats["incomplete_clusters"] += 1
            out.write(line if line.endswith("\n") else line + "\n")

    stats["dumps"] = dumps
    return stats


def default_output_path(input_path: Path) -> Path:
    """``foo.txt.gz`` -> ``foo.clean.txt.gz`` (or ``foo.clean`` when no suffix)."""
    name = input_path.name
    for double_suffix in (".txt.gz", ".tar.gz"):
        if name.endswith(double_suffix):
            return input_path.with_name(name[: -len(double_suffix)] + f".clean{double_suffix}")
    if input_path.suffix:
        return input_path.with_name(name[: -len(input_path.suffix)] + ".clean" + input_path.suffix)
    return input_path.with_name(name + ".clean")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(description="Drop structurally malformed cluster lines.")
    parser.add_argument("input", help="Cluster corpus (.txt or .txt.gz), one cluster per line.")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: <input>.clean[.ext]).")
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Also drop lines missing one or more of the twelve script tags (reported by default).",
    )
    parser.add_argument("--max-dump", type=int, default=5, help="Max dropped-line samples to log (0 to disable).")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    stats = clean_file(input_path, output_path, drop_incomplete=args.drop_incomplete, max_dump=args.max_dump)

    log.info("total lines     : %d", stats["total"])
    log.info("kept            : %d", stats["kept"])
    log.info("dropped         : %d", stats["dropped"])
    for reason, count in sorted(stats["by_reason"].items()):
        log.info("  - %-16s: %d", reason, count)
    log.info("complete (12 tags): %d", stats["complete_clusters"])
    log.info(
        "incomplete        : %d%s",
        stats["incomplete_clusters"],
        " (kept; use --drop-incomplete)" if not args.drop_incomplete else "",
    )
    for d in stats["dumps"]:
        log.warning("dropped sample: %s", d)
    log.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
