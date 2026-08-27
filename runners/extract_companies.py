"""Extract company (applicant/assignee organization) name clusters for Odin training.

Reads the inventors project's event stream parquet (built from raw PatentsView
tables by /home/gfilion/inventors/data/build_stream.py), takes the raw
original-applicant organization strings (`applicant_org_name`, as written in the
documents, organization applicants only), groups them by canonical key, and
writes one cluster-corpus line per organization:

    la{Most Frequent Form}\tla{Second Form}\t...

Cells are the distinct raw strings as they appear in the documents (the
reference surfaces), ordered by descending occurrence count. This is the
company analogue of the inventor-name raw set; `generate_companies_phase_1.py`
turns these lines into full variant clusters.

The canonical key MUST stay in sync with the inventors project's
`filter/employer_keys.py` (`canonical_key`): it is the key space of that
project's employer prior `D`, so the VAE latent and the filter's discrete
employer state refer to the same entities.

The raw assignee string (PatentsView `g_assignee.raw_assignee_organization`)
is not in the stream yet; when it becomes available, run the same grouping
over it and merge into these clusters.

Needs pyarrow, which the odin project's own venv does not carry; run it from
the inventors project venv:

    cd /home/gfilion/inventors && uv run python \
        /home/gfilion/src/gui11aume/odin/runners/extract_companies.py
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Canonical key (mirror of the inventors project's filter/employer_keys.py)
# ---------------------------------------------------------------------------

# Legal-entity / jurisdictional tokens dropped when forming a key. They name
# the entity *type*, not the organization, and are the main source of spurious
# key splits beyond case and punctuation.
_SUFFIXES = frozenset(
    {
        "inc",
        "incorp",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "llp",
        "lp",
        "corp",
        "corporation",
        "co",
        "company",
        "gmbh",
        "ag",
        "sa",
        "se",
        "bv",
        "nv",
        "plc",
        "pty",
        "kabushiki",
        "kaisha",
        "kk",
        "srl",
        "spz",
        "zoo",
        "ooo",
        "oy",
        "ab",
    }
)

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def canonical_key(raw: str) -> str:
    """Canonical key of a raw company string (see module docstring for sync).

    Lowercases, replaces every non-alphanumeric run with a single space, drops
    the legal-entity tokens in `_SUFFIXES`, and joins the remaining tokens with
    single spaces. The empty string maps to the empty string (a missing
    employer, not a key).
    """
    if not raw:
        return ""
    tokens = [t for t in _PUNCT_RE.sub(" ", raw.lower()).split() if t not in _SUFFIXES]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Grouping / line building
# ---------------------------------------------------------------------------


def group_companies(rows: list[str]) -> dict[str, list[tuple[str, int]]]:
    """Group raw company strings by canonical key.

    Args:
        rows: Raw `applicant_org_name` values (empty strings skipped).

    Returns:
        Canonical key -> list of (raw string, occurrence count), ordered by
        descending count, then descending length, then raw string.
    """
    counts: dict[str, Counter[str]] = {}
    for raw in rows:
        raw = raw.strip()
        if not raw:
            continue
        key = canonical_key(raw)
        if not key:
            continue  # suffix-only string: no organization, not a key
        counts.setdefault(key, Counter())[raw] += 1
    out: dict[str, list[tuple[str, int]]] = {}
    for key, counter in counts.items():
        out[key] = sorted(counter.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return out


def build_line(raws: list[str]) -> str:
    """Cluster-corpus line for one organization: `la{...}` cells, tab-separated."""
    return "\t".join(f"la{{{raw}}}" for raw in raws)


def read_applicant_rows(parquet_path: str) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(parquet_path).read(columns=["applicant_org_name"])
    return [x for x in table.column("applicant_org_name").to_pylist() if x]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract company name clusters from the event stream parquet.")
    parser.add_argument(
        "input",
        nargs="?",
        default="/home/gfilion/inventors/data/event_stream.parquet",
        help="Event stream parquet (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="/mnt/nvme1/odin_companies.raw.txt.gz",
        help="Output file, .gz or plain text (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    print(f"reading {args.input}", file=sys.stderr)
    rows = read_applicant_rows(args.input)
    print(f"non-empty applicant rows: {len(rows):,}", file=sys.stderr)

    groups = group_companies(rows)
    n_cells = sum(len(v) for v in groups.values())
    print(f"organizations: {len(groups):,} | distinct raw strings: {n_cells:,}", file=sys.stderr)

    open_fn = gzip.open if args.output.endswith(".gz") else open
    with open_fn(args.output, "wt", encoding="utf-8") as fh:
        for key in groups:
            fh.write(build_line([raw for raw, _ in groups[key]]) + "\n")
    print(f"wrote {len(groups):,} lines to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
