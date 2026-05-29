#!/usr/bin/env python3
"""Batch direct-correlation sweep for named pattern catalogs.

Computes max_{1<=m<=m_max} |gamma_N(m)| using direct (non-FFT) correlation
from direct_correlations.py for each named pattern set and k in a range.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from direct_correlations import run_direct_report


def named_catalog() -> dict[str, tuple[str, ...]]:
    return {
        "TM": ("1",),
        "RS": ("11",),
        "RS*TM": ("11", "1"),
        "A_{1,101,111}": ("1", "101", "111"),
        "{101,111}": ("101", "111"),
        "{01}": ("01",),
        "{10}": ("10",),
        "{01,10,11}": ("01", "10", "11"),
        "A_3": ("101", "111"),
        "A_4": ("1001", "1011", "1101", "1111"),
        "A_5": (
            "10001",
            "10011",
            "10101",
            "10111",
            "11001",
            "11011",
            "11101",
            "11111",
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1 << 14, help="Prefix length")
    ap.add_argument("--m-max", type=int, default=64, help="Maximum lag")
    ap.add_argument("--k-min", type=int, default=2, help="Minimum k")
    ap.add_argument("--k-max", type=int, default=8, help="Maximum k")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("direct_correlation_catalog.json"),
        help="JSON output path",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("direct_correlation_catalog.csv"),
        help="CSV output path",
    )
    args = ap.parse_args()

    catalog = named_catalog()
    ks = list(range(args.k_min, args.k_max + 1))

    payload: dict[str, object] = {
        "config": {
            "N": args.N,
            "m_max": args.m_max,
            "k_values": ks,
            "definition": "gamma_N(m) = (1/N) * sum_{n=0}^{N-1} a(n) * conj(a(n+m))",
            "method": "direct summation + residue histogram cross-check (no FFT)",
        },
        "catalog_scores": {},
    }

    rows: list[dict[str, object]] = []
    for name, patterns in catalog.items():
        row: dict[str, float] = {}
        print(f"{name}:", end=" ", flush=True)
        for k in ks:
            rep = run_direct_report(patterns, k, args.N, args.m_max)
            val = float(rep["max_abs_nonzero"])
            row[f"k={k}"] = val
            print(f"k={k}:{val:.4f}", end=" ", flush=True)
            rows.append(
                {
                    "name": name,
                    "patterns": " ".join(patterns),
                    "k": k,
                    "max_abs_nonzero": val,
                }
            )
        print()
        payload["catalog_scores"][name] = row

    args.out_json.write_text(json.dumps(payload, indent=2))
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "patterns", "k", "max_abs_nonzero"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()

