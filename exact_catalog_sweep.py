#!/usr/bin/env python3
"""Exact-residue catalog sweep across k values.

Computes max_{1<=m<=m_max} |gamma_{A,k,N}(m)| for each named pattern set
using exact residue counts of D_m(n) modulo k (no FFT).
"""

from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from experiments import count_pattern_set


def named_catalog() -> dict[str, tuple[str, ...]]:
    base = {
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

    # 15 empirically k=4-NC (but not k=2-NC) sequences from dichotomy census.
    k4_only = {
        "K4_01_001_101": ("01", "001", "101"),
        "K4_10_010_110": ("10", "010", "110"),
        "K4_11_011_111": ("11", "011", "111"),
        "K4_01_10": ("01", "10"),
        "K4_01_010_110": ("01", "010", "110"),
        "K4_10_001_101": ("10", "001", "101"),
        "K4_001_010_101_110": ("001", "010", "101", "110"),
        "K4_1_01_10": ("1", "01", "10"),
        "K4_1_01_001_101": ("1", "01", "001", "101"),
        "K4_1_01_010_110": ("1", "01", "010", "110"),
        "K4_1_10_001_101": ("1", "10", "001", "101"),
        "K4_1_10_010_110": ("1", "10", "010", "110"),
        "K4_1_11_011_111": ("1", "11", "011", "111"),
        "K4_001_010_100_111": ("001", "010", "100", "111"),
        "K4_001_011_100_110": ("001", "011", "100", "110"),
    }
    base.update(k4_only)
    return base


def symbolic_from_hist(hist: np.ndarray, n: int) -> str:
    terms: list[str] = []
    for r, c in enumerate(hist.tolist()):
        if c == 0:
            continue
        if r == 0:
            terms.append(str(c))
        elif c == 1:
            terms.append(f"zeta^{r}")
        else:
            terms.append(f"{c}*zeta^{r}")
    if not terms:
        terms = ["0"]
    return f"(1/{n}) * (" + " + ".join(terms) + ")"


def max_abs_from_counts(
    counts: np.ndarray, n: int, m_max: int, k: int
) -> tuple[float, int, np.ndarray, np.ndarray]:
    zeta = cmath.exp(2j * math.pi / k)
    max_abs = 0.0
    first_hist: np.ndarray | None = None
    max_hist: np.ndarray | None = None
    argmax_m = 1
    base = counts[:n]
    for m in range(1, m_max + 1):
        d = base - counts[m : n + m]
        residues = np.mod(d, k)
        hist = np.bincount(residues, minlength=k).astype(np.int64)
        if m == 1:
            first_hist = hist.copy()
        val = 0.0 + 0.0j
        for r, c in enumerate(hist.tolist()):
            if c:
                val += (c / n) * (zeta**r)
        abs_val = abs(val)
        if abs_val > max_abs:
            max_abs = abs_val
            max_hist = hist.copy()
            argmax_m = m
    if first_hist is None:
        first_hist = np.zeros(k, dtype=np.int64)
    if max_hist is None:
        max_hist = np.zeros(k, dtype=np.int64)
    return max_abs, argmax_m, max_hist, first_hist


def rational_projection_from_hist(hist: np.ndarray, n: int, k: int) -> dict[str, str] | None:
    if k == 2:
        c = hist.tolist() + [0, 0]
        return {"real": str(Fraction(c[0] - c[1], n)), "imag": "0"}
    if k == 4:
        c = hist.tolist() + [0, 0, 0, 0]
        return {
            "real": str(Fraction(c[0] - c[2], n)),
            "imag": str(Fraction(c[1] - c[3], n)),
        }
    return None


def parse_n_values(text: str) -> list[int]:
    vals = []
    for chunk in text.split(","):
        c = chunk.strip()
        if not c:
            continue
        vals.append(int(c))
    if not vals:
        raise ValueError("No N values provided")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1 << 14, help="Prefix length")
    ap.add_argument(
        "--n-values",
        type=str,
        default="",
        help="Optional comma-separated N values for multiscale exact run, e.g. 4096,8192,16384",
    )
    ap.add_argument("--m-max", type=int, default=64, help="Max lag")
    ap.add_argument("--k-min", type=int, default=2, help="Min k")
    ap.add_argument("--k-max", type=int, default=8, help="Max k")
    ap.add_argument("--threshold", type=float, default=5e-3, help="Empirical k-NC threshold")
    ap.add_argument("--out-json", type=Path, default=Path("exact_catalog_results.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("exact_catalog_results.csv"))
    args = ap.parse_args()

    n_values = parse_n_values(args.n_values) if args.n_values else [args.N]
    ks = list(range(args.k_min, args.k_max + 1))
    catalog = named_catalog()

    payload: dict[str, object] = {
        "config": {
            "N": args.N,
            "n_values": n_values,
            "m_max": args.m_max,
            "k_values": ks,
            "definition": "gamma_N(m) = (1/N) * sum_{n=0}^{N-1} a(n) * conj(a(n+m))",
            "method": "exact residue counts for D_m(n) mod k; cyclotomic reconstruction",
            "threshold_empirical_k_nc": args.threshold,
        },
        "catalog_scores_by_n": {},
        "first_lag_symbolic_by_n": {},
        "argmax_witness_by_n": {},
    }

    rows: list[dict[str, object]] = []
    rows_multiscale: list[dict[str, object]] = []

    for n in n_values:
        per_n_scores: dict[str, dict[str, float]] = {}
        per_n_first_sym: dict[str, dict[str, str]] = {}
        per_n_argmax: dict[str, dict[str, object]] = {}

        print(f"\nN={n}")
        for name, patterns in catalog.items():
            # Precompute counts once for all k and lags.
            counts = np.array([count_pattern_set(x, patterns) for x in range(n + args.m_max)], dtype=np.int64)
            row: dict[str, float] = {}
            sym: dict[str, str] = {}
            witnesses: dict[str, object] = {}

            print(f"{name}:", end=" ", flush=True)
            for k in ks:
                max_abs, argmax_m, argmax_hist, first_hist = max_abs_from_counts(counts, n, args.m_max, k)
                row[f"k={k}"] = float(max_abs)
                sym[f"k={k}"] = symbolic_from_hist(first_hist, n)
                witnesses[f"k={k}"] = {
                    "argmax_m": int(argmax_m),
                    "argmax_symbolic": symbolic_from_hist(argmax_hist, n),
                    "argmax_rational_projection": rational_projection_from_hist(argmax_hist, n, k),
                }
                print(f"k={k}:{max_abs:.4f}", end=" ", flush=True)

                entry = {
                    "name": name,
                    "patterns": " ".join(patterns),
                    "N": n,
                    "k": k,
                    "max_abs_nonzero": float(max_abs),
                    "empirical_k_nc": "Yes" if max_abs <= args.threshold else "No",
                    "argmax_m": int(argmax_m),
                    "argmax_symbolic": symbolic_from_hist(argmax_hist, n),
                }
                rows_multiscale.append(entry)
                if n == args.N:
                    rows.append(
                        {
                            "name": name,
                            "patterns": " ".join(patterns),
                            "k": k,
                            "max_abs_nonzero": float(max_abs),
                            "empirical_k_nc": "Yes" if max_abs <= args.threshold else "No",
                            "argmax_m": int(argmax_m),
                        }
                    )
            print()

            per_n_scores[name] = row
            per_n_first_sym[name] = sym
            per_n_argmax[name] = witnesses

        payload["catalog_scores_by_n"][str(n)] = per_n_scores
        payload["first_lag_symbolic_by_n"][str(n)] = per_n_first_sym
        payload["argmax_witness_by_n"][str(n)] = per_n_argmax

    # Backward-compatible aliases for single-scale readers.
    if len(n_values) == 1:
        n_key = str(n_values[0])
        payload["catalog_scores"] = payload["catalog_scores_by_n"][n_key]
        payload["first_lag_symbolic"] = payload["first_lag_symbolic_by_n"][n_key]
        payload["argmax_witness"] = payload["argmax_witness_by_n"][n_key]

    # Convergence diagnostics for last two scales.
    if len(n_values) >= 2:
        n_prev, n_last = n_values[-2], n_values[-1]
        conv: dict[str, dict[str, float]] = {}
        for name in catalog.keys():
            conv[name] = {}
            for k in ks:
                a = payload["catalog_scores_by_n"][str(n_prev)][name][f"k={k}"]
                b = payload["catalog_scores_by_n"][str(n_last)][name][f"k={k}"]
                ratio = (a / b) if b > 0 else float("inf")
                conv[name][f"k={k}"] = float(ratio)
        payload["last_step_halving_ratio"] = {
            "from_N": n_prev,
            "to_N": n_last,
            "ratios": conv,
        }

    args.out_json.write_text(json.dumps(payload, indent=2))
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "patterns", "k", "max_abs_nonzero", "empirical_k_nc", "argmax_m"]
        )
        writer.writeheader()
        writer.writerows(rows)

    out_multi = args.out_csv.with_name(args.out_csv.stem + "_multiscale.csv")
    with out_multi.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "patterns",
                "N",
                "k",
                "max_abs_nonzero",
                "empirical_k_nc",
                "argmax_m",
                "argmax_symbolic",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_multiscale)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {out_multi}")


if __name__ == "__main__":
    main()

