#!/usr/bin/env python3
"""Direct (non-FFT) correlation computation for complex-rooted sequences.

This module computes
    gamma_N(m) = (1/N) * sum_{n=0}^{N-1} a(n) * conj(a(n+m))
directly from counts, and also via residue histograms:
    gamma_N(m) = sum_{r=0}^{k-1} p_r(m;N) * omega_k^r
where p_r are exact residue frequencies of
    D_m(n) = #(A,n) - #(A,n+m) (mod k).

The implementation uses the same leading-zero convention as experiments.py:
for max pattern length L, represent n as 0^(L-1)||bin(n).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments import count_pattern_set


def omega_k(k: int) -> complex:
    return complex(np.exp(2j * np.pi / k))


def residue_histogram(patterns: tuple[str, ...], k: int, n_max: int, m: int) -> np.ndarray:
    """Return histogram h[r] = # {n < n_max : D_m(n) = r (mod k)}."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    counts = np.array([count_pattern_set(n, patterns) for n in range(n_max + m)], dtype=np.int64)
    d = counts[:n_max] - counts[m : n_max + m]
    d_mod = np.mod(d, k)
    hist = np.bincount(d_mod, minlength=k).astype(np.int64)
    return hist


def gamma_from_hist(hist: np.ndarray, k: int, n_max: int) -> complex:
    """Compute gamma from residue histogram exactly (up to floating arithmetic in roots)."""
    w = omega_k(k)
    val = 0.0 + 0.0j
    for r, c in enumerate(hist.tolist()):
        if c:
            val += (c / n_max) * (w**r)
    return val


def gamma_direct_sum(patterns: tuple[str, ...], k: int, n_max: int, m: int) -> complex:
    """Compute gamma by direct summation of a(n) * conj(a(n+m))."""
    w = omega_k(k)
    counts = np.array([count_pattern_set(n, patterns) for n in range(n_max + m)], dtype=np.int64)
    diff_mod = np.mod(counts[:n_max] - counts[m : n_max + m], k)
    # a(n)*conj(a(n+m)) = w^{count(n)-count(n+m)} = w^{diff_mod}
    vals = np.array([w ** int(r) for r in diff_mod], dtype=np.complex128)
    return np.sum(vals) / float(n_max)


def gamma_prefix_direct(patterns: tuple[str, ...], k: int, n_max: int, m_max: int) -> list[complex]:
    out: list[complex] = [1.0 + 0.0j]
    for m in range(1, m_max + 1):
        hist = residue_histogram(patterns, k, n_max, m)
        out.append(gamma_from_hist(hist, k, n_max))
    return out


@dataclass
class DirectCorrelationRecord:
    m: int
    gamma_re: float
    gamma_im: float
    gamma_abs: float
    residue_hist: list[int]


def run_direct_report(
    patterns: tuple[str, ...],
    k: int,
    n_max: int,
    m_max: int,
) -> dict:
    rows: list[DirectCorrelationRecord] = []
    max_abs = 0.0
    for m in range(1, m_max + 1):
        hist = residue_histogram(patterns, k, n_max, m)
        g = gamma_from_hist(hist, k, n_max)
        # cross-check with direct sum path
        g2 = gamma_direct_sum(patterns, k, n_max, m)
        if abs(g - g2) > 1e-10:
            raise RuntimeError(f"hist/direct mismatch at m={m}: {g} vs {g2}")
        ga = abs(g)
        max_abs = max(max_abs, ga)
        rows.append(
            DirectCorrelationRecord(
                m=m,
                gamma_re=float(g.real),
                gamma_im=float(g.imag),
                gamma_abs=float(ga),
                residue_hist=hist.tolist(),
            )
        )
    return {
        "patterns": list(patterns),
        "k": k,
        "N": n_max,
        "m_max": m_max,
        "max_abs_nonzero": max_abs,
        "rows": [r.__dict__ for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+", required=True, help="Pattern list, e.g. 01 10 001 101")
    ap.add_argument("--k", type=int, required=True, help="Root modulus")
    ap.add_argument("--N", type=int, default=1 << 14, help="Prefix length for direct computation")
    ap.add_argument("--m-max", type=int, default=64, help="Maximum lag")
    ap.add_argument("--out", type=Path, default=Path("direct_correlation_results.json"))
    args = ap.parse_args()

    pats = tuple(args.patterns)
    rep = run_direct_report(pats, args.k, args.N, args.m_max)
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"Wrote {args.out}")
    print(f"max_abs_nonzero={rep['max_abs_nonzero']:.8f}")


if __name__ == "__main__":
    main()

