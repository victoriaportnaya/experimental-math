#!/usr/bin/env python3
"""Exact finite-prefix correlations for complex-rooted pattern sequences.

This module computes (for fixed N, m):

    gamma_{A,k,N}(m) = (1/N) * sum_{n=0}^{N-1} a(n) * conj(a(n+m))
                     = (1/N) * sum_{r=0}^{k-1} c_r(m) * omega_k^r

where c_r(m) are exact integer residue counts of
    D_m(n) = #(A,n) - #(A,n+m) (mod k),
and omega_k = exp(2*pi*i/k).

No FFT and no Monte Carlo are used.
The representation by counts c_r(m) is exact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from experiments import count_pattern_set


@dataclass
class ExactLagValue:
    m: int
    n: int
    k: int
    residue_counts: list[int]

    def symbolic(self) -> str:
        """Exact cyclotomic expression of gamma(m)."""
        terms: list[str] = []
        for r, c in enumerate(self.residue_counts):
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
        return f"(1/{self.n}) * (" + " + ".join(terms) + ")"

    def rational_projection(self) -> dict[str, str] | None:
        """Exact rational real/imag parts for k=2 or k=4."""
        if self.k == 2:
            c0 = self.residue_counts[0] if len(self.residue_counts) > 0 else 0
            c1 = self.residue_counts[1] if len(self.residue_counts) > 1 else 0
            re = Fraction(c0 - c1, self.n)
            return {"real": str(re), "imag": "0"}
        if self.k == 4:
            c = self.residue_counts + [0] * (4 - len(self.residue_counts))
            re = Fraction(c[0] - c[2], self.n)
            im = Fraction(c[1] - c[3], self.n)
            return {"real": str(re), "imag": str(im)}
        return None

    def approx_abs(self) -> float:
        import cmath
        import math

        zeta = cmath.exp(2j * math.pi / self.k)
        val = 0.0 + 0.0j
        for r, c in enumerate(self.residue_counts):
            if c:
                val += (c / self.n) * (zeta**r)
        return abs(val)

    def to_json(self) -> dict:
        return {
            "m": self.m,
            "residue_counts": self.residue_counts,
            "symbolic": self.symbolic(),
            "rational_projection": self.rational_projection(),
            "approx_abs": self.approx_abs(),
        }


def residue_counts_for_lag(patterns: tuple[str, ...], k: int, n: int, m: int) -> list[int]:
    if k < 2:
        raise ValueError("k must be >= 2")
    if m < 0:
        raise ValueError("m must be nonnegative")
    counts = [count_pattern_set(x, patterns) for x in range(n + m)]
    hist = [0] * k
    for x in range(n):
        d = counts[x] - counts[x + m]
        hist[d % k] += 1
    return hist


def exact_correlation_prefix(
    patterns: tuple[str, ...],
    k: int,
    n: int,
    m_max: int,
) -> dict:
    rows: list[ExactLagValue] = []
    max_abs = 0.0
    for m in range(1, m_max + 1):
        hist = residue_counts_for_lag(patterns, k, n, m)
        rec = ExactLagValue(m=m, n=n, k=k, residue_counts=hist)
        rows.append(rec)
        max_abs = max(max_abs, rec.approx_abs())
    return {
        "config": {
            "patterns": list(patterns),
            "k": k,
            "N": n,
            "m_max": m_max,
            "definition": "gamma_N(m) = (1/N) * sum_{n=0}^{N-1} a(n) * conj(a(n+m))",
            "representation": "exact residue-count cyclotomic expansion",
        },
        "max_abs_nonzero_approx": max_abs,
        "rows": [r.to_json() for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+", required=True, help="Pattern list, e.g. 11 1")
    ap.add_argument("--k", type=int, required=True, help="Root modulus")
    ap.add_argument("--N", type=int, default=1 << 14, help="Prefix length")
    ap.add_argument("--m-max", type=int, default=64, help="Maximum lag")
    ap.add_argument("--out", type=Path, default=Path("exact_correlation_results.json"))
    args = ap.parse_args()

    pats = tuple(args.patterns)
    data = exact_correlation_prefix(pats, args.k, args.N, args.m_max)
    args.out.write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.out}")
    print(f"max_abs_nonzero_approx={data['max_abs_nonzero_approx']:.12f}")


if __name__ == "__main__":
    main()

