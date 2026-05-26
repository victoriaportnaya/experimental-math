#!/usr/bin/env python3
"""
Complex-rooted automatic sequences.

For a binary pattern set A, an integer k >= 2, and omega = exp(2*pi*i/k),
define the k-valued automatic sequence

    a_{A,k}(n) = omega^{#(A, n)}

where #(A, n) counts (with overlaps) occurrences of patterns in A in the
binary expansion of n.  The standard {+1,-1} Rudin-Shapiro / Thue-Morse
setting corresponds to k=2, omega = -1.

Noncorrelation for a complex-valued sequence means:

    gamma_a(m) = lim_{N->inf} (1/N) sum_{n=0}^{N-1} a(n) * conj(a(n+m)) = 0

for all m >= 1.  This is equivalent to the spectral measure of a being
absolutely continuous w.r.t. Lebesgue measure on T.

The correlation is purely real iff the sequence is real-valued (k=2), but
for k>2 it is complex; we track |gamma_a(m)| and max_m |gamma_a(m)|.

Key mathematical content
------------------------
* For fixed A, the map k -> max_m |gamma_{a_{A,k},N}(m)| characterises
  how the equidistribution of #(A,n)-#(A,n+m) mod k evolves with k.

* k-noncorrelation is STRICTLY STRONGER than 2-noncorrelation: the
  mod-k distribution of the difference process must be uniform, not just
  the mod-2 parity.

* For k | k': if the sequence is k'-noncorrelated, it is automatically
  k-noncorrelated (since mod-k uniformity follows from mod-k' uniformity
  when k | k').

* "Flat spectrum" in the complex sense means |hat{a}(xi)|^2 is constant,
  a classical property of Golay complementary pairs and generalised
  Rudin-Shapiro constructions.
"""

from __future__ import annotations

import itertools
import json
import math
import cmath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from experiments import count_pattern_set

ROOT = Path(__file__).parent
VIS  = ROOT / "main" / "visualizations"
VIS.mkdir(parents=True, exist_ok=True)

N_DEFAULT = 1 << 13   # 8192; large-N pass uses up to 2^16
M_MAX     = 64


# ─────────────────────────────────────────────────────────────────
# Core: build the complex sequence
# ─────────────────────────────────────────────────────────────────

def omega_sequence(patterns: tuple[str, ...], k: int, n_max: int) -> np.ndarray:
    """
    Return the complex array  a(n) = omega^{#(A,n)}  for n = 0..n_max-1,
    where omega = exp(2*pi*i/k).

    For k=2 this reduces to the standard {+1,-1} binary pattern sequence.
    The result dtype is complex128 (float for k=2).
    """
    omega = np.exp(2j * np.pi / k)
    # Vectorised: compute counts, then raise omega to those powers
    counts = np.array([count_pattern_set(n, patterns) for n in range(n_max)], dtype=np.int64)
    if k == 2:
        return np.where(counts % 2 == 0, 1.0, -1.0).astype(np.float64)
    return omega ** (counts % k)


# ─────────────────────────────────────────────────────────────────
# Correlation engine  (complex, linear autocorrelation via FFT)
# ─────────────────────────────────────────────────────────────────

def complex_gamma_prefix(a: np.ndarray, m_max: int) -> np.ndarray:
    """
    Compute  gamma(m) = (1/N) sum_{n=0}^{N-1} a(n) conj(a(n+m))
    for m = 0, 1, ..., m_max  via zero-padded FFT.

    Returns a real-valued array of |gamma(m)| for m=0..m_max.
    (gamma(0) = 1 always; for m>0 we report |gamma(m)|.)
    """
    n = len(a)
    # Linear cross-correlation: convolve a with conj(a) reversed.
    rev = np.conj(a[::-1])
    fft_len = 1
    need = 2 * n - 1
    while fft_len < need:
        fft_len <<= 1
    fa  = np.fft.fft(a,    fft_len)
    fb  = np.fft.fft(rev,  fft_len)
    conv = np.fft.ifft(fa * fb, fft_len)
    # The autocorrelation at lag m sits at index (n-1) - m
    zero_idx = n - 1
    out = np.empty(m_max + 1)
    out[0] = 1.0
    for m in range(1, m_max + 1):
        raw = conv[zero_idx - m]
        out[m] = abs(raw) / n        # full complex modulus of the correlation
    return out


def max_abs_corr_cx(a: np.ndarray, m_max: int = M_MAX) -> float:
    g = complex_gamma_prefix(a, m_max)
    return float(np.max(g[1:]))


def mean_sq_corr_cx(a: np.ndarray, m_max: int = M_MAX) -> float:
    g = complex_gamma_prefix(a, m_max)
    return float(np.mean(g[1:] ** 2))


# ─────────────────────────────────────────────────────────────────
# Experiment 1: k-sweep for known pattern sets
# ─────────────────────────────────────────────────────────────────

def run_k_sweep(n_max: int = N_DEFAULT) -> dict:
    """
    For each pattern set A in our catalog, compute max|gamma_{A,k}(m)|
    for k = 2, 3, 4, 5, 6, 7, 8 to find which k-values preserve
    noncorrelation.
    """
    catalog = {
        "TM":         ("1",),
        "RS":         ("11",),
        "RS*TM":      ("11","1"),
        "A_{1,101,111}":("1","101","111"),
        "{101,111}":  ("101","111"),
        "{01}":       ("01",),
        "{10}":       ("10",),
        "{01,10,11}": ("01","10","11"),
        "A_3":        ("101","111"),
        "A_4":        ("1001","1011","1101","1111"),
        "A_5":        tuple(f"1{m}1"
                       for m in ["000","001","010","011","100","101","110","111"]),
    }

    results: dict[str, dict] = {}
    for name, pats in catalog.items():
        print(f"  {name} ...", end=" ", flush=True)
        row: dict[str, float] = {}
        for k in range(2, 9):
            a = omega_sequence(pats, k, n_max)
            row[f"k={k}"] = max_abs_corr_cx(a)
        print(" | ".join(f"k{k}={row[f'k={k}']:.4f}" for k in range(2,9)))
        results[name] = row
    return results


# ─────────────────────────────────────────────────────────────────
# Experiment 2: find NEW noncorrelated (A, k) pairs
# ─────────────────────────────────────────────────────────────────

def run_new_nc_search(n_max: int = N_DEFAULT) -> dict:
    """
    Exhaustively test all binary pattern sets of length at most 3
    for k-noncorrelation with k in {3, 4, 6, 8}.
    Report any (A, k) pair that is k-noncorrelated but NOT 2-noncorrelated.
    """
    all_pats = []
    for ell in range(1, 4):
        for bits in itertools.product("01", repeat=ell):
            word = "".join(bits)
            if "1" in word:
                all_pats.append(word)

    # Build all non-empty subsets up to size 3
    subsets: list[tuple[str, ...]] = []
    for r in range(1, 4):
        subsets.extend(itertools.combinations(all_pats, r))

    print(f"  Searching {len(subsets)} subsets of size <= 3 ...")

    new_finds: list[dict] = []
    known_nc_2: list[tuple[str, ...]] = []

    for pats in subsets:
        # First check k=2 (real binary)
        a2 = omega_sequence(pats, 2, n_max)
        m2 = max_abs_corr_cx(a2)

        for k in [3, 4, 6, 8]:
            ak = omega_sequence(pats, k, n_max)
            mk = max_abs_corr_cx(ak)
            if mk < 0.004:
                entry = {
                    "patterns": list(pats),
                    "k": k,
                    "max_abs_k": float(mk),
                    "max_abs_2": float(m2),
                    "is_also_2_noncorr": m2 < 0.004,
                }
                new_finds.append(entry)

    # Deduplicate by (patterns, k)
    seen = set()
    unique = []
    for e in new_finds:
        key = (tuple(e["patterns"]), e["k"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    unique.sort(key=lambda x: (x["k"], x["max_abs_k"]))
    return {"candidates": unique, "total_subsets": len(subsets)}


# ─────────────────────────────────────────────────────────────────
# Experiment 3: k-noncorrelation hierarchy for fixed A
# ─────────────────────────────────────────────────────────────────

def run_hierarchy_study(n_max: int = N_DEFAULT) -> dict:
    """
    For sequences already known to be 2-noncorrelated, find the
    largest k for which k-noncorrelation holds.

    Also tests: if a sequence is k-noncorrelated for k = p*q (composite),
    does it follow that it is p- and q-noncorrelated?
    """
    nc2_catalog = {
        "RS":         ("11",),
        "RS*TM":      ("11","1"),
        "{01}":       ("01",),
        "{10}":       ("10",),
        "{01,10,11}": ("01","10","11"),
        "A_3":        ("101","111"),
        "A_4":        ("1001","1011","1101","1111"),
        "{1,101,111}":("1","101","111"),
        "{01,101,111}":("01","101","111"),
        "{10,101,111}":("10","101","111"),
    }

    results = {}
    ks_to_test = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24]

    for name, pats in nc2_catalog.items():
        print(f"  {name} ...", end=" ", flush=True)
        row = {}
        nc_ks = []
        for k in ks_to_test:
            a = omega_sequence(pats, k, n_max)
            m = max_abs_corr_cx(a)
            row[f"k={k}"] = float(m)
            if m < 0.004:
                nc_ks.append(k)
        results[name] = {"scores": row, "noncorr_ks": nc_ks}
        print(f"  NC at k={nc_ks}")

    return results


# ─────────────────────────────────────────────────────────────────
# Experiment 4: large-N verification for best hits
# ─────────────────────────────────────────────────────────────────

def run_largescale(candidates: list[tuple[tuple[str,...], int]],
                   exps: list[int] | None = None) -> dict:
    """
    Verify candidates at multiple N values to check O(1/N) decay.
    candidates: list of (patterns_tuple, k).
    """
    if exps is None:
        exps = [12, 13, 14, 15, 16]
    results = {}
    for pats, k in candidates:
        key = f"k={k}_{'+'.join(pats)}"
        print(f"  {key} ...", end=" ", flush=True)
        row: dict[str, float] = {}
        for exp in exps:
            N = 1 << exp
            a = omega_sequence(pats, k, N)
            m = max_abs_corr_cx(a)
            row[f"N=2^{exp}"] = float(m)
        # compute halving ratio
        if f"N=2^{exps[-2]}" in row and f"N=2^{exps[-1]}" in row:
            v1, v2 = row[f"N=2^{exps[-2]}"], row[f"N=2^{exps[-1]}"]
            row["halving_ratio"] = float(v1 / v2) if v2 > 0 else float("inf")
        results[key] = row
        vals_str = "  ".join(f"N=2^{e}:{row[f'N=2^{e}']:.5f}" for e in exps)
        print(f"  ratio={row.get('halving_ratio','?'):.2f}  {vals_str}")
    return results


# ─────────────────────────────────────────────────────────────────
# Experiment 5: correlation profiles at fixed N
# ─────────────────────────────────────────────────────────────────

def correlation_profiles(candidates: list[tuple[tuple[str,...], int]],
                          n_max: int = 1 << 15) -> dict:
    results = {}
    for pats, k in candidates:
        key = f"k={k}_{'+'.join(pats)}"
        a = omega_sequence(pats, k, n_max)
        g = complex_gamma_prefix(a, M_MAX)
        results[key] = {
            "patterns": list(pats), "k": k,
            "max_abs": float(max(g[1:])),
            "mean_sq": float(np.mean(g[1:] ** 2)),
            "corr_profile": [float(x) for x in g[1:M_MAX+1]],
        }
    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main(n_max: int = N_DEFAULT) -> None:
    out: dict = {}

    print(f"\n{'='*60}")
    print(f"Complex-rooted sequence experiments  (N={n_max})")
    print(f"{'='*60}\n")

    # --- 1. k-sweep for known catalog ---
    print("[1/4] k-sweep over known catalog:")
    out["k_sweep"] = run_k_sweep(n_max)

    # --- 2. Find new (A, k) NC pairs ---
    print("\n[2/4] Searching for new k-NC sequences (len<=3 patterns):")
    search_res = run_new_nc_search(n_max)
    out["new_nc_search"] = search_res
    cands = search_res["candidates"]
    print(f"  Found {len(cands)} candidates:")
    for c in cands:
        flag = "" if c["is_also_2_noncorr"] else "  *** NEW (not 2-NC) ***"
        print(f"    k={c['k']}  {c['patterns']}  "
              f"max_k={c['max_abs_k']:.4f}  max_2={c['max_abs_2']:.4f}{flag}")

    # --- 3. Hierarchy study for known 2-NC sequences ---
    print("\n[3/4] k-noncorrelation hierarchy (known 2-NC sequences):")
    out["hierarchy"] = run_hierarchy_study(n_max)

    # --- 4. Large-N check for the most interesting hits ---
    print("\n[4/4] Large-N verification:")
    # Collect the most interesting candidates from the sweep + search
    top_candidates: list[tuple[tuple[str,...], int]] = []

    # From hierarchy: sequences NC at k>2
    for name, rec in out["hierarchy"].items():
        nc_ks = rec["noncorr_ks"]
        pats_map = {
            "RS":          ("11",),
            "RS*TM":       ("11","1"),
            "{01}":        ("01",),
            "{10}":        ("10",),
            "{01,10,11}":  ("01","10","11"),
            "A_3":         ("101","111"),
            "A_4":         ("1001","1011","1101","1111"),
            "{1,101,111}": ("1","101","111"),
            "{01,101,111}":("01","101","111"),
            "{10,101,111}":("10","101","111"),
        }
        pats = pats_map.get(name)
        if pats is None:
            continue
        for k in nc_ks:
            if k != 2:  # k=2 already studied; focus on k>2
                if (pats, k) not in top_candidates:
                    top_candidates.append((pats, k))

    # Add novel finds from search (not 2-NC)
    for c in cands:
        if not c["is_also_2_noncorr"]:
            entry = (tuple(c["patterns"]), c["k"])
            if entry not in top_candidates:
                top_candidates.append(entry)

    if top_candidates:
        out["large_n_verification"] = run_largescale(
            top_candidates, exps=[12, 13, 14, 15, 16]
        )
        out["correlation_profiles"] = correlation_profiles(
            [(p, k) for p, k in top_candidates
             if out["large_n_verification"]
                .get(f"k={k}_{'+'+''.join(p) if isinstance(p,tuple) else p}", {})
                .get("halving_ratio", 0) > 1.8],
            n_max=1 << 15,
        )
    else:
        print("  No super-candidates found in this run.")

    # Save
    out_path = ROOT / "complex_rooted_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=N_DEFAULT)
    args = p.parse_args()
    main(args.N)
