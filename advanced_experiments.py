#!/usr/bin/env python3
"""Advanced experiments for noncorrelated binary pattern sequences.

Implements:
1) Exact (finite) noncorrelation decision for constant-length binary pattern sets
   using the recursive/basis-construction approach from Konieczny (2021), Section 4.
2) Symmetry-reduced census on constrained families (length 5/6, begin/end with 1).
3) Convergence-rate fitting max_{m<=M}|gamma_{N}(m)| ~ C N^{-alpha}.
4) Spectral proxies via periodogram statistics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from experiments import gamma_prefix, sequence_values


def words_of_length(ell: int) -> list[str]:
    return ["".join(bits) for bits in itertools.product("01", repeat=ell)]


def count_occurrences_len_fixed(text: str, patterns: set[str], ell: int) -> int:
    return sum(1 for i in range(len(text) - ell + 1 if len(text) >= ell else 0) if text[i : i + ell] in patterns)


def a_value(n: int, patterns: set[str], ell: int) -> int:
    padded = "0" * (ell - 1) + format(n, "b")
    c = count_occurrences_len_fixed(padded, patterns, ell)
    return -1 if (c % 2) else 1


def trailing_ones(n: int, ell: int) -> int:
    t = 0
    while t < ell and ((n >> t) & 1) == 1:
        t += 1
    return t


class RationalSpan:
    """Incremental linear-independence test over Q with row reduction."""

    def __init__(self, dim: int):
        self.dim = dim
        self.rows: list[tuple[int, list[Fraction]]] = []

    def add_if_independent(self, vec: list[Fraction]) -> bool:
        w = vec[:]
        for pivot, row in self.rows:
            if w[pivot] != 0:
                fac = w[pivot] / row[pivot]
                for j in range(pivot, self.dim):
                    if row[j] != 0:
                        w[j] -= fac * row[j]
        pivot_new = -1
        for j, x in enumerate(w):
            if x != 0:
                pivot_new = j
                break
        if pivot_new < 0:
            return False
        lead = w[pivot_new]
        for j in range(pivot_new, self.dim):
            if w[j] != 0:
                w[j] /= lead
        self.rows.append((pivot_new, w))
        self.rows.sort(key=lambda t: t[0])
        return True


@dataclass
class ExactDecisionResult:
    ell: int
    noncorrelated: bool
    basis_size: int
    witness_value: str | None

    def to_json(self) -> dict:
        return {
            "ell": self.ell,
            "noncorrelated": self.noncorrelated,
            "basis_size": self.basis_size,
            "witness_value": self.witness_value,
        }


def exact_decide_noncorr_constant_length(patterns: tuple[str, ...], ell: int) -> ExactDecisionResult:
    """Exact finite decision for constant-length pattern sets in base 2."""
    if any(len(p) != ell for p in patterns):
        raise ValueError("All patterns must have constant length ell.")
    pset = set(patterns)
    r = 1 << ell
    dim = 2 * r

    # h is 2^ell-periodic; precompute one period.
    h = [1] * r
    for n in range(r):
        h[n] = a_value(n, pset, ell) * a_value(n // 2, pset, ell)

    def h_val(n: int) -> int:
        return h[n % r]

    # Small-shift recursion from formulas (30),(31),(32), specialized to k=2.
    gamma1: list[Fraction] = [Fraction(0, 1) for _ in range(r)]
    order = list(range(r))
    order.sort(key=lambda x: (trailing_ones(x, ell), x))
    for rr in order:
        nu = trailing_ones(rr, ell)
        pref = Fraction(h_val(rr) * h_val(rr + 1), 1)
        if nu == 0:
            gamma1[rr] = pref
        elif nu < ell:
            a = rr // 2
            b = a + (1 << (ell - 1))
            gamma1[rr] = pref * Fraction(gamma1[a] + gamma1[b], 2)
        else:
            # rr = 2^ell - 1. Solve the implicit recurrence where rr appears
            # among children in (31): gamma = pref * (gamma_prev + gamma) / 2.
            idx = (1 << (ell - 1)) - 1 if ell >= 1 else 0
            gamma1[rr] = (pref * gamma1[idx]) / (Fraction(2, 1) - pref)

    def basis_idx(rr: int, e: int) -> int:
        return rr + e * r

    spans = {q: RationalSpan(dim) for q in range(r + 1)}
    queue: list[tuple[int, list[Fraction]]] = []

    # Initialization (34): for q=1..2^ell.
    for q in range(1, r + 1):
        vec = [Fraction(0, 1) for _ in range(dim)]
        for rr in range(r):
            vec[basis_idx(rr, 0)] = Fraction(1, 1)
        spans[q].add_if_independent(vec)
        queue.append((q, vec))

    u = 0
    while u < len(queue):
        q, w = queue[u]
        u += 1

        if q == 0:
            g0 = Fraction(0, 1)
            for rr in range(r):
                g0 += w[basis_idx(rr, 0)]  # gamma^{(r)}(0) = 1
                g0 += w[basis_idx(rr, 1)] * gamma1[rr]
            if g0 != 0:
                return ExactDecisionResult(ell=ell, noncorrelated=False, basis_size=len(queue), witness_value=str(g0))
            continue

        i = q & 1
        q_half = q // 2
        wprime: dict[int, list[Fraction]] = {}

        for rr in range(r):
            for e in (0, 1):
                c = w[basis_idx(rr, e)]
                if c == 0:
                    continue
                e_prime = (i + e + (rr & 1)) // 2
                fac = c * Fraction(h_val(rr) * h_val(rr + q + e), 2)

                r0 = rr // 2
                r1 = r0 + (1 << (ell - 1))
                q0 = q_half
                q1 = q_half + (1 << (ell - 1))

                for q_prime in (q0, q1):
                    if q_prime not in wprime:
                        wprime[q_prime] = [Fraction(0, 1) for _ in range(dim)]
                    wprime[q_prime][basis_idx(r0, e_prime)] += fac
                    wprime[q_prime][basis_idx(r1, e_prime)] += fac

        # Convert 1_{2^ell N0} into eta_0 + eta_{2^ell}.
        if 0 in wprime:
            add0 = wprime[0]
            if r not in wprime:
                wprime[r] = [Fraction(0, 1) for _ in range(dim)]
            for j in range(dim):
                wprime[r][j] += add0[j]

        for q_prime, vec in wprime.items():
            if spans[q_prime].add_if_independent(vec):
                queue.append((q_prime, vec))

    return ExactDecisionResult(ell=ell, noncorrelated=True, basis_size=len(queue), witness_value=None)


def lift_to_constant_length(patterns: tuple[str, ...], ell: int) -> tuple[str, ...]:
    """Convert an admissible pattern set to an equivalent constant-length set (binary)."""
    a = set(patterns)
    changed = True
    while changed:
        changed = False
        short = [w for w in a if len(w) < ell]
        if not short:
            break
        w = min(short, key=len)
        a.remove(w)
        # Symmetric difference with {0w,1w}.
        for ext in ("0" + w, "1" + w):
            if ext in a:
                a.remove(ext)
            else:
                a.add(ext)
        changed = True
    return tuple(sorted(w for w in a if len(w) == ell and set(w) != {"0"}))


def reverse_word(w: str) -> str:
    return w[::-1]


def complement_word(w: str) -> str:
    return "".join("1" if c == "0" else "0" for c in w)


def rotate_word(w: str, s: int) -> str:
    s %= len(w)
    return w[s:] + w[:s]


def transform_set(words: tuple[str, ...], op: str, shift: int = 0) -> tuple[str, ...]:
    out = list(words)
    if op in ("rev", "revcomp"):
        out = [reverse_word(w) for w in out]
    if op in ("comp", "revcomp"):
        out = [complement_word(w) for w in out]
    if op.startswith("rot"):
        out = [rotate_word(w, shift) for w in out]
    return tuple(sorted(out))


def orbit_heuristic(words: tuple[str, ...], ell: int) -> set[tuple[str, ...]]:
    ops = [("id", 0), ("rev", 0), ("comp", 0), ("revcomp", 0)]
    orbit: set[tuple[str, ...]] = set()
    for op, sh in ops:
        base = transform_set(words, op, sh)
        orbit.add(base)
        for s in range(ell):
            orbit.add(transform_set(base, "rot", s))
    return orbit


def constrained_universe(ell: int) -> list[str]:
    if ell < 2:
        return []
    return ["1" + "".join(mid) + "1" for mid in itertools.product("01", repeat=ell - 2)]


def _mid_orbit(subset_mids: tuple[str, ...], mid_len: int) -> set[tuple[str, ...]]:
    mids = list(subset_mids)
    out: set[tuple[str, ...]] = set()

    def apply(ms: list[str], op: str, shift: int = 0) -> list[str]:
        cur = ms[:]
        if op in ("rev", "revcomp"):
            cur = [m[::-1] for m in cur]
        if op in ("comp", "revcomp"):
            cur = ["".join("1" if c == "0" else "0" for c in m) for m in cur]
        if op.startswith("rot") and mid_len > 0:
            s = shift % mid_len
            cur = [m[s:] + m[:s] for m in cur]
        return cur

    for op in ("id", "rev", "comp", "revcomp"):
        base = sorted(apply(mids, op))
        out.add(tuple(base))
        for s in range(max(1, mid_len)):
            out.add(tuple(sorted(apply(base, "rot", s))))
    return out


def _bit_reverse(x: int, width: int) -> int:
    y = 0
    for _ in range(width):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def _bit_rotate_left(x: int, width: int, s: int) -> int:
    if width == 0:
        return 0
    s %= width
    mask = (1 << width) - 1
    return ((x << s) & mask) | ((x & mask) >> (width - s))


def _build_mid_transform_maps(mid_len: int) -> list[list[int]]:
    n = 1 << mid_len if mid_len > 0 else 1
    maps: list[list[int]] = []
    for op in ("id", "rev", "comp", "revcomp"):
        for s in range(max(1, mid_len)):
            mp = [0] * n
            for x in range(n):
                y = x
                if op in ("rev", "revcomp"):
                    y = _bit_reverse(y, mid_len)
                if op in ("comp", "revcomp"):
                    y = y ^ ((1 << mid_len) - 1)
                y = _bit_rotate_left(y, mid_len, s)
                mp[x] = y
            maps.append(mp)
    return maps


def _transform_subset_mask(mask: int, mp: list[int], n: int) -> int:
    out = 0
    for i in range(n):
        if (mask >> i) & 1:
            out |= 1 << mp[i]
    return out


def classify_constrained_with_symmetry(ell: int, max_orbits: int | None = None) -> dict:
    mids_universe = ["".join(bits) for bits in itertools.product("01", repeat=max(0, ell - 2))]
    mid_len = max(0, ell - 2)
    u = len(mids_universe)
    total = 1 << u
    maps = _build_mid_transform_maps(mid_len)

    canon_cache: dict[tuple[str, ...], bool] = {}
    sampled_orbits: set[tuple[str, ...]] = set()
    mismatch_orbits = 0
    orbits_evaluated = 0
    noncorr_total = 0

    truncated = False
    for subset_mask in range(total):
        orb_masks = {_transform_subset_mask(subset_mask, mp, u) for mp in maps}
        canon_mask = min(orb_masks)
        canon = tuple(
            sorted("1" + mids_universe[i] + "1" for i in range(u) if (canon_mask >> i) & 1)
        )
        if canon not in canon_cache:
            if max_orbits is not None and orbits_evaluated >= max_orbits:
                truncated = True
                break
            canon_cache[canon] = exact_decide_noncorr_constant_length(canon, ell).noncorrelated
            orbits_evaluated += 1
            # Audit invariance on a limited number of orbits.
            if len(sampled_orbits) < 24:
                vals = set()
                for m in orb_masks:
                    w = tuple(sorted("1" + mids_universe[i] + "1" for i in range(u) if (m >> i) & 1))
                    vals.add(exact_decide_noncorr_constant_length(w, ell).noncorrelated)
                if len(vals) > 1:
                    mismatch_orbits += 1
                sampled_orbits.add(canon)
        val = canon_cache[canon]
        if val:
            noncorr_total += 1

    return {
        "ell": ell,
        "universe_size": u,
        "total_sets": total,
        "evaluated_orbits": orbits_evaluated,
        "truncated": truncated,
        "symmetry_audit_orbits": len(sampled_orbits),
        "symmetry_mismatch_orbits": mismatch_orbits,
        "noncorrelated_count": noncorr_total,
    }


def fit_decay_alpha(patterns: tuple[str, ...], n_values: list[int], m_max: int) -> dict:
    y = []
    for n in n_values:
        vals = sequence_values(patterns, n)
        gam = gamma_prefix(vals, m_max)
        y.append(max(abs(v) for v in gam[1:]))
    x = np.log(np.array(n_values, dtype=np.float64))
    ly = np.log(np.array(y, dtype=np.float64))
    slope, intercept = np.polyfit(x, ly, 1)
    alpha = float(-slope)
    c = float(np.exp(intercept))
    return {
        "patterns": list(patterns),
        "n_values": n_values,
        "max_abs_values": y,
        "alpha": alpha,
        "C": c,
    }


def spectral_proxies(patterns: tuple[str, ...], n: int) -> dict:
    x = sequence_values(patterns, n).astype(np.float64)
    fft = np.fft.rfft(x)
    power = (np.abs(fft) ** 2) / (n * n)
    p = power[1:]  # drop DC
    p_sum = float(p.sum())
    if p_sum == 0.0:
        p_norm = np.zeros_like(p)
    else:
        p_norm = p / p_sum
    nz = p_norm > 0
    entropy = -float(np.sum(p_norm[nz] * np.log(p_norm[nz])))
    topk = int(min(10, len(p)))
    top_mass = float(np.sort(p_norm)[-topk:].sum()) if topk > 0 else 0.0
    return {
        "N": n,
        "patterns": list(patterns),
        "max_nonzero_periodogram": float(np.max(p) if len(p) else 0.0),
        "spectral_entropy": entropy,
        "top10_mass": top_mass,
        "l2_power": float(np.sum(p * p)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("advanced_results.json"))
    parser.add_argument("--m-max", type=int, default=128)
    args = parser.parse_args()

    exact_checks = {}
    named = {
        "Thue-Morse": ("1",),
        "Rudin-Shapiro": ("11",),
        "Rudin-Shapiro*Thue-Morse": ("11", "1"),
        "{101,111}": ("101", "111"),
    }
    for name, pats in named.items():
        ell = max(len(p) for p in pats)
        lifted = lift_to_constant_length(tuple(sorted(pats)), ell)
        exact_checks[name] = exact_decide_noncorr_constant_length(lifted, ell).to_json()

    constrained = {
        "ell5_begin_end_1": classify_constrained_with_symmetry(5),
        "ell6_begin_end_1": classify_constrained_with_symmetry(6, max_orbits=2000),
    }

    n_values = [1 << p for p in range(12, 17)]
    decay = {
        name: fit_decay_alpha(pats, n_values, args.m_max) for name, pats in named.items()
    }

    spectral = {
        name: spectral_proxies(pats, 1 << 15) for name, pats in named.items()
    }

    out = {
        "exact_decision_named": exact_checks,
        "constrained_classification": constrained,
        "decay_fits": decay,
        "spectral_proxies": spectral,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
