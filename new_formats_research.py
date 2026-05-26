#!/usr/bin/env python3
"""
Research into new sequence formats that may be noncorrelated.

Seven distinct families are explored:

  1. Base-k pattern sequences (k = 3, 4)
     Rudin-Shapiro analogues over larger alphabets: count overlapping
     occurrences of a digit-pattern in the base-k expansion of n.

  2. Reed-Muller / polynomial-over-F2 block sequences
     Evaluate a polynomial f : F_2^m -> F_2 on non-overlapping m-bit
     blocks of n and exponentiate: a(n) = (-1)^{sum_j f(block_j)}.
     Degree 1 = Thue-Morse type; degree 2 = Rudin-Shapiro type; degree 3+ new.

  3. Digit-interaction sequences
     Build a derived sequence from the bitwise differences / XOR / carry
     of consecutive digits of n, then apply a standard pattern count.

  4. Substitution / morphic sequences
     Sequences defined by letter-to-letter morphisms.  The fixed point of
     each morphism is mapped to {+1,-1} and its correlations are measured.

  5. Interleaved constructions
     a(2n) = f(n),  a(2n+1) = g(n).  We test all pairs (f, g) from the
     known noncorrelated catalog and look for flat spectra.

  6. Product / XOR combinations of pattern sequences
     a_{A+B}(n) = (-1)^{#(A,n) + #(B,n)} = a_A(n)*a_B(n).
     Both same-length and cross-length pairs are tested.

  7. Generalized digit-run sequences
     Count runs of a fixed digit d of length exactly r (or at-least r)
     in the base-k expansion of n.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from experiments import gamma_prefix, sequence_values

ROOT = Path(__file__).parent
N_DEFAULT = 1 << 13  # 8192
M_MAX = 64


# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────

def max_abs_corr(values: np.ndarray, m_max: int = M_MAX) -> float:
    g = gamma_prefix(values, m_max)
    return float(max(abs(g[m]) for m in range(1, m_max + 1)))


def mean_sq_corr(values: np.ndarray, m_max: int = M_MAX) -> float:
    g = gamma_prefix(values, m_max)
    return float(np.mean([g[m] ** 2 for m in range(1, m_max + 1)]))


def corr_vector(values: np.ndarray, m_max: int = M_MAX) -> list[float]:
    return gamma_prefix(values, m_max)[1:]  # lags 1..m_max


# ─────────────────────────────────────────────────────────────────
# 1. Base-k pattern sequences
# ─────────────────────────────────────────────────────────────────

def to_base_k(n: int, k: int, min_digits: int = 1) -> list[int]:
    """Return base-k digits of n, most-significant first, padded to min_digits."""
    if n == 0:
        digits = [0]
    else:
        digits = []
        tmp = n
        while tmp:
            digits.append(tmp % k)
            tmp //= k
        digits.reverse()
    while len(digits) < min_digits:
        digits.insert(0, 0)
    return digits


def count_basek_pattern(n: int, pattern: tuple[int, ...], k: int) -> int:
    """Count overlapping occurrences of pattern (tuple of digits) in base-k repr of n."""
    ell = len(pattern)
    digits = to_base_k(n, k, min_digits=ell)
    # pad with (ell-1) leading zeros so short n still sees the pattern
    padded = [0] * (ell - 1) + digits
    return sum(
        1 for i in range(len(padded) - ell + 1)
        if tuple(padded[i: i + ell]) == pattern
    )


def basek_sequence(pattern_set: list[tuple[int, ...]], k: int, n_max: int) -> np.ndarray:
    """a(n) = (-1)^{sum over patterns p in A of count_basek(p, n)}."""
    vals = np.empty(n_max, dtype=np.int8)
    for n in range(n_max):
        total = sum(count_basek_pattern(n, p, k) for p in pattern_set)
        vals[n] = -1 if total % 2 else 1
    return vals


def run_basek_experiments(n_max: int = N_DEFAULT) -> dict:
    results = {}

    # --- Base-3 candidates ---
    # Ternary Rudin-Shapiro: count occurrences of "11" (digits 1,1) in base-3
    for pat_label, pat in [
        ("(1,1)", ((1, 1),)),
        ("(2,2)", ((2, 2),)),
        ("(1,1)+(2,2)", ((1, 1), (2, 2))),
        ("(1,2)+(2,1)", ((1, 2), (2, 1))),
        ("(0,1)+(0,2)", ((0, 1), (0, 2))),
        ("(1,1)+(1,2)+(2,1)+(2,2)", ((1, 1), (1, 2), (2, 1), (2, 2))),
        # length-3 ternary patterns
        ("(1,1,1)", ((1, 1, 1),)),
        ("(2,2,2)", ((2, 2, 2),)),
        ("(1,1,1)+(2,2,2)", ((1, 1, 1), (2, 2, 2))),
    ]:
        v = basek_sequence(list(pat), k=3, n_max=n_max)
        results[f"base3_{pat_label}"] = {
            "k": 3, "patterns": [list(p) for p in pat],
            "max_abs": max_abs_corr(v),
            "mean_sq": mean_sq_corr(v),
            "corr": corr_vector(v),
        }

    # --- Base-4 candidates ---
    # "11" in base 4 = two consecutive 1-digits
    for pat_label, pat in [
        ("(1,1)", ((1, 1),)),
        ("(2,2)", ((2, 2),)),
        ("(3,3)", ((3, 3),)),
        ("(1,1)+(2,2)+(3,3)", ((1, 1), (2, 2), (3, 3))),
        # all non-zero same-digit pairs: Rudin-Shapiro for base-4
        ("all_same_nonzero_pairs", tuple((d, d) for d in range(1, 4))),
        # cross pairs
        ("(1,2)+(2,1)+(1,3)+(3,1)+(2,3)+(3,2)",
         ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))),
    ]:
        v = basek_sequence(list(pat), k=4, n_max=n_max)
        results[f"base4_{pat_label}"] = {
            "k": 4, "patterns": [list(p) for p in pat],
            "max_abs": max_abs_corr(v),
            "mean_sq": mean_sq_corr(v),
            "corr": corr_vector(v),
        }

    return results


# ─────────────────────────────────────────────────────────────────
# 2. Reed-Muller / polynomial-over-F2 block sequences
# ─────────────────────────────────────────────────────────────────

def eval_poly_f2(coeff_mask: int, bits: tuple[int, ...]) -> int:
    """
    Evaluate a multilinear polynomial over F_2 on `bits`.

    coeff_mask encodes which monomials are present in the polynomial.
    Bit i of coeff_mask is 1 iff the monomial corresponding to the
    i-th subset of variables (in binary enumeration of subsets) is
    present.  Singleton subsets = degree-1 terms; pairs = degree-2;
    the empty set (index 0) = constant term.

    E.g. for m=2 variables (x0, x1):
      index 0: constant 1
      index 1: x0
      index 2: x1
      index 3: x0*x1
    coeff_mask = 0b1000 = 8 means f = x0*x1.
    """
    m = len(bits)
    val = 0
    for subset_idx in range(1 << m):
        if (coeff_mask >> subset_idx) & 1:
            mono = 1
            for j in range(m):
                if (subset_idx >> j) & 1:
                    mono &= bits[j]
            val ^= mono
    return val


def rm_block_sequence(coeff_mask: int, m: int, n_max: int) -> np.ndarray:
    """
    Non-overlapping block Reed-Muller sequence.

    a(n) = (-1)^{ sum_{j=0}^{floor(b/m)-1} f(bit_{jm}, ..., bit_{jm+m-1}) }
    where b is the number of bits of n (padded to a multiple of m).
    """
    vals = np.empty(n_max, dtype=np.int8)
    for n in range(n_max):
        if n == 0:
            bit_len = m
        else:
            bit_len = int(math.log2(n)) + 1
        # pad up to multiple of m
        total_bits = math.ceil(max(bit_len, m) / m) * m
        bits_all = [(n >> (total_bits - 1 - j)) & 1 for j in range(total_bits)]
        exponent = 0
        for j in range(0, total_bits, m):
            block = tuple(bits_all[j: j + m])
            exponent ^= eval_poly_f2(coeff_mask, block)
        vals[n] = -1 if exponent else 1
    return vals


def run_rm_experiments(n_max: int = N_DEFAULT) -> dict:
    """
    Enumerate multilinear polynomials over F_2 for m=2,3,4 variables
    and test their block sequences for noncorrelation.
    """
    results = {}

    for m in [2, 3, 4]:
        n_poly = 1 << (1 << m)  # 2^{2^m} total polynomials
        # For m=4 this is 65536, too many to test all; sample interesting ones
        candidates: list[tuple[str, int]] = []

        if m == 2:
            # 16 polynomials, test all non-trivial ones
            for mask in range(1, n_poly):
                # degree: highest-order subset present
                deg = max(bin(s).count("1") for s in range(1, 1 << m) if (mask >> s) & 1) if mask > 1 else 0
                label = f"m{m}_mask{mask:04b}_deg{deg}"
                candidates.append((label, mask))
        elif m == 3:
            # 256 polynomials; test pure monomials and specific low-weight ones
            # subsets of {x0,x1,x2}
            for mask in range(1, n_poly):
                active = [s for s in range(1, 1 << m) if (mask >> s) & 1]
                if len(active) <= 3:  # at most 3 monomials
                    deg = max(bin(s).count("1") for s in active) if active else 0
                    label = f"m{m}_mask{mask:08b}_deg{deg}"
                    candidates.append((label, mask))
        elif m == 4:
            # Test: pure degree-1 (Thue-Morse), pure degree-2, pure degree-3,
            # pure degree-4 (x0x1x2x3), and the Rudin-Shapiro equivalent
            special = {
                "all_deg1": sum(1 << (1 << j) for j in range(4)),    # x0+x1+x2+x3
                "x0x1_only": 1 << 0b0011,                             # x0*x1
                "x0x1+x2x3": (1 << 0b0011) | (1 << 0b1100),          # x0x1 + x2x3
                "all_pairs": sum(1 << (3 << j) for j in range(3)),    # not right but approx
                "x0x1x2x3": 1 << 0b1111,                              # degree-4 monomial
                "x0x1+x0x1x2": (1 << 0b0011) | (1 << 0b0111),
                "x0x1x2+x1x2x3": (1 << 0b0111) | (1 << 0b1110),
            }
            # correct all_pairs: all pairs from {0,1,2,3}
            all_pairs_mask = 0
            for i in range(4):
                for j in range(i + 1, 4):
                    all_pairs_mask |= 1 << ((1 << i) | (1 << j))
            special["all_deg2_pairs"] = all_pairs_mask
            for label, mask in special.items():
                candidates.append((f"m{m}_{label}", mask))

        for label, mask in candidates:
            v = rm_block_sequence(mask, m, n_max)
            results[f"rm_{label}"] = {
                "m": m, "coeff_mask": mask,
                "max_abs": max_abs_corr(v),
                "mean_sq": mean_sq_corr(v),
                "corr": corr_vector(v),
            }

    return results


# ─────────────────────────────────────────────────────────────────
# 3. Digit-interaction sequences
# ─────────────────────────────────────────────────────────────────

def digit_interaction_sequence(interaction: str, n_max: int) -> np.ndarray:
    """
    Build a {±1} sequence from the *derived* binary string of n via a
    digit-to-digit interaction operation.

    Interactions supported:
      "xor_consec"   : d_j = bit_j XOR bit_{j+1}  (carry sequence)
      "and_consec"   : d_j = bit_j AND bit_{j+1}  (overlap sequence)
      "majority3"    : d_j = majority(bit_{j-1}, bit_j, bit_{j+1})
      "parity_prefix": d_j = XOR(bit_0, ..., bit_j)  = running parity

    The derived sequence d_j is used to define a_A for A = {1} (Thue-Morse
    on d) and A = {11} (Rudin-Shapiro on d).
    """
    vals_xor = np.empty(n_max, dtype=np.int8)
    vals_and = np.empty(n_max, dtype=np.int8)
    vals_maj = np.empty(n_max, dtype=np.int8)
    vals_par = np.empty(n_max, dtype=np.int8)

    for n in range(n_max):
        # build bit string (MSB first, padded to at least 2 bits)
        b = n.bit_length()
        bits = [(n >> (b - 1 - j)) & 1 for j in range(b)] if b >= 2 else [0, 0]
        if len(bits) < 2:
            bits = [0] + bits

        # xor_consec derived string
        d_xor = [bits[j] ^ bits[j + 1] for j in range(len(bits) - 1)]
        # and_consec
        d_and = [bits[j] & bits[j + 1] for j in range(len(bits) - 1)]
        # parity_prefix
        parity = 0
        d_par = []
        for bit in bits:
            parity ^= bit
            d_par.append(parity)
        # majority3 (pad with 0 at boundaries)
        pb = [0] + bits + [0]
        d_maj = [1 if (pb[j] + pb[j + 1] + pb[j + 2]) >= 2 else 0
                 for j in range(len(bits))]

        # Rudin-Shapiro on derived: count "11" overlapping
        def count_11(d: list[int]) -> int:
            return sum(1 for j in range(len(d) - 1) if d[j] == 1 and d[j + 1] == 1)

        vals_xor[n] = -1 if count_11(d_xor) % 2 else 1
        vals_and[n] = -1 if count_11(d_and) % 2 else 1
        vals_maj[n] = -1 if count_11(d_maj) % 2 else 1
        # Thue-Morse on parity_prefix derived
        vals_par[n] = -1 if sum(d_par) % 2 else 1

    seqs = {
        "rs_on_xor_consecutive": vals_xor,
        "rs_on_and_consecutive": vals_and,
        "rs_on_majority3": vals_maj,
        "tm_on_running_parity": vals_par,
    }
    results = {}
    for label, v in seqs.items():
        results[label] = {
            "max_abs": max_abs_corr(v),
            "mean_sq": mean_sq_corr(v),
            "corr": corr_vector(v),
        }
    return results


# ─────────────────────────────────────────────────────────────────
# 4. Substitution / morphic sequences
# ─────────────────────────────────────────────────────────────────

def morphic_fixed_point(morphism: dict[int, list[int]], seed: int,
                         n_max: int) -> np.ndarray:
    """
    Iterate morphism until we have >= n_max symbols, return the fixed point
    mapped to {+1,-1} by symbol -> (-1)^symbol.
    """
    word = [seed]
    while len(word) < n_max:
        word = [c for sym in word for c in morphism[sym]]
    arr = np.array([(1 if sym == 0 else -1) for sym in word[:n_max]], dtype=np.int8)
    return arr


def run_morphic_experiments(n_max: int = N_DEFAULT) -> dict:
    """
    Test several well-known and new substitution systems.

    Alphabet is {0, 1, ...} and the {+1,-1} assignment is symbol -> (-1)^symbol.
    """
    results = {}

    morphisms: dict[str, tuple[dict[int, list[int]], int]] = {
        # Classical Thue-Morse: 0->01, 1->10
        "thue_morse": ({0: [0, 1], 1: [1, 0]}, 0),

        # Rudin-Shapiro via 4-symbol morphism (Allouche-Shallit encoding):
        # symbols 0..3 where parity of symbol index gives {+1,-1}
        # 0->0011, 1->0010, 2->1101, 3->1100  ... produces RS after projection
        # (Use the standard 4-letter construction)
        "rudin_shapiro_4sym": ({
            0: [0, 1],  # maps to +1
            1: [0, 2],  # maps to +1
            2: [3, 1],  # maps to -1
            3: [3, 2],  # maps to -1
        }, 0),  # project: 0,1 -> +1; 2,3 -> -1  (handled below)

        # Period-doubling sequence: 0->01, 1->00
        "period_doubling": ({0: [0, 1], 1: [0, 0]}, 0),

        # Paperfolding (regular): 0->010, 1->011 ... standard 3-letter
        # In {+1,-1} by 0->+1, 1->-1
        "paper_folding": ({0: [0, 1, 0], 1: [0, 1, 1]}, 0),

        # Fibonacci morphism (binary): 0->01, 1->0
        "fibonacci": ({0: [0, 1], 1: [0]}, 0),

        # Tribonacci: 0->01, 1->02, 2->0
        "tribonacci": ({0: [0, 1], 1: [0, 2], 2: [0]}, 0),

        # "Generalized RS" type 4-symbol:
        # 0->01, 1->23, 2->01, 3->23  (4-symbol, halving the correlation)
        "gen_rs_v1": ({0: [0, 1], 1: [2, 3], 2: [0, 1], 3: [2, 3]}, 0),

        # 4-symbol: 0->01, 1->03, 2->21, 3->23
        "gen_rs_v2": ({0: [0, 1], 1: [0, 3], 2: [2, 1], 3: [2, 3]}, 0),

        # 0->0111, 1->0100 (Toeplitz-like)
        "toeplitz_like": ({0: [0, 1, 1, 1], 1: [0, 1, 0, 0]}, 0),

        # New: 3-symbol "balanced" morphism
        # 0->012, 1->120, 2->201  (cyclic permutation, each letter appears once)
        "cyclic_3sym": ({0: [0, 1, 2], 1: [1, 2, 0], 2: [2, 0, 1]}, 0),

        # 4-symbol "flat" construction (attempt to generalize Golay pairs)
        # 0->0123, 1->0132, 2->2301, 3->2310
        "golay_4sym": ({
            0: [0, 1, 2, 3],
            1: [0, 1, 3, 2],
            2: [2, 3, 0, 1],
            3: [2, 3, 1, 0],
        }, 0),
    }

    for name, (morph, seed) in morphisms.items():
        try:
            word = [seed]
            while len(word) < n_max:
                word = [c for sym in word for c in morph[sym]]
            word = word[:n_max]

            # For "rudin_shapiro_4sym": 0,1 -> +1; 2,3 -> -1
            if name == "rudin_shapiro_4sym":
                v = np.array([(1 if s in (0, 1) else -1) for s in word], dtype=np.int8)
            else:
                # generic: (-1)^symbol  (for multi-symbol alphabets, parity of symbol index)
                v = np.array([1 if s % 2 == 0 else -1 for s in word], dtype=np.int8)

            results[f"morphic_{name}"] = {
                "max_abs": max_abs_corr(v),
                "mean_sq": mean_sq_corr(v),
                "corr": corr_vector(v),
            }
        except Exception as e:
            results[f"morphic_{name}"] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────
# 5. Interleaved constructions
# ─────────────────────────────────────────────────────────────────

def interleave(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """a(2n) = f(n), a(2n+1) = g(n)."""
    n = min(len(f), len(g))
    out = np.empty(2 * n, dtype=np.int8)
    out[0::2] = f[:n]
    out[1::2] = g[:n]
    return out


def run_interleaved_experiments(n_max: int = N_DEFAULT) -> dict:
    half = n_max // 2
    # Named base sequences
    tm   = sequence_values(("1",),       n_max)
    rs   = sequence_values(("11",),      n_max)
    rtm  = sequence_values(("11", "1"),  n_max)
    a101 = sequence_values(("101", "111"), n_max)
    ones = np.ones(n_max, dtype=np.int8)

    named = {"TM": tm, "RS": rs, "RS*TM": rtm, "A101111": a101, "ONE": ones}

    results = {}
    for nf, f in named.items():
        for ng, g in named.items():
            if nf > ng:
                continue  # avoid duplicates
            key = f"interleave_{nf}_and_{ng}"
            v = interleave(f[:half], g[:half])
            results[key] = {
                "f": nf, "g": ng,
                "max_abs": max_abs_corr(v),
                "mean_sq": mean_sq_corr(v),
                "corr": corr_vector(v),
            }

    return results


# ─────────────────────────────────────────────────────────────────
# 6. Product / XOR combinations of pattern sequences
# ─────────────────────────────────────────────────────────────────

def run_product_experiments(n_max: int = N_DEFAULT) -> dict:
    """
    Test a_A * a_B = a_{A sym-diff B} for pairs of pattern sets.
    Also test direct XOR: count(A,n) + count(B,n) mod 2.
    """
    named_pats: list[tuple[str, tuple[str, ...]]] = [
        ("TM",          ("1",)),
        ("RS",          ("11",)),
        ("RS*TM",       ("11", "1")),
        ("A101111",     ("101", "111")),
        ("A_2",         ("11",)),
        ("A_3",         ("101", "111")),
        ("A_4",         ("1001", "1011", "1101", "1111")),
    ]

    # Cross-length products: A (len l1) with B (len l2), l1 != l2
    cross_pairs: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = []
    for (n1, p1), (n2, p2) in itertools.combinations(named_pats, 2):
        cross_pairs.append((n1, p1, n2, p2))

    results = {}
    for n1, p1, n2, p2 in cross_pairs:
        # product sequence
        v1 = sequence_values(p1, n_max)
        v2 = sequence_values(p2, n_max)
        vprod = (v1 * v2).astype(np.int8)
        key = f"product_{n1}_x_{n2}"
        results[key] = {
            "A": list(p1), "B": list(p2),
            "max_abs": max_abs_corr(vprod),
            "mean_sq": mean_sq_corr(vprod),
            "corr": corr_vector(vprod),
        }

    return results


# ─────────────────────────────────────────────────────────────────
# 7. Generalized digit-run sequences
# ─────────────────────────────────────────────────────────────────

def digit_run_sequence(k: int, d: int, run_len: int,
                       mode: str, n_max: int) -> np.ndarray:
    """
    a(n) = (-1)^{count of runs in base-k digits of n}.

    mode = "exact":   count runs of digit d of length exactly run_len
    mode = "atleast": count runs of digit d of length >= run_len
    """
    vals = np.empty(n_max, dtype=np.int8)
    for n in range(n_max):
        digits = to_base_k(n, k, min_digits=run_len)
        # find runs of d
        count = 0
        i = 0
        while i < len(digits):
            if digits[i] == d:
                run = 0
                while i < len(digits) and digits[i] == d:
                    run += 1
                    i += 1
                if mode == "exact" and run == run_len:
                    count += 1
                elif mode == "atleast" and run >= run_len:
                    count += 1
            else:
                i += 1
        vals[n] = -1 if count % 2 else 1
    return vals


def run_digitrun_experiments(n_max: int = N_DEFAULT) -> dict:
    results = {}
    configs = [
        (2, 1, 2, "exact"),    # binary: runs of 1 of length exactly 2 (= RS)
        (2, 1, 2, "atleast"),  # binary: runs of 1 length >= 2
        (2, 1, 3, "exact"),    # binary: runs of 1 of length exactly 3
        (2, 1, 3, "atleast"),  # binary: runs of 1 length >= 3
        (3, 1, 2, "exact"),    # ternary: runs of 1 of length exactly 2
        (3, 1, 2, "atleast"),
        (3, 2, 2, "exact"),    # ternary: runs of 2 of length exactly 2
        (3, 1, 1, "exact"),    # ternary: runs of 1 of length exactly 1 (isolated 1s)
        (4, 1, 2, "exact"),
        (4, 2, 2, "exact"),
    ]
    for (k, d, r, mode) in configs:
        label = f"run_k{k}_d{d}_r{r}_{mode}"
        v = digit_run_sequence(k, d, r, mode, n_max)
        results[label] = {
            "k": k, "digit": d, "run_len": r, "mode": mode,
            "max_abs": max_abs_corr(v),
            "mean_sq": mean_sq_corr(v),
            "corr": corr_vector(v),
        }
    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main(n_max: int = N_DEFAULT) -> None:
    print(f"Running new-format experiments with N={n_max} ...")
    all_results: dict[str, dict] = {}

    print("  [1/7] Base-k pattern sequences ...")
    all_results["basek"] = run_basek_experiments(n_max)

    print("  [2/7] Reed-Muller block sequences ...")
    all_results["reed_muller"] = run_rm_experiments(min(n_max, 1 << 12))

    print("  [3/7] Digit-interaction sequences ...")
    all_results["digit_interaction"] = digit_interaction_sequence("", n_max)

    print("  [4/7] Morphic / substitution sequences ...")
    all_results["morphic"] = run_morphic_experiments(n_max)

    print("  [5/7] Interleaved constructions ...")
    all_results["interleaved"] = run_interleaved_experiments(n_max)

    print("  [6/7] Product combinations ...")
    all_results["products"] = run_product_experiments(n_max)

    print("  [7/7] Digit-run sequences ...")
    all_results["digit_runs"] = run_digitrun_experiments(n_max)

    # Collect top candidates (max_abs < 0.005)
    candidates: list[dict] = []
    for family, fam_results in all_results.items():
        for name, rec in fam_results.items():
            if isinstance(rec, dict) and "max_abs" in rec:
                if rec["max_abs"] < 0.005:
                    candidates.append({
                        "family": family,
                        "name": name,
                        "max_abs": rec["max_abs"],
                        "mean_sq": rec["mean_sq"],
                    })

    candidates.sort(key=lambda x: x["max_abs"])
    all_results["__candidates_summary__"] = candidates

    out = ROOT / "new_formats_results.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {out}")
    print(f"\n=== Top noncorrelation candidates (max_abs < 0.005) ===")
    for c in candidates:
        print(f"  {c['name']:50s}  max={c['max_abs']:.5f}  mean_sq={c['mean_sq']:.2e}")




if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=N_DEFAULT)
    args = p.parse_args()
    main(args.N)
