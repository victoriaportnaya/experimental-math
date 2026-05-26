#!/usr/bin/env python3
"""Search parametric template families beyond 1{0,1}^{ell-2}1."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from advanced_experiments import exact_decide_noncorr_constant_length
from experiments import gamma_prefix, sequence_values


@dataclass(frozen=True)
class TemplateEval:
    family_type: str
    template: str
    ell: int
    num_patterns: int
    score: float
    exact_noncorrelated: bool
    basis_size: int

    def to_json(self) -> dict:
        return {
            "family_type": self.family_type,
            "template": self.template,
            "ell": self.ell,
            "num_patterns": self.num_patterns,
            "score": self.score,
            "exact_noncorrelated": self.exact_noncorrelated,
            "basis_size": self.basis_size,
        }


def all_words(length: int) -> list[str]:
    return ["".join(bits) for bits in itertools.product("01", repeat=length)]


def instantiate_pattern_set(
    ell: int,
    fixed_blocks: list[str],
    wildcard_blocks: list[int],
) -> tuple[str, ...]:
    # Interleave fixed/wild blocks: F0 W0 F1 W1 ... Fk
    if len(fixed_blocks) != len(wildcard_blocks) + 1:
        return ()
    fixed_total = sum(len(x) for x in fixed_blocks)
    wild_total = sum(wildcard_blocks)
    if fixed_total + wild_total != ell:
        return ()

    words = [fixed_blocks[0]]
    for wlen, f in zip(wildcard_blocks, fixed_blocks[1:]):
        mids = all_words(wlen) if wlen > 0 else [""]
        nxt = []
        for base in words:
            for mid in mids:
                nxt.append(base + mid + f)
        words = nxt

    # Admissibility: exclude all-zero word.
    words = [w for w in words if set(w) != {"0"}]
    return tuple(sorted(words))


def score_patterns(patterns: tuple[str, ...], n: int, m_max: int) -> float:
    vals = sequence_values(patterns, n)
    gam = gamma_prefix(vals, m_max)
    return max(abs(v) for v in gam[1:]) if len(gam) > 1 else 0.0


def evaluate_template(
    family_type: str,
    template: str,
    ell: int,
    patterns: tuple[str, ...],
    n: int,
    m_max: int,
    do_exact: bool,
) -> TemplateEval:
    sc = score_patterns(patterns, n, m_max)
    if do_exact:
        dec = exact_decide_noncorr_constant_length(patterns, ell)
        is_noncorr = dec.noncorrelated
        basis = dec.basis_size
    else:
        is_noncorr = False
        basis = -1
    return TemplateEval(
        family_type=family_type,
        template=template,
        ell=ell,
        num_patterns=len(patterns),
        score=sc,
        exact_noncorrelated=is_noncorr,
        basis_size=basis,
    )


def search_parametric_families(
    ell: int,
    max_anchor_len: int,
    n: int,
    m_max: int,
    exact_top_k: int,
) -> list[TemplateEval]:
    seen_sets: set[tuple[str, ...]] = set()
    candidates: list[tuple[str, str, tuple[str, ...], float]] = []

    def add_candidate(family_type: str, template: str, fixed_blocks: list[str], wildcards: list[int]) -> None:
        pats = instantiate_pattern_set(ell, fixed_blocks, wildcards)
        if not pats or pats in seen_sets:
            return
        seen_sets.add(pats)
        sc = score_patterns(pats, n, m_max)
        candidates.append((family_type, template, pats, sc))

    # Family 1: prefix + wildcard + suffix
    for lp in range(1, max_anchor_len + 1):
        for ls in range(1, max_anchor_len + 1):
            mid = ell - lp - ls
            if mid < 0:
                continue
            for p in all_words(lp):
                for s in all_words(ls):
                    template = f"{p}{{0,1}}^{mid}{s}"
                    add_candidate("prefix_wild_suffix", template, [p, s], [mid])

    # Family 2: prefix + wildcard + infix + wildcard + suffix
    for lp in range(1, max_anchor_len + 1):
        for lm in range(1, max_anchor_len + 1):
            for ls in range(1, max_anchor_len + 1):
                free = ell - lp - lm - ls
                if free < 0:
                    continue
                for a in range(free + 1):
                    b = free - a
                    for p in all_words(lp):
                        for m in all_words(lm):
                            for s in all_words(ls):
                                template = f"{p}{{0,1}}^{a}{m}{{0,1}}^{b}{s}"
                                add_candidate("prefix_wild_infix_wild_suffix", template, [p, m, s], [a, b])

    # Family 3: anchored middle block only (wild + fixed + wild)
    for lm in range(1, max_anchor_len + 1):
        free = ell - lm
        if free < 0:
            continue
        for a in range(free + 1):
            b = free - a
            for m in all_words(lm):
                template = f"{{0,1}}^{a}{m}{{0,1}}^{b}"
                add_candidate("wild_infix_wild", template, ["", m, ""], [a, b])

    candidates.sort(key=lambda t: (t[3], len(t[2]), t[1]))
    need_exact = set(range(min(exact_top_k, len(candidates))))
    results: list[TemplateEval] = []
    for i, (family_type, template, pats, _sc) in enumerate(candidates):
        results.append(
            evaluate_template(
                family_type=family_type,
                template=template,
                ell=ell,
                patterns=pats,
                n=n,
                m_max=m_max,
                do_exact=i in need_exact,
            )
        )
    results.sort(key=lambda r: (not r.exact_noncorrelated, r.score, r.num_patterns, r.template))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ell", type=int, default=6)
    parser.add_argument("--max-anchor-len", type=int, default=2)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--m-max", type=int, default=64)
    parser.add_argument("--exact-top-k", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("parametric_search_results.json"))
    args = parser.parse_args()

    rows = search_parametric_families(
        ell=args.ell,
        max_anchor_len=args.max_anchor_len,
        n=args.n,
        m_max=args.m_max,
        exact_top_k=args.exact_top_k,
    )
    out = {
        "config": {
            "ell": args.ell,
            "max_anchor_len": args.max_anchor_len,
            "N": args.n,
            "m_max": args.m_max,
            "exact_top_k": args.exact_top_k,
        },
        "num_tested_families": len(rows),
        "num_exact_noncorrelated": sum(1 for r in rows if r.exact_noncorrelated),
        "top_rows": [r.to_json() for r in rows[:60]],
        "exact_noncorrelated_rows": [r.to_json() for r in rows if r.exact_noncorrelated][:120],
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
