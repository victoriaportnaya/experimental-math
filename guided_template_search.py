#!/usr/bin/env python3
"""Guided rare-event mining + template discovery for low-correlation pattern sets."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from advanced_experiments import constrained_universe, exact_decide_noncorr_constant_length
from experiments import gamma_prefix, sequence_values


@dataclass(frozen=True)
class EvalRecord:
    mask: int
    score: float
    num_patterns: int


def mask_to_patterns(mask: int, universe: list[str]) -> tuple[str, ...]:
    return tuple(universe[i] for i in range(len(universe)) if (mask >> i) & 1)


def score_mask(mask: int, universe: list[str], n: int, m_max: int, cache: dict[int, EvalRecord]) -> EvalRecord:
    if mask in cache:
        return cache[mask]
    patterns = mask_to_patterns(mask, universe)
    vals = sequence_values(patterns, n)
    gam = gamma_prefix(vals, m_max)
    score = max(abs(x) for x in gam[1:]) if len(gam) > 1 else 0.0
    rec = EvalRecord(mask=mask, score=score, num_patterns=len(patterns))
    cache[mask] = rec
    return rec


def beam_search(
    universe: list[str],
    rng: random.Random,
    n: int,
    m_max: int,
    beam_width: int,
    iters: int,
    seeds: list[int],
    cache: dict[int, EvalRecord],
) -> list[EvalRecord]:
    bits = len(universe)
    full_mask = (1 << bits) - 1
    beam = set(seeds)
    beam.add(full_mask)
    for _ in range(iters):
        scored = [score_mask(m, universe, n, m_max, cache) for m in beam]
        scored.sort(key=lambda r: (r.score, r.num_patterns))
        top = scored[:beam_width]
        next_masks = {r.mask for r in top}
        # Local neighbors (single and double flips from top candidates).
        for rec in top:
            m = rec.mask
            flip_positions = list(range(bits))
            rng.shuffle(flip_positions)
            for i in flip_positions[: min(8, bits)]:
                next_masks.add(m ^ (1 << i))
            # A few two-bit flips for broader exploration.
            for _ in range(5):
                i = rng.randrange(bits)
                j = rng.randrange(bits)
                if i != j:
                    next_masks.add(m ^ (1 << i) ^ (1 << j))
        # Inject randoms to avoid local traps.
        for _ in range(beam_width):
            next_masks.add(rng.getrandbits(bits))
        beam = next_masks
    scored = [score_mask(m, universe, n, m_max, cache) for m in beam]
    scored.sort(key=lambda r: (r.score, r.num_patterns))
    return scored[:beam_width]


def genetic_search(
    universe: list[str],
    rng: random.Random,
    n: int,
    m_max: int,
    pop_size: int,
    generations: int,
    mutation_rate: float,
    cache: dict[int, EvalRecord],
) -> list[EvalRecord]:
    bits = len(universe)
    population = [rng.getrandbits(bits) for _ in range(pop_size - 1)] + [(1 << bits) - 1]

    def fitness(mask: int) -> float:
        return score_mask(mask, universe, n, m_max, cache).score

    for _ in range(generations):
        population.sort(key=fitness)
        elites = population[: max(2, pop_size // 8)]
        next_pop = elites[:]
        while len(next_pop) < pop_size:
            a = rng.choice(elites)
            b = rng.choice(population[: max(8, pop_size // 3)])
            # Uniform crossover.
            child = 0
            for i in range(bits):
                pick = (a >> i) & 1 if rng.random() < 0.5 else (b >> i) & 1
                child |= pick << i
            # Bit mutations.
            for i in range(bits):
                if rng.random() < mutation_rate:
                    child ^= 1 << i
            next_pop.append(child)
        population = next_pop

    scored = [score_mask(m, universe, n, m_max, cache) for m in set(population)]
    scored.sort(key=lambda r: (r.score, r.num_patterns))
    return scored[: max(10, pop_size // 4)]


def mine_templates(records: list[EvalRecord], universe: list[str], top_k: int) -> dict:
    top = records[:top_k]
    if not top:
        return {"top_k": top_k, "mandatory": [], "optional": [], "rare": []}
    counts = {w: 0 for w in universe}
    for rec in top:
        pats = set(mask_to_patterns(rec.mask, universe))
        for w in pats:
            counts[w] += 1
    mandatory = sorted([w for w, c in counts.items() if c == len(top)])
    optional = sorted([w for w, c in counts.items() if 0 < c < len(top) and c >= max(2, len(top) // 3)])
    rare = sorted([w for w, c in counts.items() if c == 1])
    return {
        "top_k": len(top),
        "mandatory": mandatory,
        "optional": optional,
        "rare": rare,
    }


def exact_validate(records: list[EvalRecord], universe: list[str], ell: int, keep: int) -> list[dict]:
    out = []
    for rec in records[:keep]:
        pats = mask_to_patterns(rec.mask, universe)
        dec = exact_decide_noncorr_constant_length(pats, ell)
        out.append(
            {
                "mask": rec.mask,
                "score": rec.score,
                "num_patterns": rec.num_patterns,
                "exact_noncorrelated": dec.noncorrelated,
                "basis_size": dec.basis_size,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ell", type=int, default=6, help="pattern length (constrained family begin/end with 1)")
    parser.add_argument("--n", type=int, default=8192, help="N for objective score")
    parser.add_argument("--m-max", type=int, default=64, help="max shift for objective score")
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--beam-iters", type=int, default=18)
    parser.add_argument("--pop-size", type=int, default=72)
    parser.add_argument("--generations", type=int, default=26)
    parser.add_argument("--mutation-rate", type=float, default=0.03)
    parser.add_argument("--validate-top", type=int, default=12, help="exactly validate this many top candidates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("guided_search_results.json"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    universe = constrained_universe(args.ell)
    bits = len(universe)
    cache: dict[int, EvalRecord] = {}

    seeds = [rng.getrandbits(bits) for _ in range(args.beam_width)]
    beam_best = beam_search(
        universe=universe,
        rng=rng,
        n=args.n,
        m_max=args.m_max,
        beam_width=args.beam_width,
        iters=args.beam_iters,
        seeds=seeds,
        cache=cache,
    )
    ga_best = genetic_search(
        universe=universe,
        rng=rng,
        n=args.n,
        m_max=args.m_max,
        pop_size=args.pop_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        cache=cache,
    )

    merged = {}
    for rec in beam_best + ga_best:
        merged[rec.mask] = rec
    all_best = sorted(merged.values(), key=lambda r: (r.score, r.num_patterns))

    template = mine_templates(all_best, universe, top_k=min(20, len(all_best)))
    validated = exact_validate(all_best, universe, args.ell, keep=min(args.validate_top, len(all_best)))

    out = {
        "config": {
            "ell": args.ell,
            "objective_N": args.n,
            "objective_m_max": args.m_max,
            "beam_width": args.beam_width,
            "beam_iters": args.beam_iters,
            "pop_size": args.pop_size,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "seed": args.seed,
        },
        "search_space_size": 1 << bits,
        "cached_evaluations": len(cache),
        "best_records": [
            {"mask": r.mask, "score": r.score, "num_patterns": r.num_patterns}
            for r in all_best[:50]
        ],
        "template_summary": template,
        "exact_validation_top": validated,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
