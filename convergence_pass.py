#!/usr/bin/env python3
"""Second-pass convergence experiments across multiple prefix lengths N."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiments import dilation_invariant_family, run_named_sequences


def write_named_csv(named: dict[str, list[dict]], out_path: Path) -> None:
    n_values = sorted(int(k) for k in named.keys())
    names = [entry["name"] for entry in named[str(n_values[0])]]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence"] + [f"N={n}" for n in n_values])
        for name in names:
            row = [name]
            for n in n_values:
                val = next(e["max_abs_nonzero"] for e in named[str(n)] if e["name"] == name)
                row.append(val)
            writer.writerow(row)


def write_family_csv(family: dict[str, list[dict]], out_path: Path) -> None:
    n_values = sorted(int(k) for k in family.keys())
    labels = [entry["name"] for entry in family[str(n_values[0])]]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family"] + [f"N={n}" for n in n_values])
        for label in labels:
            row = [label]
            for n in n_values:
                val = next(e["max_abs_nonzero"] for e in family[str(n)] if e["name"] == label)
                row.append(val)
            writer.writerow(row)


def plot_lines(data: dict[str, list[dict]], title: str, out_path: Path, key_name: str) -> None:
    n_values = sorted(int(k) for k in data.keys())
    labels = [entry[key_name] for entry in data[str(n_values[0])]]
    x = n_values
    plt.figure(figsize=(8, 4.5))
    for label in labels:
        y = [next(e["max_abs_nonzero"] for e in data[str(n)] if e[key_name] == label) for n in n_values]
        plt.plot(x, y, marker="o", label=label)
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("max_{1<=m<=64} |gamma_{a,N}(m)|")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-max", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("results_multiscale.json"))
    args = parser.parse_args()

    n_values = [1 << 14, 1 << 15, 1 << 16]

    named = {}
    fam = {}
    for n in n_values:
        named[str(n)] = [rep.to_json() for rep in run_named_sequences(n, args.m_max)]
        fam[str(n)] = dilation_invariant_family(n, args.m_max, [2, 3, 4, 5, 6])

    result = {
        "config": {
            "n_values": n_values,
            "m_max": args.m_max,
            "definition": "gamma_N(m) = (1/N) * sum_{n=0}^{N-1} a(n)a(n+m)",
        },
        "named_sequences_by_n": named,
        "dilation_family_by_n": fam,
    }
    args.out.write_text(json.dumps(result, indent=2))
    write_named_csv(named, Path("convergence_named_maxabs.csv"))
    write_family_csv(fam, Path("convergence_family_maxabs.csv"))
    plot_lines(named, "Convergence for named pattern sequences", Path("convergence_named_maxabs.png"), "name")
    plot_lines(fam, "Convergence for dilation-invariant core family", Path("convergence_family_maxabs.png"), "name")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
