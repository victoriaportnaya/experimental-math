#!/usr/bin/env python3
"""Generate project visualizations into main/visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "main" / "visualizations"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_gamma_prefix(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seq in results["named_sequences"]:
        name = seq["name"]
        gamma = seq["gamma_prefix"]
        m = np.arange(1, len(gamma))
        ax.plot(m, gamma[1:], lw=1.5, label=name)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("shift m")
    ax.set_ylabel("gamma_N(m)")
    ax.set_title("Empirical correlation coefficients (N=8192)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gamma_prefix_named.png", dpi=180)
    plt.close(fig)


def plot_multiscale_maxabs(multiscale: dict) -> None:
    n_values = multiscale["config"]["n_values"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seq in multiscale["named_sequences_by_n"][str(n_values[0])]:
        name = seq["name"]
        y = []
        for n in n_values:
            row = next(s for s in multiscale["named_sequences_by_n"][str(n)] if s["name"] == name)
            y.append(row["max_abs_nonzero"])
        ax.plot(n_values, y, marker="o", lw=1.5, label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (log2 scale)")
    ax.set_ylabel("max |gamma_N(m)|, 1<=m<=64")
    ax.set_title("Convergence of maximal correlation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multiscale_maxabs_named.png", dpi=180)
    plt.close(fig)


def plot_decay_fits(advanced: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, item in advanced["decay_fits"].items():
        n = np.array(item["n_values"], dtype=float)
        y = np.array(item["max_abs_values"], dtype=float)
        alpha = item["alpha"]
        c = item["C"]
        yfit = c * n ** (-alpha)
        ax.plot(n, y, "o", ms=4, label=f"{name} data")
        ax.plot(n, yfit, "-", lw=1.2, label=f"{name} fit (alpha={alpha:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (log2 scale)")
    ax.set_ylabel("max |gamma_N(m)|, 1<=m<=128")
    ax.set_title("Power-law fits for correlation decay")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "decay_fit_power_law.png", dpi=180)
    plt.close(fig)


def plot_spectral_bars(advanced: dict) -> None:
    names = list(advanced["spectral_proxies"].keys())
    entropy = [advanced["spectral_proxies"][k]["spectral_entropy"] for k in names]
    top10 = [advanced["spectral_proxies"][k]["top10_mass"] for k in names]
    peak = [advanced["spectral_proxies"][k]["max_nonzero_periodogram"] for k in names]

    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, entropy, width, label="spectral_entropy")
    ax.bar(x, top10, width, label="top10_mass")
    ax.bar(x + width, peak, width, label="max_nonzero_periodogram")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("Spectral proxy comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spectral_proxies_bars.png", dpi=180)
    plt.close(fig)


def plot_guided_scores(guided: dict, filename: str, title: str) -> None:
    rows = guided.get("best_records", [])
    if not rows:
        return
    k = min(25, len(rows))
    y = [rows[i]["score"] for i in range(k)]
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, y, marker="o", lw=1.5)
    ax.set_xlabel("rank among best candidates")
    ax.set_ylabel("max |gamma_N(m)| objective")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def plot_parametric_scores(parametric: dict, filename: str, title: str) -> None:
    rows = parametric.get("top_rows", [])
    if not rows:
        return
    k = min(25, len(rows))
    y = [rows[i]["score"] for i in range(k)]
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, y, marker="o", lw=1.5)
    ax.set_xlabel("rank among top parametric families")
    ax.set_ylabel("max |gamma_N(m)| objective")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_json(ROOT / "results.json")
    multiscale = load_json(ROOT / "results_multiscale.json")
    advanced = load_json(ROOT / "advanced_results.json")

    plot_gamma_prefix(results)
    plot_multiscale_maxabs(multiscale)
    plot_decay_fits(advanced)
    plot_spectral_bars(advanced)
    guided6 = ROOT / "guided_search_results_ell6_fast.json"
    guided7 = ROOT / "guided_search_results_ell7_fast.json"
    if guided6.exists():
        plot_guided_scores(load_json(guided6), "guided_ell6_scores.png", "Guided search best-score frontier (ell=6)")
    if guided7.exists():
        plot_guided_scores(load_json(guided7), "guided_ell7_scores.png", "Guided search best-score frontier (ell=7)")
    param6 = ROOT / "parametric_search_ell6.json"
    param7 = ROOT / "parametric_search_ell7.json"
    if param6.exists():
        plot_parametric_scores(load_json(param6), "parametric_ell6_scores.png", "Parametric family search frontier (ell=6)")
    if param7.exists():
        plot_parametric_scores(load_json(param7), "parametric_ell7_scores.png", "Parametric family search frontier (ell=7)")

    # Keep previously generated convergence figures in the same folder.
    for name in ("convergence_named_maxabs.png", "convergence_family_maxabs.png"):
        src = ROOT / name
        if src.exists():
            (OUT_DIR / name).write_bytes(src.read_bytes())

    print(f"Wrote visualizations to {OUT_DIR}")


if __name__ == "__main__":
    main()
