# Procedures and Findings: k-Noncorrelation Search

This note documents the full experimental procedure used to investigate
noncorrelation for complex-rooted automatic pattern sequences
\(a_{A,k}(n)=\omega_k^{\#(A,n)}\), with emphasis on the \(2/4\) dichotomy and
the \(k>6\) stress tests.

## 1) Core metric and decision proxy

- For each sequence, we compute empirical correlations
  \[
  \hat\gamma_{A,k,N}(m)=\frac1N\sum_{n=0}^{N-1}a_{A,k}(n)\overline{a_{A,k}(n+m)},
  \]
  and track
  \[
  G_{A,k,N}=\max_{1\le m\le 64}\left|\hat\gamma_{A,k,N}(m)\right|.
  \]
- Practical NC proxy:
  - near-NC candidate if \(G\) is very small (historically \(\lesssim 10^{-2}\)),
  - strong evidence of NC only when values also exhibit \(O(1/N)\) decay
    (approximately halving as \(N\) doubles).
- Persistent plateau across scales (ratio near 1.0) is interpreted as
  correlated behavior (not NC).

## 2) Search spaces

### 2.1 Exhaustive small family

- All nonempty pattern subsets from binary words with length \(\le 4\),
  using the same tractable census family as previous runs:
  - size 1, 2 from all words length \(\le 4\),
  - size 3, 4 from words length \(\le 3\).
- Total sets checked in this family: **846**.

### 2.2 Random wide sampling

- Random subsets with pattern length up to 5 or 6 and subset size typically
  between 2 and 10.
- Used to probe beyond structured families and avoid only-template bias.

### 2.3 Heavy guided search (beam + local mutations)

- Universe: all nonzero binary words of length \(\le 7\) (247 patterns).
- Multi-start guided search:
  - multiple restarts,
  - beam selection by small-\(N\) score,
  - local mutation hill-climbing (add/drop/swap/double moves),
  - random injections each generation.
- Top candidates then re-scored at large scales
  \(N=2^{12},2^{13},2^{14},2^{15},2^{16}\).

## 3) Important implementation detail

- Correlation scoring uses the **full complex modulus**:
  \[
  \left|\hat\gamma_{A,k,N}(m)\right|,
  \]
  not only the real part.
- This corrected an earlier pitfall that can falsely mark candidates as
  near-zero at \(k>2\).

## 4) Main findings

## 4.1 \(k=3\)

- No NC candidate found in exhaustive or random sweeps.
- Best values remain far from zero and stable with \(N\) (no \(1/N\) decay).

## 4.2 \(k=8\) heavy stress test

- Heavy run over length \(\le 7\) universe with **64,689 evaluations**.
- Best candidate:
  - \(G_{A,8,2^{16}} = 0.030742\),
  - scale ratio \(G_{2^{14}}/G_{2^{16}}\approx 1.003\) (plateau).
- Verdict: **no \(k=8\) NC candidate**; correlated floor persists.

## 4.3 \(k=10\) heavy stress test

- Heavy run with **63,786 evaluations**.
- Best candidate:
  - \(G_{A,10,2^{16}} = 0.038111\),
  - ratio \(G_{2^{14}}/G_{2^{16}}\approx 1.005\).
- Verdict: **no \(k=10\) NC candidate**.

## 4.4 \(k=12\) heavy stress test

- Heavy run with **63,964 evaluations**.
- Best candidate:
  - \(G_{A,12,2^{16}} = 0.036515\),
  - ratio \(G_{2^{14}}/G_{2^{16}}\approx 0.998\).
- Verdict: **no \(k=12\) NC candidate**.

## 5) Current empirical picture

- Previously established:
  - \(k=2\): many NC examples,
  - \(k=4\): new NC class discovered,
  - \(k=6\): two dual \(2+6\)-NC examples.
- New stress tests:
  - \(k=8,10,12\): no NC examples despite heavy guided search.
- Combined evidence supports a strong **correlated floor** for tested \(k>6\),
  with best values around \(3\times 10^{-2}\) to \(4\times 10^{-2}\), not
  decaying.

## 6) Reproducibility artifacts

Primary outputs saved in project root:

- `k8_guided_search_results.json`
- `k10_guided_search_results.json`
- `k12_guided_search_results.json`
- `k10_k12_guided_summary.json`

Manuscript and figures:

- `dichotomy/dichotomy_paper.tex`
- `dichotomy/figures/*.png`
- `dichotomy_overleaf.zip`

Core code:

- `complex_rooted_sequences.py`

