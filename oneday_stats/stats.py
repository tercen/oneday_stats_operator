"""Paired tests for single-acquisition (oneday) designs: subjects measured under several conditions.

Written from the method description of the CYTOSHRINK oneday pipeline's differential analysis, not from its code:
  - paired Wilcoxon signed-rank of each condition against a reference condition, paired by subject;
  - Friedman omnibus across all conditions, paired by subject;
  - Benjamini-Hochberg across each table.
The scipy calls and their options are the ones that pipeline documents, so results agree with it on the same
frequencies (checked locally by tests/parity_local.py; never in CI, never vendored).

Input is long: one value per (cluster, subject, condition). A missing value (no row) is "not measured", not zero:
that subject drops out of the pairs for that cluster, as in the reference pipeline. Repeated rows for one
(cluster, subject, condition) are averaged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

MIN_PAIRS = 3


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values; non-finite p stay NaN and are left out of the ranking."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    scaled = pv[order] * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(scaled, 0.0, 1.0)
    q[ok] = out
    return q


def _matrix(long: pd.DataFrame, cluster) -> pd.DataFrame:
    """subject x condition values for one cluster (mean over repeats)."""
    sub = long[long["cluster"] == cluster]
    return sub.pivot_table(index="subject", columns="condition", values="value", aggfunc="mean")


def paired_vs_reference(long: pd.DataFrame, reference: str, alpha: float = 0.05) -> pd.DataFrame:
    """One row per cluster x (condition vs reference): n_pairs, median_diff, W, p, q, sig."""
    conds = sorted(set(long["condition"]) - {reference})
    if reference not in set(long["condition"]):
        raise ValueError(f"reference condition {reference!r} not in the data; conditions: "
                         f"{sorted(set(long['condition']))}")
    rows = []
    for cl in sorted(long["cluster"].unique(), key=str):
        m = _matrix(long, cl)
        for c in conds:
            row = {"cluster": cl, "comparison": f"{c} vs {reference}", "n_pairs": 0,
                   "median_diff": np.nan, "W": np.nan, "p": np.nan}
            if reference in m.columns and c in m.columns:
                pair = m[[reference, c]].dropna()
                row["n_pairs"] = len(pair)
                if len(pair) >= MIN_PAIRS:
                    d = pair[c] - pair[reference]
                    row["median_diff"] = float(d.median())
                    try:
                        res = wilcoxon(d.to_numpy(), zero_method="wilcox")
                        row["W"], row["p"] = float(res.statistic), float(res.pvalue)
                    except ValueError:  # every difference is zero
                        row["W"], row["p"] = np.nan, 1.0
            rows.append(row)
    out = pd.DataFrame(rows, columns=["cluster", "comparison", "n_pairs", "median_diff", "W", "p"])
    out["q"] = bh(out["p"])
    out["sig"] = out["q"] < alpha
    return out


def friedman(long: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """One row per cluster: n_subjects, k_conditions, chi2, p, q, sig (subjects with any gap are dropped)."""
    rows = []
    for cl in sorted(long["cluster"].unique(), key=str):
        m = _matrix(long, cl).dropna(axis=0)
        row = {"cluster": cl, "n_subjects": m.shape[0], "k_conditions": m.shape[1], "chi2": np.nan, "p": np.nan}
        if m.shape[0] >= 3 and m.shape[1] >= 3:
            try:
                res = friedmanchisquare(*[m[c].to_numpy() for c in m.columns])
                row["chi2"], row["p"] = float(res.statistic), float(res.pvalue)
            except Exception:  # degenerate input: leave NaN, as the reference does
                pass
        rows.append(row)
    out = pd.DataFrame(rows, columns=["cluster", "n_subjects", "k_conditions", "chi2", "p"])
    out["q"] = bh(out["p"])
    out["sig"] = out["q"] < alpha
    return out
