"""LOCAL ONLY: compare this operator's statistics with the reference pipeline's own functions and saved outputs.

Never run in CI and never vendored: the reference code (no licence) is imported from a local clone, and the
reference outputs are study data that stay on this machine.

  .venv/bin/python tests/parity_local.py <reference clone>/python_scripts/oneday <run dir>/stages_run1
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from oneday_stats.stats import friedman, paired_vs_reference  # noqa: E402

ref_dir, stages = sys.argv[1], sys.argv[2]
sys.path.insert(0, ref_dir)
import differential_oneday as ref  # noqa: E402

freqs = pd.read_csv(f"{stages}/06_freqs.csv", dtype={"metacluster": str})
long = freqs.rename(columns={"metacluster": "cluster", "Patient": "subject", "Timepoint": "condition",
                             "freq": "value"})[["cluster", "subject", "condition", "value"]]


def check(name, ours, theirs, keys, cols):
    o = ours.set_index(keys).sort_index()
    t = theirs.set_index(keys).sort_index()
    assert list(o.index) == list(t.index), f"{name}: different rows"
    worst = 0.0
    for c in cols:
        a, b = o[c].to_numpy(float), t[c].to_numpy(float)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        d = np.nanmax(np.abs(a - b)) if np.isfinite(a).any() else 0.0
        worst = max(worst, d)
        assert same_nan and d <= 1e-12, f"{name}.{c}: max diff {d}, NaN pattern same={same_nan}"
    print(f"{name}: {len(o)} rows identical (max |diff| {worst:.1e})")


ours_w = paired_vs_reference(long, "BL").rename(columns={"cluster": "metacluster"})
ref_w = ref.paired_wilcoxon_vs_reference(freqs, subject_col="Patient", condition_col="Timepoint", reference="BL")
saved_w = pd.read_csv(f"{stages}/07_paired_wilcoxon.csv", dtype={"metacluster": str})
for label, theirs in [("wilcoxon vs reference functions", ref_w), ("wilcoxon vs notebook output", saved_w)]:
    check(label, ours_w, theirs, ["metacluster", "comparison"], ["n_pairs", "median_diff", "W", "p", "q"])

ours_f = friedman(long).rename(columns={"cluster": "metacluster"})
ref_f = ref.friedman_across_conditions(freqs, subject_col="Patient", condition_col="Timepoint")
saved_f = pd.read_csv(f"{stages}/08_friedman.csv", dtype={"metacluster": str})
for label, theirs in [("friedman vs reference functions", ref_f), ("friedman vs notebook output", saved_f)]:
    check(label, ours_f, theirs, ["metacluster"], ["n_subjects", "k_conditions", "chi2", "p", "q"])
