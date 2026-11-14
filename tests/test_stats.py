"""CI tests on synthetic data only (no study data in this repo)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import friedmanchisquare, wilcoxon

from oneday_stats.stats import bh, friedman, paired_vs_reference


def long_table(values):
    """values: {(cluster, subject, condition): value}"""
    return pd.DataFrame([{"cluster": k[0], "subject": k[1], "condition": k[2], "value": v}
                         for k, v in values.items()])


def test_bh_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan])
    q = bh(p)
    # sorted p 0.01,0.03,0.04,0.2 (m=4): 0.04, 0.0533, 0.0533, 0.2
    assert np.allclose(q[:4], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])
    assert np.isnan(q[4])


def test_paired_wilcoxon_pairs_by_subject_not_order():
    rng = np.random.default_rng(1)
    vals = {}
    for s in ["s1", "s2", "s3", "s4", "s5"]:
        base = rng.uniform(0.1, 0.3)
        vals[("c1", s, "BL")] = base
        vals[("c1", s, "D1")] = base + rng.uniform(0.01, 0.05)
    long = long_table(vals).sample(frac=1, random_state=3)  # shuffled rows
    out = paired_vs_reference(long, "BL")
    r = out.iloc[0]
    d = np.array([vals[("c1", s, "D1")] - vals[("c1", s, "BL")] for s in ["s1", "s2", "s3", "s4", "s5"]])
    ref = wilcoxon(d, zero_method="wilcox")
    assert r["n_pairs"] == 5 and r["comparison"] == "D1 vs BL"
    assert r["W"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(ref.pvalue)
    assert r["median_diff"] == pytest.approx(np.median(d))


def test_missing_value_drops_the_subject_and_too_few_pairs_is_nan():
    vals = {("c1", s, c): v for s, c, v in [
        ("s1", "BL", .1), ("s1", "D1", .2), ("s2", "BL", .1), ("s2", "D1", .3), ("s3", "BL", .2)]}
    out = paired_vs_reference(long_table(vals), "BL")
    assert out.iloc[0]["n_pairs"] == 2 and np.isnan(out.iloc[0]["p"])


def test_all_zero_differences_give_p_one():
    # Older scipy raised here (W = NaN, p = 1); the pinned scipy returns W = 0, p = 1. Either way p = 1.
    vals = {("c1", s, c): 0.1 for s in ["s1", "s2", "s3"] for c in ["BL", "D1"]}
    out = paired_vs_reference(long_table(vals), "BL")
    assert out.iloc[0]["p"] == 1.0
    assert out.iloc[0]["W"] == 0.0 or np.isnan(out.iloc[0]["W"])


def test_unknown_reference_is_an_error():
    with pytest.raises(ValueError, match="reference condition"):
        paired_vs_reference(long_table({("c1", "s1", "BL"): .1}), "DMSO")


def test_bh_is_across_clusters_and_comparisons():
    rng = np.random.default_rng(7)
    vals = {(cl, s, c): rng.uniform() for cl in ["c1", "c2"] for s in ["a", "b", "c", "d"]
            for c in ["BL", "X", "Y"]}
    out = paired_vs_reference(long_table(vals), "BL")
    assert len(out) == 4
    assert np.allclose(out["q"], bh(out["p"]))


def test_friedman_matches_scipy_and_drops_incomplete_subjects():
    rng = np.random.default_rng(5)
    vals = {("c1", s, c): rng.uniform() for s in ["a", "b", "c", "d"] for c in ["BL", "C2", "12M"]}
    del vals[("c1", "d", "C2")]
    out = friedman(long_table(vals))
    r = out.iloc[0]
    assert r["n_subjects"] == 3 and r["k_conditions"] == 3
    cols = sorted(["BL", "C2", "12M"])  # pivot orders the conditions
    ref = friedmanchisquare(*[[vals[("c1", s, c)] for s in ["a", "b", "c"]] for c in cols])
    assert r["chi2"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(ref.pvalue)
