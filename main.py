"""oneday stats: paired tests for subjects measured under several conditions (one acquisition).

Crosstab
  rows     the unit tested, e.g. the population (one test family row per row)
  columns  the subject factor and the condition factor (e.g. s.patient_id, s.group)
  y        the value compared, e.g. a population's proportion of the sample
Two tables come back, joined on the row:
  paired:   one row per condition other than the reference: comparison, n_pairs, median_diff, statistic,
            p.value, p_adjusted, neglog10_p_adjusted, significant
  friedman: one row: n_subjects, k_conditions, statistic, p.value, p_adjusted, neglog10_p_adjusted, significant
"""
import numpy as np
import pandas as pd
from tercen.client import context as ctx

from oneday_stats.stats import friedman, paired_vs_reference

tc = ctx.TercenContext()


def prop(name, default):
    try:
        v = tc.operator_property(name, typeFn=str, default=str(default))
    except Exception:
        v = default
    return default if v is None or v == "" else v


cnames = [c for c in tc.cnames if c]
if len(cnames) < 2:
    raise ValueError("Put the subject factor and the condition factor on columns (two column factors).")
subject_f = prop("subject_factor", "") or cnames[0]
condition_f = prop("condition_factor", "") or cnames[1]
for f in (subject_f, condition_f):
    if f not in cnames:
        raise ValueError(f"{f!r} is not a column factor; column factors are {cnames}")
reference = prop("reference_condition", "")
alpha = float(prop("alpha", 0.05))
tests = prop("tests", "paired and friedman")

cols = tc.cselect([subject_f, condition_f], df_lib="pandas")
cols.columns = ["subject", "condition"]
cols[".ci"] = np.arange(len(cols), dtype=np.int32)
y = tc.select([".ri", ".ci", ".y"], df_lib="pandas")
long = (y.merge(cols, on=".ci")
         .rename(columns={".ri": "cluster", ".y": "value"})[["cluster", "subject", "condition", "value"]])
long["subject"] = long["subject"].astype(str)
long["condition"] = long["condition"].astype(str)

if not reference:
    levels = sorted(long["condition"].unique())
    raise ValueError(f"Set reference_condition to one of {levels}")


def finish(df, stat_col):
    out = df.rename(columns={"cluster": ".ri", stat_col: "statistic", "p": "p.value", "q": "p_adjusted"})
    out[".ri"] = out[".ri"].astype(np.int32)
    out["neglog10_p_adjusted"] = -np.log10(out["p_adjusted"].clip(lower=1e-300))
    out["significant"] = out["sig"].astype(float)
    out = out.drop(columns=["sig"])
    for c in out.columns:
        if c not in (".ri", "comparison"):
            out[c] = out[c].astype(float)
    return tc.add_namespace(out)


tables = []
if tests in ("paired and friedman", "paired"):
    tables.append(finish(paired_vs_reference(long, reference, alpha), "W"))
if tests in ("paired and friedman", "friedman"):
    tables.append(finish(friedman(long, alpha), "chi2"))
tc.save(tables if len(tables) > 1 else tables[0])
