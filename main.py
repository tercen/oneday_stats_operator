"""oneday stats: paired tests for subjects measured under several conditions (one acquisition).

Crosstab (the convention of Tercen's Wilcoxon Test operator, so it can replace one in place)
  rows x columns  the unit tested: one test per (row, column) cell, e.g. rows = population, or
                  rows = marker and columns = population
  colors          the condition factor (e.g. s.group)
  labels          the subject factor, the pairing unit (e.g. s.patient_id)
  y               the value compared, e.g. a population's proportion of the sample
Two tables come back, joined on (.ri, .ci):
  paired:   one row per condition other than the reference: comparison, n_pairs, median_diff, statistic,
            p.value, p_adjusted, neglog10_p_adjusted, significant
  friedman: one row: n_subjects, k_conditions, statistic, p.value, p_adjusted, neglog10_p_adjusted, significant
Each table is Benjamini-Hochberg adjusted across all its rows (every cell and comparison of the step).
"""
import numpy as np
from tercen.client import context as ctx

from oneday_stats.stats import friedman, paired_vs_reference

tc = ctx.TercenContext()


def prop(name, default):
    try:
        v = tc.operator_property(name, typeFn=str, default=str(default))
    except Exception:
        v = default
    return default if v is None or v == "" else v


if len(tc.colors) != 1 or len(tc.labels) != 1:
    raise ValueError("Put the condition factor on colors and the subject (pairing) factor on labels, one each.")
condition_f, subject_f = tc.colors[0], tc.labels[0]
reference = prop("reference_condition", "")
alpha = float(prop("alpha", 0.05))
tests = prop("tests", "paired and friedman")

df = tc.select([".ri", ".ci", ".y", condition_f, subject_f], df_lib="pandas")
df = df.rename(columns={".y": "value", condition_f: "condition", subject_f: "subject"})
df["condition"] = df["condition"].astype(str)
df["subject"] = df["subject"].astype(str)
df["cluster"] = df[".ri"].astype(np.int64) * 1_000_000_000 + df[".ci"].astype(np.int64)  # one unit per cell

if not reference and tests != "friedman":
    raise ValueError(f"Set reference_condition to one of {sorted(df['condition'].unique())}")


def finish(res, stat_col):
    out = res.rename(columns={stat_col: "statistic", "p": "p.value", "q": "p_adjusted"})
    cell = out.pop("cluster").astype(np.int64)
    out.insert(0, ".ri", (cell // 1_000_000_000).astype(np.int32))
    out.insert(1, ".ci", (cell % 1_000_000_000).astype(np.int32))
    out["neglog10_p_adjusted"] = -np.log10(out["p_adjusted"].astype(float).clip(lower=1e-300))
    out["significant"] = out.pop("sig").astype(float)
    for c in out.columns:
        if c not in (".ri", ".ci", "comparison"):
            out[c] = out[c].astype(float)
    if "comparison" in out.columns:
        # pandas 3 stores strings as a StringArray, which the client's TSON encoder rejects ("Unknown object type")
        out["comparison"] = out["comparison"].astype(object)
    return tc.add_namespace(out)


long = df[["cluster", "subject", "condition", "value"]]
tables = []
if tests in ("paired and friedman", "paired"):
    tables.append(finish(paired_vs_reference(long, reference, alpha), "W"))
if tests in ("paired and friedman", "friedman"):
    tables.append(finish(friedman(long, alpha), "chi2"))
tc.save(tables if len(tables) > 1 else tables[0])
