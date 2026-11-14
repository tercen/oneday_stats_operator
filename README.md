# oneday stats operator

Paired tests for subjects measured under several conditions in one acquisition (for example donors × drug
conditions, or patients × timepoints):

- **paired Wilcoxon signed-rank** of each condition against a reference condition, paired by subject;
- **Friedman** across all conditions, paired by subject;
- **Benjamini-Hochberg** within each table.

## Crosstab

| zone | what |
|---|---|
| rows | the unit tested, e.g. the population (`ds5.metacluster_id`) |
| columns | the subject factor and the condition factor, e.g. `s.patient_id`, `s.group` |
| y | the value compared, e.g. the population's proportion of the sample |

A subject missing a value for a condition drops out of that pair (missing is "not measured", not zero).
Fewer than 3 pairs, or fewer than 3 subjects × 3 conditions for Friedman, gives NaN.

## Properties

| name | default | |
|---|---|---|
| `reference_condition` | — | required for the paired tests |
| `subject_factor`, `condition_factor` | first / second column factor | |
| `tests` | `paired and friedman` | or one of them |
| `alpha` | 0.05 | threshold on the adjusted p-value for `significant` |

## Output

Two tables joined on the row:
- **paired:** `comparison`, `n_pairs`, `median_diff`, `statistic` (W), `p.value`, `p_adjusted`,
  `neglog10_p_adjusted`, `significant`;
- **friedman:** `n_subjects`, `k_conditions`, `statistic` (chi²), `p.value`, `p_adjusted`,
  `neglog10_p_adjusted`, `significant`.

## Provenance and parity

- **Clean-room:** written from the method description of the CYTOSHRINK oneday pipeline's differential
  analysis, using the same scipy calls (`wilcoxon(d, zero_method="wilcox")`, `friedmanchisquare`). None of its
  code is included.
- **Parity:** `tests/parity_local.py` compares the results with that pipeline's functions and saved outputs on
  a local reference run. They were identical on 2026-11-14 (40 paired rows, 20 Friedman rows). The script is
  local only.
- **Pinned versions:** scipy, numpy and pandas are pinned because results depend on the scipy version.
