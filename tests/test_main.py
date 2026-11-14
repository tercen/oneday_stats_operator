"""main.py against a stub TercenContext: the crosstab glue, without a Tercen server."""
import runpy
import sys
import types

import numpy as np
import pandas as pd
import pytest

from oneday_stats.stats import friedman, paired_vs_reference


class StubCtx:
    def __init__(self, qt, props, colors=("s.group",), labels=("s.patient_id",)):
        self.qt, self.props = qt, props
        self.colors, self.labels = list(colors), list(labels)
        self.saved = None

    def operator_property(self, name, typeFn=str, default=None):
        return self.props.get(name, default)

    def select(self, names, df_lib="pandas"):
        return self.qt[names].copy()

    def add_namespace(self, df):
        return df.rename(columns={c: f"ds9.{c}" for c in df.columns if not c.startswith(".")})

    def save(self, tables):
        self.saved = tables


def run(stub, monkeypatch):
    fake = types.ModuleType("tercen.client.context")
    fake.TercenContext = lambda: stub
    monkeypatch.setitem(sys.modules, "tercen", types.ModuleType("tercen"))
    monkeypatch.setitem(sys.modules, "tercen.client", types.ModuleType("tercen.client"))
    monkeypatch.setitem(sys.modules, "tercen.client.context", fake)
    sys.modules["tercen.client"].context = fake
    runpy.run_path("main.py", run_name="__main__")
    return stub.saved


def crosstab(seed=3, n_ri=3, n_ci=2):
    rng = np.random.default_rng(seed)
    rows = []
    for ri in range(n_ri):
        for ci in range(n_ci):
            for s in ["p1", "p2", "p3", "p4"]:
                for c in ["BL", "C2", "12M"]:
                    rows.append({".ri": ri, ".ci": ci, ".y": rng.uniform(), "s.group": c, "s.patient_id": s})
    return pd.DataFrame(rows)


def test_one_test_per_cell_and_tables_match_the_stats(monkeypatch):
    qt = crosstab()
    paired, fr = run(StubCtx(qt, {"reference_condition": "BL"}), monkeypatch)
    assert len(paired) == 3 * 2 * 2 and len(fr) == 3 * 2  # cells x comparisons; cells
    assert set(map(tuple, paired[[".ri", ".ci"]].drop_duplicates().to_numpy())) == {(r, c) for r in range(3) for c in range(2)}
    assert paired[".ri"].dtype == np.int32 and paired[".ci"].dtype == np.int32
    # cell (2, 1) against the stats module directly
    cell = qt[(qt[".ri"] == 2) & (qt[".ci"] == 1)].rename(
        columns={".y": "value", "s.group": "condition", "s.patient_id": "subject"}).assign(cluster="x")
    ref = paired_vs_reference(cell[["cluster", "subject", "condition", "value"]], "BL")
    got = paired[(paired[".ri"] == 2) & (paired[".ci"] == 1)].sort_values("ds9.comparison")
    assert np.allclose(got["ds9.statistic"].to_numpy(), ref.sort_values("comparison")["W"].to_numpy())
    assert {"ds9.p.value", "ds9.p_adjusted", "ds9.neglog10_p_adjusted", "ds9.significant", "ds9.n_pairs"} <= set(paired.columns)
    assert {"ds9.n_subjects", "ds9.k_conditions", "ds9.statistic"} <= set(fr.columns)
    # BH is across the whole step, not per cell
    fam = paired_vs_reference(qt.rename(columns={".y": "value", "s.group": "condition", "s.patient_id": "subject"})
                              .assign(cluster=lambda d: d[".ri"] * 10 + d[".ci"])[["cluster", "subject", "condition", "value"]], "BL")
    assert np.allclose(np.sort(paired["ds9.p_adjusted"]), np.sort(fam["q"]))


def test_reference_is_required_for_paired(monkeypatch):
    with pytest.raises(ValueError, match="reference_condition"):
        run(StubCtx(crosstab(), {}), monkeypatch)


def test_friedman_only_needs_no_reference(monkeypatch):
    out = run(StubCtx(crosstab(), {"tests": "friedman"}), monkeypatch)
    assert isinstance(out, pd.DataFrame) and len(out) == 6


def test_needs_one_color_and_one_label(monkeypatch):
    with pytest.raises(ValueError, match="colors"):
        run(StubCtx(crosstab(), {"reference_condition": "BL"}, labels=()), monkeypatch)


def test_tables_serialise_with_the_real_client(monkeypatch):
    """Regression: pandas 3 string columns broke TSON encoding on Studio (TsonError: Unknown object type)."""
    tsu = pytest.importorskip("tercen.util.helper_functions")
    from tercen.http.HttpClientService import encodeTSON
    from tercen.model.impl import OperatorResult
    stub = StubCtx(crosstab(), {"reference_condition": "BL"})
    tables = run(stub, monkeypatch)
    monkeypatch.undo()
    r = OperatorResult()
    r.tables = [tsu.dataframe_to_table(t)[0] for t in tables]
    assert len(encodeTSON(r.toJson())) > 0
