"""Tests for the SPC chart constants, rules, and chart computations."""
from pathlib import Path

import numpy as np
import pytest

from src import control_charts as cc

ROOT = Path(__file__).resolve().parents[1]


# --- constants ---
def test_constants_known_values():
    assert cc.CONSTANTS[5]["A2"] == pytest.approx(0.577)
    assert cc.CONSTANTS[5]["D4"] == pytest.approx(2.114)
    assert cc.CONSTANTS[2]["A3"] == pytest.approx(2.659)
    assert cc.D2_N2 == pytest.approx(1.128)


def test_get_constants_unsupported_raises():
    with pytest.raises(ValueError):
        cc.get_constants(11)


# --- rules ---
def test_beyond_limits_scalar():
    v = [1, 5, 10, -2]
    assert list(cc.beyond_limits(v, ucl=8, lcl=0)) == [False, False, True, True]


def test_beyond_limits_array():
    v = np.array([0.1, 0.2, 0.3])
    ucl = np.array([0.25, 0.25, 0.25])
    lcl = np.array([0.0, 0.0, 0.0])
    assert list(cc.beyond_limits(v, ucl, lcl)) == [False, False, True]


def test_runs_rule_flags_long_run():
    # 8 consecutive points above center -> flagged
    values = [1, 1, 1, 1, 1, 1, 1, 1]
    flagged = cc.runs_about_center(values, center=0.0, run_length=8)
    assert flagged.all()


def test_runs_rule_ignores_short_run():
    values = [1, 1, -1, 1, 1, 1, 1, 1]  # no 8-in-a-row on one side
    assert not cc.runs_about_center(values, center=0.0, run_length=8).any()


# --- chart computations ---
def test_imr_limits_use_d2():
    # center +/- 3 * (MRbar / 1.128) on a known series
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "individual_measurements.csv")
    y = df["measurement"].to_numpy(float)
    mrbar = np.abs(np.diff(y)).mean()
    expected_ucl = y.mean() + 3 * mrbar / cc.D2_N2
    cc.i_mr_chart()  # writes results/i_chart_limits.csv
    saved = pd.read_csv(ROOT / "results" / "i_chart_limits.csv").iloc[0]
    assert saved["ucl"] == pytest.approx(expected_ucl)


def test_np_chart_detects_injected_violation():
    assert cc.np_chart() >= 1  # constant-n dataset has a deliberate out-of-control point


def test_c_chart_detects_injected_violation():
    assert cc.c_chart() >= 1  # constant-unit dataset has a deliberate out-of-control point


def test_np_requires_constant_n(tmp_path, monkeypatch):
    import pandas as pd
    bad = tmp_path
    (bad / "attribute_data_constant_n.csv").write_text(
        "period,sample_size,nonconforming_count\n1,100,4\n2,120,6\n3,100,5\n"
    )
    monkeypatch.setattr(cc, "DATA", bad)
    with pytest.raises(ValueError):
        cc.np_chart()


def test_all_charts_run_and_write_summaries():
    for fn in (cc.i_mr_chart, cc.xbar_r_chart, cc.xbar_s_chart, cc.p_chart,
               cc.np_chart, cc.c_chart, cc.u_chart, cc.run_chart):
        assert isinstance(fn(), int)
    for name in ("i_chart", "xbar_r_chart", "p_chart", "np_chart", "c_chart", "u_chart"):
        assert (ROOT / "results" / f"{name}_limits.csv").exists()
