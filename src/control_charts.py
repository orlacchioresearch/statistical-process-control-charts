"""Statistical process control (Shewhart) charts.

Generates the common SPC charts (I-MR, X-bar/R, X-bar/S, p, np, c, u, run),
computes 3-sigma control limits with the standard control-chart constants, and
**flags out-of-control points** (the original drew limits but never checked them):

* Rule 1 -- any point beyond a control limit (applies to every chart).
* Runs rule -- 8+ consecutive points on one side of the center line
  (variables charts: I and X-bar).

Out-of-control points are highlighted in red on each chart and counted in the
per-chart summary in results/. Synthetic data, for demonstration only.

Important: np-charts require a constant sample size and c-charts a constant
inspection opportunity, so they read the constant-size datasets; the p- and
u-charts use the variable-size datasets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

RUN_RULE_LENGTH = 8  # consecutive points on one side of center -> signal

# Standard Shewhart control-chart constants by subgroup size (Montgomery, Table).
CONSTANTS = {
    2: {"A2": 1.880, "D3": 0.000, "D4": 3.267, "A3": 2.659, "B3": 0.000, "B4": 3.267},
    3: {"A2": 1.023, "D3": 0.000, "D4": 2.574, "A3": 1.954, "B3": 0.000, "B4": 2.568},
    4: {"A2": 0.729, "D3": 0.000, "D4": 2.282, "A3": 1.628, "B3": 0.000, "B4": 2.266},
    5: {"A2": 0.577, "D3": 0.000, "D4": 2.114, "A3": 1.427, "B3": 0.000, "B4": 2.089},
    6: {"A2": 0.483, "D3": 0.000, "D4": 2.004, "A3": 1.287, "B3": 0.030, "B4": 1.970},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924, "A3": 1.182, "B3": 0.118, "B4": 1.882},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864, "A3": 1.099, "B3": 0.185, "B4": 1.815},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816, "A3": 1.032, "B3": 0.239, "B4": 1.761},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "A3": 0.975, "B3": 0.284, "B4": 1.716},
}
D2_N2 = 1.128  # d2 for moving range of 2 consecutive points


# --------------------------------------------------------------------------- #
# Out-of-control rules
# --------------------------------------------------------------------------- #
def beyond_limits(values, ucl, lcl) -> np.ndarray:
    """Rule 1: points outside the control limits (limits may be scalar or array)."""
    values = np.asarray(values, dtype=float)
    return (values > np.asarray(ucl)) | (values < np.asarray(lcl))


def runs_about_center(values, center, run_length: int = RUN_RULE_LENGTH) -> np.ndarray:
    """Flag points that belong to a run of >= run_length on one side of center."""
    values = np.asarray(values, dtype=float)
    flagged = np.zeros(len(values), dtype=bool)
    side = np.sign(values - center)
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or side[i] != side[start] or side[start] == 0:
            if side[start] != 0 and (i - start) >= run_length:
                flagged[start:i] = True
            start = i
    return flagged


def get_constants(n: int) -> Dict[str, float]:
    if n not in CONSTANTS:
        raise ValueError(f"Subgroup size {n} unsupported; constants available for n in 2..10.")
    return CONSTANTS[n]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_chart(x, y, center, ucl, lcl, title, ylabel, filename, violations=None) -> None:
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", color="#1f77b4", zorder=2, label="Value")

    plt.axhline(center, color="green", linestyle="-", label="Center")
    for lim, lab in [(ucl, "UCL"), (lcl, "LCL")]:
        if np.isscalar(lim):
            plt.axhline(lim, color="red", linestyle="--", label=lab)
        else:
            plt.plot(x, np.asarray(lim), color="red", linestyle="--", label=lab)

    if violations is not None and np.any(violations):
        plt.scatter(x[violations], y[violations], color="red", s=80, zorder=3, label="Out of control")

    plt.xlabel("Sequence")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200)
    plt.close()


def save_summary(name: str, d: dict) -> None:
    pd.DataFrame([d]).to_csv(RESULTS / f"{name}_limits.csv", index=False)


def _require_columns(df: pd.DataFrame, columns, source: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{source} missing required columns: {sorted(missing)}")


# --------------------------------------------------------------------------- #
# Charts (each returns the number of out-of-control points)
# --------------------------------------------------------------------------- #
def i_mr_chart() -> int:
    df = pd.read_csv(DATA / "individual_measurements.csv")
    _require_columns(df, ["sequence", "measurement"], "individual_measurements.csv")
    y = df["measurement"].to_numpy(float)
    mr = np.abs(np.diff(y))
    center, mrbar = y.mean(), mr.mean()
    sigma = mrbar / D2_N2
    ucl, lcl = center + 3 * sigma, center - 3 * sigma

    viol = beyond_limits(y, ucl, lcl) | runs_about_center(y, center)
    plot_chart(df["sequence"], y, center, ucl, lcl,
               "I chart: individual measurements", "Measurement", "i_chart.png", viol)
    mr_ucl = CONSTANTS[2]["D4"] * mrbar
    mr_viol = beyond_limits(mr, mr_ucl, 0.0)
    plot_chart(df["sequence"].iloc[1:], mr, mrbar, mr_ucl, 0.0,
               "MR chart: moving range", "Moving range", "mr_chart.png", mr_viol)
    n_ooc = int(viol.sum() + mr_viol.sum())
    save_summary("i_chart", {"center_line": center, "ucl": ucl, "lcl": lcl,
                             "average_moving_range": mrbar, "n_out_of_control": n_ooc})
    return n_ooc


def xbar_r_chart() -> int:
    df = pd.read_csv(DATA / "subgroup_measurements.csv")
    _require_columns(df, ["subgroup", "measurement"], "subgroup_measurements.csv")
    g = df.groupby("subgroup")["measurement"].agg(["mean", "max", "min", "count"]).reset_index()
    g["range"] = g["max"] - g["min"]
    n = int(g["count"].mode()[0])
    c = get_constants(n)
    xbarbar, rbar = g["mean"].mean(), g["range"].mean()

    xbar_ucl, xbar_lcl = xbarbar + c["A2"] * rbar, xbarbar - c["A2"] * rbar
    xviol = beyond_limits(g["mean"], xbar_ucl, xbar_lcl) | runs_about_center(g["mean"], xbarbar)
    plot_chart(g["subgroup"], g["mean"], xbarbar, xbar_ucl, xbar_lcl,
               "X-bar chart: subgroup means", "Subgroup mean", "xbar_chart.png", xviol)
    r_ucl, r_lcl = c["D4"] * rbar, c["D3"] * rbar
    rviol = beyond_limits(g["range"], r_ucl, r_lcl)
    plot_chart(g["subgroup"], g["range"], rbar, r_ucl, r_lcl,
               "R chart: subgroup ranges", "Subgroup range", "r_chart.png", rviol)
    n_ooc = int(xviol.sum() + rviol.sum())
    save_summary("xbar_r_chart", {"subgroup_size": n, "xbarbar": xbarbar, "rbar": rbar,
                                  "xbar_ucl": xbar_ucl, "xbar_lcl": xbar_lcl, "n_out_of_control": n_ooc})
    return n_ooc


def xbar_s_chart() -> int:
    df = pd.read_csv(DATA / "subgroup_measurements.csv")
    g = df.groupby("subgroup")["measurement"].agg(["mean", "std", "count"]).reset_index()
    n = int(g["count"].mode()[0])
    c = get_constants(n)
    xbarbar, sbar = g["mean"].mean(), g["std"].mean()

    xbar_ucl, xbar_lcl = xbarbar + c["A3"] * sbar, xbarbar - c["A3"] * sbar
    xviol = beyond_limits(g["mean"], xbar_ucl, xbar_lcl) | runs_about_center(g["mean"], xbarbar)
    plot_chart(g["subgroup"], g["mean"], xbarbar, xbar_ucl, xbar_lcl,
               "X-bar chart: subgroup means (S-based)", "Subgroup mean", "xbar_s_mean_chart.png", xviol)
    s_ucl, s_lcl = c["B4"] * sbar, c["B3"] * sbar
    sviol = beyond_limits(g["std"], s_ucl, s_lcl)
    plot_chart(g["subgroup"], g["std"], sbar, s_ucl, s_lcl,
               "S chart: subgroup standard deviations", "Subgroup SD", "s_chart.png", sviol)
    n_ooc = int(xviol.sum() + sviol.sum())
    save_summary("xbar_s_chart", {"subgroup_size": n, "xbarbar": xbarbar, "sbar": sbar,
                                  "n_out_of_control": n_ooc})
    return n_ooc


def p_chart() -> int:
    df = pd.read_csv(DATA / "attribute_data.csv")
    _require_columns(df, ["period", "sample_size", "nonconforming_count"], "attribute_data.csv")
    p = df["nonconforming_count"] / df["sample_size"]
    pbar = df["nonconforming_count"].sum() / df["sample_size"].sum()
    ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) / df["sample_size"])
    lcl = (pbar - 3 * np.sqrt(pbar * (1 - pbar) / df["sample_size"])).clip(lower=0)
    viol = beyond_limits(p, ucl, lcl)
    plot_chart(df["period"], p, pbar, ucl, lcl,
               "p-chart: proportion nonconforming (variable n)", "Proportion nonconforming",
               "p_chart.png", viol)
    n_ooc = int(viol.sum())
    save_summary("p_chart", {"pbar": pbar, "average_sample_size": df["sample_size"].mean(),
                             "n_out_of_control": n_ooc})
    return n_ooc


def np_chart() -> int:
    # np-chart requires CONSTANT sample size -> use the constant-n dataset.
    df = pd.read_csv(DATA / "attribute_data_constant_n.csv")
    _require_columns(df, ["period", "sample_size", "nonconforming_count"], "attribute_data_constant_n.csv")
    if df["sample_size"].nunique() != 1:
        raise ValueError("np-chart requires a constant sample size.")
    n = int(df["sample_size"].iloc[0])
    pbar = df["nonconforming_count"].sum() / df["sample_size"].sum()
    center = n * pbar
    spread = 3 * np.sqrt(n * pbar * (1 - pbar))
    ucl, lcl = center + spread, max(0.0, center - spread)
    viol = beyond_limits(df["nonconforming_count"], ucl, lcl)
    plot_chart(df["period"], df["nonconforming_count"], center, ucl, lcl,
               "np-chart: count nonconforming (constant n)", "Nonconforming count", "np_chart.png", viol)
    n_ooc = int(viol.sum())
    save_summary("np_chart", {"constant_n": n, "pbar": pbar, "center_line": center,
                              "ucl": ucl, "lcl": lcl, "n_out_of_control": n_ooc})
    return n_ooc


def c_chart() -> int:
    # c-chart requires CONSTANT opportunity -> use the constant-unit dataset.
    df = pd.read_csv(DATA / "defect_count_constant_unit.csv")
    _require_columns(df, ["period", "defect_count"], "defect_count_constant_unit.csv")
    cbar = df["defect_count"].mean()
    spread = 3 * np.sqrt(cbar)
    ucl, lcl = cbar + spread, max(0.0, cbar - spread)
    viol = beyond_limits(df["defect_count"], ucl, lcl)
    plot_chart(df["period"], df["defect_count"], cbar, ucl, lcl,
               "c-chart: defects per constant unit", "Defect count", "c_chart.png", viol)
    n_ooc = int(viol.sum())
    save_summary("c_chart", {"cbar": cbar, "ucl": ucl, "lcl": lcl, "n_out_of_control": n_ooc})
    return n_ooc


def u_chart() -> int:
    df = pd.read_csv(DATA / "defect_count_data.csv")
    _require_columns(df, ["period", "units_inspected", "defect_count"], "defect_count_data.csv")
    u = df["defect_count"] / df["units_inspected"]
    ubar = df["defect_count"].sum() / df["units_inspected"].sum()
    ucl = ubar + 3 * np.sqrt(ubar / df["units_inspected"])
    lcl = (ubar - 3 * np.sqrt(ubar / df["units_inspected"])).clip(lower=0)
    viol = beyond_limits(u, ucl, lcl)
    plot_chart(df["period"], u, ubar, ucl, lcl,
               "u-chart: defects per unit (variable opportunity)", "Defects per unit",
               "u_chart.png", viol)
    n_ooc = int(viol.sum())
    save_summary("u_chart", {"ubar": ubar, "average_units_inspected": df["units_inspected"].mean(),
                             "n_out_of_control": n_ooc})
    return n_ooc


def run_chart() -> int:
    df = pd.read_csv(DATA / "individual_measurements.csv")
    y = df["measurement"].to_numpy(float)
    median = float(np.median(y))
    # Run statistics about the median (a basic run-chart signal).
    side = np.sign(y - median)
    side = side[side != 0]
    n_runs = 1 + int(np.sum(side[1:] != side[:-1])) if len(side) else 0
    longest = 0 if len(side) == 0 else max(
        len(list(g)) for g in _groupruns(side))
    plt.figure(figsize=(8, 5))
    plt.plot(df["sequence"], y, marker="o", color="#1f77b4")
    plt.axhline(median, color="green", linestyle="--", label=f"Median ({median:.1f})")
    plt.xlabel("Sequence")
    plt.ylabel("Measurement")
    plt.title("Run chart: time-ordered trend")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "run_chart.png", dpi=200)
    plt.close()
    save_summary("run_chart", {"median": median, "n_runs": n_runs, "longest_run": longest})
    return 0


def _groupruns(arr):
    out, start = [], 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or arr[i] != arr[start]:
            out.append(arr[start:i])
            start = i
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    charts = {
        "I-MR": i_mr_chart, "X-bar/R": xbar_r_chart, "X-bar/S": xbar_s_chart,
        "p": p_chart, "np": np_chart, "c": c_chart, "u": u_chart, "Run": run_chart,
    }
    print(f"{'Chart':<10} {'Out-of-control points':>22}")
    for name, fn in charts.items():
        n_ooc = fn()
        print(f"{name:<10} {n_ooc:>22}")
    logger.info("Generated control charts in figures/ and summaries in results/.")


if __name__ == "__main__":
    main()
