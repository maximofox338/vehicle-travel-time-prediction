"""
End-to-end experiment driver.

Builds the design matrices, runs EDA figures, trains the baseline / Ridge / Lasso / MLP
models with a proper train-validation-test protocol, evaluates them, writes the June-2017
submission file and dumps every number used in the report to results.json.

The notebook reproduces these same steps interactively.

    python make_results.py           # real data (see README for where to put it)
    python make_results.py --demo    # tiny synthetic dataset, ~1 min, writes to demo_output/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEMO = "--demo" in sys.argv
if DEMO:   # must run before `src.config` is imported: it reads these variables at import time
    _root = Path(__file__).resolve().parent
    os.environ.setdefault("GY_DATA_DIR", str(_root / "sample_data"))
    os.environ.setdefault("GY_CACHE_DIR", str(_root / "sample_data" / "cache"))
    os.environ.setdefault("GY_OUT_DIR", str(_root / "demo_output"))

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src import data_prep as dp
from src import features as F
from src import models as M

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.autolayout": True})
RESULTS = {}


def hhmm(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


# ----------------------------------------------------------------------------------------
# 1. Exploratory data analysis
# ----------------------------------------------------------------------------------------
def run_eda(morning: pd.DataFrame, link_info: pd.DataFrame):
    print("\n[EDA] generating figures ...")
    d = pd.to_datetime(morning["date"])
    morning = morning.assign(weekday=d.dt.weekday.to_numpy(),
                             is_weekend=(d.dt.weekday >= 5).to_numpy())

    # Fig 1: average morning profile (mean travel time by 2-min slice over all links/days)
    prof = morning.groupby("minute_of_day")["travel_time"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(prof.index, prof.values, color="#1f77b4")
    ax.axvspan(C.TARGET_START_MIN, C.TARGET_END_MIN, color="orange", alpha=0.15,
               label="[08:00,09:00) target")
    ax.set_xticks(range(360, 541, 30))
    ax.set_xticklabels([hhmm(m) for m in range(360, 541, 30)])
    ax.set_xlabel("time of day"); ax.set_ylabel("avg travel time (s)")
    ax.set_title("Average travel time across the morning (all links)")
    ax.legend()
    fig.savefig(C.FIG_DIR / "fig1_morning_profile.png"); plt.close(fig)

    # Fig 2: distribution of travel times (log-x)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    vals = morning["travel_time"].clip(lower=0.1)
    ax.hist(np.log10(vals), bins=60, color="#2ca02c", alpha=0.85)
    ax.set_xlabel("log10(travel time in s)"); ax.set_ylabel("count")
    ax.set_title("Distribution of 2-minute travel times")
    fig.savefig(C.FIG_DIR / "fig2_distribution.png"); plt.close(fig)

    # Fig 3: travel time by link width
    mw = morning.merge(link_info[["link_idx", "width"]], on="link_idx")
    order = sorted(mw["width"].unique())
    data = [mw.loc[mw["width"] == w, "travel_time"].clip(upper=120).values for w in order]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.boxplot(data, tick_labels=[str(w) for w in order], showfliers=False)
    ax.set_xlabel("link width (m)"); ax.set_ylabel("travel time (s, clipped at 120)")
    ax.set_title("Travel time by link width")
    fig.savefig(C.FIG_DIR / "fig3_width.png"); plt.close(fig)

    # Fig 4: weekday vs weekend morning profile
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for flag, lbl, col in [(False, "weekday", "#1f77b4"), (True, "weekend", "#d62728")]:
        p = morning[morning["is_weekend"] == flag].groupby("minute_of_day")["travel_time"].mean()
        ax.plot(p.index, p.values, label=lbl, color=col)
    ax.set_xticks(range(360, 541, 30))
    ax.set_xticklabels([hhmm(m) for m in range(360, 541, 30)])
    ax.set_xlabel("time of day"); ax.set_ylabel("avg travel time (s)")
    ax.set_title("Weekday vs weekend morning profile"); ax.legend()
    fig.savefig(C.FIG_DIR / "fig4_weekday_weekend.png"); plt.close(fig)

    RESULTS["eda"] = {
        "n_morning_rows": int(len(morning)),
        "n_links": int(link_info.shape[0]),
        "peak_minute": int(prof.idxmax()), "peak_value": round(float(prof.max()), 2),
        "median_tt": round(float(morning["travel_time"].median()), 2),
        "widths": [int(w) for w in order],
    }
    print("[EDA] done.")


# ----------------------------------------------------------------------------------------
# 2. Modelling
# ----------------------------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    met = M.regression_metrics(y_true, M.clip_positive(y_pred))
    print(f"  {name:<22} MAPE={met['MAPE(%)']:.2f}%  MAE={met['MAE']:.3f}  "
          f"RMSE={met['RMSE']:.3f}  R2={met['R2']:.3f}")
    return met


HIST_FLOOR = 0.1   # small floor when dividing by the historical mean to build the ratio target
BASELINE = "Baseline (historical mean)"
APPENDIX = "Gradient boosting (appendix)"   # outside the course: reported, never selected
MODEL_MAKERS = {"Ridge": M.make_ridge, "Lasso": M.make_lasso, "MLP": M.make_mlp,
                APPENDIX: M.make_hgb}


def _ratio_target(df):
    """Multiplicative-correction target r = travel_time / historical_mean."""
    return (df["y"].to_numpy(np.float64) /
            np.clip(df["hist_slice_mean"].to_numpy(np.float64), HIST_FLOOR, None))


def _predict_y(est, df, feat_cols):
    """Invert the ratio: travel_time = correction_factor * historical_mean."""
    h = np.clip(df["hist_slice_mean"].to_numpy(np.float64), HIST_FLOOR, None)
    return M.clip_positive(est.predict(df[feat_cols].to_numpy(np.float64)) * h)


def run_models(examples, feat_cols, link_info, aux):
    train = examples[examples["role"] == "train"].copy()
    months = [int(m) for m in sorted(train["month_key"].unique())]
    print(f"\n[MODELS] {len(train):,} training rows over {len(months)} labelled months {months}, "
          f"{len(feat_cols)} features | target = travel_time / historical_mean")

    # ---- leave-one-month-out cross-validation to compare the models ----
    # The historical features are rebuilt in every fold from the *other* months only, so the
    # validation month never reaches the features of the training rows (no indirect leakage).
    print("[CV] leave-one-month-out MAPE:")
    folds = {name: [] for name in [BASELINE, *MODEL_MAKERS]}
    fold_full = {name: [] for name in folds}
    for mo in months:
        fold, _ = F.with_hist(aux["raw"], [m for m in months if m != mo], aux)
        fold = fold[fold["role"] == "train"]
        tr, va = fold[fold["month_key"] != mo], fold[fold["month_key"] == mo]
        rtr = _ratio_target(tr)
        yva = va["y"].to_numpy(np.float64)
        hva = np.clip(va["hist_slice_mean"].to_numpy(np.float64), HIST_FLOOR, None)
        folds[BASELINE].append(M.mape(yva, hva))
        fold_full[BASELINE].append(M.regression_metrics(yva, hva))
        for name in MODEL_MAKERS:
            est = MODEL_MAKERS[name]().fit(tr[feat_cols].to_numpy(np.float64), rtr)
            pred = _predict_y(est, va, feat_cols)
            folds[name].append(M.mape(yva, pred))
            fold_full[name].append(M.regression_metrics(yva, pred))

    cv_metrics = {}
    for name in folds:
        keys = fold_full[name][0].keys()
        cv_metrics[name] = {k: float(np.mean([d[k] for d in fold_full[name]])) for k in keys}
        cv_metrics[name]["per_month_MAPE"] = [round(x, 2) for x in folds[name]]
        print(f"  {name:<26} mean MAPE={cv_metrics[name]['MAPE(%)']:.2f}%  "
              f"MAE={cv_metrics[name]['MAE']:.3f}  R2={cv_metrics[name]['R2']:.3f}")

    learned = {k: v for k, v in cv_metrics.items() if k not in (BASELINE, APPENDIX)}
    best_name = min(learned, key=lambda k: learned[k]["MAPE(%)"])
    print(f"[CV] best model (lowest mean LOMO MAPE): {best_name}")

    # ---- retrain best model on ALL training months, evaluate once on the June test ----
    best = MODEL_MAKERS[best_name]().fit(train[feat_cols].to_numpy(np.float64),
                                         _ratio_target(train))
    grid_X, meta = F.build_test_grid(examples, link_info, aux)
    h = np.clip(grid_X["hist_slice_mean"].to_numpy(np.float64), HIST_FLOOR, None)
    meta["pred"] = M.clip_positive(best.predict(grid_X[feat_cols].to_numpy(np.float64)) * h)
    meta["bpred"] = M.clip_positive(grid_X["hist_slice_mean"].to_numpy())
    cmp = meta
    have = cmp["actual"].notna() & (cmp["actual"] > 0)

    test_metrics = M.regression_metrics(cmp.loc[have, "actual"], cmp.loc[have, "pred"])
    base_test_metrics = M.regression_metrics(cmp.loc[have, "actual"], cmp.loc[have, "bpred"])
    print(f"\n[TEST] June-2017 profile ({int(have.sum())} link-slices) -- {best_name}: "
          f"MAPE={test_metrics['MAPE(%)']:.2f}%  MAE={test_metrics['MAE']:.3f}  "
          f"RMSE={test_metrics['RMSE']:.3f}  R2={test_metrics['R2']:.3f}")
    print(f"[TEST] Baseline on June-2017: MAPE={base_test_metrics['MAPE(%)']:.2f}%  "
          f"MAE={base_test_metrics['MAE']:.3f}  R2={base_test_metrics['R2']:.3f}")

    # appendix model, same protocol; it does not influence the submission
    app = MODEL_MAKERS[APPENDIX]().fit(train[feat_cols].to_numpy(np.float64), _ratio_target(train))
    app_pred = M.clip_positive(app.predict(grid_X[feat_cols].to_numpy(np.float64)) * h)
    app_test_metrics = M.regression_metrics(cmp.loc[have, "actual"], app_pred[have.to_numpy()])
    print(f"[TEST] {APPENDIX} on June-2017: MAPE={app_test_metrics['MAPE(%)']:.2f}%")

    RESULTS["cv"] = cv_metrics
    RESULTS["cv_months"] = [int(m) for m in months]
    RESULTS["best_model"] = best_name
    RESULTS["best_params"] = getattr(best, "best_params_", {})
    RESULTS["test"] = {best_name: test_metrics, BASELINE: base_test_metrics,
                       APPENDIX: app_test_metrics}
    # how many labelled months feed the historical mean: one fewer in each CV fold than at test
    RESULTS["hist_months"] = {"cv_fold": len(months) - 1, "test": len(months)}
    RESULTS["n_test_link_slices"] = int(have.sum())

    return best, best_name, meta, cmp, train, feat_cols


# ----------------------------------------------------------------------------------------
# 3. Result figures + submission file
# ----------------------------------------------------------------------------------------
def result_figures(meta, cmp, best_name):
    # Fig 5: model comparison (leave-one-month-out cross-validation)
    vm = RESULTS["cv"]
    names = list(vm.keys()); mapes = [vm[n]["MAPE(%)"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(range(len(names)), mapes,
                  color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#bcbd22"][:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (historical mean)", "\n(hist. mean)")
                        .replace(" (appendix)", "\n(appendix)") for n in names],
                       fontsize=8)
    ax.set_ylabel("mean LOMO-CV MAPE (%)")
    ax.set_title("Model comparison (leave-one-month-out CV)")
    for b, v in zip(bars, mapes):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(C.FIG_DIR / "fig5_model_comparison.png"); plt.close(fig)

    # Fig 6: predicted vs actual (test averaged profile)
    have = cmp["actual"].notna()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(cmp.loc[have, "actual"], cmp.loc[have, "pred"], s=6, alpha=0.3, color="#1f77b4")
    lim = float(np.nanpercentile(cmp.loc[have, "actual"], 99))
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("actual (s)"); ax.set_ylabel("predicted (s)")
    ax.set_title(f"{best_name}: predicted vs actual\n(June-2017 link-slice averages)")
    fig.savefig(C.FIG_DIR / "fig6_pred_vs_actual.png"); plt.close(fig)

    # Fig 7: one example link's [08:00,09:00) profile, predicted vs actual
    ex_link = (cmp[have].groupby("link_idx").size().sort_values().index[-1])
    sub = cmp[(cmp["link_idx"] == ex_link)].sort_values("target_slice_idx")
    mins = C.TARGET_START_MIN + C.SLICE_MINUTES * sub["target_slice_idx"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(mins, sub["actual"].to_numpy(), "o-", ms=3, label="actual", color="#d62728")
    ax.plot(mins, sub["pred"].to_numpy(), "s-", ms=3, label="predicted", color="#1f77b4")
    ax.set_xticks(range(480, 541, 10)); ax.set_xticklabels([hhmm(m) for m in range(480, 541, 10)])
    ax.set_xlabel("time of day"); ax.set_ylabel("travel time (s)")
    ax.set_title(f"Example link {sub['link_ID'].iloc[0]}: predicted vs actual [08:00,09:00)")
    ax.legend()
    fig.savefig(C.FIG_DIR / "fig7_example_link.png"); plt.close(fig)


def write_submission(meta):
    out = meta.sort_values(["link_ID", "target_slice_idx"]).copy()
    start = pd.to_datetime("2017-06-01") + pd.to_timedelta(out["target_min"], unit="m")
    end = start + pd.to_timedelta(C.SLICE_MINUTES, unit="m")
    out["time_interval"] = ("[" + start.dt.strftime("%H:%M:%S") + "," +
                            end.dt.strftime("%H:%M:%S") + ")")
    sub = out[["link_ID", "time_interval"]].copy()
    sub["predicted_travel_time"] = out["pred"].round(4).to_numpy()
    sub.to_csv(C.PRED_DIR / "prediction.csv", index=False)
    print(f"\n[SUBMIT] wrote {len(sub):,} rows to {C.PRED_DIR / 'prediction.csv'}")
    return sub


def main():
    t0 = time.time()
    if DEMO and not C.LINK_INFO_FILE.exists():
        import make_sample_data
        make_sample_data.write_sample(C.DATA_DIR)
    morning = dp.load_morning()
    link_info = dp.load_link_info()
    run_eda(morning, link_info)

    examples, feat_cols, link_info, aux = F.build_examples()
    best, best_name, meta, cmp, train, feat_cols = run_models(
        examples, feat_cols, link_info, aux)
    result_figures(meta, cmp, best_name)
    write_submission(meta)

    RESULTS["feature_cols"] = feat_cols
    RESULTS["runtime_s"] = round(time.time() - t0, 1)
    with open(C.OUT_DIR / "results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=2, default=float)
    print(f"\nAll done in {RESULTS['runtime_s']}s. results.json written.")


if __name__ == "__main__":
    main()
