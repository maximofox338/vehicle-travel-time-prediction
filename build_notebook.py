"""Assemble the deliverable Jupyter notebook (markdown + runnable code cells)."""
import nbformat as nbf
from src import config as C

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))

md(r"""
# Vehicle Travel Time Prediction

**Machine Learning Fundamentals — Final Practical Assignment**

Authors: Angel Miwa · Miroslava Lange · Maximo Fox — 2026/06/25

This notebook reproduces the full pipeline: streaming and caching the ~2 GB of travel-time
data, exploratory analysis, feature engineering at the monthly-profile granularity, model
comparison with leave-one-month-out cross-validation, the final evaluation on the June-2017
test month, and the generation of the submission file.

The written report (`Vehicle_Travel_Time_Prediction_Report.pdf`) discusses the methodology and
results in detail.
""")

md("## 1. Setup")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from src import config as C
from src import data_prep as dp
from src import features as F
from src import models as M

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
RNG = C.RANDOM_STATE
""")

md(r"""
## 2. Data

Three sources: link static attributes, the upstream/downstream topology, and the historical
2-minute travel times (three large zipped text files). The heavy scan keeps only the morning
rows [06:00, 09:00) and caches them; if the cache already exists it is reused.
""")
code(r"""
# One-off ~2 GB streaming scan (skipped automatically once the cache exists).
dp.build_cache()

link_info = dp.load_link_info()
topology = dp.load_topology()
print("links:", len(link_info), "| topology rows:", len(topology))
link_info.head()
""")
code(r"""
morning = dp.load_morning()
print("cached morning rows:", f"{len(morning):,}")
morning.head()
""")

md("## 3. Exploratory data analysis")
code(r"""
hhmm = lambda m: f"{m//60:02d}:{m%60:02d}"
prof = morning.groupby("minute_of_day")["travel_time"].mean()
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(prof.index, prof.values)
ax.axvspan(C.TARGET_START_MIN, C.TARGET_END_MIN, color="orange", alpha=0.15, label="[08:00,09:00) target")
ax.set_xticks(range(360, 541, 30)); ax.set_xticklabels([hhmm(m) for m in range(360, 541, 30)])
ax.set_xlabel("time of day"); ax.set_ylabel("avg travel time (s)")
ax.set_title("Average travel time across the morning"); ax.legend(); plt.show()
""")
code(r"""
# travel times are positive and right-skewed; link_class is constant; width carries signal
print("link_class values:", link_info["link_class"].unique())
print("width values     :", sorted(link_info["width"].unique()))
mw = morning.merge(link_info[["link_idx", "width"]], on="link_idx")
mw.boxplot("travel_time", by="width", showfliers=False, figsize=(6.5, 3.4))
plt.ylim(0, 60); plt.suptitle(""); plt.title("Travel time by link width")
plt.xlabel("width (m)"); plt.ylabel("travel time (s)"); plt.show()
""")

md(r"""
## 4. Feature engineering

The metric is computed over a monthly-averaged profile (one value per link and slice), so we
build the supervised problem at that granularity: one example per (month, link, target slice).
Each example carries static attributes, calendar context, the month's morning [06:00, 08:00)
signal, and a **leave-one-month-out** historical mean plus a topology-based neighbour mean.

The models do not predict the raw travel time but a **multiplicative correction**
`r = travel_time / historical_mean`; the prediction is recovered as `ŷ = r̂ · historical_mean`.
""")
code(r"""
examples, feat_cols, link_info, aux = F.build_examples()
print("features:", feat_cols)
examples[["link_ID", "month_key", "target_min"] + feat_cols + ["y"]].head()
""")

md(r"""
## 5. Models and leave-one-month-out cross-validation

We compare a historical-mean baseline, Ridge and Lasso linear regression (with the
regularisation strength chosen by cross-validated MAPE), and a small MLP. Models are compared
with leave-one-month-out CV over the six labelled training months.

In every fold the historical features are **rebuilt from the other five months only**
(`F.with_hist`), so the validation month cannot leak into the training rows' features.
Gradient-boosted trees (outside the course) run through the same protocol as an appendix and
are never eligible for selection.
""")
code(r"""
HIST_FLOOR = 0.1
ratio = lambda df: df["y"].to_numpy(float) / np.clip(df["hist_slice_mean"].to_numpy(float), HIST_FLOOR, None)

def predict_y(est, df):
    h = np.clip(df["hist_slice_mean"].to_numpy(float), HIST_FLOOR, None)
    return M.clip_positive(est.predict(df[feat_cols].to_numpy(float)) * h)

train = examples[examples.role == "train"].copy()
months = sorted(train.month_key.unique())
makers = {"Ridge": M.make_ridge, "Lasso": M.make_lasso, "MLP": M.make_mlp,
          "GBoost (appendix)": M.make_hgb}

rows = {n: [] for n in ["Baseline", *makers]}
for mo in months:
    fold, _ = F.with_hist(aux["raw"], [m for m in months if m != mo], aux)
    fold = fold[fold.role == "train"]
    tr, va = fold[fold.month_key != mo], fold[fold.month_key == mo]
    yva = va["y"].to_numpy(float)
    hva = np.clip(va["hist_slice_mean"].to_numpy(float), HIST_FLOOR, None)
    rows["Baseline"].append(M.mape(yva, hva))
    for name, mk in makers.items():
        est = mk().fit(tr[feat_cols].to_numpy(float), ratio(tr))
        rows[name].append(M.mape(yva, predict_y(est, va)))

cv = pd.DataFrame({n: [np.mean(v)] for n, v in rows.items()}, index=["mean LOMO MAPE (%)"]).T.round(2)
per_month = pd.DataFrame(rows, index=months).T.round(2)   # the 2016 vs 2017 split explains the CV/test gap
display(cv, per_month)
""")
code(r"""
ax = cv["mean LOMO MAPE (%)"].plot.bar(figsize=(6, 3.4),
        color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#bcbd22"], rot=0)
ax.set_ylabel("mean LOMO-CV MAPE (%)"); ax.set_title("Model comparison")
for i, v in enumerate(cv["mean LOMO MAPE (%)"]):
    ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
plt.show()

best_name = cv.drop(["Baseline", "GBoost (appendix)"])["mean LOMO MAPE (%)"].idxmin()
print("Selected model:", best_name)
""")

md(r"""
## 6. Final evaluation on the June 2017 test month

The selected model is retrained on all six training months and evaluated once against the
actual June-2017 profile (present in the dump, never used for training/selection).
""")
code(r"""
best = makers[best_name]().fit(train[feat_cols].to_numpy(float), ratio(train))
print("chosen hyper-parameters:", getattr(best, "best_params_", "n/a"))

grid_X, meta = F.build_test_grid(examples, link_info, aux)
h = np.clip(grid_X["hist_slice_mean"].to_numpy(float), HIST_FLOOR, None)
meta["pred"]  = M.clip_positive(best.predict(grid_X[feat_cols].to_numpy(float)) * h)
meta["bpred"] = M.clip_positive(grid_X["hist_slice_mean"].to_numpy())

have = meta["actual"].notna() & (meta["actual"] > 0)
gb = makers["GBoost (appendix)"]().fit(train[feat_cols].to_numpy(float), ratio(train))
meta["gbpred"] = M.clip_positive(gb.predict(grid_X[feat_cols].to_numpy(float)) * h)
res = pd.DataFrame({
    "Baseline": M.regression_metrics(meta.loc[have, "actual"], meta.loc[have, "bpred"]),
    best_name:  M.regression_metrics(meta.loc[have, "actual"], meta.loc[have, "pred"]),
    "GBoost (appendix)": M.regression_metrics(meta.loc[have, "actual"], meta.loc[have, "gbpred"]),
}).T.round(3)
res
""")
code(r"""
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.scatter(meta.loc[have, "actual"], meta.loc[have, "pred"], s=6, alpha=0.3)
lim = float(np.nanpercentile(meta.loc[have, "actual"], 99))
ax.plot([0, lim], [0, lim], "r--", lw=1); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("actual (s)"); ax.set_ylabel("predicted (s)")
ax.set_title(f"{best_name}: predicted vs actual (June 2017)"); plt.show()
""")

md("## 7. Submission file")
code(r"""
out = meta.sort_values(["link_ID", "target_slice_idx"]).copy()
start = pd.to_datetime("2017-06-01") + pd.to_timedelta(out["target_min"], unit="m")
end = start + pd.to_timedelta(C.SLICE_MINUTES, unit="m")
out["time_interval"] = "[" + start.dt.strftime("%H:%M:%S") + "," + end.dt.strftime("%H:%M:%S") + ")"
submission = out[["link_ID", "time_interval"]].copy()
submission["predicted_travel_time"] = out["pred"].round(4).to_numpy()
submission.to_csv(C.PRED_DIR / "prediction.csv", index=False)
print("rows:", len(submission))
submission.head()
""")

md(r"""
## 8. Conclusion

Using only methods from the course, a **Lasso regression on a multiplicative-correction
target** predicts the June-2017 [08:00, 09:00) profile with a **MAPE of ~13.6%**, beating both
the strong historical-average baseline (~17%) and a neural network. Ridge is statistically tied
with it. The test score is lower than the CV score (~18.5%) because the baseline itself is easier
on June 2017; the per-month table above shows why. Outside the course, gradient-boosted trees
reach ~11.4%. See the report for the full discussion.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
path = C.PACKAGE_ROOT / "vehicle_travel_time_prediction.ipynb"
nbf.write(nb, str(path))
print("wrote", path)
