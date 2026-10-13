"""
Feature engineering.

The task asks for the *average* travel time of every 2-minute slice in [08:00, 09:00) for
June 2017, and the MAPE metric is computed over (link x slice) -- i.e. over a monthly-average
profile.  We therefore build the supervised problem at that same granularity: one example per
(month, link, target-slice), whose target is the month's average travel time for that slice.

Each example carries:
  * static link attributes        (length, width)
  * context                       (slice index, clock minute, month number, year)
  * that month's morning signal   [06:00,08:00): mean / recent means / one-hour-earlier / trend
  * historical aggregates         leave-one-month-out mean for this link&slice, neighbour mean

Roles:
  * train : 2016-03..2016-06 and 2017-03..2017-05   (the stated training period)
  * test  : June 2017                               (the month we must predict)
  * drop  : out-of-scope months (2016-07 / 2017-07)

All historical features use a leave-one-month-out scheme, so a month is never informed by
itself.  They are computed by `with_hist`, which only looks at the months it is given: the
cross-validation calls it once per fold *without* the validation month, so that month cannot
leak into the features of the training rows either.  Everything is plain feature engineering -- no time-series-specific models -- so it
stays inside the scope of the course.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import data_prep as dp

FEAT_MINS = list(range(C.FEAT_START_MIN, C.FEAT_END_MIN, C.SLICE_MINUTES))        # 60 slices, [06:00,08:00)
TARGET_MINS = list(range(C.TARGET_START_MIN, C.TARGET_END_MIN, C.SLICE_MINUTES))  # 30 slices, [08:00,09:00)
ALL_MINS = list(range(C.WINDOW_START_MIN, C.WINDOW_END_MIN, C.SLICE_MINUTES))     # 90 slices

MONTH_FEAT_COLS = ["m_feat_mean", "m_feat_first30", "m_feat_last30", "m_feat_last10",
                   "m_feat_trend", "m_feat_std", "m_feat_hour_ago"]
HIST_COLS = ["hist_slice_mean", "neighbor_slice_mean"]
STATIC_COLS = ["length", "width"]
CONTEXT_COLS = ["target_slice_idx", "month_num", "year"]   # target_min = 480 + 2*idx, redundant
FEATURE_COLS = STATIC_COLS + CONTEXT_COLS + MONTH_FEAT_COLS + HIST_COLS

TRAIN_MONTHS = [201603, 201604, 201605, 201606, 201703, 201704, 201705]
TEST_MONTH = 201706


def _month_key(dates: pd.Series) -> np.ndarray:
    dt = dates.dt
    return (dt.year * 100 + dt.month).to_numpy()


def monthly_profiles() -> pd.DataFrame:
    """Average morning profile per (month, link): index (month_key, link_idx), 90 slice cols."""
    morning = dp.load_morning()
    mk = _month_key(pd.to_datetime(morning["date"]))
    morning = morning.assign(month_key=mk)
    prof = (morning.groupby(["month_key", "link_idx", "minute_of_day"])["travel_time"]
            .mean().unstack("minute_of_day"))
    return prof.reindex(columns=ALL_MINS)


def build_examples(verbose: bool = True):
    """Return (examples, FEATURE_COLS, link_info, aux) at the (month, link, slice) granularity."""
    link_info = dp.load_link_info()
    topo = dp.load_topology()
    prof = monthly_profiles()

    months = prof.index.get_level_values("month_key").to_numpy()
    keep = np.isin(months, TRAIN_MONTHS + [TEST_MONTH])
    prof = prof[keep]

    # ---- month-level morning features (one value per (month, link)) ----
    feat = prof[FEAT_MINS]
    first30 = [m for m in FEAT_MINS if m < C.FEAT_START_MIN + 30]
    last30 = [m for m in FEAT_MINS if m >= C.FEAT_END_MIN - 30]
    last10 = [m for m in FEAT_MINS if m >= C.FEAT_END_MIN - 10]
    mf = pd.DataFrame(index=prof.index)
    mf["m_feat_mean"] = feat.mean(axis=1)
    mf["m_feat_first30"] = feat[first30].mean(axis=1)
    mf["m_feat_last30"] = feat[last30].mean(axis=1)
    mf["m_feat_last10"] = feat[last10].mean(axis=1)
    mf["m_feat_trend"] = mf["m_feat_last30"] - mf["m_feat_first30"]
    mf["m_feat_std"] = feat.std(axis=1)

    # ---- explode to one row per (month, link, target-slice) ----
    blocks = []
    for k, m in enumerate(TARGET_MINS):
        blk = mf.copy()
        blk["target_slice_idx"] = k
        blk["target_min"] = m
        blk["m_feat_hour_ago"] = prof[m - 60].to_numpy()
        blk["y"] = prof[m].to_numpy()
        blocks.append(blk.reset_index())
    ex = pd.concat(blocks, ignore_index=True)
    ex["month_num"] = (ex["month_key"] % 100).astype(int)
    ex["year"] = (ex["month_key"] // 100).astype(int)
    ex["role"] = np.where(ex["month_key"] == TEST_MONTH, "test",
                          np.where(ex["month_key"].isin(TRAIN_MONTHS), "train", "drop"))

    # ---- static attributes ----
    info_idx = link_info.set_index("link_idx")
    ex["length"] = info_idx["length"].reindex(ex["link_idx"]).to_numpy()
    ex["width"] = info_idx["width"].reindex(ex["link_idx"]).to_numpy()

    ex["link_ID"] = info_idx["link_ID"].reindex(ex["link_idx"]).to_numpy()
    # training rows need a positive target (MAPE divides by the actual value)
    drop_tr = (ex["role"] == "train") & ~(ex["y"] > 0)
    ex = ex[~drop_tr].reset_index(drop=True)

    # ---- topology: neighbour (upstream + downstream) link indices ----
    id_to_idx = dict(zip(link_info["link_ID"], link_info["link_idx"]))
    neighbors = {}
    for _, r in topo.iterrows():
        idx = id_to_idx.get(r["link_ID"])
        if idx is None:
            continue
        neighbors[idx] = [id_to_idx[x] for x in (r["in_links"] + r["out_links"]) if x in id_to_idx]

    # "raw" keeps the un-filled rows so each CV fold can rebuild its features from scratch
    aux = {"link_info": link_info, "neighbors": neighbors, "raw": ex}
    ex, aux["global_mean"] = with_hist(ex, TRAIN_MONTHS, aux)
    if verbose:
        labelled = sorted(ex.loc[ex.role == "train", "month_key"].unique().tolist())
        print(f"Built {len(ex):,} (month,link,slice) rows | "
              f"train={int((ex.role=='train').sum()):,} test={int((ex.role=='test').sum()):,}")
        print(f"Labelled training months: {labelled} | test month: {TEST_MONTH} "
              f"(June 2016 is morning-only, so it carries no [08:00,09:00) labels)")
    return ex, FEATURE_COLS, link_info, aux


def with_hist(ex: pd.DataFrame, label_months, aux: dict):
    """Add the historical features using the labels of `label_months` only.

    Rows of a label month get a leave-one-month-out mean (their own month excluded); every other
    row (a CV validation month, the test month) gets the plain mean over `label_months`.  Missing
    values are then filled.  Returns (examples_with_features, global_mean_of_label_months).
    """
    ex = ex.copy()
    keys = ["link_idx", "target_slice_idx"]
    is_lab = ex["month_key"].isin(label_months).to_numpy()
    lab = ex[is_lab]

    # ---- historical mean for (link, slice) ----
    grp = lab.groupby(keys)["y"]
    full_mean = (grp.sum() / grp.count()).rename("hm").reset_index()
    hist = np.array(ex[keys].merge(full_mean, how="left")["hm"], dtype=float)
    hist[is_lab] = ((grp.transform("sum") - lab["y"]) / (grp.transform("count") - 1)).to_numpy()
    ex["hist_slice_mean"] = hist

    # ---- neighbour mean = average of neighbours' hist_slice_mean (same month & slice) ----
    n_links = len(aux["link_info"])
    mkeys = sorted(ex["month_key"].unique())
    mpos = ex["month_key"].map({mk: i for i, mk in enumerate(mkeys)}).to_numpy()
    li = ex["link_idx"].to_numpy().astype(int)
    si = ex["target_slice_idx"].to_numpy().astype(int)
    H = np.full((len(mkeys), n_links, C.N_TARGET_SLICES), np.nan)      # H[month, link, slice]
    H[mpos, li, si] = hist
    Hn = np.full_like(H, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                 # "Mean of empty slice"
        for l in range(n_links):
            nb = aux["neighbors"].get(l, [])
            Hn[:, l, :] = np.nanmean(H[:, nb, :], axis=1) if nb else H[:, l, :]
    ex["neighbor_slice_mean"] = Hn[mpos, li, si]

    # ---- fill any remaining gaps with sensible fall-backs (no NaN for sklearn) ----
    for col in MONTH_FEAT_COLS:
        ex[col] = ex[col].fillna(ex["hist_slice_mean"])
    ex["neighbor_slice_mean"] = ex["neighbor_slice_mean"].fillna(ex["hist_slice_mean"])
    gmean = float(np.nanmean(lab["y"]))
    ex[FEATURE_COLS] = ex[FEATURE_COLS].fillna(gmean)
    return ex, gmean


def build_test_grid(examples: pd.DataFrame, link_info: pd.DataFrame, aux: dict):
    """Full 132x30 June grid for the submission (links with no June data get hist fall-backs)."""
    test = examples[examples["role"] == "test"]
    full = pd.MultiIndex.from_product(
        [link_info["link_idx"].to_numpy(), range(C.N_TARGET_SLICES)],
        names=["link_idx", "target_slice_idx"])
    grid = test.set_index(["link_idx", "target_slice_idx"]).reindex(full)

    si = grid.index.get_level_values("target_slice_idx").to_numpy()
    li = grid.index.get_level_values("link_idx").to_numpy().astype(int)
    info_idx = link_info.set_index("link_idx")
    grid["target_slice_idx"] = si
    grid["target_min"] = C.TARGET_START_MIN + C.SLICE_MINUTES * si
    grid["month_num"] = grid["month_num"].fillna(C.PREDICT_MONTH)
    grid["year"] = grid["year"].fillna(C.PREDICT_YEAR)
    grid["length"] = grid["length"].fillna(pd.Series(info_idx["length"].reindex(li).to_numpy(), index=grid.index))
    grid["width"] = grid["width"].fillna(pd.Series(info_idx["width"].reindex(li).to_numpy(), index=grid.index))
    grid[MONTH_FEAT_COLS + HIST_COLS] = grid[MONTH_FEAT_COLS + HIST_COLS].fillna(aux["global_mean"])
    for col in MONTH_FEAT_COLS:
        grid[col] = grid[col].fillna(grid["hist_slice_mean"])

    meta = pd.DataFrame({"link_idx": li, "target_slice_idx": si})
    meta["link_ID"] = info_idx["link_ID"].reindex(meta["link_idx"]).to_numpy()
    meta["target_min"] = C.TARGET_START_MIN + C.SLICE_MINUTES * si
    meta["actual"] = grid["y"].to_numpy()           # NaN where June has no data
    return grid[FEATURE_COLS].reset_index(drop=True), meta


if __name__ == "__main__":
    ex, cols, li, aux = build_examples()
    print(ex[["link_ID", "month_key", "target_min"] + cols + ["y"]].head())
