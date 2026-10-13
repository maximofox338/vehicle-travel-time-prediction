"""
Minimal checks for the parts that are easy to get silently wrong.  No data needed.

    python test_pipeline.py      (or: pytest test_pipeline.py)
"""
import numpy as np
import pandas as pd

from src import config as C
from src import features as F
from src import models as M


def _toy_examples(months=(201603, 201604, 201605), n_links=3):
    rng = np.random.default_rng(0)
    rows = [dict(month_key=mk, link_idx=l, target_slice_idx=s, role="train",
                 y=float(rng.uniform(5, 50)))
            for mk in months for l in range(n_links) for s in range(C.N_TARGET_SLICES)]
    ex = pd.DataFrame(rows)
    for col in F.STATIC_COLS + F.CONTEXT_COLS + F.MONTH_FEAT_COLS:
        if col not in ex:
            ex[col] = 1.0
    aux = {"link_info": pd.DataFrame({"link_idx": range(n_links)}),
           "neighbors": {l: [x for x in (l - 1, l + 1) if 0 <= x < n_links] for l in range(n_links)}}
    return ex, aux


def test_mape():
    assert np.isclose(M.mape([10, 20], [11, 18]), 10.0)      # (10% + 10%) / 2
    assert np.isclose(M.mape([0, 10], [5, 12]), 20.0)        # zero actuals are skipped


def test_ratio_target_round_trip():
    # the models learn r = y / hist and we predict r * hist: a perfect r must give back y
    y, hist = np.array([4.0, 9.0, 30.0]), np.array([5.0, 10.0, 20.0])
    assert np.allclose((y / hist) * hist, y)


def test_hist_is_leave_one_month_out():
    ex, aux = _toy_examples()
    out, _ = F.with_hist(ex, [201603, 201604, 201605], aux)
    row = out[(out.month_key == 201603) & (out.link_idx == 0) & (out.target_slice_idx == 0)]
    others = ex[(ex.month_key != 201603) & (ex.link_idx == 0) & (ex.target_slice_idx == 0)]
    assert np.isclose(row["hist_slice_mean"].item(), others["y"].mean())


def test_validation_month_does_not_leak():
    # A CV fold excludes the validation month: changing its labels must not move any feature.
    ex, aux = _toy_examples()
    folds = [201603, 201604]                                  # 201605 is the validation month
    a, _ = F.with_hist(ex, folds, aux)
    ex2 = ex.copy()
    ex2.loc[ex2.month_key == 201605, "y"] *= 10
    b, _ = F.with_hist(ex2, folds, aux)
    assert np.allclose(a[F.FEATURE_COLS].to_numpy(), b[F.FEATURE_COLS].to_numpy())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
