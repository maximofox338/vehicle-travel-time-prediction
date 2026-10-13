"""
Models and evaluation.

We stay strictly within the methods taught in the course:

  * a historical-average baseline (just a group-by mean, no learning),
  * Ridge and Lasso linear regression (L2 / L1 regularisation, tuned by cross-validation),
  * a small Multi-Layer Perceptron (feed-forward neural network, Ch. 8).

Outside the course, as an appendix only (never eligible as the selected model):

  * gradient-boosted trees (scikit-learn's HistGradientBoostingRegressor).

The competition metric is MAPE; we also report MAE, RMSE and R2 (the regression metrics
introduced in Ch. 1).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

EPS = 1e-6


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (%) -- the official competition metric.

    MAPE = mean_over_all(link, slice) |ttp - ttr| / ttr .  Only positive actual values
    contribute (the metric divides by the actual travel time)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > EPS
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]) * 100.0)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "MAPE(%)": mape(y_true, y_pred),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


# We do not predict the raw travel time.  The historical mean is already a strong predictor,
# so the models learn the *multiplicative correction* r = travel_time / historical_mean (built
# in make_results).  Because r is centred near 1, it is well scaled for regression, and -- since
# the historical mean cancels -- the MAPE on the travel time equals the MAPE on r.  We can
# therefore select hyper-parameters with a MAPE scorer and optimise the competition metric.
mape_scorer = make_scorer(mape, greater_is_better=False)


def make_ridge() -> GridSearchCV:
    """Standardise + Ridge, alpha chosen by MAPE cross-validation."""
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    grid = {"model__alpha": [0.1, 1.0, 10.0, 30.0, 100.0, 300.0]}
    cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    return GridSearchCV(pipe, grid, scoring=mape_scorer, cv=cv, n_jobs=-1)


def make_lasso() -> GridSearchCV:
    """Standardise + Lasso, alpha chosen by MAPE cross-validation."""
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=5000))])
    grid = {"model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}
    cv = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    return GridSearchCV(pipe, grid, scoring=mape_scorer, cv=cv, n_jobs=-1)


def make_mlp() -> Pipeline:
    """Standardise + a small MLP (ReLU, Adam, L2 weight decay, early stopping)."""
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-2,                 # L2 regularisation
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=12,
        validation_fraction=0.1,
        random_state=C.RANDOM_STATE,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def make_hgb() -> HistGradientBoostingRegressor:
    """Gradient-boosted trees -- outside the course syllabus, reported as an appendix.

    Absolute-error loss: on r (centred at 1) it is a close proxy for MAPE.  Trees need no scaling."""
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=C.RANDOM_STATE,
    )


def clip_positive(pred, floor: float = 0.1):
    """Travel times are strictly positive; clip predictions to a small positive floor."""
    return np.maximum(np.asarray(pred, dtype=float), floor)
