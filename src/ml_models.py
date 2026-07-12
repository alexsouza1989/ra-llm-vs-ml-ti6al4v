"""Five ML models evaluated under nested LOOCV with inner grid search.

Models: Power Law, RSM (2nd-order polynomial + Ridge), SVR (RBF),
Gaussian Process Regression (ARD-like RBF kernel), XGBoost.

Protocol (as in the paper):
  * Outer loop: Leave-One-Out over the 38 observations.
  * Inner loop: 5-fold GridSearchCV on the 37 training points,
    scoring = negative RMSE, random_state = 42.
  * Power Law has no hyperparameters (plain LOOCV, fitted by OLS
    on log-transformed data).
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

INNER_FOLDS = 5
SCORING = "neg_root_mean_squared_error"
SEED = 42

RSM_GRID = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 50.0]}
SVR_GRID = {
    "svr__C": [10, 100, 500],
    "svr__epsilon": [0.05, 0.1, 0.3],
    "svr__gamma": ["scale", "auto", 0.1],
}
GPR_RESTARTS = [3, 7, 15]
XGB_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
}


def _inner_cv():
    return KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)


# ── Model builders ─────────────────────────────────────────────────────────

def fit_power_law(Xtr, ytr):
    """Ra = C * Vc^b1 * f^b2 * ap^b3 * re^b4, fitted by OLS in log space."""
    lr = LinearRegression().fit(np.log(Xtr), np.log(ytr))

    class Wrap:
        def predict(self, Xt):
            return np.exp(lr.predict(np.log(Xt)))

    return Wrap()


def fit_rsm(Xtr, ytr):
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=True)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])
    gs = GridSearchCV(pipe, RSM_GRID, cv=_inner_cv(),
                      scoring=SCORING, refit=True, n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs


def fit_svr(Xtr, ytr):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])
    gs = GridSearchCV(pipe, SVR_GRID, cv=_inner_cv(),
                      scoring=SCORING, refit=True, n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs


def _gpr_kernel():
    return (C(1.0, (1e-3, 1e3))
            * RBF(length_scale=[1.0] * 4, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1)))


def fit_gpr(Xtr, ytr):
    """Select n_restarts_optimizer via inner 5-fold CV, refit on all points."""
    sc = StandardScaler()
    Xsc = sc.fit_transform(Xtr)
    inner = _inner_cv()

    best_restarts, best_rmse = GPR_RESTARTS[0], np.inf
    for n_rest in GPR_RESTARTS:
        fold_rmses = []
        for tr_idx, vl_idx in inner.split(Xsc):
            gpr = GaussianProcessRegressor(
                kernel=_gpr_kernel(), n_restarts_optimizer=n_rest,
                normalize_y=True, random_state=SEED)
            gpr.fit(Xsc[tr_idx], ytr[tr_idx])
            pv = gpr.predict(Xsc[vl_idx])
            fold_rmses.append(np.sqrt(mean_squared_error(ytr[vl_idx], pv)))
        cv_rmse = float(np.mean(fold_rmses))
        if cv_rmse < best_rmse:
            best_rmse, best_restarts = cv_rmse, n_rest

    gpr_final = GaussianProcessRegressor(
        kernel=_gpr_kernel(), n_restarts_optimizer=best_restarts,
        normalize_y=True, random_state=SEED)
    gpr_final.fit(Xsc, ytr)

    class Wrap:
        best_params_ = {"n_restarts_optimizer": best_restarts}

        def predict(self, Xt):
            return gpr_final.predict(sc.transform(Xt))

    return Wrap()


def fit_xgb(Xtr, ytr):
    gs = GridSearchCV(
        XGBRegressor(colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                     verbosity=0, random_state=SEED),
        XGB_GRID, cv=_inner_cv(), scoring=SCORING, refit=True, n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs


MODEL_BUILDERS = {
    "Power Law": fit_power_law,
    "RSM (2nd order)": fit_rsm,
    "SVR (RBF)": fit_svr,
    "Gaussian Process": fit_gpr,
    "XGBoost": fit_xgb,
}


# ── Nested LOOCV driver ────────────────────────────────────────────────────

def nested_loocv(build_fn, X, y, verbose: bool = True, name: str = ""):
    """Outer LOOCV; build_fn embeds the inner hyperparameter search."""
    n = len(y)
    preds = np.empty(n)
    best_params = []

    for train_idx, test_idx in LeaveOneOut().split(X):
        m = build_fn(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict(X[test_idx])
        if hasattr(m, "best_params_"):
            best_params.append(m.best_params_)

    if best_params and verbose:
        pdf = pd.DataFrame(best_params)
        print(f"\n  {name} — hyperparameter selection across {n} folds:")
        for col in pdf.columns:
            counts = pdf[col].value_counts()
            print(f"    {col:30s}: most selected = {counts.index[0]} "
                  f"({counts.iloc[0] / n * 100:.0f}% of folds)")

    return preds
