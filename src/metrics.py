"""Evaluation metrics shared by the ML and LLM pipelines."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, name: str = "") -> dict:
    """RMSE, MAE, R², MAPE on the valid (non-NaN) predictions.

    LLM runs may contain NaNs (parse failures / content filters);
    metrics are computed on the valid subset and n_valid is reported
    so that incomplete configurations can be identified.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~np.isnan(y_pred)
    n_valid = int(valid.sum())
    if n_valid < 5:
        return {"Model": name, "RMSE": np.nan, "MAE": np.nan,
                "R2": np.nan, "MAPE%": np.nan, "n_valid": n_valid}
    yt, yp = y_true[valid], y_pred[valid]
    return {
        "Model": name,
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "R2": float(r2_score(yt, yp)),
        "MAPE%": float(np.mean(np.abs((yt - yp) / yt)) * 100),
        "n_valid": n_valid,
    }
