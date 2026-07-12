#!/usr/bin/env python3
"""Reproduce the ML results (Table 3 of the paper).

Usage:
    python scripts/run_ml.py

Runs the five ML models under nested LOOCV (5-fold inner grid search)
on the 38-point Ti-6Al-4V CCD dataset and prints the summary table.
Predictions are saved to results/ml_loocv_predictions.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import FEATURES, load_dataset, load_xy  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.ml_models import MODEL_BUILDERS, nested_loocv  # noqa: E402

np.random.seed(42)


def main() -> None:
    X, y = load_xy()
    df = load_dataset()
    print("=" * 65)
    print("  Ra Prediction — Ti-6Al-4V | Nested LOOCV + Grid Search")
    print("=" * 65)
    print(f"  n = {len(y)} | Features: {FEATURES}")
    print(f"  Ra: min={y.min():.2f} max={y.max():.2f} "
          f"mean={y.mean():.2f} std={y.std():.2f} µm")

    results, preds_out = [], {"Ra_true": y}
    for name, builder in MODEL_BUILDERS.items():
        print(f"\n── {name} ──")
        preds = nested_loocv(builder, X, y, name=name)
        results.append(compute_metrics(y, preds, name))
        preds_out[name] = preds

    summary = (pd.DataFrame(results).set_index("Model")
               .sort_values("RMSE").round(4))
    print("\n" + "=" * 65)
    print("  NESTED LOOCV SUMMARY — sorted by RMSE")
    print("=" * 65)
    print(summary.to_string())

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(preds_out).to_csv(out_dir / "ml_loocv_predictions.csv",
                                   index=False)
    summary.to_csv(out_dir / "ml_loocv_summary.csv")
    print(f"\nSaved → {out_dir}/ml_loocv_predictions.csv, ml_loocv_summary.csv")


if __name__ == "__main__":
    main()
