"""LOOCV driver for LLM predictors, with raw-response logging.

Each configuration = (model, strategy). For each of the 38 folds, the
37 remaining observations are embedded in the prompt as demonstrations
and the model predicts Ra for the held-out point. The full raw API
response is logged for every call, enabling post-hoc inspection of
reasoning chains and parse failures.

The paper protocol runs each configuration in 3 fully independent
LOOCV sessions and reports mean ± std across runs.
"""

import time
from functools import partial

import numpy as np
import pandas as pd

from .llm_clients import CALLERS
from .llm_prompts import (SYSTEM_MSG_COT, SYSTEM_MSG_FEWSHOT,
                          build_cot_prompt, build_fewshot_prompt,
                          extract_number)
from .metrics import compute_metrics

STRATEGIES = {
    "fewshot": dict(prompt_fn=build_fewshot_prompt,
                    system=SYSTEM_MSG_FEWSHOT, max_tokens=32),
    "cot": dict(prompt_fn=build_cot_prompt,
                system=SYSTEM_MSG_COT, max_tokens=512),
}


def llm_loocv(model_key: str, strategy: str, X, y,
              delay: float = 0.5, verbose: bool = True):
    """Run one full LOOCV session for a (model, strategy) configuration.

    Returns (preds, raw_df) where raw_df logs every prompt outcome.
    """
    if model_key not in CALLERS:
        raise ValueError(f"Unknown model '{model_key}'; use {list(CALLERS)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'; use {list(STRATEGIES)}")

    cfg = STRATEGIES[strategy]
    call_fn = partial(CALLERS[model_key],
                      system=cfg["system"], max_tokens=cfg["max_tokens"])
    prompt_fn = cfg["prompt_fn"]

    n = len(y)
    preds = np.empty(n)
    rows = []

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        train_rows = [(X[j, 0], X[j, 1], X[j, 2], X[j, 3], y[j])
                      for j in train_idx]
        test_row = (X[i, 0], X[i, 1], X[i, 2], X[i, 3])

        raw = call_fn(prompt_fn(train_rows, test_row))
        val = extract_number(raw)
        preds[i] = val

        rows.append({"idx": i, "Vc": X[i, 0], "f": X[i, 1],
                     "ap": X[i, 2], "Re": X[i, 3],
                     "Ra_true": y[i], "Ra_pred": val, "raw_resp": raw})
        if verbose:
            status = f"{val:.4f}" if not np.isnan(val) else "parse error"
            print(f"    [{model_key}/{strategy}] {i + 1:2d}/{n} | "
                  f"true={y[i]:.4f} | pred={status}")
        time.sleep(delay)

    return preds, pd.DataFrame(rows)


def run_three_runs(model_key: str, strategy: str, X, y,
                   n_runs: int = 3, delay: float = 0.5, out_dir=None):
    """Paper protocol: n_runs independent LOOCV sessions, mean ± std.

    If out_dir is given, raw responses of each run are saved as CSV.
    Returns a summary dict and the list of per-run metric dicts.
    """
    per_run = []
    for run in range(1, n_runs + 1):
        print(f"\n=== {model_key}/{strategy} — run {run}/{n_runs} ===")
        preds, raw_df = llm_loocv(model_key, strategy, X, y, delay=delay)
        m = compute_metrics(y, preds, name=f"{model_key}/{strategy}")
        m["run"] = run
        per_run.append(m)
        if out_dir is not None:
            path = f"{out_dir}/raw_{model_key}_{strategy}_run{run}.csv"
            raw_df.to_csv(path, index=False)
            print(f"    raw responses saved → {path}")

    rmses = [m["RMSE"] for m in per_run]
    r2s = [m["R2"] for m in per_run]
    nvs = [m["n_valid"] for m in per_run]
    summary = {
        "Model": f"{model_key}/{strategy}",
        "RMSE_mean": float(np.nanmean(rmses)),
        "RMSE_std": float(np.nanstd(rmses, ddof=1)) if n_runs > 1 else 0.0,
        "R2_mean": float(np.nanmean(r2s)),
        "R2_std": float(np.nanstd(r2s, ddof=1)) if n_runs > 1 else 0.0,
        "n_valid_per_run": nvs,
    }
    return summary, per_run
