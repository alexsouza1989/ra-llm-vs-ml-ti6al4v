#!/usr/bin/env python3
"""Reproduce the LLM results (Table 4 of the paper).

Requires API keys in environment variables (see .env.example).
WARNING: a full reproduction is 3 runs × 38 calls × 6 configurations
= 684 API calls, which incurs cost. Use --models/--strategies/--runs
to run a subset.

Usage examples:
    python scripts/run_llm.py --models claude --strategies fewshot --runs 1
    python scripts/run_llm.py --models claude gpt gemini --runs 3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import load_xy  # noqa: E402
from src.llm_loocv import run_three_runs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+",
                    default=["claude", "gpt", "gemini"],
                    choices=["claude", "gpt", "gemini"])
    ap.add_argument("--strategies", nargs="+",
                    default=["fewshot", "cot"],
                    choices=["fewshot", "cot"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.5,
                    help="seconds between API calls (rate limiting)")
    args = ap.parse_args()

    X, y = load_xy()
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    summaries = []
    for model in args.models:
        for strategy in args.strategies:
            summary, _ = run_three_runs(model, strategy, X, y,
                                        n_runs=args.runs,
                                        delay=args.delay,
                                        out_dir=str(out_dir))
            summaries.append(summary)

    df = pd.DataFrame(summaries).sort_values("RMSE_mean")
    print("\n" + "=" * 65)
    print(f"  LLM LOOCV SUMMARY — mean ± std across {args.runs} run(s)")
    print("=" * 65)
    print(df.round(4).to_string(index=False))
    df.to_csv(out_dir / "llm_loocv_summary.csv", index=False)
    print(f"\nSaved → {out_dir}/llm_loocv_summary.csv")


if __name__ == "__main__":
    main()
