"""Prompt builders for the LLM evaluation — verbatim to the paper.

Two strategies:
  * Few-shot: 37 parameter→Ra demonstrations, no formula, numeric-only reply.
  * Chain-of-thought: kinematic formula Ra_th = f²/(32·rε) pre-computed for
    the target, followed by the demonstrations, step-by-step instruction.

NOTE ON THE CoT PRIOR: with f and rε in mm, f²/(32·rε) yields a value in
mm, not µm. The prompts used in the study label the raw value as µm
(e.g. 0.0024 µm instead of 2.39 µm for the centre point). The prompts are
kept verbatim here for reproducibility of the published results; see
README for discussion.
"""

import re

import numpy as np

SYSTEM_MSG_FEWSHOT = (
    "You are a precision machining expert. "
    "Your task is to predict surface roughness Ra (µm) in turning of "
    "Ti-6Al-4V titanium alloy. "
    "Always respond with ONLY a single positive number (the Ra value in µm). "
    "No explanation, no units, no text — just the number."
)

SYSTEM_MSG_COT = (
    "You are a precision machining expert predicting Ra (µm) "
    "for Ti-6Al-4V turning. "
    "You may reason briefly, but your LAST LINE must be "
    "ONLY the numeric Ra prediction — nothing else."
)

VALID_RANGE = (0.01, 50.0)  # µm; outside → parse failure


def build_fewshot_prompt(train_rows, test_row) -> str:
    """train_rows: iterable of (Vc, f, ap, Re, Ra); test_row: (Vc, f, ap, Re)."""
    examples = "\n".join(
        f"Vc={r[0]:.1f} m/min, f={r[1]:.4f} mm/r, "
        f"ap={r[2]:.4f} mm, re={r[3]:.2f} mm → Ra={r[4]:.4f} µm"
        for r in train_rows
    )
    query = (
        f"Vc={test_row[0]:.1f} m/min, f={test_row[1]:.4f} mm/r, "
        f"ap={test_row[2]:.4f} mm, re={test_row[3]:.2f} mm → Ra=?"
    )
    return (
        "Below are experimental measurements of surface roughness Ra (µm) "
        "during turning of Ti-6Al-4V:\n\n"
        f"{examples}\n\n"
        f"Now predict Ra for:\n{query}\n\n"
        "Respond with only the numeric value."
    )


def build_cot_prompt(train_rows, test_row) -> str:
    ra_teo = test_row[1] ** 2 / (32 * test_row[3])
    examples = "\n".join(
        f"Vc={r[0]:.1f}, f={r[1]:.4f}, ap={r[2]:.4f}, re={r[3]:.2f} "
        f"→ Ra={r[4]:.4f}"
        for r in train_rows
    )
    return (
        "You are predicting Ra (µm) for Ti-6Al-4V turning.\n\n"
        "PHYSICAL PRIOR: The theoretical roughness formula is:\n"
        "    Ra_theoretical = f² / (32 · rε)\n"
        "where f is feed rate (mm/rev) and rε is nose radius (mm).\n\n"
        f"For the target point, Ra_theoretical = "
        f"{test_row[1]:.4f}² / (32 × {test_row[3]:.2f}) = {ra_teo:.4f} µm\n\n"
        "EXPERIMENTAL DATA (use to correct for real-world deviations):\n"
        f"{examples}\n\n"
        "TARGET:\n"
        f"Vc={test_row[0]:.1f}, f={test_row[1]:.4f}, "
        f"ap={test_row[2]:.4f}, re={test_row[3]:.2f}\n\n"
        "Consider both the theoretical value and the pattern in the data. "
        "Think step by step, then on your LAST LINE write only the number."
    )


def extract_number(text: str) -> float:
    """Last numeric token in the response; NaN if none or out of range."""
    matches = re.findall(r"\d+\.?\d*", str(text).strip())
    if matches:
        val = float(matches[-1])
        lo, hi = VALID_RANGE
        return val if lo < val < hi else np.nan
    return np.nan
