# In-Context Learning vs. Trained Models — Ra Prediction in Ti-6Al-4V Dry Turning

Code and data for the paper *"In-Context Learning Versus Trained Models:
Generative AI and Machine Learning for Surface Roughness Prediction in Dry
Ti-6Al-4V Turning"* (Souza, Campos, Verri, Balestrassi).

Five machine learning models (Power Law, RSM, SVR, Gaussian Process,
XGBoost) and three large language models (Claude Sonnet 4.5, GPT-4o-mini,
Gemini 3 Flash Preview) are compared as predictors of surface roughness
Ra on a 38-point Central Composite Design, under identical Leave-One-Out
Cross-Validation. ML models use nested 5-fold grid search; LLMs are
evaluated at temperature 0 under few-shot and chain-of-thought prompting,
across 3 independent runs (mean ± std).

## Key result

| Model | Type | RMSE (µm) | R² |
|---|---|---|---|
| Claude FewShot | LLM | 0.288 ± 0.016 | 0.985 |
| SVR (RBF) | ML | 0.341 | 0.978 |
| RSM (2nd order) | ML | 0.369 | 0.975 |
| Gaussian Process | ML | 0.459 | 0.961 |
| XGBoost | ML | 0.691 | 0.912 |
| Power Law | ML | 1.033 | 0.802 |

Full table (all 11 configurations): [`results/paper_results.csv`](results/paper_results.csv).

## Repository layout

```
data/ti6al4v_ccd38.csv    38-point CCD dataset (Vc, f, ap, Re → Ra)
src/dataset.py            data loading
src/metrics.py            RMSE / MAE / R² / MAPE with n_valid tracking
src/ml_models.py          5 ML models + nested LOOCV driver
src/llm_prompts.py        few-shot / CoT prompt builders (verbatim)
src/llm_clients.py        Anthropic / OpenAI / Gemini clients (env-var keys)
src/llm_loocv.py          LLM LOOCV + 3-run protocol, raw-response logging
scripts/run_ml.py         reproduce Table 3 (ML results)
scripts/run_llm.py        reproduce Table 4 (LLM results; requires API keys)
results/                  published results + outputs of the scripts
legacy/                   original Colab export (sanitised), for provenance
```

## Setup

```bash
git clone <this-repo>
cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the ML results (no API keys needed)

```bash
python scripts/run_ml.py
```

Deterministic (seed 42): reproduces Table 3 of the paper. Runtime is a
few minutes on a laptop CPU (XGBoost's 81-combination grid × 38 folds
dominates).

## Reproducing the LLM results

1. Copy `.env.example` to `.env` and fill in your keys, or export them:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

2. Run (full protocol = 684 API calls; use flags to run subsets):

```bash
# Single cheapest sanity check: 38 calls
python scripts/run_llm.py --models claude --strategies fewshot --runs 1

# Full paper protocol
python scripts/run_llm.py --runs 3
```

Raw API responses of every call are saved to
`results/raw_<model>_<strategy>_run<k>.csv` for auditability.

**Never commit `.env` or API keys.** The `.gitignore` excludes `.env`.

### Reproducibility caveat

LLM results depend on remotely-hosted, versioned commercial models.
The identifiers used in the study are pinned in `src/llm_clients.py`
(`claude-sonnet-4-5`, `gpt-4o-mini`, `gemini-3-flash-preview`), but
providers may update or retire models; exact numeric reproduction of
the paper's LLM results is therefore not guaranteed over time. This
limitation is discussed in the paper. ML results are fully
deterministic and will reproduce exactly.

### Note on the chain-of-thought prior

The kinematic formula `Ra_th = f²/(32·rε)` with f and rε in mm yields a
value in **mm**; the CoT prompts used in the study present the raw value
labelled as µm (a factor-of-1000 unit inconsistency in the stated prior).
The prompts are kept verbatim in `src/llm_prompts.py` so that the
published results can be reproduced exactly. See the paper for the
discussion of CoT failure modes.

## Data

`data/ti6al4v_ccd38.csv` — dry turning of Ti-6Al-4V Grade 5 on a ROMI
GL240 CNC lathe. Central Composite Design, k = 3 factors (Vc, f, ap) at
five levels (α = 1.682), two insert nose radii (0.4 / 0.8 mm) treated as
a fourth continuous feature, 19 runs per insert, N = 38. Ra measured
with a Taylor Hobson Surtronic S-128 (ISO 4288), mean of 3 readings.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper if you use
this code or dataset.

## License

MIT — see [`LICENSE`](LICENSE).
