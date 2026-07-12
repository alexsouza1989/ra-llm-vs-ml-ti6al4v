"""Dataset loading for the Ti-6Al-4V dry turning CCD (N = 38).

Two insert groups (nose radius 0.4 mm and 0.8 mm), 19 runs each,
pooled into a single dataset with nose radius Re as a continuous
input feature. Feature vector: [Vc, f, ap, Re]; response: Ra (µm).
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Vc", "f", "ap", "Re"]
TARGET = "Ra"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ti6al4v_ccd38.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the full 38-point dataset as a DataFrame."""
    df = pd.read_csv(path)
    expected_cols = set(FEATURES + [TARGET])
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if len(df) != 38:
        raise ValueError(f"Expected 38 observations, found {len(df)}")
    return df


def load_xy(path: Path = DATA_PATH):
    """Return (X, y) numpy arrays in the order used throughout the paper."""
    df = load_dataset(path)
    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(float)
    return X, y


def theoretical_ra(f: np.ndarray, re: np.ndarray) -> np.ndarray:
    """Kinematic groove formula Ra_th = f^2 / (32 * r_eps).

    Note: with f and r_eps in mm this expression yields a value in mm;
    multiply by 1000 to express it in µm. The paper's prompts used the
    raw value labelled as µm (see paper erratum discussion).
    """
    return np.asarray(f) ** 2 / (32.0 * np.asarray(re))
