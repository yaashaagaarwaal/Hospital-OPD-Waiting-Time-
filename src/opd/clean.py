import numpy as np
import pandas as pd

DEPT_MAP = {
    "gen": "General", "general": "General",
    "pae": "Paediatrics", "paediatrics": "Paediatrics",
    "ort": "Orthopaedics", "orthopaedics": "Orthopaedics",
}


def fix_department(s):
    key = s.astype(str).str.strip().str.lower().str.rstrip(".")
    return key.map(DEPT_MAP)


def clean_features(df, params, median=None):
    """Cleaning that is safe for train, later and holdout."""
    df = df.copy()
    df = df.drop(columns=[params["leak_column"]], errors="ignore")
    df["department"] = fix_department(df["department"])
    col = "lab_tests_pending"
    df[col] = df[col].replace(params["sentinel"], np.nan)
    if median is None:
        median = df[col].median()
    df[col] = df[col].fillna(median)
    return df, median


def clean_train(df, params):
    """Train only: also removes rows with a mistyped target."""
    df = df[df[params["target"]] <= params["typo_threshold"]]
    df, median = clean_features(df, params)
    return df.reset_index(drop=True), median