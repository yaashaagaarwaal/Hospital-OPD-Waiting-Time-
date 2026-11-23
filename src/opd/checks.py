import pandas as pd
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def leaking_column(train, holdout, target="wait_min"):
    cols = sorted(set(train.columns) - set(holdout.columns) - {target})
    return {"present": bool(cols), "column": cols[0] if cols else None}


def unit_mixup(df, group_col, num_cols):
    return {c: df.groupby(group_col)[c].agg(["min", "mean", "max"]) for c in num_cols}


def sentinel_value(df, col, value):
    n = int((df[col] == value).sum())
    return {"present": n > 0, "column": col, "value": value, "count": n}


def duplicate_rows(df):
    n = int(df.drop(columns="record_id").duplicated().sum())
    return {"present": n > 0, "count": n}


def stuck_sensor(df, col, min_run=5):
    d = df.sort_values(["date", "record_id"])
    run_id = (d[col] != d[col].shift()).cumsum()
    runs = d.groupby(run_id).agg(
        start=("date", "first"), end=("date", "last"), n=(col, "size")
    )
    return runs[runs.n >= min_run]


def target_typos(df, target="wait_min", threshold=300):
    n = int((df[target] > threshold).sum())
    return {"present": n > 0, "count": n}


def spelling_variants(df, col):
    return df[col].value_counts()