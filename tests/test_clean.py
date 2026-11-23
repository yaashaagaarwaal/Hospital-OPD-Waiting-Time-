import pandas as pd
from opd.clean import clean_features, clean_train, fix_department

PARAMS = {
    "target": "wait_min",
    "typo_threshold": 300,
    "sentinel": -1,
    "leak_column": "token_called_after_min",
}


def make_df():
    return pd.DataFrame({
        "record_id": [1, 2, 3, 4],
        "department": ["Gen.", " Paediatrics", "ORTHOPAEDICS", "Ort."],
        "lab_tests_pending": [10, -1, 30, 20],
        "token_called_after_min": [50, 60, 70, 80],
        "wait_min": [50, 60, 700, 80],
    })


def test_department_spellings_become_three_levels():
    out = fix_department(pd.Series(["Gen.", "GENERAL", " Paediatrics", "Pae.", "Ort.", "Orthopaedics "]))
    assert sorted(out.unique()) == ["General", "Orthopaedics", "Paediatrics"]


def test_sentinel_is_replaced_by_median():
    out, median = clean_features(make_df(), PARAMS)
    assert (out["lab_tests_pending"] == -1).sum() == 0
    assert median == 20
    assert out.loc[1, "lab_tests_pending"] == 20


def test_leaking_column_is_dropped():
    out, _ = clean_features(make_df(), PARAMS)
    assert "token_called_after_min" not in out.columns


def test_typo_target_rows_are_removed():
    out, _ = clean_train(make_df(), PARAMS)
    assert len(out) == 3
    assert out["wait_min"].max() <= 300