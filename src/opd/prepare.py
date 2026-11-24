import os
import pandas as pd
from opd.checks import load_params
from opd.clean import clean_train


def main():
    p = load_params()
    tr = pd.read_csv(p["data"]["raw_train"])
    clean, median = clean_train(tr, p)

    os.makedirs(os.path.dirname(p["data"]["clean_train"]), exist_ok=True)
    clean.to_csv(p["data"]["clean_train"], index=False)

    print("rows before:", len(tr), "after:", len(clean))
    print("columns:", list(clean.columns))
    print("department:", sorted(clean["department"].unique()))
    print("lab_tests_pending min:", clean["lab_tests_pending"].min(), "median used:", median)
    print("max wait_min:", clean["wait_min"].max(), "nulls:", int(clean.isna().sum().sum()))


if __name__ == "__main__":
    main()