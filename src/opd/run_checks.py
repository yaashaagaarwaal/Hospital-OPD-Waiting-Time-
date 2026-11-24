import pandas as pd
from opd.checks import (
    load_params, leaking_column, unit_mixup, sentinel_value,
    duplicate_rows, stuck_sensor, target_typos, spelling_variants,
)


def main():
    p = load_params()
    tr = pd.read_csv(p["data"]["raw_train"])
    ho = pd.read_csv(p["data"]["holdout"])

    print("LEAK     ", leaking_column(tr, ho, p["target"]))
    print("SENTINEL ", sentinel_value(tr, "lab_tests_pending", p["sentinel"]))
    print("DUPES    ", duplicate_rows(tr))
    print("TYPOS    ", target_typos(tr, p["target"], p["typo_threshold"]))
    print("SPELLING\n", spelling_variants(tr, "department"))

    nums = ["patients_ahead", "doctors_on_duty", "arrival_hour",
            "avg_consult_min", "lab_tests_pending", "distance_km"]
    for c in nums:
        print("STUCK", c, "->", len(stuck_sensor(tr, c)), "long runs")
    for c, t in unit_mixup(tr, "wing", nums).items():
        print("\nUNIT", c)
        print(t)


if __name__ == "__main__":
    main()