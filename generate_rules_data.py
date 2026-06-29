"""import pandas as pd

dirty = pd.read_csv("datasets/flights/flights_dirty.csv")

rule_cols = [
    "flight",
    "act_dep_time",
    "act_arr_time",
    "sched_dep_time",
    "sched_arr_time",
]

rules_data = dirty.dropna(subset=rule_cols).copy()

for col in ["index", "Unnamed: 0"]:
    if col in rules_data.columns:
        rules_data = rules_data.drop(columns=[col])

rules_data.to_csv("datasets/flights/rules_data.csv", index=False)

print(rules_data.shape)
print(rules_data[rule_cols].isna().sum())"""
"""
import pandas as pd

dirty = pd.read_csv("datasets/hospital/hospital_dirty.csv")

rule_cols = [
    "ProviderNumber",
    "HospitalName",
    "Address1",
    "City",
    "State",
    "ZipCode",
    "CountyName",
    "PhoneNumber",
    "MeasureCode",
    "MeasureName",
    "Condition",
]

rules_data = dirty.dropna(subset=rule_cols).copy()
rules_data.to_csv("datasets/hospital/rules_data.csv", index=False)

print("dirty shape:", dirty.shape)
print("rules_data shape:", rules_data.shape)
print(rules_data[rule_cols].isna().sum())

print("\nProviderNumber support:")
print(rules_data.groupby("ProviderNumber").size().describe())

print("\nMeasureCode support:")
print(rules_data.groupby("MeasureCode").size().describe())
import pandas as pd

dirty = pd.read_csv("datasets/beers/beers_dirty.csv")

rule_cols = [
    "brewery_id",
    "brewery_name",
    "city",
    "state",
]

rules_data = dirty.dropna(subset=rule_cols).copy()
rules_data.to_csv("datasets/beers/rules_data.csv", index=False)

print("dirty shape:", dirty.shape)
print("rules_data shape:", rules_data.shape)
print(rules_data[rule_cols].isna().sum())

print("\nbrewery_id support:")
print(rules_data.groupby("brewery_id").size().describe())"""
import pandas as pd
from pathlib import Path

base_dir = Path("datasets/hospital")
dirty_path = base_dir / "hospital_dirty.csv"

dirty = pd.read_csv(dirty_path)

configs = {
    "2rule": {
        "cols": [
            "ProviderNumber",
            "HospitalName",
            "MeasureCode",
            "MeasureName",
        ],
        "out": base_dir / "rules_data_2.csv",
    },
    "5rule": {
        "cols": [
            "ProviderNumber",
            "HospitalName",
            "City",
            "State",
            "MeasureCode",
            "MeasureName",
            "Condition",
        ],
        "out": base_dir / "rules_data_5.csv",
    },
    "9rule": {
        "cols": [
            "ProviderNumber",
            "HospitalName",
            "Address1",
            "City",
            "State",
            "ZipCode",
            "CountyName",
            "PhoneNumber",
            "MeasureCode",
            "MeasureName",
            "Condition",
        ],
        "out": base_dir / "rules_data.csv",
    },
}

print("dirty shape:", dirty.shape)

for name, cfg in configs.items():
    cols = cfg["cols"]
    out = cfg["out"]

    missing_cols = [c for c in cols if c not in dirty.columns]
    if missing_cols:
        raise ValueError(f"{name} missing columns: {missing_cols}")

    rules_data = dirty.dropna(subset=cols).copy()
    rules_data.to_csv(out, index=False)

    print("\n==========", name, "==========")
    print("rule cols:", cols)
    print("rules_data shape:", rules_data.shape)
    print("NaN count:")
    print(rules_data[cols].isna().sum())

    if "ProviderNumber" in cols:
        print("\nProviderNumber support:")
        print(rules_data.groupby("ProviderNumber").size().describe())

    if "MeasureCode" in cols:
        print("\nMeasureCode support:")
        print(rules_data.groupby("MeasureCode").size().describe())