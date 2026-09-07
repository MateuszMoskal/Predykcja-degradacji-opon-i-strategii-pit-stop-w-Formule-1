import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

path = Path(__file__).resolve().parent.parent

categorical_columns = [
    "Driver",
    "Compound"
]

file = path / "data" / "final" / "2019" / "1_Australian Grand Prix.csv"
df = pd.read_csv(file)
df["Rainfall"] = df["Rainfall"].astype(int)
df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

df = df.drop(columns="Time")

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

compound_encoded = encoder.fit_transform(
    df[categorical_columns]
)

encoded_columns = encoder.get_feature_names_out(
            categorical_columns
        )

print(df)
