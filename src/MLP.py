import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

path = Path(__file__).resolve().parent.parent

categorical_columns = [
    "Driver",
    "Compound"
]

data = []

def load_data(years):
    for year in years:
        file = path / "data" / "final" / str(year)
        for file in file.glob("*.csv"):
            df = pd.read_csv(file)
            data.append(df)

train_data = load_data([2019,2020,2021,2022,2023])
test_data = load_data([2024,2025])


# encoder = OneHotEncoder(
#     sparse_output=False,
#     handle_unknown="ignore"
# )
#
# elements_encoded = encoder.fit_transform(
#     df[categorical_columns]
# )
#
# encoded_columns = encoder.get_feature_names_out(
#             categorical_columns
#         )
#
# print(df)
# print(pd.get_dummies(data=df, drop_first=True))
