import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from pathlib import Path

path = Path(__file__).resolve().parent.parent

history_size = 5

categorical_columns = [
    "Driver",
    "Compound",
    "Circuit"
]

history_columns = [
    f"LapTime_t-{i}"
    for i in range(1, history_size + 1)
]

numerical_columns =  history_columns + [
    "LapNumber",
    "Stint",
    "TyreLife",
    "AirTemp",
    "Humidity",
    "Rainfall",
    "TrackTemp",
    "Year"
]

def add_lap_history(df, history_size=5):
    group_columns = ["Year", "Circuit", "Driver", "Stint"]

    df = df.sort_values(group_columns + ["LapNumber"]).copy()

    grouped = df.groupby(group_columns)["LapTime"]

    for i in range(1, history_size + 1):
        df[f"LapTime_t-{i}"] = grouped.shift(i)

    df["TargetLapTime"] = grouped.shift(-1)

    return df

def load_data(years):
    data = []
    for year in years:
        file = path / "data" / "final" / str(year)
        for file in file.glob("*.csv"):
            df = pd.read_csv(file)
            df["Year"] = year
            df["Circuit"] = file.stem.split("_",1)[1]
            data.append(df)
    return pd.concat(data, ignore_index=True)

train_data = load_data([2019,2020,2021,2022,2023])
test_data = load_data([2024,2025])
train_data = add_lap_history(train_data, history_size)
test_data = add_lap_history(test_data, history_size)
train_data = train_data.dropna(subset=categorical_columns + numerical_columns + ["TargetLapTime"])
test_data = test_data.dropna(subset=categorical_columns + numerical_columns + ["TargetLapTime"])


encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

train_elements_encoded = encoder.fit_transform(
    train_data[categorical_columns]
)

# encoded_columns = encoder.get_feature_names_out(
#             categorical_columns
#         )

test_elements_encoded = encoder.transform(
    test_data[categorical_columns]
)

X_train_numerical = train_data[numerical_columns].values
X_test_numerical = test_data[numerical_columns].values
X_train = np.hstack([X_train_numerical,train_elements_encoded])
X_test = np.hstack([X_test_numerical,test_elements_encoded])
y_train = train_data["TargetLapTime"].values
y_test = test_data["TargetLapTime"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=256,
    max_iter=1000,
    tol=1e-4,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(score)
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# print(df)
# print(pd.get_dummies(data=df, drop_first=True))
