import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from pathlib import Path

path = Path(__file__).resolve().parent.parent


history_size = 5

categorical_columns = [
    "Driver",
    "Compound",
    "Circuit"
]

numerical_columns = [
    "LapTime",
    "LapNumber",
    "Stint",
    "TyreLife",
    "AirTemp",
    "Humidity",
    "Rainfall",
    "TrackTemp",
    "Year"
]

def load_data(years):
    data = []
    for year in years:
        file = path / "data" / "final" / str(year)
        for file in file.glob("*.csv"):
            df = pd.read_csv(file)
            df["Year"] = year
            df["Circuit"] = file.stem.split("_", 1)[1]
            data.append(df)
    return pd.concat(data, ignore_index=True)


train_data = load_data([2019,2020,2021,2022,2023])
test_data = load_data([2024,2025])

def create_sequences(df, history_size):

    X = []
    y = []

    group_columns = ["Year", "Circuit", "Driver", "Stint"]
    df = df.sort_values( group_columns + ["LapNumber"]).copy()
    grouped = df.groupby(group_columns)

    for _, group in grouped:
        group = group.reset_index(drop=True)
        if len(group) <= history_size:
            continue

        for i in range(history_size, len(group)):
            history = group.iloc[i - history_size:i]
            target = group.iloc[i]

            if (
                history[numerical_columns + categorical_columns]
                .isnull()
                .any()
                .any()
            ):
                continue

            if pd.isna(target["LapTime"]):
                continue

            X.append(history[numerical_columns + categorical_columns].values)
            y.append(target["LapTime"])

    return np.array(X, dtype=object), np.array(y, dtype=np.float32)


X_train, y_train = create_sequences(train_data, history_size)
X_test, y_test = create_sequences(test_data, history_size)

n_train = X_train.shape[0]
n_test = X_test.shape[0]

n_features = len(numerical_columns + categorical_columns)
numerical_indices = list(range(len(numerical_columns)))
categorical_indices = list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns)))

X_train_numerical = np.array(X_train[:, :, numerical_indices], dtype=np.float32)
X_test_numerical = np.array(X_test[:, :, numerical_indices], dtype=np.float32)
X_train_categorical = X_train[:, :, categorical_indices]
X_test_categorical = X_test[:, :, categorical_indices]

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

X_train_flat = X_train_categorical.reshape(-1, len(categorical_columns))
X_test_flat = X_test_categorical.reshape(-1, len(categorical_columns))
X_train_cat_encoded = encoder.fit_transform(X_train_flat)
X_test_cat_encoded = encoder.transform(X_test_flat)

encoded_features = X_train_cat_encoded.shape[1]
X_train_encoded = X_train_cat_encoded.reshape(n_train, history_size, encoded_features)
X_test_encoded = X_test_cat_encoded.reshape(n_test, history_size, encoded_features)

samples = X_train_numerical.shape[0]
timesteps = X_train_numerical.shape[1]
numeric_features = X_train_numerical.shape[2]

scaler = StandardScaler()


X_train_numerical = (scaler.fit_transform(X_train_numerical.reshape(-1, numeric_features))
    .reshape(samples, timesteps, numeric_features))


X_test_numerical = (scaler.transform(X_test_numerical.reshape(-1, numeric_features))
    .reshape(X_test_numerical.shape[0], timesteps, numeric_features))


X_train = np.concatenate([X_train_numerical, X_train_encoded], axis=2)
X_test = np.concatenate([X_test_numerical, X_test_encoded], axis=2)

X_train = np.transpose(X_train,(0, 2, 1))
X_test = np.transpose(X_test,(0, 2, 1))

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class TCNBlock(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2
    ):

        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if input_channels != output_channels:

            self.residual = nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1
            )

        else:

            self.residual = nn.Identity()


    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = out[:, :, :x.size(2)]

        return torch.relu(out + residual)

class TCNModel(nn.Module):

    def __init__(
        self,
        input_size,
        channels=(64, 64),
        kernel_size=3,
        dropout=0.2
    ):

        super().__init__()

        layers = []

        for i, output_channels in enumerate(channels):

            if i == 0:
                input_channels = input_size
            else:
                input_channels = channels[i - 1]

            dilation = 2 ** i

            layers.append(
                TCNBlock(
                    input_channels,
                    output_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1],1)

    def forward(self, x):

        output = self.tcn(x)
        last_output = output[:, :, -1]
        prediction = self.fc(last_output)
        return prediction


input_size = X_train.shape[1]

model = TCNModel(
    input_size=input_size,
    channels=(64, 64),
    kernel_size=3,
    dropout=0.2
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions,y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = (total_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{epochs} "f"- Loss: {average_loss:.6f}")

model.eval()

with torch.no_grad():
    y_train_pred = (model(X_train_tensor).numpy().flatten())
    y_test_pred = (model(X_test_tensor).numpy().flatten())


train_mae = mean_absolute_error(y_train,y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
train_r2 = r2_score(y_train,y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\nTEST")
print(f"R²:  {r2_score(y_test,y_test_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.4f} s")
print(f"MSE: {mean_squared_error(y_test, y_test_pred):.4f}")


