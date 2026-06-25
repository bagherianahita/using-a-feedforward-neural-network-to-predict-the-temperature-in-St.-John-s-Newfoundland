"""Core temperature forecast logic — feedforward NN on synthetic St. John's hourly data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def build_series(hours: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    # St. John's–style cool maritime signal: ~8°C mean, daily swing
    temp = 8 + 6 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, size=hours)
    return pd.DataFrame({"hour": t, "temp_c": temp})


def make_windows(series: np.ndarray, lookback: int = 12, horizon: int = 6):
    x, y = [], []
    for i in range(len(series) - lookback - horizon):
        x.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(x), np.array(y)


def train_and_forecast(
    hours: int = 500,
    lookback: int = 12,
    horizon: int = 6,
    seed: int = 42,
) -> dict:
    df = build_series(hours=hours, seed=seed)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["temp_c"]]).flatten()

    x, y = make_windows(scaled, lookback=lookback, horizon=horizon)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=200,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    pred_scaled = model.predict(x_test[:1])[0]
    actual_scaled = y_test[0]
    pred_c = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual_c = scaler.inverse_transform(actual_scaled.reshape(-1, 1)).flatten()

    # Persistence baseline: repeat last observed hour
    last_hour = scaler.inverse_transform(x_test[0][-1:].reshape(-1, 1)).flatten()[0]
    baseline_c = np.full(horizon, last_hour)

    mae_nn = float(np.mean(np.abs(pred_c - actual_c)))
    mae_baseline = float(np.mean(np.abs(baseline_c - actual_c)))

    history = df.tail(lookback + horizon)
    return {
        "df": df,
        "lookback": lookback,
        "horizon": horizon,
        "pred_c": pred_c,
        "actual_c": actual_c,
        "baseline_c": baseline_c,
        "mae_nn": mae_nn,
        "mae_baseline": mae_baseline,
        "history_hours": history["hour"].tolist(),
        "history_temp": history["temp_c"].tolist(),
    }
