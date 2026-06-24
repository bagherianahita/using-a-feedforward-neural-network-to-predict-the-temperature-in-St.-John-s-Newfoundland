"""St. John's temperature forecast demo — synthetic data, no download required."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow import keras
except ImportError:
    raise SystemExit("Install: pip install tensorflow scikit-learn pandas numpy")


def build_series(hours: int = 500) -> pd.DataFrame:
    t = np.arange(hours)
    temp = 8 + 6 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, size=hours)
    return pd.DataFrame({"hour": t, "temp_c": temp})


def make_windows(series: np.ndarray, lookback: int = 12, horizon: int = 6):
    x, y = [], []
    for i in range(len(series) - lookback - horizon):
        x.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(x), np.array(y)


def main() -> None:
    print("Temperature prediction demo — St. John's, NL (synthetic hourly data)\n")
    df = build_series()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["temp_c"]])

    x, y = make_windows(scaled.flatten())
    x = x.reshape((x.shape[0], x.shape[1], 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x.shape[1], 1)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(6),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1, validation_split=0.1)

    pred = model.predict(x_test[:1], verbose=0)
    actual = y_test[:1]
    pred_c = scaler.inverse_transform(pred.T).flatten()
    actual_c = scaler.inverse_transform(actual.T).flatten()

    print("\nNext 6 hours forecast (°C):", np.round(pred_c, 2))
    print("Baseline actual (°C):       ", np.round(actual_c, 2))
    mae = np.mean(np.abs(pred_c - actual_c))
    print(f"MAE on sample: {mae:.2f}°C")
    print("\nDemo complete — no API keys or external data required.")


if __name__ == "__main__":
    main()
