"""Data normalization techniques for financial machine learning."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def main() -> None:
    frame = pd.DataFrame(
        {
            "Return": [0.010, -0.015, 0.008, 0.040, -0.030],
            "Volume": [1_000_000, 1_250_000, 980_000, 2_500_000, 1_100_000],
            "Volatility": [0.18, 0.22, 0.17, 0.35, 0.24],
        }
    )

    standard = pd.DataFrame(
        StandardScaler().fit_transform(frame),
        columns=frame.columns,
    )
    minmax = pd.DataFrame(
        MinMaxScaler().fit_transform(frame),
        columns=frame.columns,
    )
    robust = pd.DataFrame(
        RobustScaler().fit_transform(frame),
        columns=frame.columns,
    )

    print("Original data:\n", frame, "\n")
    print("Standardized data:\n", standard, "\n")
    print("Min-Max scaled data:\n", minmax, "\n")
    print("Robust-scaled data:\n", robust, "\n")

    positive_volume = frame["Volume"]
    frame["Log_Volume"] = np.log1p(positive_volume)
    print("Log-transformed volume:\n", frame[["Volume", "Log_Volume"]])

    print(
        "\nImportant: fit preprocessing objects only on training data, then use transform() on validation/test data."
    )


if __name__ == "__main__":
    main()
