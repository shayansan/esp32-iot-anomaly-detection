from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_binary"
    / "train.npz"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_binary"
    / "subset_metadata.json"
)


def main():
    data = np.load(DATA_PATH)

    x = data["x"]
    x_stats = x.astype(np.float64)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    feature_names = metadata["feature_names"]

    print(f"Samples: {x.shape[0]}")
    print(f"Features: {x.shape[1]}")

    print(
        f"NaN values: {np.isnan(x).sum()}"
    )

    print(
        f"Inf values: {np.isinf(x).sum()}"
    )

    variances = np.var(
    x_stats,
    axis=0,
    )

    constant_indices = np.where(
        variances == 0
    )[0]

    near_constant_indices = np.where(
        variances < 1e-12
    )[0]

    print(
        f"Constant features: "
        f"{len(constant_indices)}"
    )

    print(
        f"Near-constant features: "
        f"{len(near_constant_indices)}"
    )

    if len(constant_indices) > 0:
        print("\nConstant feature names:")

        for index in constant_indices:
            print(
                f"- {feature_names[index]}"
            )

    print("\nFeature value ranges:")

    for index, name in enumerate(
        feature_names
    ):
        column = x_stats[:, index]

    print(
        f"{name}: "
        f"min={column.min():.6f}, "
        f"max={column.max():.6f}, "
        f"mean={column.mean():.6f}, "
        f"std={column.std():.6f}"
    )


if __name__ == "__main__":
    main()