from pathlib import Path
import json

import numpy as np
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_mixed_baseline"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_mixed_preprocessed"
    / "full_115"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "multiclass_mixed_preprocessing"
)


def load_split(name):
    data = np.load(
        INPUT_ROOT / f"{name}.npz"
    )

    return {
        "x": data["x"],
        "y": data["y"],
        "source_id": data["source_id"],
    }


def validate(name, split):
    x = split["x"]

    if x.shape[1] != 115:
        raise ValueError(
            f"{name}: expected 115 features"
        )

    if np.isnan(x).any():
        raise ValueError(
            f"{name}: NaN detected"
        )

    if np.isinf(x).any():
        raise ValueError(
            f"{name}: Inf detected"
        )


def save_split(
    name,
    x,
    y,
    source_id,
):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_ROOT / f"{name}.npz",
        x=x.astype(np.float32),
        y=y,
        source_id=source_id,
    )

    print(
        f"Saved {name}: {x.shape}"
    )


def main():
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        validate(name, split)

    scaler = StandardScaler()

    scaler.fit(
        train["x"].astype(np.float64)
    )

    train_x = scaler.transform(
        train["x"]
    )

    validation_x = scaler.transform(
        validation["x"]
    )

    test_x = scaler.transform(
        test["x"]
    )

    save_split(
        "train",
        train_x,
        train["y"],
        train["source_id"],
    )

    save_split(
        "validation",
        validation_x,
        validation["y"],
        validation["source_id"],
    )

    save_split(
        "test",
        test_x,
        test["y"],
        test["source_id"],
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler_data = {
        "protocol":
            "mixed_stratified_diagnostic",
        "feature_count": 115,
        "fit_split": "train",
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "variance": scaler.var_.tolist(),
    }

    scaler_path = (
        ARTIFACT_ROOT
        / "full_115_scaler.json"
    )

    with scaler_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scaler_data,
            file,
            indent=2,
        )

    print(
        f"Saved scaler: {scaler_path}"
    )

    for name, x in [
        ("train", train_x),
        ("validation", validation_x),
        ("test", test_x),
    ]:
        print(f"\n{name}")
        print(
            "mean(abs):",
            float(
                np.mean(
                    np.abs(
                        np.mean(
                            x,
                            axis=0,
                        )
                    )
                )
            ),
        )
        print(
            "std(mean):",
            float(
                np.mean(
                    np.std(
                        x,
                        axis=0,
                    )
                )
            ),
        )
        print(
            "NaN:",
            int(np.isnan(x).sum()),
        )
        print(
            "Inf:",
            int(np.isinf(x).sum()),
        )


if __name__ == "__main__":
    main()