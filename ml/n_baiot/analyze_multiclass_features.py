from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_multiclass"
    / "train.npz"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_multiclass"
    / "subset_metadata.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "multiclass_feature_analysis"
)

CORRELATION_THRESHOLD = 0.99


def main():
    data = np.load(TRAIN_PATH)

    # Use float64 for statistical calculations.
    x = data["x"].astype(np.float64)
    y = data["y"]

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    feature_names = metadata[
        "feature_names"
    ]

    if x.shape[1] != len(feature_names):
        raise ValueError(
            "Feature count does not match metadata"
        )

    print("Train shape:", x.shape)
    print("Feature count:", len(feature_names))
    print("NaN:", int(np.isnan(x).sum()))
    print("Inf:", int(np.isinf(x).sum()))

    print(
        "Class counts:",
        np.bincount(
            y.astype(int),
            minlength=11,
        ),
    )

    feature_min = np.min(
        x,
        axis=0,
    )

    feature_max = np.max(
        x,
        axis=0,
    )

    feature_mean = np.mean(
        x,
        axis=0,
    )

    feature_std = np.std(
        x,
        axis=0,
    )

    constant_indices = np.flatnonzero(
        feature_std == 0.0
    )

    constant_features = [
        feature_names[i]
        for i in constant_indices
    ]

    print(
        "Constant features:",
        len(constant_features),
    )

    # Pearson correlation using training data only.
    correlation_matrix = np.corrcoef(
        x,
        rowvar=False,
    )

    correlated_pairs = []

    for i in range(len(feature_names)):
        for j in range(
            i + 1,
            len(feature_names),
        ):
            correlation = (
                correlation_matrix[i, j]
            )

            if (
                np.isfinite(correlation)
                and abs(correlation)
                >= CORRELATION_THRESHOLD
            ):
                correlated_pairs.append(
                    {
                        "feature_1_index": i,
                        "feature_1":
                            feature_names[i],
                        "feature_2_index": j,
                        "feature_2":
                            feature_names[j],
                        "correlation":
                            float(correlation),
                        "absolute_correlation":
                            float(
                                abs(correlation)
                            ),
                    }
                )

    correlated_pairs.sort(
        key=lambda item:
            item[
                "absolute_correlation"
            ],
        reverse=True,
    )

    print(
        "Highly correlated pairs:",
        len(correlated_pairs),
    )

    print(
        "\nTop correlated pairs:"
    )

    for pair in correlated_pairs[:10]:
        print(
            f"{pair['feature_1']} "
            f"<-> "
            f"{pair['feature_2']} "
            f"r={pair['correlation']:.6f}"
        )

    output = {
        "analysis_split": "train",
        "samples": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "class_count": 11,
        "correlation_threshold":
            CORRELATION_THRESHOLD,
        "nan_count":
            int(np.isnan(x).sum()),
        "inf_count":
            int(np.isinf(x).sum()),
        "constant_feature_count":
            len(constant_features),
        "constant_features":
            constant_features,
        "highly_correlated_pair_count":
            len(correlated_pairs),
        "highly_correlated_pairs":
            correlated_pairs,
        "feature_statistics": [
            {
                "index": i,
                "name": feature_names[i],
                "min":
                    float(feature_min[i]),
                "max":
                    float(feature_max[i]),
                "mean":
                    float(feature_mean[i]),
                "std":
                    float(feature_std[i]),
            }
            for i in range(
                len(feature_names)
            )
        ],
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / "train_feature_analysis.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\nSaved:",
        output_path,
    )


if __name__ == "__main__":
    main()