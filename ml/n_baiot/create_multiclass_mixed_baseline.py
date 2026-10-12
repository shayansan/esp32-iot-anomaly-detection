from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_multiclass"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_mixed_baseline"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "multiclass_mixed_baseline"
)

RANDOM_SEED = 42

TRAIN_PER_CLASS = 5600
VALIDATION_PER_CLASS = 1200
TEST_PER_CLASS = 1200

NUM_CLASSES = 11


def load_original_splits():
    parts = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        data = np.load(
            INPUT_ROOT
            / f"{split_name}.npz"
        )

        parts.append(
            (
                data["x"],
                data["y"],
                data["source_id"],
            )
        )

    x = np.concatenate(
        [part[0] for part in parts],
        axis=0,
    )

    y = np.concatenate(
        [part[1] for part in parts],
        axis=0,
    )

    source_id = np.concatenate(
        [part[2] for part in parts],
        axis=0,
    )

    return x, y, source_id


def build_mixed_splits(
    x,
    y,
    source_id,
):
    split_parts = {
        "train": [],
        "validation": [],
        "test": [],
    }

    source_parts = {
        "train": [],
        "validation": [],
        "test": [],
    }

    label_parts = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(
            y == class_id
        )

        if len(indices) != 8000:
            raise ValueError(
                f"Class {class_id}: "
                f"expected 8000 samples, "
                f"got {len(indices)}"
            )

        rng = np.random.default_rng(
            RANDOM_SEED + class_id
        )

        indices = rng.permutation(
            indices
        )

        train_idx = indices[
            :TRAIN_PER_CLASS
        ]

        validation_idx = indices[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS
        ]

        test_idx = indices[
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS:
        ]

        for name, idx in [
            ("train", train_idx),
            (
                "validation",
                validation_idx,
            ),
            ("test", test_idx),
        ]:
            split_parts[name].append(
                x[idx]
            )

            label_parts[name].append(
                y[idx]
            )

            source_parts[name].append(
                source_id[idx]
            )

    result = {}

    for split_index, name in enumerate(
        [
            "train",
            "validation",
            "test",
        ]
    ):
        split_x = np.concatenate(
            split_parts[name],
            axis=0,
        )

        split_y = np.concatenate(
            label_parts[name],
            axis=0,
        )

        split_source = np.concatenate(
            source_parts[name],
            axis=0,
        )

        rng = np.random.default_rng(
            RANDOM_SEED
            + 100
            + split_index
        )

        permutation = rng.permutation(
            len(split_y)
        )

        result[name] = {
            "x": split_x[
                permutation
            ],
            "y": split_y[
                permutation
            ],
            "source_id":
                split_source[
                    permutation
                ],
        }

    return result


def validate_split(
    name,
    split,
    expected_per_class,
):
    x = split["x"]
    y = split["y"]

    counts = np.bincount(
        y.astype(int),
        minlength=NUM_CLASSES,
    )

    if not np.all(
        counts == expected_per_class
    ):
        raise ValueError(
            f"{name}: class imbalance "
            f"{counts}"
        )

    if np.isnan(x).any():
        raise ValueError(
            f"{name}: NaN detected"
        )

    if np.isinf(x).any():
        raise ValueError(
            f"{name}: Inf detected"
        )

    print(
        name,
        x.shape,
        counts,
    )


def save_split(
    name,
    split,
):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_ROOT
        / f"{name}.npz",
        x=split["x"],
        y=split["y"],
        source_id=split[
            "source_id"
        ],
    )


def main():
    x, y, source_id = (
        load_original_splits()
    )

    print(
        "Original pool:",
        x.shape,
        y.shape,
    )

    splits = build_mixed_splits(
        x,
        y,
        source_id,
    )

    expected = {
        "train": 5600,
        "validation": 1200,
        "test": 1200,
    }

    for name in [
        "train",
        "validation",
        "test",
    ]:
        validate_split(
            name,
            splits[name],
            expected[name],
        )

        save_split(
            name,
            splits[name],
        )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "dataset": "N-BaIoT",
        "device": "Danmini Doorbell",
        "protocol":
            "mixed_stratified_diagnostic",
        "purpose":
            "diagnose source-range distribution shift",
        "sample_pool":
            "same 88000 samples as initial_multiclass",
        "random_seed":
            RANDOM_SEED,
        "classes":
            NUM_CLASSES,
        "samples_per_class": {
            "train": 5600,
            "validation": 1200,
            "test": 1200,
        },
        "warning":
            (
                "This protocol mixes samples "
                "from the previously separated "
                "source regions and is intended "
                "as a diagnostic IID-style "
                "baseline, not the primary "
                "generalization estimate."
            ),
    }

    with (
        ARTIFACT_ROOT
        / "subset_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\nMixed diagnostic baseline "
        "created successfully."
    )


if __name__ == "__main__":
    main()