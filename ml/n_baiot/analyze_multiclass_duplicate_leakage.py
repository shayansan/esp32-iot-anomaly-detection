from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_mixed_baseline"
)

NUM_CLASSES = 11

CLASS_NAMES = [
    "benign",
    "gafgyt_combo",
    "gafgyt_junk",
    "gafgyt_scan",
    "gafgyt_tcp",
    "gafgyt_udp",
    "mirai_ack",
    "mirai_scan",
    "mirai_syn",
    "mirai_udp",
    "mirai_udpplain",
]


def load_split(name):
    data = np.load(
        DATA_ROOT / f"{name}.npz"
    )

    return (
        data["x"].astype(np.float32),
        data["y"].astype(np.int32),
    )


def row_keys(x):
    x = np.ascontiguousarray(x)

    return x.view(
        np.dtype(
            (
                np.void,
                x.dtype.itemsize
                * x.shape[1],
            )
        )
    ).reshape(-1)


def analyze_against_train(
    train_x,
    train_y,
    split_x,
    split_y,
    split_name,
):
    train_keys = row_keys(train_x)
    split_keys = row_keys(split_x)

    unique_train = np.unique(
        train_keys
    )

    seen_in_train = np.isin(
        split_keys,
        unique_train,
    )

    print(
        f"\n=== {split_name.upper()} ==="
    )

    print(
        "Samples:",
        len(split_y),
    )

    print(
        "Samples whose exact feature vector "
        "already exists in train:",
        int(np.sum(seen_in_train)),
    )

    print(
        "Exact-vector train coverage:",
        f"{np.mean(seen_in_train) * 100:.2f}%",
    )

    print(
        "\nPer-class exact-vector coverage:"
    )

    for class_id in range(NUM_CLASSES):
        train_class_keys = np.unique(
            train_keys[
                train_y == class_id
            ]
        )

        mask = (
            split_y == class_id
        )

        class_seen = np.isin(
            split_keys[mask],
            train_class_keys,
        )

        print(
            f"{CLASS_NAMES[class_id]:16s}",
            f"{np.mean(class_seen) * 100:6.2f}%",
            f"({np.sum(class_seen)}/{np.sum(mask)})",
        )


def print_unique_counts(
    x,
    y,
    split_name,
):
    keys = row_keys(x)

    print(
        f"\n=== {split_name.upper()} "
        "UNIQUE VECTORS ==="
    )

    print(
        "Total samples:",
        len(y),
    )

    print(
        "Unique feature vectors:",
        len(np.unique(keys)),
    )

    for class_id in range(NUM_CLASSES):
        mask = (
            y == class_id
        )

        unique_count = len(
            np.unique(
                keys[mask]
            )
        )

        print(
            f"{CLASS_NAMES[class_id]:16s}",
            f"samples={np.sum(mask):5d}",
            f"unique={unique_count:5d}",
        )


def main():
    train_x, train_y = load_split(
        "train"
    )

    validation_x, validation_y = (
        load_split(
            "validation"
        )
    )

    test_x, test_y = load_split(
        "test"
    )

    print_unique_counts(
        train_x,
        train_y,
        "train",
    )

    print_unique_counts(
        validation_x,
        validation_y,
        "validation",
    )

    print_unique_counts(
        test_x,
        test_y,
        "test",
    )

    analyze_against_train(
        train_x,
        train_y,
        validation_x,
        validation_y,
        "validation",
    )

    analyze_against_train(
        train_x,
        train_y,
        test_x,
        test_y,
        "test",
    )


if __name__ == "__main__":
    main()