from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "raw"
    / "Danmini_Doorbell"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_multiclass"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_multiclass"
)

RANDOM_SEED = 42

TRAIN_PER_CLASS = 5600
VALIDATION_PER_CLASS = 1200
TEST_PER_CLASS = 1200

CLASSES = [
    {
        "id": 0,
        "name": "benign",
        "path": RAW_ROOT / "benign_traffic.csv",
    },
    {
        "id": 1,
        "name": "gafgyt_combo",
        "path": RAW_ROOT / "gafgyt" / "combo.csv",
    },
    {
        "id": 2,
        "name": "gafgyt_junk",
        "path": RAW_ROOT / "gafgyt" / "junk.csv",
    },
    {
        "id": 3,
        "name": "gafgyt_scan",
        "path": RAW_ROOT / "gafgyt" / "scan.csv",
    },
    {
        "id": 4,
        "name": "gafgyt_tcp",
        "path": RAW_ROOT / "gafgyt" / "tcp.csv",
    },
    {
        "id": 5,
        "name": "gafgyt_udp",
        "path": RAW_ROOT / "gafgyt" / "udp.csv",
    },
    {
        "id": 6,
        "name": "mirai_ack",
        "path": RAW_ROOT / "mirai" / "ack.csv",
    },
    {
        "id": 7,
        "name": "mirai_scan",
        "path": RAW_ROOT / "mirai" / "scan.csv",
    },
    {
        "id": 8,
        "name": "mirai_syn",
        "path": RAW_ROOT / "mirai" / "syn.csv",
    },
    {
        "id": 9,
        "name": "mirai_udp",
        "path": RAW_ROOT / "mirai" / "udp.csv",
    },
    {
        "id": 10,
        "name": "mirai_udpplain",
        "path": RAW_ROOT / "mirai" / "udpplain.csv",
    },
]
def calculate_split_ranges(total_rows):
    required_rows = (
        TRAIN_PER_CLASS
        + VALIDATION_PER_CLASS
        + TEST_PER_CLASS
    )

    if total_rows < required_rows:
        raise ValueError(
            f"Not enough rows: "
            f"required={required_rows}, "
            f"available={total_rows}"
        )

    unused_rows = (
        total_rows - required_rows
    )

    gap1 = unused_rows // 2
    gap2 = unused_rows - gap1

    train_start = 0
    train_end = TRAIN_PER_CLASS

    validation_start = (
        train_end + gap1
    )

    validation_end = (
        validation_start
        + VALIDATION_PER_CLASS
    )

    test_start = (
        total_rows
        - TEST_PER_CLASS
    )

    test_end = total_rows

    return {
        "train": {
            "start": train_start,
            "end": train_end,
        },
        "validation": {
            "start": validation_start,
            "end": validation_end,
        },
        "test": {
            "start": test_start,
            "end": test_end,
        },
        "gap1": gap1,
        "gap2": gap2,
    }

def count_csv_rows(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        total_lines = sum(
            1 for _ in file
        )

    # First line is the CSV header.
    return total_lines - 1

def load_csv_range(
    path,
    start,
    end,
    feature_names=None,
):
    row_count = end - start

    if feature_names is None:
        header = pd.read_csv(
            path,
            nrows=0,
        )

        feature_names = list(
            header.columns
        )

    data = pd.read_csv(
        path,
        skiprows=start + 1,
        nrows=row_count,
        header=None,
        names=feature_names,
        dtype=np.float32,
    )

    if len(data) != row_count:
        raise ValueError(
            f"Expected {row_count} rows "
            f"from {path}, "
            f"but loaded {len(data)}"
        )

    return (
        data.to_numpy(dtype=np.float32),
        feature_names,
    )

def build_split(
    split_name,
    feature_names=None,
):
    x_parts = []
    y_parts = []
    source_parts = []

    metadata = []

    for class_info in CLASSES:
        class_id = class_info["id"]
        class_name = class_info["name"]
        path = class_info["path"]

        total_rows = count_csv_rows(
            path
        )

        ranges = calculate_split_ranges(
            total_rows
        )

        split_range = ranges[
            split_name
        ]

        start = split_range["start"]
        end = split_range["end"]

        x, feature_names = load_csv_range(
            path,
            start,
            end,
            feature_names,
        )

        y = np.full(
            len(x),
            class_id,
            dtype=np.uint8,
        )

        source_id = np.full(
            len(x),
            class_id,
            dtype=np.uint8,
        )

        x_parts.append(x)
        y_parts.append(y)
        source_parts.append(source_id)

        metadata.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "source": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "source_total_rows": total_rows,
                "start": start,
                "end": end,
                "samples": len(x),
                "gap1": ranges["gap1"],
                "gap2": ranges["gap2"],
            }
        )

    x = np.concatenate(
        x_parts,
        axis=0,
    )

    y = np.concatenate(
        y_parts,
        axis=0,
    )

    source_id = np.concatenate(
        source_parts,
        axis=0,
    )

    return (
        x,
        y,
        source_id,
        feature_names,
        metadata,
    )

def shuffle_split(
    x,
    y,
    source_id,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    indices = rng.permutation(
        len(y)
    )

    return (
        x[indices],
        y[indices],
        source_id[indices],
    )

def validate_split(
    split_name,
    x,
    y,
    source_id,
    expected_per_class,
):
    expected_total = (
        expected_per_class
        * len(CLASSES)
    )

    if x.shape != (
        expected_total,
        115,
    ):
        raise ValueError(
            f"{split_name}: "
            f"unexpected x shape "
            f"{x.shape}"
        )

    if y.shape != (
        expected_total,
    ):
        raise ValueError(
            f"{split_name}: "
            f"unexpected y shape "
            f"{y.shape}"
        )

    if source_id.shape != (
        expected_total,
    ):
        raise ValueError(
            f"{split_name}: "
            f"unexpected source_id shape "
            f"{source_id.shape}"
        )

    if np.isnan(x).any():
        raise ValueError(
            f"{split_name}: NaN detected"
        )

    if np.isinf(x).any():
        raise ValueError(
            f"{split_name}: Inf detected"
        )

    for class_info in CLASSES:
        class_id = class_info["id"]

        count = int(
            np.sum(
                y == class_id
            )
        )

        if count != expected_per_class:
            raise ValueError(
                f"{split_name}: "
                f"class {class_id} "
                f"has {count} samples, "
                f"expected "
                f"{expected_per_class}"
            )

def save_split(
    split_name,
    x,
    y,
    source_id,
):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / f"{split_name}.npz"
    )

    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        source_id=source_id,
    )

    print(
        f"Saved {split_name}: "
        f"{output_path}"
    )

def build_metadata(
    feature_names,
    split_metadata,
):
    class_mapping = {
        str(class_info["id"]):
            class_info["name"]
        for class_info in CLASSES
    }

    return {
        "dataset": "N-BaIoT",
        "device": "Danmini Doorbell",
        "experiment": "initial_multiclass",
        "random_seed": RANDOM_SEED,
        "feature_count": len(
            feature_names
        ),
        "feature_names": feature_names,
        "class_count": len(CLASSES),
        "class_mapping": class_mapping,
        "samples_per_class": {
            "train": TRAIN_PER_CLASS,
            "validation":
                VALIDATION_PER_CLASS,
            "test": TEST_PER_CLASS,
        },
        "total_samples": {
            "train":
                TRAIN_PER_CLASS
                * len(CLASSES),
            "validation":
                VALIDATION_PER_CLASS
                * len(CLASSES),
            "test":
                TEST_PER_CLASS
                * len(CLASSES),
        },
        "splits": split_metadata,
    }

def save_metadata(metadata):
    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        ARTIFACT_ROOT
        / "subset_metadata.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Saved metadata: "
        f"{output_path}"
    )

def inspect_class_sources():
    for class_info in CLASSES:
        class_id = class_info["id"]
        class_name = class_info["name"]
        path = class_info["path"]

        total_rows = count_csv_rows(
            path
        )

        ranges = calculate_split_ranges(
            total_rows
        )

        print(
            f"\nClass {class_id}: "
            f"{class_name}"
        )

        print(
            f"Source: {path}"
        )

        print(
            f"Total rows: {total_rows}"
        )

        print(
            "Train:",
            ranges["train"],
        )

        print(
            "Validation:",
            ranges["validation"],
        )

        print(
            "Test:",
            ranges["test"],
        )

        print(
            f"Gap 1: {ranges['gap1']}"
        )

        print(
            f"Gap 2: {ranges['gap2']}"
        )
def main():
    split_config = {
        "train": TRAIN_PER_CLASS,
        "validation":
            VALIDATION_PER_CLASS,
        "test": TEST_PER_CLASS,
    }

    feature_names = None
    split_metadata = {}

    for split_index, (
        split_name,
        expected_per_class,
    ) in enumerate(
        split_config.items()
    ):
        print(
            f"\nBuilding "
            f"{split_name} split..."
        )

        (
            x,
            y,
            source_id,
            feature_names,
            metadata,
        ) = build_split(
            split_name,
            feature_names,
        )

        validate_split(
            split_name,
            x,
            y,
            source_id,
            expected_per_class,
        )

        x, y, source_id = (
            shuffle_split(
                x,
                y,
                source_id,
                RANDOM_SEED
                + split_index,
            )
        )

        validate_split(
            split_name,
            x,
            y,
            source_id,
            expected_per_class,
        )

        save_split(
            split_name,
            x,
            y,
            source_id,
        )

        split_metadata[
            split_name
        ] = metadata

        print(
            f"{split_name}: "
            f"x={x.shape}, "
            f"y={y.shape}"
        )

    metadata = build_metadata(
        feature_names,
        split_metadata,
    )

    save_metadata(
        metadata
    )

    print(
        "\nMulticlass subset "
        "created successfully."
    )


if __name__ == "__main__":
    main()