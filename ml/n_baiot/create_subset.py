from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "raw"
    / "Danmini_Doorbell"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "initial_binary"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "initial_binary"
)

RANDOM_SEED = 42
EXPECTED_FEATURE_COUNT = 115


SOURCES = [
    {
        "name": "benign",
        "path": RAW_DIR / "benign_traffic.csv",
        "label": 0,
        "train": 28000,
        "validation": 6000,
        "test": 6000,
    },

    {
        "name": "gafgyt_combo",
        "path": RAW_DIR / "gafgyt" / "combo.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
    {
        "name": "gafgyt_junk",
        "path": RAW_DIR / "gafgyt" / "junk.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
    {
        "name": "gafgyt_scan",
        "path": RAW_DIR / "gafgyt" / "scan.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
    {
        "name": "gafgyt_tcp",
        "path": RAW_DIR / "gafgyt" / "tcp.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
    {
        "name": "gafgyt_udp",
        "path": RAW_DIR / "gafgyt" / "udp.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },

    {
        "name": "mirai_ack",
        "path": RAW_DIR / "mirai" / "ack.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
    {
        "name": "mirai_scan",
        "path": RAW_DIR / "mirai" / "scan.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
    {
        "name": "mirai_syn",
        "path": RAW_DIR / "mirai" / "syn.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
    {
        "name": "mirai_udp",
        "path": RAW_DIR / "mirai" / "udp.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
    {
        "name": "mirai_udpplain",
        "path": RAW_DIR / "mirai" / "udpplain.csv",
        "label": 1,
        "train": 2800,
        "validation": 600,
        "test": 600,
    },
]


def calculate_ranges(
    total_rows: int,
    train_count: int,
    validation_count: int,
    test_count: int,
):
    selected = (
        train_count
        + validation_count
        + test_count
    )

    if selected > total_rows:
        raise ValueError(
            f"Requested {selected} samples "
            f"from file containing only {total_rows}"
        )

    unused = total_rows - selected

    gap_1 = unused // 2
    gap_2 = unused - gap_1

    train_start = 0
    train_end = train_start + train_count

    validation_start = train_end + gap_1
    validation_end = (
        validation_start
        + validation_count
    )

    test_start = validation_end + gap_2
    test_end = test_start + test_count

    assert test_end == total_rows

    return {
        "train": [
            train_start,
            train_end,
        ],
        "validation": [
            validation_start,
            validation_end,
        ],
        "test": [
            test_start,
            test_end,
        ],
        "gap_1": gap_1,
        "gap_2": gap_2,
    }


def shuffle_split(
    x: np.ndarray,
    y: np.ndarray,
    source_id: np.ndarray,
    rng: np.random.Generator,
):
    indices = rng.permutation(len(y))

    return (
        x[indices],
        y[indices],
        source_id[indices],
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_data = {
        "train": {
            "x": [],
            "y": [],
            "source_id": [],
        },
        "validation": {
            "x": [],
            "y": [],
            "source_id": [],
        },
        "test": {
            "x": [],
            "y": [],
            "source_id": [],
        },
    }

    feature_names = None
    source_metadata = []

    print("Creating N-BaIoT initial binary subset")
    print("=" * 60)

    for source_id, source in enumerate(SOURCES):

        path = source["path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Missing source file: {path}"
            )

        print(
            f"\nLoading: {source['name']}"
        )

        df = pd.read_csv(
            path,
            dtype=np.float32,
        )

        if len(df.columns) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"{source['name']} has "
                f"{len(df.columns)} features, "
                f"expected {EXPECTED_FEATURE_COUNT}"
            )

        current_features = list(df.columns)

        if feature_names is None:
            feature_names = current_features

        elif current_features != feature_names:
            raise ValueError(
                f"Feature mismatch in {source['name']}"
            )

        total_rows = len(df)

        ranges = calculate_ranges(
            total_rows=total_rows,
            train_count=source["train"],
            validation_count=source["validation"],
            test_count=source["test"],
        )

        values = df.to_numpy(
            dtype=np.float32,
            copy=False,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in "
                f"{source['name']}"
            )

        print(
            f"Rows: {total_rows:,}"
        )

        print(
            f"Train: "
            f"{ranges['train'][0]:,}"
            f":{ranges['train'][1]:,}"
        )

        print(
            f"Validation: "
            f"{ranges['validation'][0]:,}"
            f":{ranges['validation'][1]:,}"
        )

        print(
            f"Test: "
            f"{ranges['test'][0]:,}"
            f":{ranges['test'][1]:,}"
        )

        print(
            f"Gaps: "
            f"{ranges['gap_1']:,}, "
            f"{ranges['gap_2']:,}"
        )

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            start, end = ranges[split_name]

            x_part = values[start:end].copy()

            y_part = np.full(
                len(x_part),
                source["label"],
                dtype=np.uint8,
            )

            source_part = np.full(
                len(x_part),
                source_id,
                dtype=np.uint8,
            )

            split_data[split_name]["x"].append(
                x_part
            )

            split_data[split_name]["y"].append(
                y_part
            )

            split_data[split_name][
                "source_id"
            ].append(source_part)

        source_metadata.append(
            {
                "source_id": source_id,
                "name": source["name"],
                "path": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "binary_label": source["label"],
                "total_rows": total_rows,
                "selected": {
                    "train": source["train"],
                    "validation": source[
                        "validation"
                    ],
                    "test": source["test"],
                },
                "ranges": ranges,
            }
        )

        del df
        del values

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    final_metadata = {
        "dataset": "N-BaIoT",
        "device": "Danmini Doorbell",
        "problem": "binary_classification",
        "labels": {
            "0": "benign",
            "1": "attack",
        },
        "random_seed": RANDOM_SEED,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "split_strategy": (
            "Contiguous regions from each "
            "source CSV with unused gaps "
            "between train, validation and test. "
            "Rows are shuffled only after "
            "splitting."
        ),
        "row_index_convention": (
            "Zero-based data-row indices; "
            "end index is exclusive."
        ),
        "sources": source_metadata,
        "splits": {},
    }

    print("\n" + "=" * 60)
    print("Combining splits")

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        x = np.concatenate(
            split_data[split_name]["x"],
            axis=0,
        )

        y = np.concatenate(
            split_data[split_name]["y"],
            axis=0,
        )

        source_id = np.concatenate(
            split_data[split_name][
                "source_id"
            ],
            axis=0,
        )

        x, y, source_id = shuffle_split(
            x,
            y,
            source_id,
            rng,
        )

        output_path = (
            OUTPUT_DIR
            / f"{split_name}.npz"
        )

        np.savez_compressed(
            output_path,
            x=x,
            y=y,
            source_id=source_id,
        )

        benign_count = int(
            np.sum(y == 0)
        )

        attack_count = int(
            np.sum(y == 1)
        )

        final_metadata["splits"][
            split_name
        ] = {
            "samples": int(len(y)),
            "benign": benign_count,
            "attack": attack_count,
            "shape": list(x.shape),
        }

        print(
            f"{split_name}: "
            f"{x.shape}, "
            f"benign={benign_count:,}, "
            f"attack={attack_count:,}"
        )

    metadata_path = (
        ARTIFACT_DIR
        / "subset_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_metadata,
            file,
            indent=2,
        )

        file.write("\n")

    print("\nSubset created successfully.")
    print(
        f"Processed data: {OUTPUT_DIR}"
    )
    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()