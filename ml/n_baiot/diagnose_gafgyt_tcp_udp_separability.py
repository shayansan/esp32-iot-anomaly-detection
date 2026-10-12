from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_mixed_baseline"
)

TCP_ID = 4
UDP_ID = 5


def load_binary_split(name):
    data = np.load(
        DATA_ROOT / f"{name}.npz"
    )

    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int32)

    mask = (
        (y == TCP_ID)
        | (y == UDP_ID)
    )

    x = x[mask]
    y = y[mask]

    # Binary labels:
    # TCP = 0
    # UDP = 1
    y = (y == UDP_ID).astype(np.int32)

    return x, y


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


def print_exact_overlap(x, y, split_name):
    tcp = x[y == 0]
    udp = x[y == 1]

    tcp_keys = np.unique(
        row_keys(tcp)
    )

    udp_keys = np.unique(
        row_keys(udp)
    )

    shared = np.intersect1d(
        tcp_keys,
        udp_keys,
    )

    print(
        f"\n=== {split_name.upper()} "
        "EXACT OVERLAP ==="
    )

    print(
        "TCP samples:",
        len(tcp),
    )

    print(
        "UDP samples:",
        len(udp),
    )

    print(
        "Unique TCP rows:",
        len(tcp_keys),
    )

    print(
        "Unique UDP rows:",
        len(udp_keys),
    )

    print(
        "Exact feature vectors "
        "present in both classes:",
        len(shared),
    )


def evaluate(
    model,
    name,
    x,
    y,
):
    prediction = model.predict(x)

    probability = (
        model.predict_proba(x)[:, 1]
    )

    print(
        f"\n=== {name.upper()} ==="
    )

    print(
        "Accuracy:",
        accuracy_score(
            y,
            prediction,
        ),
    )

    print(
        "Balanced accuracy:",
        balanced_accuracy_score(
            y,
            prediction,
        ),
    )

    print(
        "ROC AUC:",
        roc_auc_score(
            y,
            probability,
        ),
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y,
            prediction,
            target_names=[
                "gafgyt_tcp",
                "gafgyt_udp",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y,
            prediction,
        )
    )


def main():
    train_x, train_y = (
        load_binary_split("train")
    )

    validation_x, validation_y = (
        load_binary_split(
            "validation"
        )
    )

    test_x, test_y = (
        load_binary_split("test")
    )

    print_exact_overlap(
        train_x,
        train_y,
        "train",
    )

    print_exact_overlap(
        validation_x,
        validation_y,
        "validation",
    )

    print_exact_overlap(
        test_x,
        test_y,
        "test",
    )

    print(
        "\nTraining ExtraTrees "
        "binary diagnostic..."
    )

    model = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train_x,
        train_y,
    )

    evaluate(
        model,
        "train",
        train_x,
        train_y,
    )

    evaluate(
        model,
        "validation",
        validation_x,
        validation_y,
    )

    evaluate(
        model,
        "test",
        test_x,
        test_y,
    )

    importances = (
        model.feature_importances_
    )

    top_indices = np.argsort(
        importances
    )[-10:][::-1]

    print(
        "\nTop 10 feature indices "
        "by ExtraTrees importance:"
    )

    for index in top_indices:
        print(
            f"{index:3d}",
            f"{importances[index]:.6f}",
        )


if __name__ == "__main__":
    main()