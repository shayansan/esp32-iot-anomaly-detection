from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_mixed_preprocessed"
    / "full_115"
)

TCP_ID = 4
UDP_ID = 5


def load_split(name):
    data = np.load(
        DATA_ROOT / f"{name}.npz"
    )

    return (
        data["x"].astype(np.float64),
        data["y"].astype(np.int32),
    )


def summarize_pair(
    split_name,
    x,
    y,
):
    tcp = x[y == TCP_ID]
    udp = x[y == UDP_ID]

    tcp_mean = np.mean(
        tcp,
        axis=0,
    )

    udp_mean = np.mean(
        udp,
        axis=0,
    )

    mean_difference = np.abs(
        tcp_mean - udp_mean
    )

    tcp_std = np.std(
        tcp,
        axis=0,
    )

    udp_std = np.std(
        udp,
        axis=0,
    )

    pooled_scale = (
        tcp_std + udp_std
    ) / 2.0

    standardized_difference = (
        mean_difference
        / (pooled_scale + 1e-12)
    )

    print(
        f"\n=== {split_name.upper()} ==="
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
        "Centroid L2 distance:",
        float(
            np.linalg.norm(
                tcp_mean - udp_mean
            )
        ),
    )

    print(
        "Mean absolute centroid difference:",
        float(
            np.mean(
                mean_difference
            )
        ),
    )

    print(
        "Maximum absolute centroid difference:",
        float(
            np.max(
                mean_difference
            )
        ),
    )

    print(
        "Mean standardized difference:",
        float(
            np.mean(
                standardized_difference
            )
        ),
    )

    print(
        "Features with standardized "
        "difference > 0.5:",
        int(
            np.sum(
                standardized_difference > 0.5
            )
        ),
    )

    print(
        "Features with standardized "
        "difference > 1.0:",
        int(
            np.sum(
                standardized_difference > 1.0
            )
        ),
    )

    top_indices = np.argsort(
        standardized_difference
    )[-10:][::-1]

    print(
        "\nTop 10 separating "
        "feature indices:"
    )

    for index in top_indices:
        print(
            f"{index:3d}",
            "mean diff=",
            f"{mean_difference[index]:.6f}",
            "standardized=",
            f"{standardized_difference[index]:.6f}",
        )


def main():
    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        x, y = load_split(
            split_name
        )

        summarize_pair(
            split_name,
            x,
            y,
        )


if __name__ == "__main__":
    main()