import numpy as np
import tensorflow as tf


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def print_class_distribution(
    labels: np.ndarray,
    dataset_name: str,
) -> None:
    flat_labels = labels.flatten()

    classes, counts = np.unique(
        flat_labels,
        return_counts=True,
    )

    print(f"\n{dataset_name} class distribution:")

    for class_id, count in zip(classes, counts):
        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[int(class_id)]:10s} "
            f"-> {count}"
        )


def main() -> None:
    (
        (x_train, y_train),
        (x_test, y_test),
    ) = tf.keras.datasets.cifar10.load_data()

    print("CIFAR-10 dataset inspection")
    print("=" * 40)

    print("\nTraining images:")
    print("  shape:", x_train.shape)
    print("  dtype:", x_train.dtype)
    print("  min:", x_train.min())
    print("  max:", x_train.max())

    print("\nTraining labels:")
    print("  shape:", y_train.shape)
    print("  dtype:", y_train.dtype)

    print("\nTest images:")
    print("  shape:", x_test.shape)
    print("  dtype:", x_test.dtype)
    print("  min:", x_test.min())
    print("  max:", x_test.max())

    print("\nTest labels:")
    print("  shape:", y_test.shape)
    print("  dtype:", y_test.dtype)

    print(
        "\nPixels per image:",
        int(np.prod(x_train.shape[1:])),
    )

    print_class_distribution(
        y_train,
        "Training",
    )

    print_class_distribution(
        y_test,
        "Test",
    )


if __name__ == "__main__":
    main()
