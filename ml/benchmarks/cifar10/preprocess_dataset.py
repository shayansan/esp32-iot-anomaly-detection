import numpy as np
import tensorflow as tf


def load_and_preprocess_data():
    (
        (x_train, y_train),
        (x_test, y_test),
    ) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    x_train /= 255.0
    x_test /= 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


def main() -> None:
    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = load_and_preprocess_data()

    print("CIFAR-10 preprocessing")
    print("=" * 40)

    print("\nTraining images:")
    print("  shape:", x_train.shape)
    print("  dtype:", x_train.dtype)
    print("  min:", x_train.min())
    print("  max:", x_train.max())

    print("\nTraining labels:")
    print("  shape:", y_train.shape)
    print("  dtype:", y_train.dtype)
    print("  min:", y_train.min())
    print("  max:", y_train.max())

    print("\nTest images:")
    print("  shape:", x_test.shape)
    print("  dtype:", x_test.dtype)
    print("  min:", x_test.min())
    print("  max:", x_test.max())

    print("\nTest labels:")
    print("  shape:", y_test.shape)
    print("  dtype:", y_test.dtype)
    print("  min:", y_test.min())
    print("  max:", y_test.max())


if __name__ == "__main__":
    main()
