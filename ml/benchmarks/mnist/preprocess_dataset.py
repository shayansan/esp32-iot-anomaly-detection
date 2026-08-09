import numpy as np
from tensorflow import keras


def load_and_preprocess_data():
    """
    Load MNIST and prepare the images for model training.

    Returns:
        x_train: Preprocessed training images.
        y_train: Training labels.
        x_test: Preprocessed test images.
        y_test: Test labels.
    """

    # Load the original MNIST dataset.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Convert pixel values from uint8 to float32.
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Normalize pixel values from [0, 255] to [0.0, 1.0].
    x_train /= 255.0
    x_test /= 255.0

    # Add a channel dimension:
    # (samples, 28, 28) -> (samples, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


def main() -> None:
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    print("Training images shape:", x_train.shape)
    print("Test images shape:", x_test.shape)

    print("Training image dtype:", x_train.dtype)
    print("Test image dtype:", x_test.dtype)

    print(
        "Training pixel range:",
        x_train.min(),
        "to",
        x_train.max(),
    )

    print("Training labels shape:", y_train.shape)
    print("Test labels shape:", y_test.shape)


if __name__ == "__main__":
    main()
