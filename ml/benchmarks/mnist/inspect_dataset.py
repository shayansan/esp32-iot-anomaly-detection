import numpy as np
from tensorflow import keras


def main() -> None:
    # Load the MNIST dataset.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Inspect dataset shapes.
    print("Training images shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)

    print("Test images shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)

    # Inspect data types.
    print("Image data type:", x_train.dtype)
    print("Label data type:", y_train.dtype)

    # Inspect pixel value range.
    print("Pixel value range:", x_train.min(), "to", x_train.max())

    # Inspect one example label.
    print("First label:", y_train[0])

    # Find all classes and count their training samples.
    classes, counts = np.unique(y_train, return_counts=True)

    print("Classes:", classes)
    print("Training samples per class:")

    for digit, count in zip(classes, counts):
        print(f"  {digit}: {count}")


if __name__ == "__main__":
    main()
