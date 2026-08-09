from pathlib import Path

from tensorflow import keras

from preprocess_dataset import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIRECTORY = PROJECT_ROOT / "artifacts" / "mnist" / "models"
MODEL_PATH = MODEL_DIRECTORY / "mnist_cnn.keras"


def build_model() -> keras.Model:
    """
    Build a small CNN suitable for the MNIST TinyML benchmark.
    """

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),

            keras.layers.Conv2D(
                filters=8,
                kernel_size=(3, 3),
                activation="relu",
            ),

            keras.layers.MaxPooling2D(
                pool_size=(2, 2),
            ),

            keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation="relu",
            ),

            keras.layers.MaxPooling2D(
                pool_size=(2, 2),
            ),

            keras.layers.Flatten(),

            keras.layers.Dense(
                units=16,
                activation="relu",
            ),

            keras.layers.Dense(
                units=10,
                activation="softmax",
            ),
        ]
    )

    return model


def main() -> None:
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    model = build_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=2,
    )

    print()
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(MODEL_PATH)

    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
