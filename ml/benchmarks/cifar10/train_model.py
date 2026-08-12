from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from preprocess_dataset import load_and_preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cifar10"
    / "models"
    / "cifar10_cnn.keras"
)


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(
                shape=(32, 32, 3)
            ),

            keras.layers.Conv2D(
                16,
                kernel_size=3,
                activation="relu",
            ),

            keras.layers.MaxPooling2D(
                pool_size=2
            ),

            keras.layers.Conv2D(
                24,
                kernel_size=3,
                activation="relu",
            ),

            keras.layers.MaxPooling2D(
                pool_size=2
            ),

            keras.layers.Conv2D(
                32,
                kernel_size=3,
                activation="relu",
            ),

            keras.layers.MaxPooling2D(
                pool_size=2
            ),

            keras.layers.Flatten(),

            keras.layers.Dense(
                32,
                activation="relu",
            ),

            keras.layers.Dense(
                10,
                activation="softmax",
            ),
        ]
    )

    return model


def main() -> None:
    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = load_and_preprocess_data()

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
        epochs=15,
        batch_size=128,
        validation_split=0.1,
        shuffle=True,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print()
    print("CIFAR-10 test results")
    print("=" * 40)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(MODEL_PATH)

    print()
    print("Saved model to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
