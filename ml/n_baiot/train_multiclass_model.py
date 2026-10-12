from pathlib import Path
import argparse
import json

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "n_baiot"
    / "processed"
    / "multiclass_preprocessed"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "multiclass_models"
)

RANDOM_SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 11

def load_split(
    variant,
    split_name,
):
    path = (
        DATA_ROOT
        / variant
        / f"{split_name}.npz"
    )

    data = np.load(path)

    return (
        data["x"].astype(np.float32),
        data["y"].astype(np.int32),
    )

def build_model(
    input_features,
):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(input_features,)
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                NUM_CLASSES,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"],
    )

    return model

def train_variant(
    variant,
):
    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    train_x, train_y = load_split(
        variant,
        "train",
    )

    validation_x, validation_y = (
        load_split(
            variant,
            "validation",
        )
    )

    test_x, test_y = load_split(
        variant,
        "test",
    )

    print(
        "Train:",
        train_x.shape,
        train_y.shape,
    )

    print(
        "Validation:",
        validation_x.shape,
        validation_y.shape,
    )

    print(
        "Test:",
        test_x.shape,
        test_y.shape,
    )

    model = build_model(
        train_x.shape[1]
    )

    model.summary()

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    )

    history = model.fit(
        train_x,
        train_y,
        validation_data=(
            validation_x,
            validation_y,
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping
        ],
        verbose=2,
        shuffle=True,
    )

    test_loss, test_accuracy = (
        model.evaluate(
            test_x,
            test_y,
            verbose=0,
        )
    )

    print(
        "\nTest loss:",
        test_loss,
    )

    print(
        "Test accuracy:",
        test_accuracy,
    )

    output_root = (
        ARTIFACT_ROOT
        / variant
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_root
        / "model.keras"
    )

    model.save(
        model_path
    )

    history_path = (
        output_root
        / "training_history.json"
    )

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history_data,
            file,
            indent=2,
        )

    result = {
        "variant": variant,
        "input_features":
            int(train_x.shape[1]),
        "classes": NUM_CLASSES,
        "epochs_trained":
            len(history.history["loss"]),
        "parameters":
            int(model.count_params()),
        "test_loss":
            float(test_loss),
        "test_accuracy":
            float(test_accuracy),
    }

    with (
        output_root
        / "training_result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        "\nSaved model:",
        model_path,
    )

    print(
        "Parameters:",
        model.count_params(),
    )

    print(
        "Epochs trained:",
        len(history.history["loss"]),
    )
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "full_115",
            "reduced_54",
        ],
    )

    args = parser.parse_args()

    train_variant(
        args.variant
    )


if __name__ == "__main__":
    main()