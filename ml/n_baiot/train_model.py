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
    / "preprocessed"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "n_baiot"
    / "models"
)

RANDOM_SEED = 42
EPOCHS = 30
BATCH_SIZE = 128


def load_split(config, split):
    data = np.load(
        DATA_ROOT
        / config
        / f"{split}.npz"
    )

    return data["x"], data["y"]


def build_model(input_features):
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
                1,
                activation="sigmoid",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
            tf.keras.metrics.AUC(
                name="roc_auc"
            ),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR",
            ),
        ],
    )

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        choices=[
            "full_115",
            "reduced_60",
        ],
    )

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    x_train, y_train = load_split(
        args.config,
        "train",
    )

    x_val, y_val = load_split(
        args.config,
        "validation",
    )

    x_test, y_test = load_split(
        args.config,
        "test",
    )

    print(f"Configuration: {args.config}")
    print(f"Train: {x_train.shape}")
    print(f"Validation: {x_val.shape}")
    print(f"Test: {x_test.shape}")

    model = build_model(
        x_train.shape[1]
    )

    model.summary()

    output_dir = (
        ARTIFACT_ROOT
        / args.config
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    test_results = model.evaluate(
        x_test,
        y_test,
        verbose=0,
        return_dict=True,
    )

    model_path = (
        output_dir
        / "model.keras"
    )

    model.save(model_path)

    results = {
        "configuration": args.config,
        "input_features": int(
            x_train.shape[1]
        ),
        "train_samples": int(
            len(y_train)
        ),
        "validation_samples": int(
            len(y_val)
        ),
        "test_samples": int(
            len(y_test)
        ),
        "epochs_completed": len(
            history.history["loss"]
        ),
        "test_metrics": {
            key: float(value)
            for key, value
            in test_results.items()
        },
    }

    with open(
        output_dir / "training_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )
        file.write("\n")

    print("\nTest results:")

    for name, value in test_results.items():
        print(
            f"{name}: {value:.6f}"
        )

    print(
        f"\nModel saved to: {model_path}"
    )


if __name__ == "__main__":
    main()