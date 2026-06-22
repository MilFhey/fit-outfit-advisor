from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config.paths import (
    DEFAULT_MODCLOTH_DATASET_PATH,
    ENCODERS_DIR,
    FIGURES_DIR,
    FIT_LABEL_ENCODER_PATH,
    FIT_METADATA_PATH,
    FIT_MODEL_PATH,
    FIT_PREPROCESSOR_PATH,
)
from src.preprocessing.tabular_preprocessing import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NUMERIC_FEATURES,
    FIT_LABELS,
    prepare_fit_training_frame,
)


def _read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        expected = ["fit", "height", "weight", "body type", "size", "category"]
        raise FileNotFoundError(
            f"Dataset ModCloth introuvable: {path}\n"
            "Place le fichier dans data/raw/ ou passe --dataset.\n"
            "Colonnes attendues au minimum: " + ", ".join(expected)
        )

    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Format dataset non supporte. Utilise .json, .jsonl ou .csv.")


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fit": "fit", "height": "5ft 5in", "weight": "135lbs", "body type": "hourglass", "size": "M", "category": "tops"},
            {"fit": "small", "height": "5ft 8in", "weight": "170lbs", "body type": "athletic", "size": "S", "category": "dresses"},
            {"fit": "large", "height": "5ft 2in", "weight": "115lbs", "body type": "petite", "size": "XL", "category": "bottoms"},
            {"fit": "fit", "height": "170 cm", "weight": "65 kg", "body type": "straight", "size": "L", "category": "tops"},
            {"fit": "small", "height": "5ft 7in", "weight": "150lbs", "body type": "curvy", "size": "S", "category": "tops"},
            {"fit": "large", "height": "5ft 3in", "weight": "125lbs", "body type": "petite", "size": "XXL", "category": "dresses"},
        ]
    )


def build_model(input_dim: int, class_count: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _stratify_or_none(labels, test_size: float):
    counts = pd.Series(labels).value_counts()
    class_count = len(counts)
    test_count = int(len(labels) * test_size)
    train_count = len(labels) - test_count
    if counts.empty or counts.min() < 2:
        return None
    if test_count < class_count or train_count < class_count:
        return None
    return labels


def train(args: argparse.Namespace) -> None:
    df = _sample_dataframe() if args.sample else _read_dataset(args.dataset)
    if args.sample:
        print("Mode --sample: jeu artificiel uniquement destine a verifier le pipeline.")

    x, y = prepare_fit_training_frame(df)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y_encoded,
        test_size=0.30,
        random_state=args.seed,
        stratify=_stratify_or_none(y_encoded, 0.30),
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=args.seed,
        stratify=_stratify_or_none(y_temp, 0.50),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), DEFAULT_NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), DEFAULT_CATEGORICAL_FEATURES),
        ]
    )
    x_train_ready = preprocessor.fit_transform(x_train)
    x_val_ready = preprocessor.transform(x_val)
    x_test_ready = preprocessor.transform(x_test)

    model = build_model(x_train_ready.shape[1], len(label_encoder.classes_))
    model.fit(
        x_train_ready,
        y_train,
        validation_data=(x_val_ready, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test_ready, y_test, verbose=0)
    probabilities = model.predict(x_test_ready, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    FIT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model.save(FIT_MODEL_PATH)
    joblib.dump(preprocessor, FIT_PREPROCESSOR_PATH)
    joblib.dump(label_encoder, FIT_LABEL_ENCODER_PATH)
    FIT_METADATA_PATH.write_text(
        json.dumps(
            {
                "feature_columns": DEFAULT_FEATURE_COLUMNS,
                "numeric_features": DEFAULT_NUMERIC_FEATURES,
                "categorical_features": DEFAULT_CATEGORICAL_FEATURES,
                "class_labels": list(label_encoder.classes_) or list(FIT_LABELS),
                "dataset": "sample" if args.sample else str(args.dataset),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    example = x.head(1)
    example_ready = preprocessor.transform(example)
    example_prediction = model.predict(example_ready, verbose=0)[0]
    print("Example prediction:")
    print(
        {
            "input": example.to_dict(orient="records")[0],
            "predicted_fit": label_encoder.inverse_transform([int(example_prediction.argmax())])[0],
            "confidence": float(example_prediction.max()),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TensorFlow MLP on ModCloth fit data.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_MODCLOTH_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use a tiny artificial dataset only to smoke-test the pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
