from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config.paths import DEFAULT_MODCLOTH_DATASET_PATH, MODELS_DIR
from src.preprocessing.tabular_preprocessing import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    FIT_LABELS,
    prepare_fit_training_frame,
)


FIT_V2_DIR = MODELS_DIR / "fit_v2"
FIT_V2_MODEL_PATH = FIT_V2_DIR / "fit_model.keras"
FIT_V2_PREPROCESSOR_PATH = FIT_V2_DIR / "fit_preprocessor.joblib"
FIT_V2_LABEL_ENCODER_PATH = FIT_V2_DIR / "fit_label_encoder.joblib"
FIT_V2_METADATA_PATH = FIT_V2_DIR / "metadata.json"
FIT_V2_METRICS_PATH = FIT_V2_DIR / "metrics.json"
FIT_V2_RAW_CM_PATH = FIT_V2_DIR / "confusion_matrix_raw.png"
FIT_V2_NORM_CM_PATH = FIT_V2_DIR / "confusion_matrix_normalized.png"
FIT_V2_HISTORY_PATH = FIT_V2_DIR / "training_history.png"


@dataclass(frozen=True)
class DatasetWarningColumn:
    index: int
    name: str | None
    observed_python_types: dict[str, int]
    sample_values: list[Any]
    cleaning_strategy: str


def _read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        expected = ["fit", "height", "body type", "size", "category"]
        raise FileNotFoundError(
            f"Dataset ModCloth introuvable: {path}\n"
            "Place le fichier dans data/raw/ ou passe --dataset.\n"
            "Colonnes attendues au minimum pour V2: " + ", ".join(expected)
        )

    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Format dataset non supporte. Utilise .json, .jsonl ou .csv.")


def inspect_warning_column(path: Path, column_index: int = 8) -> DatasetWarningColumn:
    """
    Identifie sans supposition la colonne signalee par pandas DtypeWarning.

    Pandas indexe les colonnes a partir de 0 dans `Columns (8)`.
    """
    if path.suffix.lower() != ".csv":
        return DatasetWarningColumn(
            index=column_index,
            name=None,
            observed_python_types={},
            sample_values=[],
            cleaning_strategy="DtypeWarning pandas non applicable directement: dataset non CSV.",
        )

    columns = list(pd.read_csv(path, nrows=0).columns)
    if column_index >= len(columns):
        return DatasetWarningColumn(
            index=column_index,
            name=None,
            observed_python_types={},
            sample_values=[],
            cleaning_strategy="Index hors limites pour ce CSV.",
        )

    column_name = columns[column_index]
    sample = pd.read_csv(path, usecols=[column_name], dtype={column_name: "object"}, low_memory=False)
    series = sample[column_name].dropna()
    observed_types = series.map(lambda value: type(value).__name__).value_counts().to_dict()
    sample_values = series.astype(str).drop_duplicates().head(20).tolist()
    strategy = (
        "Conserver cette colonne hors features V2 sauf justification metier explicite. "
        "Si elle devient utile, la parser dans une fonction dediee et documenter les valeurs invalides."
    )
    return DatasetWarningColumn(
        index=column_index,
        name=str(column_name),
        observed_python_types={str(key): int(value) for key, value in observed_types.items()},
        sample_values=sample_values,
        cleaning_strategy=strategy,
    )


def _sample_dataframe() -> pd.DataFrame:
    rows = [
        {"fit": "fit", "height": "5ft 5in", "body type": "hourglass", "size": 8, "category": "new"},
        {"fit": "small", "height": "5ft 8in", "body type": "athletic", "size": 4, "category": "dresses"},
        {"fit": "large", "height": "5ft 2in", "body type": "petite", "size": 16, "category": "tops"},
        {"fit": "fit", "height": "170 cm", "body type": "straight", "size": 12, "category": "bottoms"},
        {"fit": "small", "height": "5ft 7in", "body type": "curvy", "size": 2, "category": "new"},
        {"fit": "large", "height": "5ft 3in", "body type": "petite", "size": 18, "category": "dresses"},
    ]
    return pd.DataFrame(rows * 12)


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


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers)


def build_model(input_dim: int, class_count: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _metrics_dict(y_true, y_pred, labels: list[int], class_names: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    raw_cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_cm = raw_cm.astype("float") / raw_cm.sum(axis=1, keepdims=True)
        normalized_cm = np.nan_to_num(normalized_cm)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": report,
        "confusion_matrix_raw": raw_cm.tolist(),
        "confusion_matrix_normalized": normalized_cm.tolist(),
    }


def _plot_confusion_matrix(matrix: list[list[float]], labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Truth")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            text = f"{value:.2f}" if isinstance(value, float) else str(value)
            ax.text(j, i, text, ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_history(history_by_experiment: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in history_by_experiment.items():
        axes[0].plot(history.get("loss", []), label=f"{name} train")
        axes[0].plot(history.get("val_loss", []), label=f"{name} val")
        axes[1].plot(history.get("accuracy", []), label=f"{name} train")
        axes[1].plot(history.get("val_accuracy", []), label=f"{name} val")
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _train_experiment(
    name: str,
    x_train_ready,
    y_train,
    x_val_ready,
    y_val,
    class_names: list[str],
    class_weight: dict[int, float] | None,
    epochs: int,
    batch_size: int,
):
    import tensorflow as tf

    labels = list(range(len(class_names)))
    model = build_model(x_train_ready.shape[1], len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train_ready,
        y_train,
        validation_data=(x_val_ready, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    probabilities = model.predict(x_val_ready, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    metrics = _metrics_dict(y_val, y_pred, labels, class_names)
    metrics["best_val_loss"] = float(min(history.history.get("val_loss", [np.nan])))
    metrics["epochs_run"] = int(len(history.history.get("loss", [])))
    print(f"\nValidation experiment: {name}")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}, indent=2))
    print(classification_report(y_val, y_pred, labels=labels, target_names=class_names, zero_division=0))
    print("Validation confusion matrix:")
    print(np.array(metrics["confusion_matrix_raw"]))
    return model, metrics, history.history


def _select_best_experiment(validation_metrics: dict[str, dict]) -> str:
    model_experiments = {name: value for name, value in validation_metrics.items() if name != "majority_baseline"}
    return max(
        model_experiments,
        key=lambda name: (
            model_experiments[name]["macro_f1"],
            model_experiments[name]["balanced_accuracy"],
            model_experiments[name]["per_class"].get("small", {}).get("recall", 0.0),
            model_experiments[name]["per_class"].get("large", {}).get("recall", 0.0),
        ),
    )


def _selection_reason(selected_experiment: str, validation_metrics: dict[str, dict]) -> str:
    selected = validation_metrics[selected_experiment]
    baseline = validation_metrics["majority_baseline"]
    return (
        f"{selected_experiment} selected on validation only: "
        f"macro_f1={selected['macro_f1']:.4f} vs majority_baseline={baseline['macro_f1']:.4f}, "
        f"balanced_accuracy={selected['balanced_accuracy']:.4f} vs majority_baseline={baseline['balanced_accuracy']:.4f}, "
        f"small_recall={selected['per_class'].get('small', {}).get('recall', 0.0):.4f}, "
        f"large_recall={selected['per_class'].get('large', {}).get('recall', 0.0):.4f}."
    )


def _diagnose_raw_dataset(df: pd.DataFrame, dataset_path: Path, warning_column: DatasetWarningColumn) -> dict:
    missing_values = df.isna().sum().sort_values(ascending=False).to_dict()
    category_values = (
        {str(key): int(value) for key, value in df["category"].astype(str).value_counts(dropna=False).head(20).items()}
        if "category" in df.columns
        else {}
    )
    return {
        "dataset_path": str(dataset_path),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "available_expected_columns": {
            "target": "fit" in df.columns,
            "size": "size" in df.columns,
            "height": "height" in df.columns,
            "body type": "body type" in df.columns,
            "category": "category" in df.columns,
            "weight": "weight" in df.columns,
        },
        "missing_values_by_column": {str(key): int(value) for key, value in missing_values.items()},
        "target_column": "fit",
        "target_distribution_before_cleaning": (
            {
                str(key): int(value)
                for key, value in df["fit"].astype(str).str.lower().str.strip().value_counts(dropna=False).items()
            }
            if "fit" in df.columns
            else {}
        ),
        "category_top_values": category_values,
        "dtype_warning_column": asdict(warning_column),
        "missing_value_strategy": {
            "numeric": "Median imputation fitted on train only, then StandardScaler fitted on train only.",
            "categorical": "Constant 'missing' imputation fitted on train only, then OneHotEncoder fitted on train only.",
            "weight": "Not used in V2 because it is absent in the observed ModCloth schema / not reliable.",
        },
    }


def train(args: argparse.Namespace) -> None:
    df = _sample_dataframe() if args.sample else _read_dataset(args.dataset)
    dataset_path = Path("sample") if args.sample else args.dataset
    if args.sample:
        print("Mode --sample: jeu artificiel uniquement destine a verifier le pipeline.")

    warning_column = inspect_warning_column(args.dataset, args.dtype_warning_column) if not args.sample else DatasetWarningColumn(
        index=args.dtype_warning_column,
        name=None,
        observed_python_types={},
        sample_values=[],
        cleaning_strategy="Non applicable en mode sample.",
    )

    print("\n=== Diagnostic dataset brut ===")
    raw_diagnostic = _diagnose_raw_dataset(df, dataset_path, warning_column)
    print(json.dumps(raw_diagnostic, indent=2, ensure_ascii=False))

    x, y, preprocessing_diagnostic = prepare_fit_training_frame(df)
    print("\n=== Diagnostic preprocessing ===")
    print(json.dumps(preprocessing_diagnostic, indent=2, ensure_ascii=False))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    labels = list(range(len(class_names)))
    label_mapping = {label: int(index) for index, label in enumerate(class_names)}
    print("\nMapping label -> index:")
    print(json.dumps(label_mapping, indent=2, ensure_ascii=False))

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

    print("\nDistribution classes:")
    class_distribution_train = {
        str(key): int(value)
        for key, value in pd.Series(y_train).map(lambda idx: class_names[idx]).value_counts().items()
    }
    class_distribution_validation = {
        str(key): int(value)
        for key, value in pd.Series(y_val).map(lambda idx: class_names[idx]).value_counts().items()
    }
    class_distribution_test = {
        str(key): int(value)
        for key, value in pd.Series(y_test).map(lambda idx: class_names[idx]).value_counts().items()
    }
    dataset_row_counts = {
        "raw": int(len(df)),
        "after_cleaning": int(len(x)),
        "train": int(len(x_train)),
        "validation": int(len(x_val)),
        "test": int(len(x_test)),
    }
    print("train:", class_distribution_train)
    print("validation:", class_distribution_validation)
    print("test:", class_distribution_test)

    numeric_features = preprocessing_diagnostic["numeric_features"]
    categorical_features = preprocessing_diagnostic["categorical_features"]
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    x_train_ready = preprocessor.fit_transform(x_train)
    x_val_ready = preprocessor.transform(x_val)

    majority_label = int(pd.Series(y_train).mode().iloc[0])
    majority_val_pred = np.full_like(y_val, majority_label)
    validation_metrics = {
        "majority_baseline": _metrics_dict(y_val, majority_val_pred, labels, class_names)
    }
    print("\n=== Majority baseline validation ===")
    print(json.dumps({k: validation_metrics["majority_baseline"][k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}, indent=2))

    models = {}
    histories = {}
    unweighted_model, validation_metrics["mlp_unweighted"], histories["mlp_unweighted"] = _train_experiment(
        "mlp_unweighted",
        x_train_ready,
        y_train,
        x_val_ready,
        y_val,
        class_names,
        class_weight=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    models["mlp_unweighted"] = unweighted_model

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array(labels), y=y_train)
    class_weight_dict = {int(label): float(weight) for label, weight in zip(labels, class_weights)}
    weighted_model, validation_metrics["mlp_class_weight"], histories["mlp_class_weight"] = _train_experiment(
        "mlp_class_weight",
        x_train_ready,
        y_train,
        x_val_ready,
        y_val,
        class_names,
        class_weight=class_weight_dict,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    models["mlp_class_weight"] = weighted_model

    best_experiment = _select_best_experiment(validation_metrics)
    best_model = models[best_experiment]
    reason_for_selection = _selection_reason(best_experiment, validation_metrics)

    print("\n=== Final test evaluation for selected experiment only ===")
    x_test_ready = preprocessor.transform(x_test)
    test_probabilities = best_model.predict(x_test_ready, verbose=0)
    test_predictions = test_probabilities.argmax(axis=1)
    test_metrics = _metrics_dict(y_test, test_predictions, labels, class_names)
    print(json.dumps({k: test_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}, indent=2))
    print(classification_report(y_test, test_predictions, labels=labels, target_names=class_names, zero_division=0))
    print("Final test confusion matrix:")
    print(np.array(test_metrics["confusion_matrix_raw"]))

    promotable = (
        test_metrics["macro_f1"] > validation_metrics["majority_baseline"]["macro_f1"]
        and test_metrics["balanced_accuracy"] > validation_metrics["majority_baseline"]["balanced_accuracy"]
        and test_metrics["per_class"].get("small", {}).get("recall", 0.0) > 0
        and test_metrics["per_class"].get("large", {}).get("recall", 0.0) > 0
    )

    FIT_V2_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save(FIT_V2_MODEL_PATH)
    joblib.dump(preprocessor, FIT_V2_PREPROCESSOR_PATH)
    joblib.dump(label_encoder, FIT_V2_LABEL_ENCODER_PATH)

    metadata = {
        "version": "fit_v2",
        "dataset": str(dataset_path),
        "target_column": "fit",
        "feature_columns": preprocessing_diagnostic["feature_columns"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "class_labels": class_names,
        "label_mapping": label_mapping,
        "selected_experiment": best_experiment,
        "reason_for_selection": reason_for_selection,
        "class_weight": class_weight_dict,
        "promotable_to_streamlit": promotable,
        "dataset_row_counts": dataset_row_counts,
        "class_distribution_train": class_distribution_train,
        "class_distribution_validation": class_distribution_validation,
        "class_distribution_test": class_distribution_test,
        "inference_contract": {
            "user_profile": ["height_cm", "body_type"],
            "item_features": ["item_size", "category"],
            "excluded_previous_fields": ["weight_kg", "usual_size", "brand", "color"],
        },
        "diagnostics": {
            "raw_dataset": raw_diagnostic,
            "preprocessing": preprocessing_diagnostic,
        },
    }
    FIT_V2_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_payload = {
        "version": "fit_v2",
        "selected_experiment": best_experiment,
        "reason_for_selection": reason_for_selection,
        "feature_columns": preprocessing_diagnostic["feature_columns"],
        "dataset_row_counts": dataset_row_counts,
        "class_distribution_train": class_distribution_train,
        "class_distribution_validation": class_distribution_validation,
        "class_distribution_test": class_distribution_test,
        "promotable_to_streamlit": promotable,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    FIT_V2_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_confusion_matrix(
        test_metrics["confusion_matrix_raw"],
        class_names,
        FIT_V2_RAW_CM_PATH,
        f"{best_experiment} - final test confusion matrix",
    )
    _plot_confusion_matrix(
        test_metrics["confusion_matrix_normalized"],
        class_names,
        FIT_V2_NORM_CM_PATH,
        f"{best_experiment} - final test normalized confusion matrix",
    )
    _plot_history(histories, FIT_V2_HISTORY_PATH)

    print("\n=== Decision ===")
    print(f"Selected experiment: {best_experiment}")
    print(f"Reason for selection: {reason_for_selection}")
    print(f"Promotable to Streamlit: {promotable}")
    print("\nArtefacts fit_v2:")
    for path in [
        FIT_V2_MODEL_PATH,
        FIT_V2_PREPROCESSOR_PATH,
        FIT_V2_LABEL_ENCODER_PATH,
        FIT_V2_METADATA_PATH,
        FIT_V2_METRICS_PATH,
        FIT_V2_RAW_CM_PATH,
        FIT_V2_NORM_CM_PATH,
        FIT_V2_HISTORY_PATH,
    ]:
        print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train comparative TensorFlow MLP experiments on ModCloth fit data.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_MODCLOTH_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype-warning-column", type=int, default=8)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use a tiny artificial dataset only to smoke-test the pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("default", category=pd.errors.DtypeWarning)
    train(parse_args())
