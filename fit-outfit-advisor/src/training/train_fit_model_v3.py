from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

from src.config.paths import DEFAULT_MODCLOTH_DATASET_PATH, FIT_V3_DIR
from src.preprocessing.tabular_preprocessing import (
    AMBIGUOUS_COMMERCIAL_CATEGORIES,
    EXPLICIT_CLOTHING_CATEGORIES,
    build_fit_v3_inference_contract,
    prepare_fit_training_frame_v3,
)


FIT_V3_MODEL_PATH = FIT_V3_DIR / "fit_model.keras"
FIT_V3_ESTIMATOR_PATH = FIT_V3_DIR / "fit_estimator.joblib"
FIT_V3_PREPROCESSOR_PATH = FIT_V3_DIR / "fit_preprocessor.joblib"
FIT_V3_LABEL_ENCODER_PATH = FIT_V3_DIR / "fit_label_encoder.joblib"
FIT_V3_METADATA_PATH = FIT_V3_DIR / "metadata.json"
FIT_V3_METRICS_PATH = FIT_V3_DIR / "metrics.json"
FIT_V3_RAW_CM_PATH = FIT_V3_DIR / "confusion_matrix_raw.png"
FIT_V3_NORM_CM_PATH = FIT_V3_DIR / "confusion_matrix_normalized.png"
FIT_V3_HISTORY_PATH = FIT_V3_DIR / "training_history.png"


def _read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset ModCloth introuvable: {path}\n"
            "Place le fichier dans data/raw/ ou passe --dataset.\n"
            "Colonnes attendues pour V3: fit, size, category, height, hips, bra size, cup size."
        )

    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Format dataset non supporte. Utilise .json, .jsonl ou .csv.")


def _sample_dataframe() -> pd.DataFrame:
    rows = [
        {"fit": "fit", "height": "5ft 5in", "size": 8, "category": "tops", "hips": 38, "bra size": 34, "cup size": "c"},
        {"fit": "small", "height": "5ft 7in", "size": 4, "category": "dresses", "hips": 36, "bra size": 32, "cup size": "b"},
        {"fit": "large", "height": "5ft 2in", "size": 20, "category": "bottoms", "hips": 44, "bra size": 38, "cup size": "dd/e"},
        {"fit": "fit", "height": "170 cm", "size": 12, "category": "outerwear", "hips": None, "bra size": 36, "cup size": "d"},
        {"fit": "small", "height": "7ft 11in", "size": 15, "category": "new", "hips": 42, "bra size": None, "cup size": None},
        {"fit": "large", "height": "3ft", "size": 26, "category": "sale", "hips": 50, "bra size": 42, "cup size": "ddd/f"},
    ]
    return pd.DataFrame(rows * 12)


def _apply_category_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return df
    if "category" not in df.columns:
        raise ValueError("La colonne category est requise pour filtrer le scope V3.")

    category = df["category"].astype(str).str.lower().str.strip()
    if scope in {"explicit", "no-commercial"}:
        return df[category.isin(EXPLICIT_CLOTHING_CATEGORIES)].copy()
    raise ValueError("category_scope doit etre: all, explicit ou no-commercial.")


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
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def _build_mlp(input_dim: int, class_count: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dropout(0.20),
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


def _metrics_by_category(
    x_frame: pd.DataFrame,
    y_true,
    y_pred,
    labels: list[int],
    class_names: list[str],
) -> dict[str, dict]:
    if "category" not in x_frame.columns:
        return {}

    payload = {}
    categories = x_frame["category"].astype(str).fillna("missing")
    for category in sorted(categories.unique()):
        mask = categories == category
        if int(mask.sum()) == 0:
            continue
        payload[str(category)] = {
            "row_count": int(mask.sum()),
            **_metrics_dict(np.asarray(y_true)[mask.to_numpy()], np.asarray(y_pred)[mask.to_numpy()], labels, class_names),
        }
    return payload


def _summary_metrics(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    }


def _plot_confusion_matrix(matrix: list[list[float]], labels: list[str], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

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
    if not history_by_experiment:
        return

    import matplotlib.pyplot as plt

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


def _train_logistic_regression(x_train_ready, y_train, x_val_ready, y_val, labels, class_names):
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x_train_ready, y_train)
    y_pred = model.predict(x_val_ready)
    metrics = _metrics_dict(y_val, y_pred, labels, class_names)
    print("\nValidation experiment: logistic_regression")
    print(json.dumps(_summary_metrics(metrics), indent=2))
    return model, metrics


def _train_mlp_experiment(
    name: str,
    x_train_ready,
    y_train,
    x_val_ready,
    y_val,
    labels: list[int],
    class_names: list[str],
    class_weight: dict[int, float] | None,
    epochs: int,
    batch_size: int,
):
    import tensorflow as tf

    model = _build_mlp(x_train_ready.shape[1], len(class_names))
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
    print(json.dumps(_summary_metrics(metrics), indent=2))
    return model, metrics, history.history


def _select_best_experiment(validation_metrics: dict[str, dict]) -> str:
    candidates = {
        name: value
        for name, value in validation_metrics.items()
        if name != "majority_baseline"
    }
    return max(
        candidates,
        key=lambda name: (
            candidates[name]["macro_f1"],
            candidates[name]["balanced_accuracy"],
            candidates[name]["per_class"].get("small", {}).get("recall", 0.0),
            candidates[name]["per_class"].get("large", {}).get("recall", 0.0),
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


def _count_labels(y_values, class_names: list[str]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in pd.Series(y_values).map(lambda idx: class_names[idx]).value_counts().items()
    }


def _raw_diagnostic(df: pd.DataFrame, dataset_path: Path, category_scope: str) -> dict:
    missing_values = df.isna().sum().sort_values(ascending=False).to_dict()
    return {
        "dataset_path": str(dataset_path),
        "category_scope": category_scope,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "missing_values_by_column": {str(key): int(value) for key, value in missing_values.items()},
        "target_distribution_before_cleaning": (
            {
                str(key): int(value)
                for key, value in df["fit"].astype(str).str.lower().str.strip().value_counts(dropna=False).items()
            }
            if "fit" in df.columns
            else {}
        ),
        "category_distribution_before_cleaning": (
            {
                str(key): int(value)
                for key, value in df["category"].astype(str).str.lower().str.strip().value_counts(dropna=False).items()
            }
            if "category" in df.columns
            else {}
        ),
        "category_groups": {
            "explicit_clothing_categories": list(EXPLICIT_CLOTHING_CATEGORIES),
            "ambiguous_commercial_categories": list(AMBIGUOUS_COMMERCIAL_CATEGORIES),
        },
    }


def train(args: argparse.Namespace) -> None:
    df_raw = _sample_dataframe() if args.sample else _read_dataset(args.dataset)
    dataset_path = Path("sample") if args.sample else args.dataset
    if args.sample:
        print("Mode --sample: jeu artificiel uniquement destine a verifier le pipeline V3.")

    df = _apply_category_scope(df_raw, args.category_scope)
    if df.empty:
        raise ValueError(f"Aucune ligne apres application du category_scope={args.category_scope}.")

    print("\n=== Diagnostic dataset brut V3 ===")
    raw_diagnostic = _raw_diagnostic(df, dataset_path, args.category_scope)
    print(json.dumps(raw_diagnostic, indent=2, ensure_ascii=False))

    x, y, preprocessing_diagnostic = prepare_fit_training_frame_v3(
        df,
        min_height_cm=args.min_height_cm,
        max_height_cm=args.max_height_cm,
    )
    print("\n=== Diagnostic preprocessing V3 ===")
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

    class_distribution_train = _count_labels(y_train, class_names)
    class_distribution_validation = _count_labels(y_val, class_names)
    class_distribution_test = _count_labels(y_test, class_names)
    dataset_row_counts = {
        "raw": int(len(df_raw)),
        "after_category_scope": int(len(df)),
        "after_cleaning": int(len(x)),
        "train": int(len(x_train)),
        "validation": int(len(x_val)),
        "test": int(len(x_test)),
    }
    print("\nDistribution classes:")
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
    validation_metrics["majority_baseline"]["by_category"] = _metrics_by_category(
        x_val, y_val, majority_val_pred, labels, class_names
    )
    print("\n=== Majority baseline validation ===")
    print(json.dumps(_summary_metrics(validation_metrics["majority_baseline"]), indent=2))

    models: dict[str, Any] = {}
    model_types: dict[str, str] = {}
    histories: dict[str, Any] = {}

    logistic_model, validation_metrics["logistic_regression"] = _train_logistic_regression(
        x_train_ready,
        y_train,
        x_val_ready,
        y_val,
        labels,
        class_names,
    )
    validation_metrics["logistic_regression"]["by_category"] = _metrics_by_category(
        x_val,
        y_val,
        logistic_model.predict(x_val_ready),
        labels,
        class_names,
    )
    models["logistic_regression"] = logistic_model
    model_types["logistic_regression"] = "sklearn_logistic_regression"

    unweighted_model, validation_metrics["mlp_unweighted"], histories["mlp_unweighted"] = _train_mlp_experiment(
        "mlp_unweighted",
        x_train_ready,
        y_train,
        x_val_ready,
        y_val,
        labels,
        class_names,
        class_weight=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    validation_metrics["mlp_unweighted"]["by_category"] = _metrics_by_category(
        x_val,
        y_val,
        unweighted_model.predict(x_val_ready, verbose=0).argmax(axis=1),
        labels,
        class_names,
    )
    models["mlp_unweighted"] = unweighted_model
    model_types["mlp_unweighted"] = "keras_mlp"

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array(labels), y=y_train)
    class_weight_dict = {int(label): float(weight) for label, weight in zip(labels, class_weights)}
    weighted_model, validation_metrics["mlp_class_weight"], histories["mlp_class_weight"] = _train_mlp_experiment(
        "mlp_class_weight",
        x_train_ready,
        y_train,
        x_val_ready,
        y_val,
        labels,
        class_names,
        class_weight=class_weight_dict,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    validation_metrics["mlp_class_weight"]["by_category"] = _metrics_by_category(
        x_val,
        y_val,
        weighted_model.predict(x_val_ready, verbose=0).argmax(axis=1),
        labels,
        class_names,
    )
    models["mlp_class_weight"] = weighted_model
    model_types["mlp_class_weight"] = "keras_mlp"

    selected_experiment = _select_best_experiment(validation_metrics)
    selected_model = models[selected_experiment]
    selected_model_type = model_types[selected_experiment]
    reason_for_selection = _selection_reason(selected_experiment, validation_metrics)

    print("\n=== Final test evaluation after validation-only selection ===")
    x_test_ready = preprocessor.transform(x_test)
    test_metrics: dict[str, Any] = {}

    majority_test_pred = np.full_like(y_test, majority_label)
    test_metrics["majority_baseline"] = _metrics_dict(y_test, majority_test_pred, labels, class_names)
    test_metrics["majority_baseline"]["by_category"] = _metrics_by_category(
        x_test, y_test, majority_test_pred, labels, class_names
    )

    if selected_model_type == "keras_mlp":
        selected_predictions = selected_model.predict(x_test_ready, verbose=0).argmax(axis=1)
    else:
        selected_predictions = selected_model.predict(x_test_ready)

    test_metrics["selected_experiment"] = _metrics_dict(y_test, selected_predictions, labels, class_names)
    test_metrics["selected_experiment"]["by_category"] = _metrics_by_category(
        x_test, y_test, selected_predictions, labels, class_names
    )

    print(json.dumps(_summary_metrics(test_metrics["selected_experiment"]), indent=2))
    print(classification_report(y_test, selected_predictions, labels=labels, target_names=class_names, zero_division=0))
    print("Final selected test confusion matrix:")
    print(np.array(test_metrics["selected_experiment"]["confusion_matrix_raw"]))

    academically_improved_over_baseline = (
        test_metrics["selected_experiment"]["macro_f1"] > test_metrics["majority_baseline"]["macro_f1"]
        and test_metrics["selected_experiment"]["balanced_accuracy"] > test_metrics["majority_baseline"]["balanced_accuracy"]
        and test_metrics["selected_experiment"]["per_class"].get("small", {}).get("recall", 0.0) > 0
        and test_metrics["selected_experiment"]["per_class"].get("large", {}).get("recall", 0.0) > 0
    )

    FIT_V3_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, FIT_V3_PREPROCESSOR_PATH)
    joblib.dump(label_encoder, FIT_V3_LABEL_ENCODER_PATH)
    if selected_model_type == "keras_mlp":
        selected_model.save(FIT_V3_MODEL_PATH)
    else:
        joblib.dump(selected_model, FIT_V3_ESTIMATOR_PATH)

    metadata = {
        "version": "fit_v3",
        "dataset": str(dataset_path),
        "category_scope": args.category_scope,
        "target_column": "fit",
        "feature_columns": preprocessing_diagnostic["feature_columns"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "class_labels": class_names,
        "label_mapping": label_mapping,
        "selected_experiment": selected_experiment,
        "selected_model_type": selected_model_type,
        "reason_for_selection": reason_for_selection,
        "class_weight": class_weight_dict,
        "promotable_to_streamlit": False,
        "model_status": "experimental_only",
        "streamlit_promotion_decision": (
            "V3 is experimental only. It must not be copied to models/fit_active/ "
            "without a separate promotion review."
        ),
        "academically_improved_over_majority_baseline": academically_improved_over_baseline,
        "abstention_strategy": {
            "enabled": True,
            "minimum_confidence": 0.60,
            "low_confidence_prediction": "uncertain",
            "service_policy": "Do not provide firm small/large advice while V3 is experimental.",
        },
        "dataset_row_counts": dataset_row_counts,
        "class_distribution_train": class_distribution_train,
        "class_distribution_validation": class_distribution_validation,
        "class_distribution_test": class_distribution_test,
        "category_groups": {
            "explicit_clothing_categories": list(EXPLICIT_CLOTHING_CATEGORIES),
            "ambiguous_commercial_categories": list(AMBIGUOUS_COMMERCIAL_CATEGORIES),
        },
        "inference_contract": build_fit_v3_inference_contract(preprocessing_diagnostic["feature_columns"]),
        "diagnostics": {
            "raw_dataset": raw_diagnostic,
            "preprocessing": preprocessing_diagnostic,
        },
    }
    FIT_V3_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_payload = {
        "version": "fit_v3",
        "selected_experiment": selected_experiment,
        "selected_model_type": selected_model_type,
        "reason_for_selection": reason_for_selection,
        "feature_columns": preprocessing_diagnostic["feature_columns"],
        "dataset_row_counts": dataset_row_counts,
        "class_distribution_train": class_distribution_train,
        "class_distribution_validation": class_distribution_validation,
        "class_distribution_test": class_distribution_test,
        "promotable_to_streamlit": False,
        "model_status": "experimental_only",
        "academically_improved_over_majority_baseline": academically_improved_over_baseline,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    FIT_V3_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_confusion_matrix(
        test_metrics["selected_experiment"]["confusion_matrix_raw"],
        class_names,
        FIT_V3_RAW_CM_PATH,
        f"{selected_experiment} - final test confusion matrix",
    )
    _plot_confusion_matrix(
        test_metrics["selected_experiment"]["confusion_matrix_normalized"],
        class_names,
        FIT_V3_NORM_CM_PATH,
        f"{selected_experiment} - final test normalized confusion matrix",
    )
    _plot_history(histories, FIT_V3_HISTORY_PATH)

    print("\n=== Decision V3 ===")
    print(f"Selected experiment: {selected_experiment}")
    print(f"Selected model type: {selected_model_type}")
    print(f"Reason for selection: {reason_for_selection}")
    print("Promotable to Streamlit: False")
    print("\nArtefacts fit_v3:")
    for path in [
        FIT_V3_ESTIMATOR_PATH if selected_model_type != "keras_mlp" else None,
        FIT_V3_MODEL_PATH if selected_model_type == "keras_mlp" else None,
        FIT_V3_PREPROCESSOR_PATH,
        FIT_V3_LABEL_ENCODER_PATH,
        FIT_V3_METADATA_PATH,
        FIT_V3_METRICS_PATH,
        FIT_V3_RAW_CM_PATH,
        FIT_V3_NORM_CM_PATH,
        FIT_V3_HISTORY_PATH,
    ]:
        if path is not None:
            print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train experimental V3 fit models on ModCloth data.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_MODCLOTH_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-height-cm", type=float, default=130.0)
    parser.add_argument("--max-height-cm", type=float, default=210.0)
    parser.add_argument(
        "--category-scope",
        choices=["all", "explicit", "no-commercial"],
        default="all",
        help="Use all categories or only explicit clothing categories.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use a tiny artificial dataset only to smoke-test the V3 pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("default", category=pd.errors.DtypeWarning)
    train(parse_args())
