from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config.paths import (
    FASHION_V1_CLASSES_PATH,
    FASHION_V1_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from src.mappings.fashion_v1_mapping import load_fashion_v1_class_config
from src.training.train_fashion_model_v1 import (
    build_image_dataset,
    prepare_fashion_v1_training_frame,
    split_fashion_frame,
)


UNKNOWN_LABEL = "unknown"
DEFAULT_THRESHOLDS = [
    float(Decimal("0.50") + Decimal("0.05") * index)
    for index in range(10)
]
DEFAULT_FASHION_METADATA_CSV = RAW_DATA_DIR / "fashion-product-images-small" / "styles.csv"
DEFAULT_FASHION_IMAGE_DIR = RAW_DATA_DIR / "fashion-product-images-small" / "images"


def apply_image_abstention(
    probabilities: np.ndarray,
    class_labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Convertit des probabilites image en predictions avec sortie unknown sous seuil."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities doit etre une matrice 2D.")
    if probabilities.shape[1] != len(class_labels):
        raise ValueError("Le nombre de colonnes doit correspondre aux labels.")

    raw_indexes = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    raw_predictions = [class_labels[index] for index in raw_indexes]
    predictions = [
        prediction if confidence >= threshold else UNKNOWN_LABEL
        for prediction, confidence in zip(raw_predictions, confidences)
    ]
    return {
        "threshold": float(threshold),
        "raw_predictions": raw_predictions,
        "predictions": predictions,
        "confidences": confidences.tolist(),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _class_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_labels: list[str],
) -> dict[str, dict[str, float]]:
    metrics = {}
    for label in class_labels:
        tp = sum(true == label and pred == label for true, pred in zip(y_true, y_pred))
        fp = sum(true != label and pred == label for true, pred in zip(y_true, y_pred))
        fn = sum(true == label and pred != label for true, pred in zip(y_true, y_pred))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(sum(true == label for true in y_true)),
        }
    return metrics


def evaluate_image_abstention(
    y_true: list[str],
    y_pred_with_unknown: list[str],
    class_labels: list[str],
) -> dict[str, Any]:
    """Calcule coverage et qualite sur les predictions non unknown."""
    if len(y_true) != len(y_pred_with_unknown):
        raise ValueError("y_true et y_pred_with_unknown doivent avoir la meme longueur.")

    total = len(y_true)
    covered_mask = [prediction != UNKNOWN_LABEL for prediction in y_pred_with_unknown]
    covered_count = sum(covered_mask)
    unknown_count = total - covered_count

    covered_true = [true for true, keep in zip(y_true, covered_mask) if keep]
    covered_pred = [pred for pred, keep in zip(y_pred_with_unknown, covered_mask) if keep]
    correct_covered = sum(true == pred for true, pred in zip(covered_true, covered_pred))
    per_class = _class_metrics(covered_true, covered_pred, class_labels)
    macro_f1 = float(np.mean([metrics["f1"] for metrics in per_class.values()])) if class_labels else 0.0

    confusion_labels = class_labels + [UNKNOWN_LABEL]
    confusion_matrix = []
    for true_label in class_labels:
        row = []
        for pred_label in confusion_labels:
            row.append(
                int(
                    sum(
                        true == true_label and pred == pred_label
                        for true, pred in zip(y_true, y_pred_with_unknown)
                    )
                )
            )
        confusion_matrix.append(row)

    unknown_by_true_class = {}
    for label in class_labels:
        support = sum(true == label for true in y_true)
        unknown = sum(
            true == label and pred == UNKNOWN_LABEL
            for true, pred in zip(y_true, y_pred_with_unknown)
        )
        unknown_by_true_class[label] = {
            "support": int(support),
            "unknown": int(unknown),
            "unknown_rate": _safe_divide(unknown, support),
        }

    return {
        "total_count": int(total),
        "covered_count": int(covered_count),
        "unknown_count": int(unknown_count),
        "coverage": _safe_divide(covered_count, total),
        "unknown_rate": _safe_divide(unknown_count, total),
        "accuracy_non_unknown": _safe_divide(correct_covered, covered_count),
        "macro_f1_non_unknown": macro_f1,
        "per_class_non_unknown": per_class,
        "unknown_by_true_class": unknown_by_true_class,
        "confusion_labels": confusion_labels,
        "confusion_matrix": confusion_matrix,
    }


def evaluate_thresholds(
    y_true: list[str],
    probabilities: np.ndarray,
    class_labels: list[str],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        abstained = apply_image_abstention(probabilities, class_labels, threshold)
        metrics = evaluate_image_abstention(y_true, abstained["predictions"], class_labels)
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def _minimum_precision(row: dict[str, Any], monitored_classes: list[str]) -> float:
    per_class = row["per_class_non_unknown"]
    if not monitored_classes:
        return 0.0
    return min(per_class.get(label, {}).get("precision", 0.0) for label in monitored_classes)


def select_threshold(
    threshold_rows: list[dict[str, Any]],
    *,
    min_coverage: float = 0.70,
    min_macro_f1: float = 0.86,
    min_precision_per_monitored_class: float = 0.70,
    monitored_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Selectionne un seuil uniquement sur validation selon des contraintes image."""
    if not threshold_rows:
        raise ValueError("threshold_rows ne doit pas etre vide.")

    monitored_classes = monitored_classes or list(
        threshold_rows[0]["per_class_non_unknown"].keys()
    )
    eligible = []
    for row in threshold_rows:
        if (
            row["coverage"] >= min_coverage
            and row["macro_f1_non_unknown"] >= min_macro_f1
            and _minimum_precision(row, monitored_classes) >= min_precision_per_monitored_class
        ):
            eligible.append(row)

    constraints = {
        "min_coverage": min_coverage,
        "min_macro_f1": min_macro_f1,
        "min_precision_per_monitored_class": min_precision_per_monitored_class,
        "monitored_classes": monitored_classes,
    }
    if not eligible:
        diagnostic_best = max(
            threshold_rows,
            key=lambda row: (
                _minimum_precision(row, monitored_classes),
                row["macro_f1_non_unknown"],
                row["coverage"],
                -row["threshold"],
            ),
        )
        return {
            "selected": False,
            "selected_threshold": None,
            "reason": (
                "Aucun seuil validation ne respecte coverage, macro F1 et precision minimale "
                "par classe surveillee."
            ),
            "constraints": constraints,
            "diagnostic_best_threshold": diagnostic_best["threshold"],
        }

    selected = max(
        eligible,
        key=lambda row: (
            row["macro_f1_non_unknown"],
            row["coverage"],
            -row["threshold"],
        ),
    )
    return {
        "selected": True,
        "selected_threshold": selected["threshold"],
        "reason": "Seuil selectionne sur validation uniquement selon les contraintes Fashion V1.",
        "constraints": constraints,
    }


def _load_keras_model(artifact_dir: Path):
    import tensorflow as tf

    return tf.keras.models.load_model(artifact_dir / "fashion_model.keras")


def _predict_probabilities(model, dataset) -> np.ndarray:
    return np.asarray(model.predict(dataset, verbose=0), dtype=float)


def analyze_artifact_dir(
    *,
    metadata_csv: Path,
    image_dir: Path,
    artifact_dir: Path,
    output_path: Path,
    class_config_path: Path = FASHION_V1_CLASSES_PATH,
    thresholds: list[float] | None = None,
    seed: int = 42,
    batch_size: int = 32,
    image_size: int | None = None,
    min_coverage: float = 0.70,
    min_macro_f1: float = 0.86,
    min_precision_per_monitored_class: float = 0.70,
    monitored_classes: list[str] | None = None,
) -> dict[str, Any]:
    """Analyse les seuils confidence -> unknown pour un dossier Fashion V1."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    architecture = metadata.get("architecture") or metadata.get("selected_experiment")
    if architecture not in {"simple_cnn", "mobilenet_v2"}:
        raise ValueError(f"Architecture Fashion inconnue: {architecture}")

    class_config = load_fashion_v1_class_config(class_config_path)
    prepared_frame, diagnostics = prepare_fashion_v1_training_frame(
        metadata_csv,
        image_dir,
        class_config,
    )
    train_frame, validation_frame, test_frame = split_fashion_frame(prepared_frame, seed=seed)
    label_encoder = joblib.load(artifact_dir / "label_encoder.joblib")
    class_labels = [str(label) for label in label_encoder.classes_]
    if class_labels != list(metadata.get("class_labels", [])):
        raise ValueError("Les labels du label_encoder et des metadata ne correspondent pas.")

    validation_labels = label_encoder.transform(validation_frame["product_type_v0"])
    test_labels = label_encoder.transform(test_frame["product_type_v0"])
    resolved_image_size = int(image_size or metadata.get("image_size", 224))
    model = _load_keras_model(artifact_dir)

    validation_dataset = build_image_dataset(
        [str(path) for path in validation_frame["image_path"]],
        validation_labels.tolist(),
        image_size=resolved_image_size,
        batch_size=batch_size,
        architecture=architecture,
        shuffle=False,
        seed=seed,
    )
    validation_probabilities = _predict_probabilities(model, validation_dataset)
    validation_true = label_encoder.inverse_transform(validation_labels).tolist()
    validation_table = evaluate_thresholds(
        validation_true,
        validation_probabilities,
        class_labels,
        thresholds,
    )
    selection = select_threshold(
        validation_table,
        min_coverage=min_coverage,
        min_macro_f1=min_macro_f1,
        min_precision_per_monitored_class=min_precision_per_monitored_class,
        monitored_classes=monitored_classes,
    )
    selected_threshold = (
        selection["selected_threshold"]
        if selection["selected"]
        else selection["diagnostic_best_threshold"]
    )

    test_dataset = build_image_dataset(
        [str(path) for path in test_frame["image_path"]],
        test_labels.tolist(),
        image_size=resolved_image_size,
        batch_size=batch_size,
        architecture=architecture,
        shuffle=False,
        seed=seed,
    )
    test_probabilities = _predict_probabilities(model, test_dataset)
    test_true = label_encoder.inverse_transform(test_labels).tolist()
    test_predictions = apply_image_abstention(
        test_probabilities,
        class_labels,
        selected_threshold,
    )["predictions"]
    test_evaluation = evaluate_image_abstention(test_true, test_predictions, class_labels)
    test_evaluation["threshold"] = selected_threshold

    report = {
        "version": "fashion_v1_abstention",
        "artifact_dir": str(artifact_dir),
        "metadata_csv": str(metadata_csv),
        "image_dir": str(image_dir),
        "selected_experiment": metadata.get("selected_experiment"),
        "architecture": architecture,
        "model_status": metadata.get("model_status"),
        "promotable_to_streamlit_before_review": False,
        "threshold_source": "validation",
        "threshold_selection": selection,
        "threshold_table_validation": validation_table,
        "test_evaluation_at_selected_threshold": test_evaluation,
        "dataset_diagnostics": diagnostics,
        "class_labels": class_labels,
        "decision": (
            "Review threshold report before any copy to models/fashion_active. "
            "If promoted, store the chosen threshold in metadata.abstention_strategy.minimum_confidence."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse confidence thresholds for Fashion V1 artifacts.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_FASHION_METADATA_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_FASHION_IMAGE_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=FASHION_V1_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "fashion_v1_abstention.json",
    )
    parser.add_argument("--class-config", type=Path, default=FASHION_V1_CLASSES_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument("--min-macro-f1", type=float, default=0.86)
    parser.add_argument("--min-precision", type=float, default=0.70)
    parser.add_argument(
        "--monitored-classes",
        nargs="*",
        default=None,
        help="Classes dont la precision minimale doit respecter --min-precision. Defaut: toutes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_artifact_dir(
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        artifact_dir=args.artifact_dir,
        output_path=args.output,
        class_config_path=args.class_config,
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=args.image_size,
        min_coverage=args.min_coverage,
        min_macro_f1=args.min_macro_f1,
        min_precision_per_monitored_class=args.min_precision,
        monitored_classes=args.monitored_classes,
    )
    selection = report["threshold_selection"]
    print(selection["reason"])
    if selection["selected"]:
        print(f"Selected threshold: {selection['selected_threshold']}")
    else:
        print(f"Diagnostic threshold: {selection['diagnostic_best_threshold']}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
