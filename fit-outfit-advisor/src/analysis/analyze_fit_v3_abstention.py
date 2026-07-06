from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from src.config.paths import DEFAULT_MODCLOTH_DATASET_PATH, MODELS_DIR, REPORTS_DIR


UNCERTAIN_LABEL = "uncertain"
DEFAULT_THRESHOLDS = [float(Decimal("0.35") + Decimal("0.05") * index) for index in range(12)]


def apply_abstention(
    probabilities: np.ndarray,
    class_labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Convertit des probabilites en predictions avec abstention par seuil."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities doit etre une matrice 2D.")
    if probabilities.shape[1] != len(class_labels):
        raise ValueError("Le nombre de colonnes de probabilities doit correspondre aux labels.")

    raw_indexes = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    raw_predictions = [class_labels[index] for index in raw_indexes]
    final_predictions = [
        prediction if confidence >= threshold else UNCERTAIN_LABEL
        for prediction, confidence in zip(raw_predictions, confidences)
    ]
    return {
        "threshold": float(threshold),
        "raw_predictions": raw_predictions,
        "predictions": final_predictions,
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


def evaluate_abstention(
    y_true: list[str],
    y_pred_with_uncertain: list[str],
    class_labels: list[str],
) -> dict[str, Any]:
    """Calcule les metriques d'abstention, avec metriques de qualite sur les cas couverts."""
    if len(y_true) != len(y_pred_with_uncertain):
        raise ValueError("y_true et y_pred_with_uncertain doivent avoir la meme longueur.")

    total = len(y_true)
    covered_mask = [prediction != UNCERTAIN_LABEL for prediction in y_pred_with_uncertain]
    covered_count = sum(covered_mask)
    uncertain_count = total - covered_count

    covered_true = [true for true, keep in zip(y_true, covered_mask) if keep]
    covered_pred = [pred for pred, keep in zip(y_pred_with_uncertain, covered_mask) if keep]
    correct_covered = sum(true == pred for true, pred in zip(covered_true, covered_pred))
    per_class = _class_metrics(covered_true, covered_pred, class_labels)
    macro_f1 = float(np.mean([metrics["f1"] for metrics in per_class.values()])) if class_labels else 0.0

    confusion_labels = class_labels + [UNCERTAIN_LABEL]
    confusion_matrix = []
    for true_label in class_labels:
        row = []
        for pred_label in confusion_labels:
            row.append(
                int(
                    sum(
                        true == true_label and pred == pred_label
                        for true, pred in zip(y_true, y_pred_with_uncertain)
                    )
                )
            )
        confusion_matrix.append(row)

    abstention_by_true_class = {}
    for label in class_labels:
        support = sum(true == label for true in y_true)
        abstained = sum(
            true == label and pred == UNCERTAIN_LABEL
            for true, pred in zip(y_true, y_pred_with_uncertain)
        )
        abstention_by_true_class[label] = {
            "support": int(support),
            "abstained": int(abstained),
            "abstention_rate": _safe_divide(abstained, support),
        }

    return {
        "total_count": int(total),
        "covered_count": int(covered_count),
        "uncertain_count": int(uncertain_count),
        "coverage": _safe_divide(covered_count, total),
        "abstention_rate": _safe_divide(uncertain_count, total),
        "accuracy_non_abstained": _safe_divide(correct_covered, covered_count),
        "macro_f1_non_abstained": macro_f1,
        "per_class_non_abstained": per_class,
        "abstention_by_true_class": abstention_by_true_class,
        "confusion_labels": confusion_labels,
        "confusion_matrix": confusion_matrix,
    }


def evaluate_thresholds(
    y_true: list[str],
    probabilities: np.ndarray,
    class_labels: list[str],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Evalue une grille de seuils pour des probabilites donnees."""
    rows = []
    for threshold in thresholds:
        abstained = apply_abstention(probabilities, class_labels, threshold)
        metrics = evaluate_abstention(y_true, abstained["predictions"], class_labels)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
            }
        )
    return rows


def select_threshold(
    threshold_rows: list[dict[str, Any]],
    min_coverage: float = 0.25,
    min_precision_small: float = 0.40,
    min_precision_large: float = 0.40,
) -> dict[str, Any]:
    """Selectionne un seuil sur validation uniquement selon les contraintes metier."""
    if not threshold_rows:
        raise ValueError("threshold_rows ne doit pas etre vide.")

    eligible = []
    for row in threshold_rows:
        per_class = row["per_class_non_abstained"]
        small_precision = per_class.get("small", {}).get("precision", 0.0)
        large_precision = per_class.get("large", {}).get("precision", 0.0)
        if (
            row["coverage"] >= min_coverage
            and small_precision >= min_precision_small
            and large_precision >= min_precision_large
        ):
            eligible.append(row)

    if not eligible:
        diagnostic_best = max(
            threshold_rows,
            key=lambda row: (
                min(
                    row["per_class_non_abstained"].get("small", {}).get("precision", 0.0),
                    row["per_class_non_abstained"].get("large", {}).get("precision", 0.0),
                ),
                row["coverage"],
                row["macro_f1_non_abstained"],
            ),
        )
        return {
            "selected": False,
            "selected_threshold": None,
            "reason": "Aucun seuil validation ne respecte coverage >= 25% et precision small/large >= 0.40.",
            "constraints": {
                "min_coverage": min_coverage,
                "min_precision_small": min_precision_small,
                "min_precision_large": min_precision_large,
            },
            "diagnostic_best_threshold": diagnostic_best["threshold"],
        }

    selected = max(
        eligible,
        key=lambda row: (
            row["macro_f1_non_abstained"],
            row["coverage"],
            -row["threshold"],
        ),
    )
    return {
        "selected": True,
        "selected_threshold": selected["threshold"],
        "reason": (
            "Seuil selectionne sur validation parmi les seuils respectant coverage >= 25% "
            "et precision small/large >= 0.40."
        ),
        "constraints": {
            "min_coverage": min_coverage,
            "min_precision_small": min_precision_small,
            "min_precision_large": min_precision_large,
        },
    }


def _read_dataset(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Format dataset non supporte. Utilise .json, .jsonl ou .csv.")


def _load_model(artifact_dir: Path, metadata: dict):
    if metadata.get("selected_model_type") == "keras_mlp":
        import tensorflow as tf

        return tf.keras.models.load_model(artifact_dir / "fit_model.keras")

    import joblib

    return joblib.load(artifact_dir / "fit_estimator.joblib")


def _predict_probabilities(model, transformed, metadata: dict) -> np.ndarray:
    if metadata.get("selected_model_type") == "keras_mlp":
        return np.asarray(model.predict(transformed, verbose=0), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(transformed), dtype=float)
    raise ValueError("Le modele selectionne ne fournit pas de probabilites.")


def _split_for_artifact(df, metadata: dict, seed: int):
    from sklearn.model_selection import train_test_split

    from src.preprocessing.tabular_preprocessing import prepare_fit_training_frame_v3
    from src.training.train_fit_model_v3 import _apply_category_scope, _stratify_or_none

    scoped = _apply_category_scope(df, metadata.get("category_scope", "all"))
    x, y, _ = prepare_fit_training_frame_v3(scoped)

    import joblib

    label_encoder = joblib.load(Path(metadata["_artifact_dir"]) / "fit_label_encoder.joblib")
    y_encoded = label_encoder.transform(y)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y_encoded,
        test_size=0.30,
        random_state=seed,
        stratify=_stratify_or_none(y_encoded, 0.30),
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=_stratify_or_none(y_temp, 0.50),
    )
    return x_val, x_test, y_val, y_test, label_encoder


def analyze_artifact_dir(
    dataset_path: Path,
    artifact_dir: Path,
    output_path: Path,
    thresholds: list[float] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyse l'abstention d'un dossier d'artefacts V3 et ecrit un rapport JSON."""
    import joblib

    thresholds = thresholds or DEFAULT_THRESHOLDS
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["_artifact_dir"] = str(artifact_dir)
    class_labels = list(metadata["class_labels"])

    df = _read_dataset(dataset_path)
    x_val, x_test, y_val, y_test, label_encoder = _split_for_artifact(df, metadata, seed)

    preprocessor = joblib.load(artifact_dir / "fit_preprocessor.joblib")
    model = _load_model(artifact_dir, metadata)

    val_ready = preprocessor.transform(x_val)
    val_probabilities = _predict_probabilities(model, val_ready, metadata)
    val_true = label_encoder.inverse_transform(y_val).tolist()
    validation_table = evaluate_thresholds(val_true, val_probabilities, class_labels, thresholds)
    selection = select_threshold(validation_table)

    if selection["selected"]:
        selected_threshold = selection["selected_threshold"]
    else:
        selected_threshold = selection["diagnostic_best_threshold"]

    test_ready = preprocessor.transform(x_test)
    test_probabilities = _predict_probabilities(model, test_ready, metadata)
    test_true = label_encoder.inverse_transform(y_test).tolist()
    test_predictions = apply_abstention(test_probabilities, class_labels, selected_threshold)["predictions"]
    test_evaluation = evaluate_abstention(test_true, test_predictions, class_labels)
    test_evaluation["threshold"] = selected_threshold

    report = {
        "version": "fit_v3_abstention",
        "artifact_dir": str(artifact_dir),
        "dataset_path": str(dataset_path),
        "category_scope": metadata.get("category_scope"),
        "selected_experiment": metadata.get("selected_experiment"),
        "model_status": metadata.get("model_status"),
        "promotable_to_streamlit": False,
        "threshold_source": "validation",
        "threshold_selection": selection,
        "threshold_table_validation": validation_table,
        "test_evaluation_at_selected_threshold": test_evaluation,
        "decision": (
            "No firm Streamlit recommendation. Use uncertain when confidence is below the selected or diagnostic threshold."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse abstention thresholds for ModCloth V3 artifacts.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_MODCLOTH_DATASET_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run",
        choices=["all", "explicit", "both"],
        default="both",
        help="Which imported V3 artifact directory to analyse.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = ["all", "explicit"] if args.run == "both" else [args.run]
    for run in runs:
        artifact_dir = MODELS_DIR / f"fit_v3_{run}"
        output_path = REPORTS_DIR / f"modcloth_v3_abstention_{run}.json"
        report = analyze_artifact_dir(args.dataset, artifact_dir, output_path, seed=args.seed)
        selection = report["threshold_selection"]
        print(f"{run}: {selection['reason']}")
        print(f"{run}: report written to {output_path}")


if __name__ == "__main__":
    main()
