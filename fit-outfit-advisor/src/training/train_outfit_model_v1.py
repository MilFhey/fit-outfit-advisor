from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.analysis.analyze_polyvore_v0_schema_mapping import (
    CATEGORIES_FILE,
    DEFAULT_RAW_ROOT_CANDIDATES,
    METADATA_FILE,
    load_categories_lookup,
    load_json,
)
from src.analysis.build_polyvore_v0_cooccurrence_baseline import (
    load_mapped_split_items,
)
from src.config.paths import OUTFIT_V1_CONFIG_PATH, OUTFIT_V1_DIR
from src.mappings.polyvore_mapping import (
    load_outfit_v1_config,
    validate_outfit_v1_config,
)
from src.preprocessing.outfit_preprocessing import (
    OUTFIT_PAIR_FEATURE_COLUMNS,
    build_outfit_feature_frame,
    build_positive_outfit_pairs,
    generate_hard_negative_pairs,
)


OUTFIT_MODEL_PATH = OUTFIT_V1_DIR / "outfit_model.keras"
OUTFIT_PREPROCESSOR_PATH = OUTFIT_V1_DIR / "outfit_preprocessor.joblib"
OUTFIT_METADATA_PATH = OUTFIT_V1_DIR / "metadata.json"
OUTFIT_METRICS_PATH = OUTFIT_V1_DIR / "metrics.json"
OUTFIT_HISTORY_PATH = OUTFIT_V1_DIR / "training_history.png"
OUTFIT_RAW_CM_PATH = OUTFIT_V1_DIR / "confusion_matrix_raw.png"
OUTFIT_NORM_CM_PATH = OUTFIT_V1_DIR / "confusion_matrix_normalized.png"

DEFAULT_CONFIG = "disjoint"
DEFAULT_SPLITS = ("train", "valid", "test")


def resolve_raw_root(raw_root: Path | None) -> Path | None:
    if raw_root is not None:
        return raw_root
    for candidate in DEFAULT_RAW_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _split_path(raw_root: Path, config_name: str, split_name: str) -> Path:
    return raw_root / config_name / f"{split_name}.json"


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "outfit_id",
        "item_id",
        "product_type_v0",
        "canonical_category",
        "outfit_role",
        "source_field",
    ]
    return pd.DataFrame(rows, columns=columns)


def load_mapped_item_splits(
    raw_root: Path,
    *,
    config: dict[str, Any],
    config_name: str = DEFAULT_CONFIG,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    metadata_path = raw_root / METADATA_FILE
    categories_path = raw_root / CATEGORIES_FILE
    missing_files = [
        str(path)
        for path in [
            metadata_path,
            categories_path,
            *[_split_path(raw_root, config_name, split) for split in DEFAULT_SPLITS],
        ]
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Fichiers Polyvore raw manquants: {missing_files}")

    metadata = load_json(metadata_path)
    categories_lookup = load_categories_lookup(raw_root)
    item_splits: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {}

    for split_name in DEFAULT_SPLITS:
        rows, split_diagnostics = load_mapped_split_items(
            _split_path(raw_root, config_name, split_name),
            metadata=metadata,
            categories_lookup=categories_lookup,
            config=config,
        )
        item_splits[split_name] = _rows_to_frame(rows)
        diagnostics[split_name] = split_diagnostics

    return item_splits, diagnostics


def _filter_positive_overlap(
    pairs: pd.DataFrame,
    *,
    forbidden_positive_keys: set[tuple[str, str]],
) -> tuple[pd.DataFrame, int]:
    if pairs.empty:
        return pairs.copy(), 0
    positive_overlap = (pairs["label"] == 1) & pairs["pair_key"].isin(forbidden_positive_keys)
    filtered_count = int(positive_overlap.sum())
    return pairs.loc[~positive_overlap].copy(), filtered_count


def build_labeled_pair_splits(
    item_splits: dict[str, pd.DataFrame],
    *,
    config: dict[str, Any],
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    positive_splits = {
        split_name: build_positive_outfit_pairs(items, config)
        for split_name, items in item_splits.items()
    }
    all_positive_keys = {
        pair_key
        for frame in positive_splits.values()
        if not frame.empty
        for pair_key in frame["pair_key"]
    }

    pair_splits: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {
        "negative_ratio": float(negative_ratio),
        "positive_pair_counts_before_filtering": {},
        "filtered_positive_overlap_with_train": {},
        "labeled_pair_counts": {},
        "label_distribution": {},
    }
    train_positive_keys = (
        set(positive_splits["train"]["pair_key"])
        if not positive_splits["train"].empty
        else set()
    )

    for split_name in DEFAULT_SPLITS:
        positive_pairs = positive_splits[split_name]
        diagnostics["positive_pair_counts_before_filtering"][split_name] = int(len(positive_pairs))
        negative_pairs = generate_hard_negative_pairs(
            positive_pairs,
            item_splits[split_name],
            positive_pair_keys=all_positive_keys,
            ratio=negative_ratio,
            seed=seed,
        )
        labeled = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        if split_name != "train":
            labeled, filtered_count = _filter_positive_overlap(
                labeled,
                forbidden_positive_keys=train_positive_keys,
            )
        else:
            filtered_count = 0
        labeled = labeled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        pair_splits[split_name] = labeled
        diagnostics["filtered_positive_overlap_with_train"][split_name] = filtered_count
        diagnostics["labeled_pair_counts"][split_name] = int(len(labeled))
        diagnostics["label_distribution"][split_name] = {
            str(label): int(count)
            for label, count in labeled["label"].value_counts().sort_index().items()
        }

    return pair_splits, diagnostics


def build_outfit_training_splits(
    *,
    raw_root: Path,
    config_path: Path = OUTFIT_V1_CONFIG_PATH,
    config_name: str = DEFAULT_CONFIG,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    config = load_outfit_v1_config(config_path)
    validate_outfit_v1_config(config, require_ready=True)
    item_splits, item_diagnostics = load_mapped_item_splits(
        raw_root,
        config=config,
        config_name=config_name,
    )
    pair_splits, pair_diagnostics = build_labeled_pair_splits(
        item_splits,
        config=config,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    diagnostics = {
        "raw_root": str(raw_root),
        "config_path": str(config_path),
        "config_name": config_name,
        "feature_columns": OUTFIT_PAIR_FEATURE_COLUMNS,
        "item_splits": item_diagnostics,
        "pair_splits": pair_diagnostics,
    }
    return pair_splits, diagnostics


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                OUTFIT_PAIR_FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )


def build_outfit_mlp(input_dim: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="outfit_v1_pair_compatibility_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (probabilities >= threshold).astype(int)
    raw_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(
        raw_cm,
        row_sums,
        out=np.zeros_like(raw_cm, dtype=float),
        where=row_sums != 0,
    )
    try:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        roc_auc = None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_compatible": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_compatible": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["not_compatible", "compatible"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_raw": raw_cm.tolist(),
        "confusion_matrix_normalized": normalized_cm.round(6).tolist(),
    }


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, Any]]:
    thresholds = thresholds or [round(value, 2) for value in np.linspace(0.05, 0.95, 19)]
    metrics_by_threshold = {
        str(threshold): _classification_metrics(y_true, probabilities, threshold=threshold)
        for threshold in thresholds
    }
    selected_key = max(
        metrics_by_threshold,
        key=lambda key: (
            metrics_by_threshold[key]["macro_f1"],
            metrics_by_threshold[key]["balanced_accuracy"],
            metrics_by_threshold[key]["recall_compatible"],
        ),
    )
    return float(selected_key), metrics_by_threshold[selected_key]


def build_cooccurrence_score_table(train_pairs: pd.DataFrame) -> dict[tuple[str, str], float]:
    positive_pairs = train_pairs[train_pairs["label"] == 1]
    directed_counts: Counter[tuple[str, str]] = Counter()
    input_counts: Counter[str] = Counter()
    for row in positive_pairs.to_dict("records"):
        key = (row["input_product_type"], row["candidate_product_type"])
        directed_counts[key] += 1
        input_counts[row["input_product_type"]] += 1
    return {
        key: count / max(input_counts[key[0]], 1)
        for key, count in directed_counts.items()
    }


def score_pairs_with_cooccurrence(pairs: pd.DataFrame, score_table: dict[tuple[str, str], float]) -> np.ndarray:
    return np.array(
        [
            float(score_table.get((row["input_product_type"], row["candidate_product_type"]), 0.0))
            for row in pairs.to_dict("records")
        ],
        dtype=float,
    )


def ranking_metrics_for_products(
    pairs: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    by_input: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for row, probability in zip(pairs.to_dict("records"), probabilities):
        by_input[row["input_product_type"]].append(
            (row["candidate_product_type"], float(probability), int(row["label"]))
        )

    hit_counts = {k: 0 for k in top_ks}
    total_positive_groups = 0
    reciprocal_rank_sum = 0.0
    for rows in by_input.values():
        ranked = sorted(rows, key=lambda item: item[1], reverse=True)
        positives = {candidate for candidate, _, label in rows if label == 1}
        if not positives:
            continue
        total_positive_groups += len(positives)
        for positive in positives:
            rank = next(
                (index for index, (candidate, _, _) in enumerate(ranked, start=1) if candidate == positive),
                None,
            )
            if rank is None:
                continue
            reciprocal_rank_sum += 1 / rank
            for k in top_ks:
                if rank <= k:
                    hit_counts[k] += 1

    if total_positive_groups == 0:
        return {
            "positive_group_count": 0,
            "recall_at_k": {str(k): None for k in top_ks},
            "mrr": None,
        }
    return {
        "positive_group_count": int(total_positive_groups),
        "recall_at_k": {
            str(k): round(hit_counts[k] / total_positive_groups, 6)
            for k in top_ks
        },
        "mrr": round(reciprocal_rank_sum / total_positive_groups, 6),
    }


def plot_training_history(history_payload: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_payload.get("loss", []), label="train")
    axes[0].plot(history_payload.get("val_loss", []), label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_payload.get("accuracy", []), label="train accuracy")
    axes[1].plot(history_payload.get("val_accuracy", []), label="validation accuracy")
    axes[1].plot(history_payload.get("auc", []), label="train auc")
    axes[1].plot(history_payload.get("val_auc", []), label="validation auc")
    axes[1].set_title("Accuracy / AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: list[list[float]], output_path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = ["not_compatible", "compatible"]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            text = f"{value:.2f}" if isinstance(value, float) else str(int(value))
            ax.text(col_index, row_index, text, ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    resolved_raw_root = resolve_raw_root(args.raw_root)
    if resolved_raw_root is None or not resolved_raw_root.exists():
        raise FileNotFoundError(
            "Raw Polyvore files missing. Provide --raw-root pointing to the raw HF files cache."
        )

    model_path = args.output_dir / "outfit_model.keras"
    preprocessor_path = args.output_dir / "outfit_preprocessor.joblib"
    metadata_path = args.output_dir / "metadata.json"
    metrics_path = args.output_dir / "metrics.json"
    history_path = args.output_dir / "training_history.png"
    raw_cm_path = args.output_dir / "confusion_matrix_raw.png"
    norm_cm_path = args.output_dir / "confusion_matrix_normalized.png"

    pair_splits, diagnostics = build_outfit_training_splits(
        raw_root=resolved_raw_root,
        config_path=args.config,
        config_name=args.config_name,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pretraining_path = args.output_dir / "pretraining_diagnostics.json"
    pretraining_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print("Dry-run OK: outfit pair splits prepared.")
        print(json.dumps(diagnostics["pair_splits"], indent=2, ensure_ascii=False))
        return

    x_train, y_train = build_outfit_feature_frame(pair_splits["train"])
    x_valid, y_valid = build_outfit_feature_frame(pair_splits["valid"])
    x_test, y_test = build_outfit_feature_frame(pair_splits["test"])

    preprocessor = build_preprocessor()
    x_train_ready = preprocessor.fit_transform(x_train)
    x_valid_ready = preprocessor.transform(x_valid)
    x_test_ready = preprocessor.transform(x_test)

    cooccurrence_scores = build_cooccurrence_score_table(pair_splits["train"])
    cooccurrence_valid_proba = score_pairs_with_cooccurrence(pair_splits["valid"], cooccurrence_scores)
    cooccurrence_threshold, cooccurrence_validation_metrics = select_threshold(
        y_valid.to_numpy(),
        cooccurrence_valid_proba,
    )
    cooccurrence_test_proba = score_pairs_with_cooccurrence(pair_splits["test"], cooccurrence_scores)
    cooccurrence_test_metrics = _classification_metrics(
        y_test.to_numpy(),
        cooccurrence_test_proba,
        threshold=cooccurrence_threshold,
    )
    cooccurrence_test_metrics["ranking"] = ranking_metrics_for_products(
        pair_splits["test"],
        cooccurrence_test_proba,
    )

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    model = build_outfit_mlp(x_train_ready.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train_ready,
        y_train.to_numpy(),
        validation_data=(x_valid_ready, y_valid.to_numpy()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    history_payload = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    model_valid_proba = model.predict(x_valid_ready, verbose=0).reshape(-1)
    selected_threshold, model_validation_metrics = select_threshold(
        y_valid.to_numpy(),
        model_valid_proba,
    )
    model_test_proba = model.predict(x_test_ready, verbose=0).reshape(-1)
    model_test_metrics = _classification_metrics(
        y_test.to_numpy(),
        model_test_proba,
        threshold=selected_threshold,
    )
    model_test_metrics["ranking"] = ranking_metrics_for_products(pair_splits["test"], model_test_proba)

    beats_cooccurrence_baseline = (
        model_test_metrics["macro_f1"] > cooccurrence_test_metrics["macro_f1"]
        and model_test_metrics["balanced_accuracy"] > cooccurrence_test_metrics["balanced_accuracy"]
        and (model_test_metrics["roc_auc"] or 0.0) >= (cooccurrence_test_metrics["roc_auc"] or 0.0)
    )

    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    plot_training_history(history_payload, history_path)
    plot_confusion_matrix(
        model_test_metrics["confusion_matrix_raw"],
        raw_cm_path,
        title="Outfit V1 TensorFlow - final test confusion matrix",
    )
    plot_confusion_matrix(
        model_test_metrics["confusion_matrix_normalized"],
        norm_cm_path,
        title="Outfit V1 TensorFlow - final test normalized confusion matrix",
    )

    metadata = {
        "version": "outfit_v1",
        "target": "binary_outfit_compatibility",
        "positive_source": "mapped Polyvore same-outfit compatible role pairs",
        "negative_sampling": {
            "strategy": "hard_negatives_same_candidate_role_when_possible",
            "positive_to_negative_ratio": args.negative_ratio,
            "exclude_known_positive_pairs": True,
            "seed": args.seed,
        },
        "feature_columns": OUTFIT_PAIR_FEATURE_COLUMNS,
        "forbidden_direct_features": ["item_id", "outfit_id"],
        "config_name": args.config_name,
        "raw_root": str(resolved_raw_root),
        "selected_model_type": "keras_mlp_binary_classifier",
        "model_status": "experimental_only",
        "promotable_to_streamlit": False,
        "streamlit_promotion_decision": (
            "Do not copy to models/outfit_active without a separate promotion review. "
            "The cooccurrence baseline remains the fail-closed MVP path."
        ),
        "threshold_selection": {
            "source": "validation_only",
            "selected_threshold": selected_threshold,
        },
        "beats_cooccurrence_baseline_on_test": beats_cooccurrence_baseline,
        "dataset_diagnostics": diagnostics,
        "artifacts": {
            "model": model_path.name,
            "preprocessor": preprocessor_path.name,
            "metrics": metrics_path.name,
            "training_history": history_path.name,
            "confusion_matrix_raw": raw_cm_path.name,
            "confusion_matrix_normalized": norm_cm_path.name,
        },
    }
    metrics_payload = {
        "version": "outfit_v1",
        "model_status": "experimental_only",
        "promotable_to_streamlit": False,
        "validation_metrics": {
            "cooccurrence_baseline": cooccurrence_validation_metrics,
            "tensorflow_mlp": model_validation_metrics,
        },
        "test_metrics": {
            "cooccurrence_baseline": cooccurrence_test_metrics,
            "tensorflow_mlp": model_test_metrics,
        },
        "thresholds": {
            "cooccurrence_baseline": cooccurrence_threshold,
            "tensorflow_mlp": selected_threshold,
        },
        "beats_cooccurrence_baseline_on_test": beats_cooccurrence_baseline,
        "training_history": history_payload,
        "dataset_diagnostics": diagnostics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Outfit V1 final test ===")
    print("TensorFlow MLP:")
    print(json.dumps({key: model_test_metrics[key] for key in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]}, indent=2))
    print("Cooccurrence baseline:")
    print(json.dumps({key: cooccurrence_test_metrics[key] for key in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]}, indent=2))
    print(f"Beats cooccurrence baseline on test: {beats_cooccurrence_baseline}")
    print(f"Artifacts written to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train experimental TensorFlow Outfit Compatibility V1 on Polyvore raw metadata."
    )
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=OUTFIT_V1_CONFIG_PATH)
    parser.add_argument("--config-name", choices=["disjoint", "nondisjoint"], default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTFIT_V1_DIR)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
