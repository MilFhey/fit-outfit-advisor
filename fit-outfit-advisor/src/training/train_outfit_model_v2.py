from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.paths import OUTFIT_V1_CONFIG_PATH, OUTFIT_V2_DIR, REPORTS_DIR
from src.mappings.polyvore_mapping import load_outfit_v1_config, validate_outfit_v1_config
from src.preprocessing.outfit_v2_features import (
    ImageVisualFeatures,
    MOBILENET_V2_EMBEDDING_DIM,
    OUTFIT_V2_CATEGORICAL_FEATURES,
    OUTFIT_V2_NUMERIC_FEATURES,
    build_pair_embedding_block,
    build_pair_numeric_features,
    classify_color_family,
    extract_dominant_rgb,
    load_mobilenet_v2_embedding_model,
    preprocess_for_mobilenet_embedding,
    rgb_to_hsv01,
    validate_no_forbidden_v2_features,
)
from src.training.train_outfit_model_v1 import (
    DEFAULT_CONFIG,
    build_cooccurrence_score_table,
    build_labeled_pair_splits,
    configure_tensorflow_runtime,
    load_mapped_item_splits,
    ranking_metrics_for_products,
    resolve_raw_root,
    select_threshold,
    _classification_metrics,
)


DEFAULT_HF_DATASET = "mvasil/polyvore-outfits"
OUTFIT_V2_DATASET_AUDIT_PATH = REPORTS_DIR / "outfit_v2_dataset_audit.json"


def _hf_split_name(split_name: str) -> str:
    return "validation" if split_name == "valid" else split_name


def load_hf_split(config_name: str, split_name: str, *, hf_dataset_root: Path | None, dataset_id: str):
    from datasets import load_dataset, load_from_disk

    hf_split = _hf_split_name(split_name)
    if hf_dataset_root is not None and hf_dataset_root.exists():
        for candidate in [
            hf_dataset_root / config_name,
            hf_dataset_root,
        ]:
            if not candidate.exists():
                continue
            try:
                loaded = load_from_disk(str(candidate))
                if hasattr(loaded, "keys") and hf_split in loaded:
                    return loaded[hf_split]
                if hasattr(loaded, "keys") and split_name in loaded:
                    return loaded[split_name]
                if not hasattr(loaded, "keys"):
                    return loaded
            except Exception:
                continue
    return load_dataset(dataset_id, config_name, split=hf_split)


def load_image_lookup(
    item_ids: set[str],
    *,
    config_name: str,
    split_name: str,
    hf_dataset_root: Path | None,
    dataset_id: str = DEFAULT_HF_DATASET,
) -> dict[str, Any]:
    if not item_ids:
        return {}
    dataset = load_hf_split(
        config_name,
        split_name,
        hf_dataset_root=hf_dataset_root,
        dataset_id=dataset_id,
    )
    lookup: dict[str, Any] = {}
    remaining = set(str(item_id) for item_id in item_ids)
    for row in dataset:
        item_id = str(row.get("item_id"))
        if item_id not in remaining:
            continue
        lookup[item_id] = row.get("image")
        remaining.remove(item_id)
        if not remaining:
            break
    return lookup


def collect_pair_item_ids(pairs: pd.DataFrame) -> set[str]:
    if pairs.empty:
        return set()
    return set(pairs["input_item_id"].astype(str)).union(set(pairs["candidate_item_id"].astype(str)))


def extract_item_visual_features(
    item_ids: set[str],
    image_lookup: dict[str, Any],
    *,
    embedding_model: Any,
    embedding_batch_size: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features: dict[str, Any] = {}
    missing = []
    failed = []
    image_batches: list[tuple[str, np.ndarray, tuple[int, int, int], tuple[float, float, float], str]] = []

    for item_id in sorted(item_ids):
        image = image_lookup.get(str(item_id))
        if image is None:
            missing.append(str(item_id))
            continue
        try:
            dominant_rgb = extract_dominant_rgb(image)
            hsv = rgb_to_hsv01(dominant_rgb)
            color_family = classify_color_family(dominant_rgb)
            image_batches.append(
                (
                    str(item_id),
                    preprocess_for_mobilenet_embedding(image)[0],
                    dominant_rgb,
                    hsv,
                    color_family,
                )
            )
        except Exception:
            failed.append(str(item_id))

    for start in range(0, len(image_batches), embedding_batch_size):
        batch_rows = image_batches[start : start + embedding_batch_size]
        batch = np.asarray([row[1] for row in batch_rows], dtype="float32")
        try:
            embeddings = embedding_model.predict(
                batch,
                batch_size=embedding_batch_size,
                verbose=0,
            )
        except Exception:
            failed.extend(row[0] for row in batch_rows)
            continue
        for (item_id, _, dominant_rgb, hsv, color_family), embedding in zip(batch_rows, embeddings):
            features[item_id] = ImageVisualFeatures(
                embedding=np.asarray(embedding, dtype="float32"),
                dominant_rgb=dominant_rgb,
                hsv=hsv,
                color_family=color_family,
            )

    diagnostics = {
        "requested_item_count": len(item_ids),
        "feature_item_count": len(features),
        "missing_image_count": len(missing),
        "failed_feature_count": len(failed),
        "missing_image_examples": missing[:10],
        "failed_feature_examples": failed[:10],
        "embedding_batch_size": int(embedding_batch_size),
    }
    return features, diagnostics


def save_visual_feature_cache(cache_path: Path, features: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_item_ids = sorted(features)
    np.savez_compressed(
        cache_path,
        item_ids=np.asarray(ordered_item_ids),
        embeddings=np.asarray([features[item_id].embedding for item_id in ordered_item_ids], dtype="float32"),
        dominant_rgbs=np.asarray([features[item_id].dominant_rgb for item_id in ordered_item_ids], dtype="int16"),
        hsvs=np.asarray([features[item_id].hsv for item_id in ordered_item_ids], dtype="float32"),
        color_families=np.asarray([features[item_id].color_family for item_id in ordered_item_ids]),
    )


def load_visual_feature_cache(cache_path: Path, item_ids: set[str]) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        cached_item_ids = [str(item_id) for item_id in payload["item_ids"]]
        requested_item_ids = set(str(item_id) for item_id in item_ids)
        if not requested_item_ids.issubset(set(cached_item_ids)):
            return None
        from src.preprocessing.outfit_v2_features import ImageVisualFeatures

        features = {}
        for index, item_id in enumerate(cached_item_ids):
            if item_id not in requested_item_ids:
                continue
            features[item_id] = ImageVisualFeatures(
                embedding=np.asarray(payload["embeddings"][index], dtype="float32"),
                dominant_rgb=tuple(int(value) for value in payload["dominant_rgbs"][index]),
                hsv=tuple(float(value) for value in payload["hsvs"][index]),
                color_family=str(payload["color_families"][index]),
            )
    except Exception:
        return None
    return features


def cooccurrence_score_for_pair(row: dict[str, Any], score_table: dict[tuple[str, str], float]) -> float:
    return float(score_table.get((row["input_product_type"], row["candidate_product_type"]), 0.0))


def build_v2_pair_matrices(
    pairs: pd.DataFrame,
    item_visual_features: dict[str, Any],
    *,
    cooccurrence_scores: dict[tuple[str, str], float],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    categorical_rows = []
    numeric_rows = []
    embedding_rows = []
    labels = []
    usable_rows = []
    skipped_missing_features = 0

    for row in pairs.to_dict("records"):
        input_features = item_visual_features.get(str(row["input_item_id"]))
        candidate_features = item_visual_features.get(str(row["candidate_item_id"]))
        if input_features is None or candidate_features is None:
            skipped_missing_features += 1
            continue

        cooccurrence_score = cooccurrence_score_for_pair(row, cooccurrence_scores)
        categorical_rows.append(
            {
                "input_product_type": row["input_product_type"],
                "input_canonical_category": row["input_canonical_category"],
                "input_outfit_role": row["input_outfit_role"],
                "input_color_family": input_features.color_family,
                "candidate_product_type": row["candidate_product_type"],
                "candidate_canonical_category": row["candidate_canonical_category"],
                "candidate_outfit_role": row["candidate_outfit_role"],
                "candidate_color_family": candidate_features.color_family,
            }
        )
        numeric_rows.append(
            build_pair_numeric_features(
                input_features,
                candidate_features,
                cooccurrence_score=cooccurrence_score,
            )
        )
        embedding_rows.append(build_pair_embedding_block(input_features, candidate_features))
        labels.append(int(row["label"]))
        usable_rows.append(row)

    validate_no_forbidden_v2_features(OUTFIT_V2_CATEGORICAL_FEATURES + OUTFIT_V2_NUMERIC_FEATURES)
    categorical = pd.DataFrame(categorical_rows, columns=OUTFIT_V2_CATEGORICAL_FEATURES)
    numeric = np.asarray(numeric_rows, dtype="float32")
    embeddings = np.asarray(embedding_rows, dtype="float32")
    target = np.asarray(labels, dtype="float32")
    usable_pairs = pd.DataFrame(usable_rows)
    diagnostics = {
        "input_pair_count": int(len(pairs)),
        "usable_pair_count": int(len(target)),
        "skipped_missing_visual_feature_count": int(skipped_missing_features),
        "label_distribution": {
            str(label): int(count)
            for label, count in pd.Series(target.astype(int)).value_counts().sort_index().items()
        },
    }
    return categorical, numeric, embeddings, target, usable_pairs, diagnostics


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
                OUTFIT_V2_CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline(steps=[("scaler", StandardScaler())]),
                OUTFIT_V2_NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )


def combine_feature_blocks(
    categorical: pd.DataFrame,
    numeric: np.ndarray,
    embeddings: np.ndarray,
    *,
    preprocessor: ColumnTransformer,
    fit: bool,
) -> np.ndarray:
    numeric_frame = pd.DataFrame(numeric, columns=OUTFIT_V2_NUMERIC_FEATURES)
    structured = pd.concat([categorical.reset_index(drop=True), numeric_frame], axis=1)
    transformed = preprocessor.fit_transform(structured) if fit else preprocessor.transform(structured)
    return np.hstack([embeddings, transformed.astype("float32", copy=False)]).astype("float32", copy=False)


def build_outfit_v2_model(input_dim: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="outfit_v2_image_color_pair_compatibility",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def save_pair_dataset(
    output_path: Path,
    categorical: pd.DataFrame,
    numeric: np.ndarray,
    target: np.ndarray,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [
            categorical.reset_index(drop=True),
            pd.DataFrame(numeric, columns=OUTFIT_V2_NUMERIC_FEATURES),
            pd.Series(target.astype(int), name="label"),
        ],
        axis=1,
    )
    try:
        frame.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        fallback_path = output_path.with_suffix(".csv")
        frame.to_csv(fallback_path, index=False)
        return fallback_path


def build_product_type_prototypes(
    item_splits: dict[str, pd.DataFrame],
    item_visual_features: dict[str, Any],
) -> dict[str, Any]:
    rows = item_splits["train"].to_dict("records")
    grouped: dict[str, list[tuple[dict[str, Any], Any]]] = defaultdict(list)
    for row in rows:
        features = item_visual_features.get(str(row["item_id"]))
        if features is not None:
            grouped[row["product_type_v0"]].append((row, features))

    prototypes: dict[str, Any] = {}
    for product_type, values in grouped.items():
        embeddings = np.asarray([features.embedding for _, features in values], dtype="float32")
        rgbs = np.asarray([features.dominant_rgb for _, features in values], dtype="float32")
        color_counts = pd.Series([features.color_family for _, features in values]).value_counts()
        first_row = values[0][0]
        mean_rgb = np.clip(np.rint(rgbs.mean(axis=0)), 0, 255).astype(int)
        prototypes[product_type] = {
            "product_type_v0": product_type,
            "canonical_category": first_row["canonical_category"],
            "outfit_role": first_row["outfit_role"],
            "mean_embedding": embeddings.mean(axis=0).round(6).tolist(),
            "mean_rgb": [int(value) for value in mean_rgb],
            "color_family": str(color_counts.index[0]),
            "support": int(len(values)),
        }
    return prototypes


def plot_training_history(history_payload: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_payload.get("loss", []), label="train")
    axes[0].plot(history_payload.get("val_loss", []), label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history_payload.get("accuracy", []), label="train accuracy")
    axes[1].plot(history_payload.get("val_accuracy", []), label="validation accuracy")
    axes[1].plot(history_payload.get("auc", []), label="train auc")
    axes[1].plot(history_payload.get("val_auc", []), label="validation auc")
    axes[1].set_title("Accuracy / AUC")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix_raw(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true.astype(int), predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred negative", "pred positive"])
    ax.set_yticks([0, 1], labels=["true negative", "true positive"])
    ax.set_title("Outfit V2 confusion matrix")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, str(matrix[row_index, col_index]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_ranking_examples(
    pairs: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    output_path: Path,
    top_k: int = 12,
) -> None:
    import matplotlib.pyplot as plt

    if pairs.empty:
        output_path.write_text("No ranking examples available.", encoding="utf-8")
        return
    frame = pairs.copy()
    frame["score"] = probabilities
    frame["pair_label"] = (
        frame["input_product_type"].astype(str)
        + " -> "
        + frame["candidate_product_type"].astype(str)
        + " (y="
        + frame["label"].astype(str)
        + ")"
    )
    examples = frame.sort_values("score", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(examples["pair_label"], examples["score"], color="#2f6f9f")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted compatibility")
    ax.set_title("Top Outfit V2 ranking examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    resolved_raw_root = resolve_raw_root(args.raw_root)
    if resolved_raw_root is None or not resolved_raw_root.exists():
        raise FileNotFoundError("Raw Polyvore files missing. Provide --raw-root.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_outfit_v1_config(args.config)
    validate_outfit_v1_config(config, require_ready=True)
    item_splits, item_diagnostics = load_mapped_item_splits(
        resolved_raw_root,
        config=config,
        config_name=args.config_name,
    )
    pair_splits, pair_diagnostics = build_labeled_pair_splits(
        item_splits,
        config=config,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    if args.max_pairs_per_split:
        pair_splits = {
            split: frame.sample(
                n=min(len(frame), args.max_pairs_per_split),
                random_state=args.seed,
            ).reset_index(drop=True)
            for split, frame in pair_splits.items()
        }

    import tensorflow as tf

    tensorflow_device_summary = configure_tensorflow_runtime(tf, require_gpu=args.require_gpu)
    embedding_model = load_mobilenet_v2_embedding_model()
    cooccurrence_scores = build_cooccurrence_score_table(pair_splits["train"])

    split_features: dict[str, dict[str, Any]] = {}
    visual_diagnostics: dict[str, Any] = {}
    all_train_visual_features: dict[str, Any] = {}
    for split_name, pairs in pair_splits.items():
        item_ids = collect_pair_item_ids(pairs)
        cache_path = args.output_dir / f"{split_name}_item_visual_features.npz"
        cached_features = load_visual_feature_cache(cache_path, item_ids)
        if cached_features is not None:
            item_features = cached_features
            feature_diag = {
                "requested_item_count": len(item_ids),
                "feature_item_count": len(item_features),
                "missing_image_count": 0,
                "failed_feature_count": 0,
                "cache_hit": True,
                "cache_path": str(cache_path),
            }
        else:
            image_lookup = load_image_lookup(
                item_ids,
                config_name=args.config_name,
                split_name=split_name,
                hf_dataset_root=args.hf_dataset_root,
                dataset_id=args.hf_dataset_id,
            )
            item_features, feature_diag = extract_item_visual_features(
                item_ids,
                image_lookup,
                embedding_model=embedding_model,
                embedding_batch_size=args.embedding_batch_size,
            )
            save_visual_feature_cache(cache_path, item_features)
            feature_diag["cache_hit"] = False
            feature_diag["cache_path"] = str(cache_path)
        if split_name == "train":
            all_train_visual_features.update(item_features)
        categorical, numeric, embeddings, target, usable_pairs, pair_diag = build_v2_pair_matrices(
            pairs,
            item_features,
            cooccurrence_scores=cooccurrence_scores,
        )
        split_features[split_name] = {
            "categorical": categorical,
            "numeric": numeric,
            "embeddings": embeddings,
            "target": target,
            "pairs": usable_pairs,
        }
        visual_diagnostics[split_name] = {
            "item_features": feature_diag,
            "pair_features": pair_diag,
        }
        actual_dataset_path = save_pair_dataset(
            args.output_dir / f"{split_name}_pairs.parquet",
            categorical,
            numeric,
            target,
        )
        visual_diagnostics[split_name]["pair_dataset_path"] = str(actual_dataset_path)

    preprocessor = build_preprocessor()
    x_train = combine_feature_blocks(
        split_features["train"]["categorical"],
        split_features["train"]["numeric"],
        split_features["train"]["embeddings"],
        preprocessor=preprocessor,
        fit=True,
    )
    x_valid = combine_feature_blocks(
        split_features["valid"]["categorical"],
        split_features["valid"]["numeric"],
        split_features["valid"]["embeddings"],
        preprocessor=preprocessor,
        fit=False,
    )
    x_test = combine_feature_blocks(
        split_features["test"]["categorical"],
        split_features["test"]["numeric"],
        split_features["test"]["embeddings"],
        preprocessor=preprocessor,
        fit=False,
    )
    y_train = split_features["train"]["target"]
    y_valid = split_features["valid"]["target"]
    y_test = split_features["test"]["target"]

    tf.keras.utils.set_random_seed(args.seed)
    model = build_outfit_v2_model(x_train.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    history_payload = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    valid_proba = model.predict(x_valid, verbose=0).reshape(-1)
    selected_threshold, validation_metrics = select_threshold(y_valid, valid_proba)
    test_proba = model.predict(x_test, verbose=0).reshape(-1)
    test_metrics = _classification_metrics(y_test, test_proba, threshold=selected_threshold)
    test_metrics["ranking"] = ranking_metrics_for_products(
        split_features["test"]["pairs"].assign(label=y_test.astype(int)),
        test_proba,
    )

    model_path = args.output_dir / "outfit_model.keras"
    preprocessor_path = args.output_dir / "outfit_preprocessor.joblib"
    metadata_path = args.output_dir / "metadata.json"
    metrics_path = args.output_dir / "metrics.json"
    prototypes_path = args.output_dir / "product_type_prototypes.json"
    history_path = args.output_dir / "training_history.png"
    confusion_matrix_path = args.output_dir / "confusion_matrix_raw.png"
    ranking_examples_path = args.output_dir / "ranking_examples.png"

    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    prototypes = build_product_type_prototypes(item_splits, all_train_visual_features)
    prototypes_path.write_text(
        json.dumps({"version": "outfit_v2", "product_type_prototypes": prototypes}, indent=2),
        encoding="utf-8",
    )
    plot_training_history(history_payload, history_path)
    plot_confusion_matrix_raw(
        y_test,
        test_proba,
        threshold=selected_threshold,
        output_path=confusion_matrix_path,
    )
    plot_ranking_examples(
        split_features["test"]["pairs"].assign(label=y_test.astype(int)),
        test_proba,
        output_path=ranking_examples_path,
    )

    baseline_report = None
    baseline_report_path = REPORTS_DIR / "polyvore_v0_cooccurrence_baseline.json"
    if baseline_report_path.exists():
        baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8-sig"))
    baseline_test = (baseline_report or {}).get("leakage_filtered_evaluation", {}).get("test", {})
    beats_baseline = bool(
        test_metrics.get("roc_auc", 0.0) >= args.min_test_roc_auc
        and test_metrics.get("ranking", {}).get("recall_at_k", {}).get("3", 0.0)
        >= float(baseline_test.get("recall_at_k", {}).get("3", 1.0))
    )

    metadata = {
        "version": "outfit_v2",
        "model_status": "experimental_only",
        "promotable_to_streamlit": False,
        "target": "pairwise_outfit_compatibility",
        "uses_image_embeddings": True,
        "uses_color_features": True,
        "uses_cooccurrence_feature": True,
        "forbidden_direct_features": ["item_id", "outfit_id", "set_id"],
        "feature_policy": {
            "categorical_features": OUTFIT_V2_CATEGORICAL_FEATURES,
            "numeric_features": OUTFIT_V2_NUMERIC_FEATURES,
            "embedding_dim": MOBILENET_V2_EMBEDDING_DIM,
        },
        "threshold_selection": {
            "source": "validation_only",
            "selected_threshold": selected_threshold,
        },
        "promotion_decision": "experimental_only_until_metrics_review",
        "beats_cooccurrence_baseline_on_test": beats_baseline,
        "tensorflow_device_summary": tensorflow_device_summary,
        "dataset_diagnostics": {
            "item_splits": item_diagnostics,
            "pair_splits": pair_diagnostics,
            "visual_features": visual_diagnostics,
            "max_pairs_per_split": args.max_pairs_per_split,
            "embedding_batch_size": args.embedding_batch_size,
        },
        "artifact_names": [
            model_path.name,
            preprocessor_path.name,
            metadata_path.name,
            metrics_path.name,
            prototypes_path.name,
            history_path.name,
            confusion_matrix_path.name,
            ranking_examples_path.name,
            "train_item_visual_features.npz",
            "valid_item_visual_features.npz",
            "test_item_visual_features.npz",
        ],
    }
    metrics_payload = {
        "version": "outfit_v2",
        "model_status": "experimental_only",
        "promotable_to_streamlit": False,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "threshold": selected_threshold,
        "training_history": history_payload,
        "baseline_reference": baseline_test,
        "beats_cooccurrence_baseline_on_test": beats_baseline,
        "dataset_diagnostics": metadata["dataset_diagnostics"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTFIT_V2_DATASET_AUDIT_PATH.write_text(
        json.dumps(metadata["dataset_diagnostics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: test_metrics[key] for key in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]}, indent=2))
    print(f"Artifacts written to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Outfit V2 multimodal image/color compatibility model.")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--hf-dataset-root", type=Path, default=None)
    parser.add_argument("--hf-dataset-id", default=DEFAULT_HF_DATASET)
    parser.add_argument("--config", type=Path, default=OUTFIT_V1_CONFIG_PATH)
    parser.add_argument("--config-name", choices=["disjoint", "nondisjoint"], default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTFIT_V2_DIR)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-split", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-test-roc-auc", type=float, default=0.60)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
