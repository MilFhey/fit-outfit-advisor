import argparse
import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from src.config.paths import (
    FASHION_V1_CLASSES_PATH,
    FASHION_V1_DIR,
    RAW_DATA_DIR,
)
from src.mappings.fashion_v1_mapping import (
    build_article_type_to_product_type_mapping,
    load_fashion_v1_class_config,
    map_product_type_to_canonical_category,
    validate_fashion_v1_class_config,
)


FASHION_DATASET_DIR = RAW_DATA_DIR / "fashion-product-images-small"
FASHION_METADATA_FILENAME = "styles.csv"
FASHION_IMAGES_DIRNAME = "images"
FASHION_ARTIFACT_NAMES = (
    "fashion_model.keras",
    "label_encoder.joblib",
    "metadata.json",
    "metrics.json",
    "confusion_matrix_raw.png",
    "confusion_matrix_normalized.png",
    "training_history.png",
    "sample_predictions.png",
)

SIMPLE_CNN_EXPERIMENT = "simple_cnn"
MOBILENET_V2_EXPERIMENT = "mobilenet_v2"
SUPPORTED_EXPERIMENTS = (SIMPLE_CNN_EXPERIMENT, MOBILENET_V2_EXPERIMENT)


def resolve_fashion_dataset_paths(
    dataset_root: Path = FASHION_DATASET_DIR,
    metadata_csv: Path | None = None,
    image_dir: Path | None = None,
) -> tuple[Path, Path]:
    metadata_path = metadata_csv or dataset_root / FASHION_METADATA_FILENAME
    images_path = image_dir or dataset_root / FASHION_IMAGES_DIRNAME

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata Fashion absent : {metadata_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Dossier images Fashion absent : {images_path}")

    return metadata_path, images_path


def verify_image_is_readable(image_path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def prepare_fashion_v1_training_frame(
    metadata_csv: Path,
    image_dir: Path,
    class_config: dict[str, Any],
) -> tuple["pd.DataFrame", dict[str, Any]]:
    import pandas as pd

    validate_fashion_v1_class_config(class_config, require_ready=True)
    article_type_mapping = build_article_type_to_product_type_mapping(class_config)
    minimum_count = int(class_config["minimum_readable_images_per_class"])

    frame = pd.read_csv(metadata_csv, on_bad_lines="skip")
    required_columns = {"id", "articleType"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Colonnes Fashion absentes: {sorted(missing_columns)}")

    prepared = frame.copy()
    prepared["articleType"] = prepared["articleType"].astype(str).str.strip()
    prepared["product_type_v0"] = prepared["articleType"].map(article_type_mapping)
    prepared = prepared.dropna(subset=["product_type_v0"]).copy()
    prepared["canonical_category"] = prepared["product_type_v0"].map(
        lambda product_type: map_product_type_to_canonical_category(product_type, class_config)
    )
    prepared["image_path"] = prepared["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    prepared["image_path"] = prepared["image_path"].map(lambda image_id: image_dir / f"{image_id}.jpg")
    prepared["image_present"] = prepared["image_path"].map(Path.exists)
    prepared = prepared[prepared["image_present"]].copy()
    prepared["image_readable"] = prepared["image_path"].map(verify_image_is_readable)
    prepared = prepared[prepared["image_readable"]].copy()

    class_counts = prepared["product_type_v0"].value_counts().sort_index().to_dict()
    below_threshold = {
        class_name: int(count)
        for class_name, count in class_counts.items()
        if int(count) < minimum_count
    }
    selected_classes = sorted(class_config["product_type_mapping"].keys())
    missing_classes = [
        class_name for class_name in selected_classes if class_counts.get(class_name, 0) < minimum_count
    ]
    if below_threshold or missing_classes:
        raise ValueError(
            "Classes sous le seuil minimal apres verification images lisibles: "
            f"{missing_classes or below_threshold}."
        )

    diagnostics = {
        "metadata_row_count": int(len(frame)),
        "mapped_row_count": int(len(prepared)),
        "selected_product_types": selected_classes,
        "class_counts_after_readability_check": {
            class_name: int(count) for class_name, count in class_counts.items()
        },
        "minimum_readable_images_per_class": minimum_count,
        "artifact_names": list(FASHION_ARTIFACT_NAMES),
    }
    return prepared, diagnostics


def build_simple_cnn_model(input_shape: tuple[int, int, int], num_classes: int):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="fashion_v1_simple_cnn",
    )


def build_mobilenet_v2_model(input_shape: tuple[int, int, int], num_classes: int):
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="fashion_v1_mobilenet_v2")


def split_fashion_frame(
    prepared_frame,
    *,
    seed: int = 42,
    test_size: float = 0.15,
    validation_size: float = 0.15,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    from sklearn.model_selection import train_test_split

    train_validation_frame, test_frame = train_test_split(
        prepared_frame,
        test_size=test_size,
        random_state=seed,
        stratify=prepared_frame["product_type_v0"],
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train_frame, validation_frame = train_test_split(
        train_validation_frame,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=train_validation_frame["product_type_v0"],
    )
    return train_frame.copy(), validation_frame.copy(), test_frame.copy()


def encode_labels(train_frame, validation_frame, test_frame):
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_frame["product_type_v0"])
    validation_labels = label_encoder.transform(validation_frame["product_type_v0"])
    test_labels = label_encoder.transform(test_frame["product_type_v0"])
    return label_encoder, train_labels, validation_labels, test_labels


def build_image_dataset(
    image_paths: list[str],
    labels: list[int] | None,
    *,
    image_size: int,
    batch_size: int,
    architecture: str,
    shuffle: bool = False,
    seed: int = 42,
):
    import tensorflow as tf

    path_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    if labels is None:
        dataset = path_dataset
    else:
        label_dataset = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((path_dataset, label_dataset))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    def load_and_preprocess(path):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32)
        if architecture == SIMPLE_CNN_EXPERIMENT:
            return image / 255.0
        if architecture == MOBILENET_V2_EXPERIMENT:
            return tf.keras.applications.mobilenet_v2.preprocess_input(image)
        raise ValueError(f"Architecture image inconnue : {architecture}")

    if labels is None:
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(
            lambda path, label: (load_and_preprocess(path), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(architecture: str, input_shape: tuple[int, int, int], num_classes: int):
    if architecture == SIMPLE_CNN_EXPERIMENT:
        return build_simple_cnn_model(input_shape, num_classes)
    if architecture == MOBILENET_V2_EXPERIMENT:
        return build_mobilenet_v2_model(input_shape, num_classes)
    raise ValueError(f"Architecture image inconnue : {architecture}")


def predict_probabilities(model, dataset):
    probabilities = model.predict(dataset, verbose=0)
    return probabilities


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    *,
    labels: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    import numpy as np

    raw_cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = raw_cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(
        raw_cm,
        row_sums,
        out=np.zeros_like(raw_cm, dtype=float),
        where=row_sums != 0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_raw": raw_cm.tolist(),
        "confusion_matrix_normalized": normalized_cm.round(6).tolist(),
    }


def evaluate_model(model, dataset, y_true: list[int], class_names: list[str]) -> dict[str, Any]:
    import numpy as np

    probabilities = predict_probabilities(model, dataset)
    predictions = np.argmax(probabilities, axis=1)
    metrics = compute_classification_metrics(
        list(y_true),
        predictions.tolist(),
        labels=list(range(len(class_names))),
        class_names=class_names,
    )
    metrics["mean_confidence"] = float(np.max(probabilities, axis=1).mean())
    return metrics


def select_experiment(validation_metrics: dict[str, dict[str, Any]]) -> tuple[str, str]:
    ordered = sorted(
        validation_metrics.items(),
        key=lambda item: (
            item[1]["macro_f1"],
            item[1]["balanced_accuracy"],
            item[1]["accuracy"],
        ),
        reverse=True,
    )
    selected_name, selected_metrics = ordered[0]
    reason = (
        f"{selected_name} selected on validation only: "
        f"macro_f1={selected_metrics['macro_f1']:.4f}, "
        f"balanced_accuracy={selected_metrics['balanced_accuracy']:.4f}, "
        f"accuracy={selected_metrics['accuracy']:.4f}."
    )
    return selected_name, reason


def count_classes(frame) -> dict[str, int]:
    return {
        str(class_name): int(count)
        for class_name, count in frame["product_type_v0"].value_counts().sort_index().items()
    }


def plot_training_history(history_payload: dict[str, list[float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_payload.get("loss", []), label="train")
    axes[0].plot(history_payload.get("val_loss", []), label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_payload.get("accuracy", []), label="train")
    axes[1].plot(history_payload.get("val_accuracy", []), label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: list[list[float]],
    class_names: list[str],
    output_path: Path,
    *,
    title: str,
    normalized: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.array(matrix)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    threshold = values.max() / 2 if values.size else 0
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            text = f"{value:.2f}" if normalized else str(int(value))
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_sample_predictions(
    model,
    sample_frame,
    label_encoder,
    *,
    architecture: str,
    image_size: int,
    output_path: Path,
    seed: int = 42,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    samples = sample_frame.sample(n=min(12, len(sample_frame)), random_state=seed)
    paths = [str(path) for path in samples["image_path"]]
    dataset = build_image_dataset(
        paths,
        labels=None,
        image_size=image_size,
        batch_size=32,
        architecture=architecture,
        shuffle=False,
        seed=seed,
    )
    probabilities = predict_probabilities(model, dataset)
    predicted_indexes = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_indexes)

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for ax, (_, row), predicted_label, confidence in zip(
        axes.flatten(),
        samples.iterrows(),
        predicted_labels,
        confidences,
    ):
        image = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(
            f"true: {row['product_type_v0']}\npred: {predicted_label} ({confidence:.0%})",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes.flatten()[len(samples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def train_single_experiment(
    architecture: str,
    train_frame,
    validation_frame,
    *,
    train_labels,
    validation_labels,
    image_size: int,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    num_classes: int,
    class_names: list[str],
) -> tuple[Any, dict[str, list[float]], dict[str, Any]]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    train_dataset = build_image_dataset(
        [str(path) for path in train_frame["image_path"]],
        train_labels.tolist(),
        image_size=image_size,
        batch_size=batch_size,
        architecture=architecture,
        shuffle=True,
        seed=seed,
    )
    validation_dataset = build_image_dataset(
        [str(path) for path in validation_frame["image_path"]],
        validation_labels.tolist(),
        image_size=image_size,
        batch_size=batch_size,
        architecture=architecture,
        shuffle=False,
        seed=seed,
    )
    model = build_model(architecture, (image_size, image_size, 3), num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    history_payload = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    validation_metrics = evaluate_model(
        model,
        validation_dataset,
        validation_labels.tolist(),
        class_names,
    )
    return model, history_payload, validation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare/train Fashion Product Images Small CNN V1."
    )
    parser.add_argument("--dataset-root", type=Path, default=FASHION_DATASET_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--class-config", type=Path, default=FASHION_V1_CLASSES_PATH)
    parser.add_argument("--output-dir", type=Path, default=FASHION_V1_DIR)
    parser.add_argument(
        "--architecture",
        choices=["simple_cnn", "mobilenet_v2", "both"],
        default="both",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_csv, image_dir = resolve_fashion_dataset_paths(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
    )
    class_config = load_fashion_v1_class_config(args.class_config)
    prepared_frame, diagnostics = prepare_fashion_v1_training_frame(
        metadata_csv,
        image_dir,
        class_config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.output_dir / "pretraining_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(
            "Dry-run OK: "
            f"{len(prepared_frame)} images lisibles, "
            f"{prepared_frame['product_type_v0'].nunique()} classes."
        )
        print(f"Diagnostics ecrits : {diagnostics_path}")
        return

    train_frame, validation_frame, test_frame = split_fashion_frame(
        prepared_frame,
        seed=args.seed,
    )
    label_encoder, train_labels, validation_labels, test_labels = encode_labels(
        train_frame,
        validation_frame,
        test_frame,
    )
    class_names = [str(class_name) for class_name in label_encoder.classes_]
    experiments = (
        list(SUPPORTED_EXPERIMENTS)
        if args.architecture == "both"
        else [args.architecture]
    )

    trained_models: dict[str, Any] = {}
    training_histories: dict[str, dict[str, list[float]]] = {}
    validation_metrics: dict[str, dict[str, Any]] = {}

    for experiment in experiments:
        print(f"\n=== Training Fashion V1 experiment: {experiment} ===")
        model, history_payload, metrics_payload = train_single_experiment(
            experiment,
            train_frame,
            validation_frame,
            train_labels=train_labels,
            validation_labels=validation_labels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            num_classes=len(class_names),
            class_names=class_names,
        )
        trained_models[experiment] = model
        training_histories[experiment] = history_payload
        validation_metrics[experiment] = metrics_payload
        print(
            f"{experiment} validation: "
            f"macro_f1={metrics_payload['macro_f1']:.4f}, "
            f"balanced_accuracy={metrics_payload['balanced_accuracy']:.4f}, "
            f"accuracy={metrics_payload['accuracy']:.4f}"
        )

    selected_experiment, reason_for_selection = select_experiment(validation_metrics)
    selected_model = trained_models[selected_experiment]
    test_dataset = build_image_dataset(
        [str(path) for path in test_frame["image_path"]],
        test_labels.tolist(),
        image_size=args.image_size,
        batch_size=args.batch_size,
        architecture=selected_experiment,
        shuffle=False,
        seed=args.seed,
    )
    test_metrics = evaluate_model(
        selected_model,
        test_dataset,
        test_labels.tolist(),
        class_names,
    )

    model_path = args.output_dir / "fashion_model.keras"
    label_encoder_path = args.output_dir / "label_encoder.joblib"
    metadata_path = args.output_dir / "metadata.json"
    metrics_path = args.output_dir / "metrics.json"
    raw_cm_path = args.output_dir / "confusion_matrix_raw.png"
    norm_cm_path = args.output_dir / "confusion_matrix_normalized.png"
    history_path = args.output_dir / "training_history.png"
    sample_predictions_path = args.output_dir / "sample_predictions.png"

    selected_model.save(model_path)
    joblib.dump(label_encoder, label_encoder_path)
    plot_confusion_matrix(
        test_metrics["confusion_matrix_raw"],
        class_names,
        raw_cm_path,
        title=f"{selected_experiment} - final test confusion matrix",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix_normalized"],
        class_names,
        norm_cm_path,
        title=f"{selected_experiment} - final test normalized confusion matrix",
        normalized=True,
    )
    plot_training_history(training_histories[selected_experiment], history_path)
    plot_sample_predictions(
        selected_model,
        test_frame,
        label_encoder,
        architecture=selected_experiment,
        image_size=args.image_size,
        output_path=sample_predictions_path,
        seed=args.seed,
    )

    split_counts = {
        "train": int(len(train_frame)),
        "validation": int(len(validation_frame)),
        "test": int(len(test_frame)),
        "total": int(len(prepared_frame)),
    }
    metadata = {
        "version": "fashion_v1",
        "target": "product_type_v0",
        "source_column": "articleType",
        "canonical_category_source": "deterministic_mapping_after_prediction",
        "selected_experiment": selected_experiment,
        "selected_model_type": "keras_cnn",
        "candidate_experiments": experiments,
        "model_status": "experimental_only",
        "promotable_to_streamlit": False,
        "reason_for_selection": reason_for_selection,
        "class_labels": class_names,
        "image_size": args.image_size,
        "preprocessing": (
            "rescale_1_over_255"
            if selected_experiment == SIMPLE_CNN_EXPERIMENT
            else "mobilenet_v2_preprocess_input"
        ),
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": args.seed},
        "dataset_row_counts": split_counts,
        "class_distribution_train": count_classes(train_frame),
        "class_distribution_validation": count_classes(validation_frame),
        "class_distribution_test": count_classes(test_frame),
        "class_config": class_config,
        "artifact_names": list(FASHION_ARTIFACT_NAMES),
    }
    metrics = {
        "validation_metrics": validation_metrics,
        "test_metrics": {"selected_experiment": test_metrics},
        "selected_experiment": selected_experiment,
        "reason_for_selection": reason_for_selection,
        "class_labels": class_names,
        "dataset_row_counts": split_counts,
        "class_distribution_train": metadata["class_distribution_train"],
        "class_distribution_validation": metadata["class_distribution_validation"],
        "class_distribution_test": metadata["class_distribution_test"],
        "training_history": training_histories,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print("\n=== Fashion V1 final selection ===")
    print(reason_for_selection)
    print(
        "Final test selected: "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}, "
        f"accuracy={test_metrics['accuracy']:.4f}"
    )
    print(f"Artefacts ecrits dans : {args.output_dir}")


if __name__ == "__main__":
    main()
