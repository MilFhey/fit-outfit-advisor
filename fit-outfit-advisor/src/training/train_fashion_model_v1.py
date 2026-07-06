import argparse
import json
from pathlib import Path
from typing import Any

from src.config.paths import (
    FASHION_V1_CLASSES_PATH,
    FASHION_V1_DIR,
    RAW_DATA_DIR,
)
from src.mappings.fashion_v1_mapping import (
    build_article_type_to_canonical_mapping,
    load_fashion_v1_class_config,
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
    article_type_mapping = build_article_type_to_canonical_mapping(class_config)
    minimum_count = int(class_config["minimum_readable_images_per_class"])

    frame = pd.read_csv(metadata_csv, on_bad_lines="skip")
    required_columns = {"id", "articleType"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Colonnes Fashion absentes: {sorted(missing_columns)}")

    prepared = frame.copy()
    prepared["articleType"] = prepared["articleType"].astype(str).str.strip()
    prepared["canonical_category"] = prepared["articleType"].map(article_type_mapping)
    prepared = prepared.dropna(subset=["canonical_category"]).copy()
    prepared["image_path"] = prepared["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    prepared["image_path"] = prepared["image_path"].map(lambda image_id: image_dir / f"{image_id}.jpg")
    prepared["image_present"] = prepared["image_path"].map(Path.exists)
    prepared = prepared[prepared["image_present"]].copy()
    prepared["image_readable"] = prepared["image_path"].map(verify_image_is_readable)
    prepared = prepared[prepared["image_readable"]].copy()

    class_counts = prepared["canonical_category"].value_counts().sort_index().to_dict()
    below_threshold = {
        class_name: int(count)
        for class_name, count in class_counts.items()
        if int(count) < minimum_count
    }
    selected_classes = sorted(class_config["mapping"].keys())
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
        choices=["simple_cnn", "mobilenet_v2"],
        default="simple_cnn",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
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

    raise NotImplementedError(
        "Entrainement volontairement non lance dans cette etape. "
        f"Dataset valide avec {len(prepared_frame)} images lisibles. "
        "Valider les classes V0 puis implementer/executer la cellule d'entrainement dediee."
    )


if __name__ == "__main__":
    main()
