import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.paths import (
    OUTFIT_METADATA_PATH,
    OUTFIT_MODEL_PATH,
    OUTFIT_PREPROCESSOR_PATH,
    OUTFIT_PROTOTYPES_PATH,
)


@dataclass
class OutfitModelArtifacts:
    model: Any
    preprocessor: Any
    metadata: dict
    product_type_prototypes: dict[str, Any]


def read_outfit_metadata(metadata_path: Path = OUTFIT_METADATA_PATH) -> dict | None:
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_outfit_metadata_promoted(metadata: dict | None) -> bool:
    if not metadata:
        return False
    return (
        metadata.get("model_status") == "promoted"
        and metadata.get("promotable_to_streamlit") is True
        and metadata.get("version") == "outfit_v2"
        and metadata.get("uses_image_embeddings") is True
        and metadata.get("uses_color_features") is True
    )


def read_product_type_prototypes(
    prototypes_path: Path = OUTFIT_PROTOTYPES_PATH,
) -> dict[str, Any] | None:
    if not prototypes_path.exists():
        return None
    try:
        with prototypes_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    prototypes = payload.get("product_type_prototypes") if isinstance(payload, dict) else None
    return prototypes if isinstance(prototypes, dict) else None


def load_outfit_artifacts(
    model_path: Path = OUTFIT_MODEL_PATH,
    preprocessor_path: Path = OUTFIT_PREPROCESSOR_PATH,
    metadata_path: Path = OUTFIT_METADATA_PATH,
    prototypes_path: Path = OUTFIT_PROTOTYPES_PATH,
) -> OutfitModelArtifacts | None:
    metadata = read_outfit_metadata(metadata_path)
    if not is_outfit_metadata_promoted(metadata):
        return None
    if not model_path.exists() or not preprocessor_path.exists():
        return None

    prototypes = read_product_type_prototypes(prototypes_path)
    if not prototypes:
        return None

    try:
        import joblib
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        preprocessor = joblib.load(preprocessor_path)
    except Exception:
        return None

    return OutfitModelArtifacts(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        product_type_prototypes=prototypes,
    )
