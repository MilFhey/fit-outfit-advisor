import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config.paths import (
    FASHION_LABEL_ENCODER_PATH,
    FASHION_METADATA_PATH,
    FASHION_MODEL_PATH,
)


@dataclass
class ImageModelArtifacts:
    model: object
    label_encoder: object | None
    metadata: dict


def read_image_metadata(metadata_path: Path = FASHION_METADATA_PATH) -> dict | None:
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except Exception:
        return None


def is_image_metadata_promoted(metadata: dict | None) -> bool:
    if not metadata:
        return False
    return (
        metadata.get("model_status") == "promoted"
        and metadata.get("promotable_to_streamlit") is True
    )


def load_image_model(
    model_path: Path = FASHION_MODEL_PATH,
    metadata_path: Path = FASHION_METADATA_PATH,
) -> Optional[object]:
    """
    Charge le futur modèle CNN Fashion Product Images Small.

    Retourne None si le modèle n'existe pas ou si ses metadata ne sont pas explicitement promues.
    """
    metadata = read_image_metadata(metadata_path)
    if not is_image_metadata_promoted(metadata):
        return None
    if not model_path.exists():
        return None

    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    except Exception:
        return None


def load_image_artifacts(
    model_path: Path = FASHION_MODEL_PATH,
    label_encoder_path: Path = FASHION_LABEL_ENCODER_PATH,
    metadata_path: Path = FASHION_METADATA_PATH,
) -> ImageModelArtifacts | None:
    metadata = read_image_metadata(metadata_path)
    if not is_image_metadata_promoted(metadata):
        return None

    model = load_image_model(model_path=model_path, metadata_path=metadata_path)
    if model is None:
        return None

    label_encoder = None
    if label_encoder_path.exists():
        try:
            import joblib

            label_encoder = joblib.load(label_encoder_path)
        except Exception:
            return None

    return ImageModelArtifacts(model=model, label_encoder=label_encoder, metadata=metadata)
