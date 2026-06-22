import json
from dataclasses import dataclass
from typing import Optional

import joblib

from src.config.paths import (
    FIT_LABEL_ENCODER_PATH,
    FIT_METADATA_PATH,
    FIT_MODEL_PATH,
    FIT_PREPROCESSOR_PATH,
)


@dataclass(frozen=True)
class FitModelArtifacts:
    model: object
    preprocessor: object
    label_encoder: object | None
    metadata: dict


def load_fit_model(model_path=FIT_MODEL_PATH) -> Optional[object]:
    """
    Charge le futur modèle MLP ModCloth.

    Retourne None si le modèle n'existe pas encore, afin de permettre au MVP simulé de fonctionner.
    """
    if not model_path.exists():
        return None

    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger le modèle fit : {exc}") from exc


def load_fit_artifacts(
    model_path=FIT_MODEL_PATH,
    preprocessor_path=FIT_PREPROCESSOR_PATH,
    label_encoder_path=FIT_LABEL_ENCODER_PATH,
    metadata_path=FIT_METADATA_PATH,
) -> Optional[FitModelArtifacts]:
    """
    Charge les artefacts necessaires a l'inference ModCloth.

    Retourne None tant que le modele, le preprocessor ou les metadonnees ne sont pas
    disponibles. Le label encoder est optionnel si les classes sont dans metadata.
    """
    required_paths = [model_path, preprocessor_path, metadata_path]
    if any(not path.exists() for path in required_paths):
        return None

    model = load_fit_model(model_path)
    if model is None:
        return None

    try:
        preprocessor = joblib.load(preprocessor_path)
        label_encoder = joblib.load(label_encoder_path) if label_encoder_path.exists() else None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger les artefacts fit : {exc}") from exc

    return FitModelArtifacts(
        model=model,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        metadata=metadata,
    )
