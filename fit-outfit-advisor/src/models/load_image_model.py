from typing import Optional

from src.config.paths import IMAGE_MODEL_PATH


def load_image_model(model_path=IMAGE_MODEL_PATH) -> Optional[object]:
    """
    Charge le futur modèle CNN Fashion Product Images Small.

    Retourne None si le modèle n'existe pas encore, afin de permettre au MVP simulé de fonctionner.
    """
    if not model_path.exists():
        return None

    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    except Exception as exc:
        raise RuntimeError(f"Impossible de charger le modèle image : {exc}") from exc
