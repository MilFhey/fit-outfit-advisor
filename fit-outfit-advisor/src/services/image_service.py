from typing import Any

from src.mappings.category_mapping import map_to_common_category
from src.mappings.fashion_v1_mapping import map_product_type_to_canonical_category
from src.models.load_image_model import load_image_artifacts
from src.preprocessing.image_preprocessing import preprocess_image_for_cnn


DEFAULT_SIMULATED_PRODUCT_TYPE = "tshirt"


def predict_image(image: Any, use_real_model: bool = False) -> dict:
    """
    Prédit la catégorie d'un vêtement à partir d'une image.

    Version MVP : simulation contrôlée.
    Version future : CNN TensorFlow/Keras branché via models/fashion_active/.
    """
    if use_real_model:
        artifacts = load_image_artifacts()
        if artifacts is None:
            fallback = predict_image(image, use_real_model=False)
            fallback["fallback_reason"] = (
                "Artefacts image actifs absents ou non promus. Attendu: models/fashion_active/."
            )
            return fallback

        architecture = (
            artifacts.metadata.get("architecture")
            or artifacts.metadata.get("selected_experiment")
            or "simple_cnn"
        )
        image_size = int(artifacts.metadata.get("image_size", 128))
        batch = preprocess_image_for_cnn(
            image,
            image_size=(image_size, image_size),
            architecture=architecture,
        )
        probabilities = artifacts.model.predict(batch, verbose=0)[0]
        class_index = int(probabilities.argmax())
        confidence = float(probabilities[class_index])

        if artifacts.label_encoder is not None:
            product_type = str(artifacts.label_encoder.inverse_transform([class_index])[0])
        else:
            class_labels = artifacts.metadata.get("class_labels", [])
            product_type = str(class_labels[class_index]) if class_index < len(class_labels) else "unknown"

        raw_product_type = product_type
        minimum_confidence = float(
            artifacts.metadata.get("abstention_strategy", {}).get("minimum_confidence", 0.0)
        )
        if minimum_confidence and confidence < minimum_confidence:
            product_type = "unknown"

        category_config = artifacts.metadata.get("class_config", artifacts.metadata)
        canonical_category = map_product_type_to_canonical_category(product_type, category_config)
        if canonical_category == "unknown":
            canonical_category = map_to_common_category(product_type)

        return {
            "product_type": product_type,
            "canonical_category": canonical_category,
            "predicted_class": product_type,
            "common_category": canonical_category,
            "confidence": confidence,
            "raw_product_type": raw_product_type,
            "minimum_confidence": minimum_confidence,
            "image_size": image_size,
            "model_status": artifacts.metadata.get("model_status", "promoted"),
            "mode": "real_model",
        }

    product_type = DEFAULT_SIMULATED_PRODUCT_TYPE
    canonical_category = map_to_common_category(product_type)
    return {
        "product_type": product_type,
        "canonical_category": canonical_category,
        "predicted_class": product_type,
        "common_category": canonical_category,
        "confidence": 0.82,
        "model_status": "fallback",
        "mode": "simulation",
    }
