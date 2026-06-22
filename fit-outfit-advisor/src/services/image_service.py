from typing import Any

from src.mappings.category_mapping import map_to_common_category
from src.models.load_image_model import load_image_model
from src.preprocessing.image_preprocessing import preprocess_image_for_cnn


DEFAULT_SIMULATED_CLASS = "Tshirts"


def predict_image(image: Any, use_real_model: bool = False) -> dict:
    """
    Prédit la catégorie d'un vêtement à partir d'une image.

    Version MVP : simulation contrôlée.
    Version future : CNN TensorFlow/Keras branché via models/fashion_model.keras.
    """
    if use_real_model:
        model = load_image_model()
        if model is None:
            fallback = predict_image(image, use_real_model=False)
            fallback["fallback_reason"] = (
                "Artefact image absent. Attendu: models/fashion_model.keras."
            )
            return fallback

        batch = preprocess_image_for_cnn(image)
        probabilities = model.predict(batch, verbose=0)[0]
        class_index = int(probabilities.argmax())
        confidence = float(probabilities[class_index])

        # À remplacer par le vrai fichier class_indices.json généré à l'entraînement.
        index_to_class = {
            0: "Tshirts",
            1: "Shirts",
            2: "Jeans",
            3: "Dresses",
            4: "Casual Shoes",
        }
        predicted_class = index_to_class.get(class_index, "unknown")

        return {
            "predicted_class": predicted_class,
            "common_category": map_to_common_category(predicted_class),
            "confidence": confidence,
            "mode": "real_model",
        }

    predicted_class = DEFAULT_SIMULATED_CLASS
    return {
        "predicted_class": predicted_class,
        "common_category": map_to_common_category(predicted_class),
        "confidence": 0.82,
        "mode": "simulation",
    }
