from src.models.load_fit_model import load_fit_artifacts
from src.preprocessing.tabular_preprocessing import (
    FIT_LABELS,
    build_runtime_fit_features,
    convert_size_to_order,
)


def _risk_level(fit_prediction: str, confidence: float) -> str:
    if fit_prediction == "uncertain":
        return "medium"
    if confidence < 0.60:
        return "medium"
    if fit_prediction == "fit":
        return "low"
    return "high"


def _is_experimental_only(metadata: dict) -> bool:
    return (
        metadata.get("model_status") == "experimental_only"
        or metadata.get("promotable_to_streamlit") is False
    )


def _uncertain_result(confidence: float, reason: str, mode: str = "tensorflow") -> dict:
    return {
        "fit_prediction": "uncertain",
        "confidence": confidence,
        "risk_level": "medium",
        "reason": reason,
        "mode": mode,
    }


def _simulate_fit(user_profile: dict, item_features: dict) -> dict:
    """
    Simulation simple basée sur l'écart entre taille habituelle et taille du vêtement.
    """
    usual_size_order = convert_size_to_order(user_profile.get("usual_size"))
    item_size_order = convert_size_to_order(item_features.get("item_size"))

    if usual_size_order == -1 or item_size_order == -1:
        return {
            "fit_prediction": "fit",
            "confidence": 0.50,
            "risk_level": "medium",
            "reason": "Taille inconnue, prédiction neutre.",
            "mode": "simulation",
        }

    diff = item_size_order - usual_size_order

    if diff <= -1:
        return {
            "fit_prediction": "small",
            "confidence": 0.72,
            "risk_level": "high",
            "reason": "La taille choisie est inférieure à la taille habituelle.",
            "mode": "simulation",
        }

    if diff >= 1:
        return {
            "fit_prediction": "large",
            "confidence": 0.70,
            "risk_level": "high",
            "reason": "La taille choisie est supérieure à la taille habituelle.",
            "mode": "simulation",
        }

    return {
        "fit_prediction": "fit",
        "confidence": 0.76,
        "risk_level": "low",
        "reason": "La taille choisie correspond à la taille habituelle.",
        "mode": "simulation",
    }


def _predict_with_artifacts(user_profile: dict, item_features: dict, artifacts) -> dict:
    features = build_runtime_fit_features(user_profile, item_features)
    feature_columns = artifacts.metadata.get("feature_columns") or list(features.columns)
    features = features.reindex(columns=feature_columns, fill_value="unknown")
    transformed = artifacts.preprocessor.transform(features)

    probabilities = artifacts.model.predict(transformed, verbose=0)[0]
    class_index = int(probabilities.argmax())
    confidence = float(probabilities[class_index])

    if artifacts.label_encoder is not None:
        fit_prediction = str(artifacts.label_encoder.inverse_transform([class_index])[0])
    else:
        labels = artifacts.metadata.get("class_labels", FIT_LABELS)
        fit_prediction = str(labels[class_index])

    if _is_experimental_only(artifacts.metadata):
        result = _uncertain_result(
            confidence,
            "Artefact ModCloth experimental : V2 n'est pas promouvable comme conseil de taille fiable.",
        )
        result["raw_fit_prediction"] = fit_prediction
        return result

    threshold = float(
        artifacts.metadata.get("abstention_strategy", {}).get("minimum_confidence", 0.60)
    )
    if confidence < threshold:
        result = _uncertain_result(
            confidence,
            "Confiance trop faible : aucune recommandation ferme small/large.",
        )
        result["raw_fit_prediction"] = fit_prediction
        return result

    return {
        "fit_prediction": fit_prediction,
        "confidence": confidence,
        "risk_level": _risk_level(fit_prediction, confidence),
        "reason": "Prédiction issue du modèle TensorFlow/Keras ModCloth.",
        "mode": "tensorflow",
    }


def predict_fit(user_profile: dict, item_features: dict, use_real_model: bool = True) -> dict:
    """
    Prédit si un vêtement risque d'être small / fit / large.

    Version MVP : simulation contrôlée.
    Version future : MLP TensorFlow/Keras ModCloth.
    """
    if use_real_model:
        artifacts = load_fit_artifacts()
        if artifacts is not None:
            return _predict_with_artifacts(user_profile, item_features, artifacts)

        fallback = _simulate_fit(user_profile, item_features)
        fallback["fallback_reason"] = (
            "Artefacts ModCloth absents ou incomplets. Attendus: "
            "models/fit_v2/fit_model.keras, models/fit_v2/fit_preprocessor.joblib, "
            "models/fit_v2/metadata.json."
        )
        return fallback

    return _simulate_fit(user_profile, item_features)
