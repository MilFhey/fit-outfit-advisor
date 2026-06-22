def _confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "élevée"
    if confidence >= 0.60:
        return "moyenne"
    return "faible"


def generate_advice(
    image_result: dict,
    fit_result: dict,
    outfit_result: dict,
    context: str,
) -> dict:
    """
    Génère un conseil final compréhensible.

    Le but n'est pas seulement de prédire, mais d'expliquer la décision à l'utilisateur.
    """
    clothing_type = image_result.get("predicted_class", "vêtement inconnu")
    image_confidence = float(image_result.get("confidence", 0.0))

    fit_prediction = fit_result.get("fit_prediction", "fit")
    fit_confidence = float(fit_result.get("confidence", 0.0))

    compatibility_score = float(outfit_result.get("compatibility_score", 0.0))
    compatible_items = outfit_result.get("compatible_items", [])
    compatible_colors = outfit_result.get("compatible_colors", [])

    warnings = []
    if image_confidence < 0.60:
        warnings.append("Confiance image faible : la catégorie détectée doit être vérifiée.")
    if fit_confidence < 0.60:
        warnings.append("Confiance fit faible : la prédiction de taille est prudente.")

    advice = (
        f"Le vêtement détecté est probablement : {clothing_type} "
        f"avec une confiance {_confidence_label(image_confidence)} ({image_confidence:.0%}). "
        f"La prédiction de taille indique : {fit_prediction} "
        f"avec une confiance {_confidence_label(fit_confidence)} ({fit_confidence:.0%}). "
        f"Pour un contexte {context}, la compatibilité de tenue est estimée à {compatibility_score:.0%}. "
    )

    if compatible_items:
        advice += "Tu peux l'associer avec : " + ", ".join(compatible_items[:3]) + ". "

    if compatible_colors:
        advice += "Couleurs compatibles conseillées : " + ", ".join(compatible_colors[:4]) + ". "

    if fit_prediction == "small":
        advice += "Conseil final : la taille risque d'être trop petite, envisage une taille au-dessus."
    elif fit_prediction == "large":
        advice += "Conseil final : la taille risque d'être trop grande, vérifie les mesures avant achat."
    else:
        advice += "Conseil final : la taille semble adaptée, sous réserve des mesures réelles du vêtement."

    return {
        "advice": advice,
        "warnings": warnings,
        "mode": "rule_based_mvp",
    }
