from src.mappings.color_mapping import get_compatible_colors


BASE_SUGGESTIONS = {
    "top": ["jean brut", "pantalon beige", "sneakers blanches"],
    "bottom": ["t-shirt uni", "chemise casual", "sneakers sobres"],
    "dress": ["chaussures sobres", "sac minimaliste", "veste légère"],
    "shoes": ["jean", "t-shirt basique", "surchemise"],
    "accessory": ["tenue sobre", "couleurs neutres", "pièce principale simple"],
    "unknown": ["pièces neutres", "couleurs simples", "coupe classique"],
}

RECOMMENDED_PRODUCT_TYPES = {
    "top": ["jeans", "trousers", "shorts", "casual_shoes"],
    "bottom": ["tshirt", "shirt", "top", "casual_shoes"],
    "dress": ["heels", "dress_shoes", "bag", "outerwear"],
    "shoes": ["jeans", "trousers", "tshirt", "shirt"],
    "outerwear": ["shirt", "top", "jeans", "trousers"],
    "bag": ["dress", "heels", "shirt", "trousers"],
    "accessory": ["shirt", "dress", "jeans", "outerwear"],
    "unknown": ["tshirt", "jeans", "casual_shoes"],
}

COMPATIBLE_ROLES = {
    "top": ["bottom", "shoes", "outerwear", "bag"],
    "bottom": ["top", "shoes", "outerwear"],
    "dress": ["shoes", "outerwear", "bag"],
    "shoes": ["top", "bottom", "dress", "bag"],
    "outerwear": ["top", "bottom", "dress"],
    "bag": ["top", "dress", "shoes"],
    "accessory": ["top", "bottom", "dress"],
    "unknown": ["top", "bottom", "shoes"],
}

CONTEXT_RULES = {
    "casual": "Privilégier une association simple, confortable et facile à porter.",
    "travail": "Privilégier des pièces sobres, structurées et peu voyantes.",
    "soirée": "Ajouter une pièce plus habillée ou une couleur plus marquée.",
    "sport": "Privilégier confort, respirabilité et liberté de mouvement.",
}


def recommend_outfit(common_category: str, context: str, color: str) -> dict:
    """
    Recommandation de tenue simplifiée.

    Cette version MVP repose sur des règles lisibles, défendables et faciles à expliquer.
    """
    category = common_category or "unknown"
    context = context or "casual"
    color = (color or "noir").lower()

    base_items = BASE_SUGGESTIONS.get(category, BASE_SUGGESTIONS["unknown"])
    compatible_colors = get_compatible_colors(color)
    context_reason = CONTEXT_RULES.get(context, CONTEXT_RULES["casual"])

    score = 0.78
    if category == "unknown":
        score = 0.50
    elif context == "travail" and color in {"noir", "blanc", "beige", "bleu"}:
        score = 0.84
    elif context == "soirée" and color in {"noir", "bordeaux", "rouge"}:
        score = 0.86

    return {
        "input_product_type": category,
        "recommended_product_types": RECOMMENDED_PRODUCT_TYPES.get(
            category, RECOMMENDED_PRODUCT_TYPES["unknown"]
        ),
        "compatible_roles": COMPATIBLE_ROLES.get(category, COMPATIBLE_ROLES["unknown"]),
        "raw_compatibility_score": score,
        "compatible_items": base_items,
        "compatible_colors": compatible_colors,
        "compatibility_score": score,
        "reason": context_reason,
        "mode": "rule_based",
        "model_status": "fallback",
    }
