import json
from pathlib import Path
from typing import Any

from src.config.paths import REPORTS_DIR
from src.mappings.category_mapping import map_to_common_category
from src.mappings.color_mapping import get_compatible_colors


COOCCURRENCE_REPORT_PATH = REPORTS_DIR / "polyvore_v0_cooccurrence_baseline.json"
READY_BASELINE_DECISION = "train_only_baseline_ready_with_leakage_filtered_evaluation"
MIN_VALIDATION_MRR = 0.65
MIN_VALIDATION_RECALL_AT_3 = 0.85


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


def read_cooccurrence_report(
    report_path: Path = COOCCURRENCE_REPORT_PATH,
) -> dict[str, Any] | None:
    try:
        with report_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_cooccurrence_report_usable(report: dict[str, Any] | None) -> bool:
    if not report:
        return False
    if report.get("baseline_decision") != READY_BASELINE_DECISION:
        return False
    if report.get("leakage_filtered_evaluation_ready") is not True:
        return False
    if report.get("tensorflow_used") is not False:
        return False
    if report.get("model_status") != "experimental_only":
        return False

    evaluation = report.get("leakage_filtered_evaluation")
    if not isinstance(evaluation, dict) or not evaluation:
        return False
    validation_metrics = evaluation.get("valid") or next(iter(evaluation.values()), {})
    if not isinstance(validation_metrics, dict):
        return False
    recall_at_3 = validation_metrics.get("recall_at_k", {}).get("3")
    return (
        float(validation_metrics.get("mrr", 0.0)) >= MIN_VALIDATION_MRR
        and float(recall_at_3 or 0.0) >= MIN_VALIDATION_RECALL_AT_3
        and int(validation_metrics.get("evaluable_directed_pair_count", 0)) > 0
    )


def _compatible_roles_for_product_types(product_types: list[str]) -> list[str]:
    roles = []
    for product_type in product_types:
        role = map_to_common_category(product_type)
        if role != "unknown" and role not in roles:
            roles.append(role)
    return roles


def _build_rule_based_result(category: str, context: str, color: str, fallback_reason: str) -> dict:
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
        "reason": f"{context_reason} {fallback_reason}",
        "mode": "rule_based",
        "model_status": "fallback",
    }


def _build_cooccurrence_result(
    category: str,
    context: str,
    color: str,
    report: dict[str, Any],
) -> dict | None:
    recommendations = (
        report.get("primary_baseline", {})
        .get("recommendations_by_product_type", {})
        .get(category)
    )
    if not recommendations:
        return None

    recommended_product_types = [row["product_type_v0"] for row in recommendations]
    top_score = float(recommendations[0]["raw_compatibility_score"])
    compatible_roles = _compatible_roles_for_product_types(recommended_product_types)
    evaluation = report.get("leakage_filtered_evaluation", {})
    validation_metrics = evaluation.get("valid") or {}
    context_reason = CONTEXT_RULES.get(context, CONTEXT_RULES["casual"])

    return {
        "input_product_type": category,
        "recommended_product_types": recommended_product_types,
        "compatible_roles": compatible_roles,
        "raw_compatibility_score": top_score,
        "compatible_items": [
            product_type.replace("_", " ") for product_type in recommended_product_types
        ],
        "compatible_colors": get_compatible_colors(color),
        "compatibility_score": top_score,
        "reason": (
            f"{context_reason} Baseline cooccurrence Polyvore V0 issue de "
            f"{report.get('primary_training_split', 'disjoint_train')} ; "
            f"evaluation filtree validee "
            f"(MRR valid={float(validation_metrics.get('mrr', 0.0)):.3f}, "
            f"Recall@3 valid={float(validation_metrics.get('recall_at_k', {}).get('3', 0.0)):.3f})."
        ),
        "mode": "cooccurrence_baseline",
        "model_status": "experimental_only",
        "baseline_decision": report.get("baseline_decision"),
    }


def recommend_outfit(common_category: str, context: str, color: str) -> dict:
    """
    Recommandation de tenue V0.

    La baseline Polyvore n'est utilisée que si le rapport local est explicitement prêt
    et évalué avec filtrage de fuite. Sinon, le service reste fail-closed en fallback.
    """
    category = common_category or "unknown"
    context = context or "casual"
    color = (color or "noir").lower()

    report = read_cooccurrence_report()
    if is_cooccurrence_report_usable(report):
        result = _build_cooccurrence_result(category, context, color, report)
        if result is not None:
            return result

    return _build_rule_based_result(
        category,
        context,
        color,
        "Baseline cooccurrence absente, incomplete ou non applicable a ce product_type.",
    )
