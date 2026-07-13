from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.mappings.category_mapping import map_to_common_category
from src.models.load_outfit_model import OutfitModelArtifacts, load_outfit_artifacts
from src.preprocessing.outfit_v2_features import (
    ImageVisualFeatures,
    OUTFIT_V2_CATEGORICAL_FEATURES,
    OUTFIT_V2_NUMERIC_FEATURES,
    build_pair_embedding_block,
    build_pair_numeric_features,
    classify_color_family,
    color_harmony_score,
    empty_visual_features,
    extract_visual_features,
    rgb_to_hsv01,
)
from src.services.image_service import predict_image
from src.services.outfit_service import (
    COMPATIBLE_ROLES,
    RECOMMENDED_PRODUCT_TYPES,
    _compatible_roles_for_product_types,
    recommend_outfit,
    read_cooccurrence_report,
)


COLOR_FAMILY_TO_FR = {
    "black": "noir",
    "white": "blanc",
    "gray": "gris",
    "beige": "beige",
    "brown": "marron",
    "blue": "bleu",
    "red": "rouge",
    "green": "vert",
    "violet": "violet",
    "yellow": "jaune",
    "orange": "orange",
    "pink": "rose",
    "unknown": "noir",
}

CORE_OUTFIT_ROLES = ["top", "bottom", "shoes"]


def _normalize_product_type(image_result: dict[str, Any]) -> str:
    product_type = str(image_result.get("product_type") or image_result.get("raw_product_type") or "unknown")
    if product_type == "unknown":
        return str(image_result.get("common_category") or "unknown")
    return product_type


def _canonical_for_product_type(product_type: str) -> str:
    mapped = map_to_common_category(product_type)
    return mapped if mapped != "unknown" else product_type


def _color_label(features: ImageVisualFeatures) -> str:
    return COLOR_FAMILY_TO_FR.get(features.color_family, "noir")


def _candidate_feature_from_prototype(prototype: dict[str, Any]) -> ImageVisualFeatures:
    rgb_values = prototype.get("mean_rgb") or [0, 0, 0]
    rgb = tuple(int(np.clip(value, 0, 255)) for value in rgb_values[:3])
    color_family = str(prototype.get("color_family") or classify_color_family(rgb))
    embedding = np.asarray(prototype.get("mean_embedding", []), dtype="float32")
    if embedding.size == 0:
        embedding = empty_visual_features().embedding
    return ImageVisualFeatures(
        embedding=embedding,
        dominant_rgb=rgb,
        hsv=rgb_to_hsv01(rgb),
        color_family=color_family,
    )


def _recommendations_from_report(product_type: str) -> list[dict[str, Any]]:
    report = read_cooccurrence_report()
    rows = (
        (report or {})
        .get("primary_baseline", {})
        .get("recommendations_by_product_type", {})
        .get(product_type)
    )
    return rows if isinstance(rows, list) else []


def _cooccurrence_score(input_product_type: str, candidate_product_type: str) -> float:
    for row in _recommendations_from_report(input_product_type):
        if row.get("product_type_v0") == candidate_product_type:
            return float(row.get("raw_compatibility_score", 0.0))
    return 0.0


def _candidate_product_types(input_product_type: str, artifacts: OutfitModelArtifacts) -> list[str]:
    candidates = [row.get("product_type_v0") for row in _recommendations_from_report(input_product_type)]
    if not candidates:
        input_role = _canonical_for_product_type(input_product_type)
        candidates = RECOMMENDED_PRODUCT_TYPES.get(input_role, RECOMMENDED_PRODUCT_TYPES["unknown"])

    supported = artifacts.product_type_prototypes
    ordered = []
    for product_type in candidates:
        if product_type in supported and product_type not in ordered and product_type != input_product_type:
            ordered.append(product_type)

    if len(ordered) < 5:
        input_role = _canonical_for_product_type(input_product_type)
        compatible_roles = set(COMPATIBLE_ROLES.get(input_role, COMPATIBLE_ROLES["unknown"]))
        for product_type, prototype in supported.items():
            candidate_role = str(prototype.get("outfit_role") or _canonical_for_product_type(product_type))
            if candidate_role in compatible_roles and product_type not in ordered and product_type != input_product_type:
                ordered.append(product_type)
    return ordered[:12]


def _build_model_row(
    artifacts: OutfitModelArtifacts,
    *,
    input_product_type: str,
    input_features: ImageVisualFeatures,
    candidate_product_type: str,
    candidate_features: ImageVisualFeatures,
    candidate_prototype: dict[str, Any],
) -> np.ndarray:
    cooccurrence_score = _cooccurrence_score(input_product_type, candidate_product_type)
    categorical = pd.DataFrame(
        [
            {
                "input_product_type": input_product_type,
                "input_canonical_category": _canonical_for_product_type(input_product_type),
                "input_outfit_role": _canonical_for_product_type(input_product_type),
                "input_color_family": input_features.color_family,
                "candidate_product_type": candidate_product_type,
                "candidate_canonical_category": str(
                    candidate_prototype.get("canonical_category")
                    or _canonical_for_product_type(candidate_product_type)
                ),
                "candidate_outfit_role": str(
                    candidate_prototype.get("outfit_role")
                    or _canonical_for_product_type(candidate_product_type)
                ),
                "candidate_color_family": candidate_features.color_family,
            }
        ],
        columns=OUTFIT_V2_CATEGORICAL_FEATURES,
    )
    numeric = np.asarray(
        [
            build_pair_numeric_features(
                input_features,
                candidate_features,
                cooccurrence_score=cooccurrence_score,
            )
        ],
        dtype="float32",
    )
    structured = pd.concat(
        [categorical, pd.DataFrame(numeric, columns=OUTFIT_V2_NUMERIC_FEATURES)],
        axis=1,
    )
    transformed = artifacts.preprocessor.transform(structured)
    embeddings = build_pair_embedding_block(input_features, candidate_features).reshape(1, -1)
    return np.hstack([embeddings, transformed.astype("float32", copy=False)]).astype("float32")


def _score_candidate(
    artifacts: OutfitModelArtifacts,
    *,
    input_product_type: str,
    input_features: ImageVisualFeatures,
    candidate_product_type: str,
    candidate_prototype: dict[str, Any],
) -> dict[str, Any]:
    candidate_features = _candidate_feature_from_prototype(candidate_prototype)
    cooccurrence = _cooccurrence_score(input_product_type, candidate_product_type)
    harmony = color_harmony_score(input_features.color_family, candidate_features.color_family)

    feature_row = _build_model_row(
        artifacts,
        input_product_type=input_product_type,
        input_features=input_features,
        candidate_product_type=candidate_product_type,
        candidate_features=candidate_features,
        candidate_prototype=candidate_prototype,
    )

    ml_score = float(artifacts.model.predict(feature_row, verbose=0).reshape(-1)[0])
    return {
        "product_type_v0": candidate_product_type,
        "canonical_category": str(
            candidate_prototype.get("canonical_category") or _canonical_for_product_type(candidate_product_type)
        ),
        "outfit_role": str(candidate_prototype.get("outfit_role") or _canonical_for_product_type(candidate_product_type)),
        "ml_score": ml_score,
        "color_harmony_score": harmony,
        "cooccurrence_score": cooccurrence,
        "raw_compatibility_score": ml_score,
        "color_family": candidate_features.color_family,
    }


def _predict_visual_item(image: Any, *, use_real_image_model: bool, require_embedding: bool) -> dict[str, Any]:
    image_result = predict_image(image, use_real_model=use_real_image_model)
    try:
        visual_features = extract_visual_features(image, zero_embedding=not require_embedding)
    except Exception:
        visual_features = empty_visual_features("unknown")
    product_type = _normalize_product_type(image_result)
    canonical_category = image_result.get("canonical_category") or _canonical_for_product_type(product_type)
    return {
        "image_result": image_result,
        "product_type": product_type,
        "canonical_category": canonical_category,
        "outfit_role": canonical_category,
        "visual_features": visual_features,
        "color_family": visual_features.color_family,
        "color_label": _color_label(visual_features),
        "confidence": float(image_result.get("confidence", 0.0)),
    }


def recommend_associations_from_image(
    image: Any,
    context: str = "casual",
    *,
    use_real_image_model: bool = True,
) -> dict[str, Any]:
    artifacts = load_outfit_artifacts()
    require_embedding = artifacts is not None
    detected = _predict_visual_item(
        image,
        use_real_image_model=use_real_image_model,
        require_embedding=require_embedding,
    )
    input_product_type = detected["product_type"]
    outfit_input_product_type = input_product_type

    if artifacts is None:
        if not _recommendations_from_report(outfit_input_product_type):
            outfit_input_product_type = str(detected["canonical_category"] or input_product_type)
        fallback = recommend_outfit(outfit_input_product_type, context, detected["color_label"])
        fallback.update(
            {
                "input_product_type": outfit_input_product_type,
                "ml_score": None,
                "color_harmony_score": None,
                "cooccurrence_score": fallback.get("raw_compatibility_score", 0.0),
                "detected_item": {
                    "product_type": input_product_type,
                    "canonical_category": detected["canonical_category"],
                    "color_family": detected["color_family"],
                    "confidence": detected["confidence"],
                },
            }
        )
        return fallback

    if outfit_input_product_type not in artifacts.product_type_prototypes:
        outfit_input_product_type = str(detected["canonical_category"] or input_product_type)

    scored = [
        _score_candidate(
            artifacts,
            input_product_type=outfit_input_product_type,
            input_features=detected["visual_features"],
            candidate_product_type=product_type,
            candidate_prototype=artifacts.product_type_prototypes[product_type],
        )
        for product_type in _candidate_product_types(outfit_input_product_type, artifacts)
    ]
    scored.sort(key=lambda row: row["ml_score"], reverse=True)
    top_rows = scored[:5]

    return {
        "input_product_type": outfit_input_product_type,
        "recommended_product_types": [row["product_type_v0"] for row in top_rows],
        "compatible_roles": _compatible_roles_for_product_types([row["product_type_v0"] for row in top_rows]),
        "raw_compatibility_score": float(top_rows[0]["ml_score"]) if top_rows else 0.0,
        "compatible_items": [row["product_type_v0"].replace("_", " ") for row in top_rows],
        "compatible_colors": [COLOR_FAMILY_TO_FR.get(row["color_family"], row["color_family"]) for row in top_rows],
        "compatibility_score": float(top_rows[0]["ml_score"]) if top_rows else 0.0,
        "mode": "outfit_v2_model",
        "model_status": str(artifacts.metadata.get("model_status", "promoted")),
        "reason": "Outfit V2 actif: score TensorFlow image+couleur+taxonomie, avec cooccurrence V0 comme feature.",
        "ml_score": float(top_rows[0]["ml_score"]) if top_rows else 0.0,
        "color_harmony_score": float(top_rows[0]["color_harmony_score"]) if top_rows else 0.0,
        "cooccurrence_score": float(top_rows[0]["cooccurrence_score"]) if top_rows else 0.0,
        "ranked_recommendations": top_rows,
        "detected_item": {
            "product_type": input_product_type,
            "outfit_product_type": outfit_input_product_type,
            "canonical_category": detected["canonical_category"],
            "color_family": detected["color_family"],
            "confidence": detected["confidence"],
        },
    }


def _missing_roles(detected_roles: list[str]) -> list[str]:
    present = set(role for role in detected_roles if role != "unknown")
    return [role for role in CORE_OUTFIT_ROLES if role not in present]


def _fallback_outfit_score(detected_items: list[dict[str, Any]]) -> float:
    if len(detected_items) < 2:
        return 0.0
    scores = []
    for left, right in combinations(detected_items, 2):
        role_score = 0.55
        if right["outfit_role"] in COMPATIBLE_ROLES.get(left["outfit_role"], []):
            role_score = 0.72
        harmony = color_harmony_score(left["color_family"], right["color_family"])
        scores.append(0.65 * role_score + 0.35 * harmony)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_outfit_images(
    images: list[Any],
    context: str = "casual",
    *,
    use_real_image_model: bool = True,
) -> dict[str, Any]:
    artifacts = load_outfit_artifacts()
    require_embedding = artifacts is not None
    detected_items = [
        _predict_visual_item(
            image,
            use_real_image_model=use_real_image_model,
            require_embedding=require_embedding,
        )
        for image in images
    ]
    warnings = []
    if len(detected_items) < 2:
        warnings.append("Ajoute au moins deux images pour evaluer une tenue complete.")

    public_detected = [
        {
            "product_type": item["product_type"],
            "canonical_category": item["canonical_category"],
            "outfit_role": item["outfit_role"],
            "color_family": item["color_family"],
            "color_label": item["color_label"],
            "confidence": item["confidence"],
        }
        for item in detected_items
    ]
    missing_roles = _missing_roles([item["outfit_role"] for item in detected_items])

    if artifacts is None or len(detected_items) < 2:
        outfit_score = _fallback_outfit_score(public_detected)
        suggested = []
        for role in missing_roles:
            suggested.extend(RECOMMENDED_PRODUCT_TYPES.get(role, []))
        return {
            "outfit_score": outfit_score,
            "pair_scores": [],
            "detected_items": public_detected,
            "missing_roles": missing_roles,
            "suggested_associations": suggested[:5],
            "warnings": warnings,
            "mode": "cooccurrence_baseline" if artifacts is None else "fallback",
            "model_status": "experimental_only" if artifacts is None else "fallback",
            "reason": "Outfit V2 non promu: evaluation via roles, cooccurrence et harmonie couleur.",
        }

    pair_scores = []
    for left, right in combinations(detected_items, 2):
        right_prototype = {
            "canonical_category": right["canonical_category"],
            "outfit_role": right["outfit_role"],
            "mean_embedding": right["visual_features"].embedding.tolist(),
            "mean_rgb": list(right["visual_features"].dominant_rgb),
            "color_family": right["visual_features"].color_family,
        }
        score = _score_candidate(
            artifacts,
            input_product_type=left["product_type"],
            input_features=left["visual_features"],
            candidate_product_type=right["product_type"],
            candidate_prototype=right_prototype,
        )
        pair_scores.append(
            {
                "input_product_type": left["product_type"],
                "candidate_product_type": right["product_type"],
                "ml_score": score["ml_score"],
                "color_harmony_score": score["color_harmony_score"],
                "cooccurrence_score": score["cooccurrence_score"],
                "raw_compatibility_score": score["raw_compatibility_score"],
            }
        )

    ml_average = float(np.mean([row["ml_score"] for row in pair_scores])) if pair_scores else 0.0
    color_average = float(np.mean([row["color_harmony_score"] for row in pair_scores])) if pair_scores else 0.0
    missing_penalty = 0.06 * len(missing_roles)
    outfit_score = float(np.clip(0.82 * ml_average + 0.18 * color_average - missing_penalty, 0.0, 1.0))
    suggested = []
    for role in missing_roles:
        suggested.extend(RECOMMENDED_PRODUCT_TYPES.get(role, []))

    return {
        "outfit_score": outfit_score,
        "pair_scores": pair_scores,
        "detected_items": public_detected,
        "missing_roles": missing_roles,
        "suggested_associations": suggested[:5],
        "warnings": warnings,
        "mode": "outfit_v2_model",
        "model_status": str(artifacts.metadata.get("model_status", "promoted")),
        "reason": "Outfit V2 actif: score agrege des paires TensorFlow, ajuste par couleur et roles manquants.",
    }
