import json
from pathlib import Path
from typing import Any

from src.config.paths import OUTFIT_V1_CONFIG_PATH
from src.mappings.fashion_v1_mapping import (
    FASHION_CANONICAL_CATEGORIES,
    FASHION_PRODUCT_TYPES_V0,
)


OUTFIT_ROLES_V0 = FASHION_CANONICAL_CATEGORIES


class OutfitConfigError(ValueError):
    """Raised when the Outfit V1 configuration is incomplete or incoherent."""


def load_outfit_v1_config(config_path: Path = OUTFIT_V1_CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


def validate_outfit_v1_config(config: dict[str, Any], *, require_ready: bool = False) -> None:
    if config.get("target") != "outfit_compatibility_v0":
        raise OutfitConfigError("La cible Outfit V1 doit etre outfit_compatibility_v0.")
    if config.get("taxonomy_source") != "fashion_v1":
        raise OutfitConfigError("Outfit V1 doit utiliser la taxonomie fashion_v1.")

    allowed_roles = config.get("allowed_outfit_roles")
    if not isinstance(allowed_roles, list) or not allowed_roles:
        raise OutfitConfigError("allowed_outfit_roles doit etre une liste non vide.")
    invalid_roles = [role for role in allowed_roles if role not in OUTFIT_ROLES_V0]
    if invalid_roles:
        raise OutfitConfigError(f"Roles outfit invalides: {invalid_roles}.")

    compatible_pairs = config.get("compatible_role_pairs")
    if not isinstance(compatible_pairs, list):
        raise OutfitConfigError("compatible_role_pairs doit etre une liste.")
    for pair in compatible_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise OutfitConfigError(f"Paire de roles invalide: {pair}.")
        invalid_pair_roles = [role for role in pair if role not in allowed_roles]
        if invalid_pair_roles:
            raise OutfitConfigError(f"Paire avec roles inconnus: {pair}.")

    feature_policy = config.get("feature_policy", {})
    forbidden_direct_features = set(feature_policy.get("forbidden_direct_features", []))
    allowed_features = set(feature_policy.get("allowed_features", []))
    if forbidden_direct_features.intersection(allowed_features):
        raise OutfitConfigError("item_id/outfit_id ne doivent pas etre des features directes.")

    mapping = config.get("polyvore_label_mapping")
    if not isinstance(mapping, dict):
        raise OutfitConfigError("polyvore_label_mapping doit etre un dictionnaire.")
    for label, payload in mapping.items():
        if not label or not isinstance(payload, dict):
            raise OutfitConfigError(f"Mapping Polyvore invalide pour: {label}.")
        product_type = payload.get("product_type_v0")
        canonical_category = payload.get("canonical_category")
        outfit_role = payload.get("outfit_role")
        if product_type not in FASHION_PRODUCT_TYPES_V0:
            raise OutfitConfigError(f"product_type_v0 inconnu pour {label}: {product_type}.")
        if canonical_category not in FASHION_CANONICAL_CATEGORIES:
            raise OutfitConfigError(
                f"canonical_category inconnue pour {label}: {canonical_category}."
            )
        if outfit_role not in allowed_roles:
            raise OutfitConfigError(f"outfit_role inconnu pour {label}: {outfit_role}.")

    if not require_ready:
        return

    if config.get("status") == "draft_requires_dataset_inspection":
        raise OutfitConfigError("La config Outfit V1 requiert encore l'audit dataset.")
    if not mapping:
        raise OutfitConfigError("polyvore_label_mapping ne peut pas etre vide pour entrainer.")
    if config.get("source_label_column") is None:
        raise OutfitConfigError("source_label_column doit etre renseigne apres audit.")


def map_polyvore_label_to_fashion(
    polyvore_label: str,
    config: dict[str, Any],
) -> dict[str, str] | None:
    mapping = config.get("polyvore_label_mapping", {})
    payload = mapping.get(str(polyvore_label).strip())
    if payload is None:
        payload = mapping.get(_normalize_label(polyvore_label))
    if payload is None:
        return None
    return {
        "polyvore_label": str(polyvore_label).strip(),
        "product_type_v0": str(payload["product_type_v0"]),
        "canonical_category": str(payload["canonical_category"]),
        "outfit_role": str(payload["outfit_role"]),
    }


def compatible_role_pair_key(role_a: str, role_b: str) -> tuple[str, str]:
    return tuple(sorted((str(role_a), str(role_b))))


def build_compatible_role_pair_set(config: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        compatible_role_pair_key(pair[0], pair[1])
        for pair in config.get("compatible_role_pairs", [])
    }
