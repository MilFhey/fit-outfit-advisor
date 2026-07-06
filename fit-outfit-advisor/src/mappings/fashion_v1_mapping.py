import json
from pathlib import Path
from typing import Any

from src.config.paths import FASHION_V1_CLASSES_PATH


FASHION_PRODUCT_TYPES_V0 = (
    "tshirt",
    "shirt",
    "top",
    "jeans",
    "trousers",
    "shorts",
    "dress",
    "outerwear",
    "casual_shoes",
    "sports_shoes",
    "dress_shoes",
    "bag",
    "watch",
    "sunglasses",
    "cap",
)

FASHION_CANONICAL_CATEGORIES = (
    "top",
    "bottom",
    "dress",
    "shoes",
    "outerwear",
    "bag",
    "accessory",
)


class FashionClassConfigError(ValueError):
    """Raised when the Fashion V1 class configuration is incomplete."""


def load_fashion_v1_class_config(config_path: Path = FASHION_V1_CLASSES_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_article_type_to_product_type_mapping(config: dict[str, Any]) -> dict[str, str]:
    article_type_mapping: dict[str, str] = {}
    for product_type, article_types in config.get("product_type_mapping", {}).items():
        if product_type not in FASHION_PRODUCT_TYPES_V0:
            continue
        for article_type in article_types or []:
            normalized_article_type = str(article_type).strip()
            if normalized_article_type:
                article_type_mapping[normalized_article_type] = product_type
    return article_type_mapping


def map_article_type_to_product_type(
    article_type: str,
    config: dict[str, Any],
) -> str | None:
    article_type_mapping = build_article_type_to_product_type_mapping(config)
    return article_type_mapping.get(str(article_type).strip())


def map_product_type_to_canonical_category(
    product_type: str,
    config: dict[str, Any],
) -> str:
    canonical_mapping = config.get("canonical_mapping", {})
    return str(canonical_mapping.get(str(product_type).strip(), "unknown"))


def map_article_type_to_canonical_category(
    article_type: str,
    config: dict[str, Any],
) -> str | None:
    product_type = map_article_type_to_product_type(article_type, config)
    if product_type is None:
        return None
    return map_product_type_to_canonical_category(product_type, config)


def validate_fashion_v1_class_config(
    config: dict[str, Any],
    *,
    require_ready: bool = False,
) -> None:
    if config.get("target") != "product_type_v0":
        raise FashionClassConfigError("La cible Fashion V1 doit etre product_type_v0.")
    if config.get("source_column") != "articleType":
        raise FashionClassConfigError("La colonne source Fashion V1 doit etre articleType.")

    product_type_mapping = config.get("product_type_mapping")
    if not isinstance(product_type_mapping, dict):
        raise FashionClassConfigError("product_type_mapping doit etre un dictionnaire.")

    canonical_mapping = config.get("canonical_mapping")
    if not isinstance(canonical_mapping, dict):
        raise FashionClassConfigError("canonical_mapping doit etre un dictionnaire.")

    unknown_product_types = [
        product_type
        for product_type in product_type_mapping
        if product_type not in FASHION_PRODUCT_TYPES_V0
    ]
    if unknown_product_types:
        raise FashionClassConfigError(
            f"Product types non reconnus: {unknown_product_types}."
        )

    missing_canonical = [
        product_type
        for product_type in product_type_mapping
        if product_type not in canonical_mapping
    ]
    if missing_canonical:
        raise FashionClassConfigError(
            f"Mapping canonique absent pour: {missing_canonical}."
        )

    invalid_canonical_categories = [
        canonical_category
        for canonical_category in canonical_mapping.values()
        if canonical_category not in FASHION_CANONICAL_CATEGORIES
    ]
    if invalid_canonical_categories:
        raise FashionClassConfigError(
            f"Categories canoniques invalides: {invalid_canonical_categories}."
        )

    if not require_ready:
        return

    if config.get("status") == "draft_requires_dataset_inspection":
        raise FashionClassConfigError(
            "Le mapping Fashion V1 est encore en brouillon et requiert l'inspection dataset."
        )

    minimum_count = config.get("minimum_readable_images_per_class")
    if minimum_count is None:
        raise FashionClassConfigError(
            "minimum_readable_images_per_class doit etre renseigne avant entrainement."
        )
    if int(minimum_count) <= 0:
        raise FashionClassConfigError(
            "minimum_readable_images_per_class doit etre strictement positif."
        )

    empty_product_types = [
        product_type
        for product_type, article_types in product_type_mapping.items()
        if not article_types
    ]
    if empty_product_types:
        raise FashionClassConfigError(
            f"ArticleType absents pour les product types: {empty_product_types}."
        )


def is_fashion_v1_class_config_ready(config: dict[str, Any]) -> bool:
    try:
        validate_fashion_v1_class_config(config, require_ready=True)
    except FashionClassConfigError:
        return False
    return True
