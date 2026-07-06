import json
from pathlib import Path
from typing import Any

from src.config.paths import FASHION_V1_CLASSES_PATH


FASHION_CANONICAL_CATEGORIES = (
    "top",
    "bottom",
    "dress",
    "shoes",
    "outerwear",
    "accessory",
)


class FashionClassConfigError(ValueError):
    """Raised when the Fashion V1 class configuration is incomplete."""


def load_fashion_v1_class_config(config_path: Path = FASHION_V1_CLASSES_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_article_type_to_canonical_mapping(config: dict[str, Any]) -> dict[str, str]:
    article_type_mapping: dict[str, str] = {}
    for canonical_category, article_types in config.get("mapping", {}).items():
        if canonical_category not in FASHION_CANONICAL_CATEGORIES:
            continue
        for article_type in article_types or []:
            normalized_article_type = str(article_type).strip()
            if normalized_article_type:
                article_type_mapping[normalized_article_type] = canonical_category
    return article_type_mapping


def map_article_type_to_canonical_category(
    article_type: str,
    config: dict[str, Any],
) -> str | None:
    article_type_mapping = build_article_type_to_canonical_mapping(config)
    return article_type_mapping.get(str(article_type).strip())


def validate_fashion_v1_class_config(
    config: dict[str, Any],
    *,
    require_ready: bool = False,
) -> None:
    if config.get("target") != "canonical_category":
        raise FashionClassConfigError("La cible Fashion V1 doit etre canonical_category.")
    if config.get("source_column") != "articleType":
        raise FashionClassConfigError("La colonne source Fashion V1 doit etre articleType.")

    mapping = config.get("mapping")
    if not isinstance(mapping, dict):
        raise FashionClassConfigError("Le mapping Fashion V1 doit etre un dictionnaire.")

    missing_categories = [
        category for category in FASHION_CANONICAL_CATEGORIES if category not in mapping
    ]
    if missing_categories:
        raise FashionClassConfigError(
            f"Categories canoniques absentes du mapping: {missing_categories}."
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

    empty_categories = [
        category for category, article_types in mapping.items() if not article_types
    ]
    if empty_categories:
        raise FashionClassConfigError(
            f"ArticleType absents pour les categories: {empty_categories}."
        )


def is_fashion_v1_class_config_ready(config: dict[str, Any]) -> bool:
    try:
        validate_fashion_v1_class_config(config, require_ready=True)
    except FashionClassConfigError:
        return False
    return True
