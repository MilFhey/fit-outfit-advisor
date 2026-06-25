from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


FIT_TARGET_MAPPING = {
    "small": 0,
    "fit": 1,
    "large": 2,
}

INVERSE_FIT_TARGET_MAPPING = {v: k for k, v in FIT_TARGET_MAPPING.items()}
FIT_LABELS = tuple(FIT_TARGET_MAPPING.keys())

EXPLICIT_CLOTHING_CATEGORIES = ("tops", "dresses", "bottoms", "outerwear", "wedding")
AMBIGUOUS_COMMERCIAL_CATEGORIES = ("new", "sale")

# V2 uses only fields that can be traced to ModCloth columns and can plausibly be
# provided later at inference. It deliberately excludes previous placeholders
# such as weight_kg, usual_size, brand and color.
DEFAULT_NUMERIC_FEATURES = ["height_cm", "height_cm_missing", "item_size_order"]
DEFAULT_CATEGORICAL_FEATURES = ["body_type", "category"]
DEFAULT_FEATURE_COLUMNS = DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES

RENAME_MAP = {
    "body type": "body_type",
    "height": "height_cm",
    "size": "item_size",
    "category": "category",
    "fit": "fit",
}


class TabularPreprocessingError(ValueError):
    """Erreur explicite pour les donnees tabulaires ModCloth invalides."""


def _count_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def basic_clean_modcloth_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage minimal de la cible ModCloth."""
    df = df.copy().drop_duplicates()

    if "fit" in df.columns:
        df["fit"] = df["fit"].astype(str).str.lower().str.strip()
        df = df[df["fit"].isin(FIT_TARGET_MAPPING.keys())]

    return df


def convert_size_to_order(size: Any) -> int:
    """Encode une taille textuelle simple en ordre numerique."""
    order = {
        "XS": 0,
        "S": 1,
        "M": 2,
        "L": 3,
        "XL": 4,
        "XXL": 5,
    }
    return order.get(str(size).upper(), -1)


def convert_modcloth_size_to_order(size: Any) -> float | None:
    """Encode une taille textuelle ou numerique ModCloth en valeur ordonnee."""
    text_order = convert_size_to_order(size)
    if text_order >= 0:
        return float(text_order)

    numeric_size = pd.to_numeric(size, errors="coerce")
    if pd.isna(numeric_size):
        return None
    return float(numeric_size)


def parse_height_to_cm(value: Any) -> float | None:
    """Convertit une taille ModCloth courante (`5ft 5in`, `165 cm`) en cm."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    numeric = re.findall(r"\d+(?:\.\d+)?", text)
    if not numeric:
        return None

    if "cm" in text:
        return float(numeric[0])

    if "ft" in text or "'" in text:
        feet = float(numeric[0])
        inches = float(numeric[1]) if len(numeric) > 1 else 0.0
        return round((feet * 12 + inches) * 2.54, 2)

    number = float(numeric[0])
    if number < 3:
        return round(number * 100, 2)
    if number < 100:
        return round(number * 2.54, 2)
    return number


def parse_weight_to_kg(value: Any) -> float | None:
    """Legacy helper kept for compatibility; weight is not used by ModCloth v2."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()
    numeric = re.findall(r"\d+(?:\.\d+)?", text)
    if not numeric:
        return None

    weight = float(numeric[0])
    if "kg" in text:
        return weight
    return round(weight * 0.453592, 2)


def normalize_modcloth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise uniquement les colonnes ModCloth utiles au fit V2."""
    normalized = df.rename(columns={src: dst for src, dst in RENAME_MAP.items() if src in df.columns}).copy()

    if "fit" in normalized.columns:
        normalized["fit"] = normalized["fit"].astype(str).str.lower().str.strip()

    if "height_cm" in normalized.columns:
        normalized["height_cm"] = normalized["height_cm"].map(parse_height_to_cm)
    else:
        normalized["height_cm"] = np.nan
    normalized["height_cm_missing"] = normalized["height_cm"].isna().astype(int)

    if "item_size" in normalized.columns:
        normalized["item_size_order"] = normalized["item_size"].map(convert_modcloth_size_to_order)
    else:
        normalized["item_size_order"] = np.nan

    if "body_type" in normalized.columns:
        normalized["body_type"] = normalized["body_type"].replace("", np.nan)

    if "category" in normalized.columns:
        normalized["category"] = normalized["category"].replace("", np.nan)

    for column in DEFAULT_NUMERIC_FEATURES:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized


def available_fit_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retourne les colonnes numeriques/categorielles reellement disponibles."""
    numeric = [column for column in DEFAULT_NUMERIC_FEATURES if column in df.columns]
    categorical = [column for column in DEFAULT_CATEGORICAL_FEATURES if column in df.columns]
    return numeric, categorical


def build_runtime_fit_features(user_profile: dict, item_features: dict) -> pd.DataFrame:
    """
    Construit une ligne d'inference compatible avec les artefacts fit V2.

    Ce contrat volontairement court exclut les champs non appris par ModCloth V2 :
    poids, marque, couleur et taille habituelle.
    """
    item_size = item_features.get("item_size", np.nan)
    height_cm = user_profile.get("height_cm", np.nan)
    record = {
        "height_cm": height_cm,
        "height_cm_missing": int(pd.isna(height_cm)),
        "item_size_order": convert_modcloth_size_to_order(item_size),
        "body_type": user_profile.get("body_type", np.nan),
        "category": item_features.get("category", np.nan),
    }
    return pd.DataFrame([record], columns=DEFAULT_FEATURE_COLUMNS)


def build_fit_inference_contract(feature_columns: list[str]) -> dict[str, Any]:
    """Construit le contrat d'inference a partir des colonnes reellement apprises."""
    user_profile = []
    item_features = []
    retained_measurements = []
    excluded_fields = ["weight_kg", "usual_size", "brand", "color"]

    if "height_cm" in feature_columns:
        user_profile.append("height_cm")
        retained_measurements.append("height_cm")
    if "height_cm_missing" in feature_columns:
        user_profile.append("height_cm_missing")
    if "body_type" in feature_columns:
        user_profile.append("body_type")
    if "item_size_order" in feature_columns:
        item_features.append("item_size")
    if "category" in feature_columns:
        item_features.append("category")

    return {
        "user_profile": user_profile,
        "item_features": item_features,
        "retained_measurements": retained_measurements,
        "missing_value_indicators": [
            column for column in feature_columns if column.endswith("_missing")
        ],
        "excluded_previous_fields": excluded_fields,
        "body_type_policy": (
            "Included only when present in feature_columns. Exclude from V3 by default "
            "unless analysis proves it is really trained, useful, and reasonably askable."
        ),
    }


def prepare_fit_training_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Nettoie ModCloth et retourne X/y plus diagnostics de selection."""
    target_before = _count_dict(df["fit"].astype(str).str.lower().str.strip()) if "fit" in df.columns else {}
    cleaned = normalize_modcloth_columns(basic_clean_modcloth_dataframe(df))

    if "fit" not in cleaned.columns:
        raise TabularPreprocessingError("Colonne cible `fit` absente du dataset ModCloth.")

    numeric_features, categorical_features = available_fit_feature_columns(cleaned)
    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        raise TabularPreprocessingError("Aucune colonne feature exploitable apres normalisation.")

    required_numeric = ["item_size_order"]
    missing_required = [column for column in required_numeric if column not in numeric_features]
    if missing_required:
        raise TabularPreprocessingError(
            "Colonnes ModCloth requises absentes apres normalisation: "
            + ", ".join(missing_required)
        )

    cleaned = cleaned[cleaned["fit"].isin(FIT_TARGET_MAPPING.keys())].copy()
    invalid_size_count = int(cleaned["item_size_order"].isna().sum())
    cleaned = cleaned.dropna(subset=["fit", "item_size_order"]).copy()

    if cleaned.empty:
        raise TabularPreprocessingError("Aucune ligne exploitable apres nettoyage ModCloth.")

    diagnostics = {
        "target_distribution_before_cleaning": target_before,
        "target_distribution_after_cleaning": _count_dict(cleaned["fit"]),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": feature_columns,
        "explicit_clothing_categories": list(EXPLICIT_CLOTHING_CATEGORIES),
        "ambiguous_commercial_categories": list(AMBIGUOUS_COMMERCIAL_CATEGORIES),
        "category_distribution": (
            _count_dict(cleaned["category"].astype(str).str.lower().str.strip())
            if "category" in cleaned.columns
            else {}
        ),
        "ambiguous_category_row_count": (
            int(
                cleaned["category"]
                .astype(str)
                .str.lower()
                .str.strip()
                .isin(AMBIGUOUS_COMMERCIAL_CATEGORIES)
                .sum()
            )
            if "category" in cleaned.columns
            else 0
        ),
        "invalid_item_size_count": invalid_size_count,
        "missing_values_after_normalization": cleaned[feature_columns + ["fit"]]
        .isna()
        .sum()
        .to_dict(),
    }

    return cleaned[feature_columns], cleaned["fit"], diagnostics
