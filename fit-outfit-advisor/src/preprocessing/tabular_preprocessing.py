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

V3_NUMERIC_FEATURES = [
    "item_size_order",
    "height_cm",
    "height_cm_missing",
    "hips",
    "hips_missing",
    "bra_size",
    "bra_size_missing",
    "cup_size_missing",
]
V3_CATEGORICAL_FEATURES = ["category", "cup_size"]
V3_FEATURE_COLUMNS = V3_NUMERIC_FEATURES + V3_CATEGORICAL_FEATURES

RENAME_MAP = {
    "body type": "body_type",
    "height": "height_cm",
    "size": "item_size",
    "category": "category",
    "fit": "fit",
}

V3_RENAME_MAP = {
    "height": "height_cm",
    "size": "item_size",
    "category": "category",
    "fit": "fit",
    "hips": "hips",
    "bra size": "bra_size",
    "cup size": "cup_size",
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


def _normalize_text_category(value: Any) -> Any:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    return text if text else np.nan


def _missing_indicator(series: pd.Series) -> pd.Series:
    return series.isna().astype(int)


def normalize_modcloth_columns_v3(
    df: pd.DataFrame,
    min_height_cm: float = 130.0,
    max_height_cm: float = 210.0,
) -> tuple[pd.DataFrame, dict]:
    """Normalise les colonnes ModCloth retenues pour l'experience fit V3."""
    normalized = df.rename(columns={src: dst for src, dst in V3_RENAME_MAP.items() if src in df.columns}).copy()

    if "fit" in normalized.columns:
        normalized["fit"] = normalized["fit"].astype(str).str.lower().str.strip()

    if "height_cm" in normalized.columns:
        normalized["height_cm"] = normalized["height_cm"].map(parse_height_to_cm)
    else:
        normalized["height_cm"] = np.nan

    height_outlier_mask = normalized["height_cm"].notna() & (
        (normalized["height_cm"] < min_height_cm) | (normalized["height_cm"] > max_height_cm)
    )
    height_outlier_count = int(height_outlier_mask.sum())
    normalized.loc[height_outlier_mask, "height_cm"] = np.nan
    normalized["height_cm_missing"] = _missing_indicator(normalized["height_cm"])

    if "item_size" in normalized.columns:
        normalized["item_size_order"] = normalized["item_size"].map(convert_modcloth_size_to_order)
    else:
        normalized["item_size_order"] = np.nan

    if "hips" in normalized.columns:
        normalized["hips"] = pd.to_numeric(normalized["hips"], errors="coerce")
    else:
        normalized["hips"] = np.nan
    normalized["hips_missing"] = _missing_indicator(normalized["hips"])

    if "bra_size" in normalized.columns:
        normalized["bra_size"] = pd.to_numeric(normalized["bra_size"], errors="coerce")
    else:
        normalized["bra_size"] = np.nan
    normalized["bra_size_missing"] = _missing_indicator(normalized["bra_size"])

    if "cup_size" in normalized.columns:
        normalized["cup_size"] = normalized["cup_size"].map(_normalize_text_category)
    else:
        normalized["cup_size"] = np.nan
    normalized["cup_size_missing"] = _missing_indicator(normalized["cup_size"])

    if "category" in normalized.columns:
        normalized["category"] = normalized["category"].map(_normalize_text_category)
    else:
        normalized["category"] = np.nan

    for column in V3_NUMERIC_FEATURES:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    diagnostics = {
        "height_outlier_count": height_outlier_count,
        "height_plausible_range_cm": [min_height_cm, max_height_cm],
        "feature_policy": {
            "retained": V3_FEATURE_COLUMNS,
            "excluded": [
                "waist",
                "bust",
                "shoe size",
                "shoe width",
                "quality",
                "length",
                "review_summary",
                "review_text",
                "user_id",
                "user_name",
                "item_id",
            ],
            "cup_size": "Categorical raw values are retained; no arbitrary numeric parsing.",
            "size": "Used as item_size_order, representing the selected item size.",
        },
    }
    return normalized, diagnostics


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


def build_fit_v3_inference_contract(feature_columns: list[str]) -> dict[str, Any]:
    """Contrat d'inference V3 experimental, limite aux variables pre-achat."""
    user_profile = []
    item_features = []
    retained_measurements = []

    measurement_fields = {
        "height_cm": "height_cm",
        "hips": "hips",
        "bra_size": "bra_size",
        "cup_size": "cup_size",
    }
    item_fields = {
        "item_size_order": "item_size",
        "category": "category",
    }

    for feature, external_name in measurement_fields.items():
        if feature in feature_columns:
            user_profile.append(external_name)
            retained_measurements.append(external_name)
        missing_feature = f"{feature}_missing"
        if missing_feature in feature_columns:
            user_profile.append(missing_feature)

    if "cup_size_missing" in feature_columns and "cup_size_missing" not in user_profile:
        user_profile.append("cup_size_missing")

    for feature, external_name in item_fields.items():
        if feature in feature_columns:
            item_features.append(external_name)

    return {
        "user_profile": user_profile,
        "item_features": item_features,
        "retained_measurements": retained_measurements,
        "missing_value_indicators": [
            column for column in feature_columns if column.endswith("_missing")
        ],
        "excluded_fields": [
            "waist",
            "bust",
            "shoe_size",
            "shoe_width",
            "quality",
            "length",
            "review_summary",
            "review_text",
            "user_id",
            "user_name",
            "item_id",
            "body_type",
        ],
        "status": "experimental_only",
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


def prepare_fit_training_frame_v3(
    df: pd.DataFrame,
    min_height_cm: float = 130.0,
    max_height_cm: float = 210.0,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Nettoie ModCloth et retourne X/y pour l'experience fit V3."""
    target_before = _count_dict(df["fit"].astype(str).str.lower().str.strip()) if "fit" in df.columns else {}
    normalized, normalization_diagnostics = normalize_modcloth_columns_v3(
        basic_clean_modcloth_dataframe(df),
        min_height_cm=min_height_cm,
        max_height_cm=max_height_cm,
    )

    if "fit" not in normalized.columns:
        raise TabularPreprocessingError("Colonne cible `fit` absente du dataset ModCloth.")

    cleaned = normalized[normalized["fit"].isin(FIT_TARGET_MAPPING.keys())].copy()
    invalid_size_count = int(cleaned["item_size_order"].isna().sum())
    cleaned = cleaned.dropna(subset=["fit", "item_size_order"]).copy()

    if cleaned.empty:
        raise TabularPreprocessingError("Aucune ligne exploitable apres nettoyage ModCloth V3.")

    feature_columns = list(V3_FEATURE_COLUMNS)
    diagnostics = {
        "target_distribution_before_cleaning": target_before,
        "target_distribution_after_cleaning": _count_dict(cleaned["fit"]),
        "numeric_features": list(V3_NUMERIC_FEATURES),
        "categorical_features": list(V3_CATEGORICAL_FEATURES),
        "feature_columns": feature_columns,
        "explicit_clothing_categories": list(EXPLICIT_CLOTHING_CATEGORIES),
        "ambiguous_commercial_categories": list(AMBIGUOUS_COMMERCIAL_CATEGORIES),
        "category_distribution": _count_dict(cleaned["category"].astype(str).str.lower().str.strip()),
        "ambiguous_category_row_count": int(
            cleaned["category"]
            .astype(str)
            .str.lower()
            .str.strip()
            .isin(AMBIGUOUS_COMMERCIAL_CATEGORIES)
            .sum()
        ),
        "invalid_item_size_count": invalid_size_count,
        "missing_values_after_normalization": cleaned[feature_columns + ["fit"]]
        .isna()
        .sum()
        .to_dict(),
        "normalization": normalization_diagnostics,
    }

    return cleaned[feature_columns], cleaned["fit"], diagnostics
