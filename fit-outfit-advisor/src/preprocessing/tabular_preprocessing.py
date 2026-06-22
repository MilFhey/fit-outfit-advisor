from __future__ import annotations

import re
from typing import Any

import pandas as pd


FIT_TARGET_MAPPING = {
    "small": 0,
    "fit": 1,
    "large": 2,
}

INVERSE_FIT_TARGET_MAPPING = {v: k for k, v in FIT_TARGET_MAPPING.items()}

FIT_LABELS = tuple(FIT_TARGET_MAPPING.keys())
DEFAULT_NUMERIC_FEATURES = ["height_cm", "weight_kg", "item_size_order"]
DEFAULT_CATEGORICAL_FEATURES = [
    "body_type",
    "usual_size",
    "item_size",
    "category",
    "brand",
    "color",
]
DEFAULT_FEATURE_COLUMNS = DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES


class TabularPreprocessingError(ValueError):
    """Erreur explicite pour les donnees tabulaires ModCloth invalides."""


def basic_clean_modcloth_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage minimal pour ModCloth.

    Cette fonction sert de point de départ. Le notebook d'entraînement devra l'enrichir
    selon les colonnes réellement présentes dans le dataset.
    """
    df = df.copy()
    df = df.drop_duplicates()

    if "fit" in df.columns:
        df["fit"] = df["fit"].astype(str).str.lower().str.strip()
        df = df[df["fit"].isin(FIT_TARGET_MAPPING.keys())]

    return df


def convert_size_to_order(size: Any) -> int:
    """Encode une taille textuelle simple en ordre numérique."""
    order = {
        "XS": 0,
        "S": 1,
        "M": 2,
        "L": 3,
        "XL": 4,
        "XXL": 5,
    }
    return order.get(str(size).upper(), -1)


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
    """Convertit un poids ModCloth courant (`140lbs`, `63 kg`) en kg."""
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
    """Normalise quelques noms de colonnes ModCloth vers les contrats internes."""
    rename_map = {
        "body type": "body_type",
        "height": "height_cm",
        "weight": "weight_kg",
        "size": "item_size",
        "item_size": "item_size",
        "category": "category",
        "fit": "fit",
    }
    normalized = df.rename(columns={src: dst for src, dst in rename_map.items() if src in df.columns}).copy()

    if "height_cm" in normalized.columns:
        normalized["height_cm"] = normalized["height_cm"].map(parse_height_to_cm)

    if "weight_kg" in normalized.columns:
        normalized["weight_kg"] = normalized["weight_kg"].map(parse_weight_to_kg)

    if "item_size" in normalized.columns:
        normalized["item_size"] = normalized["item_size"].astype(str).str.upper().str.strip()
        normalized["item_size_order"] = normalized["item_size"].map(convert_size_to_order)

    for column in DEFAULT_CATEGORICAL_FEATURES:
        if column not in normalized.columns:
            normalized[column] = "unknown"
        normalized[column] = normalized[column].fillna("unknown").astype(str).str.strip()

    return normalized


def build_runtime_fit_features(user_profile: dict, item_features: dict) -> pd.DataFrame:
    """Construit une ligne d'inference compatible avec le futur preprocessor sklearn."""
    usual_size = user_profile.get("usual_size", "unknown")
    item_size = item_features.get("item_size", "unknown")
    record = {
        "height_cm": float(user_profile.get("height_cm") or 0),
        "weight_kg": float(user_profile.get("weight_kg") or 0),
        "item_size_order": convert_size_to_order(item_size),
        "body_type": user_profile.get("body_type", "unknown") or "unknown",
        "usual_size": str(usual_size).upper(),
        "item_size": str(item_size).upper(),
        "category": item_features.get("category", "unknown") or "unknown",
        "brand": item_features.get("brand", "unknown") or "unknown",
        "color": item_features.get("color", "unknown") or "unknown",
    }
    return pd.DataFrame([record], columns=DEFAULT_FEATURE_COLUMNS)


def prepare_fit_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Nettoie ModCloth et retourne X/y avec colonnes stables pour l'entrainement."""
    cleaned = normalize_modcloth_columns(basic_clean_modcloth_dataframe(df))
    missing = [column for column in ["fit", *DEFAULT_NUMERIC_FEATURES] if column not in cleaned.columns]
    if missing:
        raise TabularPreprocessingError(
            "Colonnes ModCloth manquantes apres normalisation: " + ", ".join(missing)
        )

    cleaned = cleaned.dropna(subset=["fit", *DEFAULT_NUMERIC_FEATURES]).copy()
    cleaned = cleaned[cleaned["item_size_order"] >= 0]
    if cleaned.empty:
        raise TabularPreprocessingError("Aucune ligne exploitable apres nettoyage ModCloth.")

    return cleaned[DEFAULT_FEATURE_COLUMNS], cleaned["fit"]
