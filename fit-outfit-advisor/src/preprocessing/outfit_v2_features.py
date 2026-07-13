from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image


MOBILENET_V2_EMBEDDING_DIM = 1280
OUTFIT_V2_NUMERIC_FEATURES = [
    "embedding_cosine_similarity",
    "embedding_l2_distance",
    "input_red",
    "input_green",
    "input_blue",
    "candidate_red",
    "candidate_green",
    "candidate_blue",
    "input_hue",
    "input_saturation",
    "input_value",
    "candidate_hue",
    "candidate_saturation",
    "candidate_value",
    "color_harmony_score",
    "cooccurrence_score",
]
OUTFIT_V2_CATEGORICAL_FEATURES = [
    "input_product_type",
    "input_canonical_category",
    "input_outfit_role",
    "input_color_family",
    "candidate_product_type",
    "candidate_canonical_category",
    "candidate_outfit_role",
    "candidate_color_family",
]
OUTFIT_V2_FORBIDDEN_DIRECT_FEATURES = ["item_id", "outfit_id", "set_id"]

NEUTRAL_COLOR_FAMILIES = {"black", "white", "gray", "beige", "brown"}
COLOR_FAMILY_HUES = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "blue": 220.0,
    "violet": 275.0,
    "pink": 330.0,
}


@dataclass(frozen=True)
class ImageVisualFeatures:
    embedding: np.ndarray
    dominant_rgb: tuple[int, int, int]
    hsv: tuple[float, float, float]
    color_family: str


def _as_rgb_image(image: Image.Image | Any) -> Image.Image:
    if image is None:
        raise ValueError("Image absente pour l'extraction Outfit V2.")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def extract_dominant_rgb(
    image: Image.Image | Any,
    *,
    resize: tuple[int, int] = (96, 96),
    sample_pixels: int = 2048,
    random_state: int = 42,
) -> tuple[int, int, int]:
    rgb_image = _as_rgb_image(image).resize(resize)
    pixels = np.asarray(rgb_image, dtype=np.uint8).reshape(-1, 3)
    # Ignore almost-transparent-looking white backgrounds common in product shots.
    non_white = pixels[np.mean(pixels, axis=1) < 245]
    if len(non_white) >= 32:
        pixels = non_white
    if len(pixels) > sample_pixels:
        rng = np.random.default_rng(random_state)
        pixels = pixels[rng.choice(len(pixels), size=sample_pixels, replace=False)]

    try:
        from sklearn.cluster import MiniBatchKMeans

        clusters = min(3, len(pixels))
        kmeans = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=random_state,
            batch_size=min(512, len(pixels)),
            n_init="auto",
        )
        labels = kmeans.fit_predict(pixels)
        counts = np.bincount(labels)
        dominant = kmeans.cluster_centers_[int(counts.argmax())]
    except Exception:
        dominant = pixels.mean(axis=0)

    clipped = np.clip(np.rint(dominant), 0, 255).astype(int)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def rgb_to_hsv01(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    red, green, blue = [channel / 255.0 for channel in rgb]
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    return float(hue), float(saturation), float(value)


def classify_color_family(rgb: tuple[int, int, int]) -> str:
    hue, saturation, value = rgb_to_hsv01(rgb)
    if value < 0.16:
        return "black"
    if saturation < 0.12 and value > 0.86:
        return "white"
    if saturation < 0.16:
        return "gray"
    hue_degrees = hue * 360.0
    red, green, blue = rgb
    if saturation < 0.32 and red > green > blue:
        return "beige"
    if value < 0.45 and 15 <= hue_degrees <= 55:
        return "brown"
    if hue_degrees < 15 or hue_degrees >= 345:
        return "red"
    if hue_degrees < 45:
        return "orange"
    if hue_degrees < 75:
        return "yellow"
    if hue_degrees < 165:
        return "green"
    if hue_degrees < 255:
        return "blue"
    if hue_degrees < 300:
        return "violet"
    return "pink"


def _hue_distance_degrees(left: float, right: float) -> float:
    diff = abs(left - right) % 360.0
    return min(diff, 360.0 - diff)


def color_harmony_score(left_family: str, right_family: str) -> float:
    if left_family == "unknown" or right_family == "unknown":
        return 0.50
    if left_family == right_family:
        return 0.74 if left_family in NEUTRAL_COLOR_FAMILIES else 0.62
    if left_family in NEUTRAL_COLOR_FAMILIES or right_family in NEUTRAL_COLOR_FAMILIES:
        return 0.86
    left_hue = COLOR_FAMILY_HUES.get(left_family)
    right_hue = COLOR_FAMILY_HUES.get(right_family)
    if left_hue is None or right_hue is None:
        return 0.50
    distance = _hue_distance_degrees(left_hue, right_hue)
    if distance <= 45:
        return 0.80
    if 135 <= distance <= 210:
        return 0.76
    if distance <= 90:
        return 0.64
    return 0.42


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype="float32")
    right = np.asarray(right, dtype="float32")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denominator)


def l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype="float32") - np.asarray(right, dtype="float32")))


@lru_cache(maxsize=1)
def load_mobilenet_v2_embedding_model():
    import tensorflow as tf

    return tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(224, 224, 3),
    )


def preprocess_for_mobilenet_embedding(image: Image.Image | Any) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    rgb_image = _as_rgb_image(image).resize((224, 224))
    array = np.asarray(rgb_image, dtype="float32")
    return np.expand_dims(preprocess_input(array), axis=0)


def extract_mobilenet_embedding(image: Image.Image | Any, model: Any | None = None) -> np.ndarray:
    model = model or load_mobilenet_v2_embedding_model()
    embedding = model.predict(preprocess_for_mobilenet_embedding(image), verbose=0)[0]
    return np.asarray(embedding, dtype="float32")


def extract_visual_features(
    image: Image.Image | Any,
    *,
    embedding_model: Any | None = None,
    zero_embedding: bool = False,
) -> ImageVisualFeatures:
    dominant_rgb = extract_dominant_rgb(image)
    hsv = rgb_to_hsv01(dominant_rgb)
    color_family = classify_color_family(dominant_rgb)
    if zero_embedding:
        embedding = np.zeros(MOBILENET_V2_EMBEDDING_DIM, dtype="float32")
    else:
        embedding = extract_mobilenet_embedding(image, model=embedding_model)
    return ImageVisualFeatures(
        embedding=embedding,
        dominant_rgb=dominant_rgb,
        hsv=hsv,
        color_family=color_family,
    )


def empty_visual_features(color_family: str = "unknown") -> ImageVisualFeatures:
    return ImageVisualFeatures(
        embedding=np.zeros(MOBILENET_V2_EMBEDDING_DIM, dtype="float32"),
        dominant_rgb=(0, 0, 0),
        hsv=(0.0, 0.0, 0.0),
        color_family=color_family,
    )


def build_pair_numeric_features(
    input_features: ImageVisualFeatures,
    candidate_features: ImageVisualFeatures,
    *,
    cooccurrence_score: float = 0.0,
) -> list[float]:
    input_hsv = input_features.hsv
    candidate_hsv = candidate_features.hsv
    return [
        cosine_similarity(input_features.embedding, candidate_features.embedding),
        l2_distance(input_features.embedding, candidate_features.embedding),
        *(channel / 255.0 for channel in input_features.dominant_rgb),
        *(channel / 255.0 for channel in candidate_features.dominant_rgb),
        *input_hsv,
        *candidate_hsv,
        color_harmony_score(input_features.color_family, candidate_features.color_family),
        float(cooccurrence_score),
    ]


def build_pair_embedding_block(
    input_features: ImageVisualFeatures,
    candidate_features: ImageVisualFeatures,
) -> np.ndarray:
    input_embedding = np.asarray(input_features.embedding, dtype="float32")
    candidate_embedding = np.asarray(candidate_features.embedding, dtype="float32")
    return np.concatenate(
        [
            input_embedding,
            candidate_embedding,
            np.abs(input_embedding - candidate_embedding),
        ]
    ).astype("float32")


def validate_no_forbidden_v2_features(feature_names: list[str]) -> None:
    joined = " ".join(feature_names)
    forbidden = [feature for feature in OUTFIT_V2_FORBIDDEN_DIRECT_FEATURES if feature in joined]
    if forbidden:
        raise ValueError(f"Forbidden direct Outfit V2 features present: {forbidden}")
