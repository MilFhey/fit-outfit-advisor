from dataclasses import dataclass
from typing import List


@dataclass
class ImagePrediction:
    predicted_class: str
    common_category: str
    confidence: float


@dataclass
class FitPrediction:
    fit_prediction: str
    confidence: float


@dataclass
class OutfitRecommendation:
    compatible_items: List[str]
    compatibility_score: float
    reason: str


@dataclass
class UserProfile:
    height_cm: int
    weight_kg: int
    usual_size: str


@dataclass
class ItemFeatures:
    item_size: str
    brand: str
    color: str
