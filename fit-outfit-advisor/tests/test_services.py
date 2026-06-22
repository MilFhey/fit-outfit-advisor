from src.mappings.category_mapping import map_to_common_category
from src.mappings.color_mapping import get_compatible_colors
from src.config.paths import PROJECT_ROOT, FIT_MODEL_PATH, IMAGE_MODEL_PATH
from src.models.load_fit_model import load_fit_artifacts, load_fit_model
from src.models.load_image_model import load_image_model
from src.preprocessing.tabular_preprocessing import (
    DEFAULT_FEATURE_COLUMNS,
    prepare_fit_training_frame,
)
from src.services.image_service import predict_image
from src.services.fit_service import predict_fit
from src.services.outfit_service import recommend_outfit
from src.services.advice_service import generate_advice


def test_category_mapping_known_and_unknown():
    assert map_to_common_category("Tshirts") == "top"
    assert map_to_common_category("Jeans") == "bottom"
    assert map_to_common_category("Unmapped") == "unknown"
    assert map_to_common_category("") == "unknown"


def test_color_mapping_fallback():
    assert "blanc" in get_compatible_colors("noir")
    assert get_compatible_colors("couleur-inconnue") == ["noir", "blanc", "beige"]


def test_image_service_simulated_output_keys():
    image_result = predict_image(None)
    assert image_result["predicted_class"] == "Tshirts"
    assert image_result["common_category"] == "top"
    assert image_result["mode"] == "simulation"
    assert {"predicted_class", "common_category", "confidence", "mode"} <= set(image_result)


def test_image_service_real_mode_falls_back_without_model():
    image_result = predict_image(None, use_real_model=True)
    assert image_result["predicted_class"] == "Tshirts"
    assert image_result["mode"] == "simulation"
    assert "fallback_reason" in image_result


def test_model_paths_and_missing_loaders_do_not_crash():
    assert PROJECT_ROOT.name == "fit-outfit-advisor"
    assert FIT_MODEL_PATH.is_absolute()
    assert IMAGE_MODEL_PATH.is_absolute()
    assert load_fit_model() is None
    assert load_fit_artifacts() is None
    assert load_image_model() is None


def test_modcloth_v2_feature_contract_excludes_incoherent_placeholders():
    assert "weight_kg" not in DEFAULT_FEATURE_COLUMNS
    assert "usual_size" not in DEFAULT_FEATURE_COLUMNS
    assert "brand" not in DEFAULT_FEATURE_COLUMNS
    assert "color" not in DEFAULT_FEATURE_COLUMNS


def test_modcloth_preprocessing_keeps_missing_values_for_imputer():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"fit": "fit", "height": None, "body type": "hourglass", "size": 8, "category": "new"},
            {"fit": "small", "height": "5ft 7in", "body type": None, "size": 4, "category": "dresses"},
            {"fit": "large", "height": "5ft 2in", "body type": "petite", "size": 16, "category": "tops"},
        ]
    )

    features, target, diagnostics = prepare_fit_training_frame(frame)

    assert list(target) == ["fit", "small", "large"]
    assert "weight_kg" not in features.columns
    assert features["height_cm"].isna().sum() == 1
    assert diagnostics["feature_columns"] == list(features.columns)


def test_fit_service_fallback_output_keys_without_real_artifacts():
    user_profile = {
        "height_cm": 175,
        "weight_kg": 75,
        "usual_size": "M",
    }
    item_features = {
        "item_size": "M",
        "brand": "Test Brand",
        "color": "noir",
    }

    fit_result = predict_fit(user_profile, item_features, use_real_model=True)
    assert fit_result["fit_prediction"] == "fit"
    assert fit_result["mode"] == "simulation"
    assert {"fit_prediction", "confidence", "risk_level", "reason", "mode"} <= set(fit_result)


def test_outfit_service_output_keys():
    outfit_result = recommend_outfit("top", "casual", "noir")
    assert outfit_result["compatibility_score"] > 0
    assert outfit_result["compatible_items"]
    assert outfit_result["compatible_colors"]
    assert {"compatible_items", "compatible_colors", "compatibility_score", "reason", "mode"} <= set(outfit_result)


def test_advice_service_output_keys():
    image_result = predict_image(None)
    fit_result = predict_fit(
        {"height_cm": 175, "weight_kg": 75, "usual_size": "M"},
        {"item_size": "M", "brand": "Test Brand", "color": "noir"},
        use_real_model=False,
    )
    outfit_result = recommend_outfit(image_result["common_category"], "casual", "noir")
    advice = generate_advice(image_result, fit_result, outfit_result, "casual")

    assert {"advice", "warnings", "mode"} <= set(advice)
    assert "Conseil final" in advice["advice"]


def test_mvp_services_pipeline():
    image_result = predict_image(None)
    fit_result = predict_fit(
        {"height_cm": 175, "weight_kg": 75, "usual_size": "M"},
        {"item_size": "M", "brand": "Test Brand", "color": "noir"},
    )
    outfit_result = recommend_outfit(image_result["common_category"], "casual", "noir")
    advice = generate_advice(image_result, fit_result, outfit_result, "casual")

    assert image_result["predicted_class"] == "Tshirts"
    assert fit_result["fit_prediction"] == "fit"
    assert outfit_result["compatibility_score"] > 0
    assert "Conseil final" in advice["advice"]
