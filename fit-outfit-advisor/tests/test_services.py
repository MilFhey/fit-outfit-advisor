from src.mappings.category_mapping import map_to_common_category
from src.mappings.color_mapping import get_compatible_colors
from src.config.paths import PROJECT_ROOT, FIT_MODEL_PATH, FIT_METADATA_PATH, IMAGE_MODEL_PATH
from src.models.load_fit_model import (
    FitModelArtifacts,
    is_fit_metadata_promoted,
    load_fit_artifacts,
    load_fit_model,
    read_fit_metadata,
)
from src.models.load_image_model import load_image_model
from src.preprocessing.tabular_preprocessing import (
    AMBIGUOUS_COMMERCIAL_CATEGORIES,
    DEFAULT_FEATURE_COLUMNS,
    EXPLICIT_CLOTHING_CATEGORIES,
    V3_FEATURE_COLUMNS,
    build_fit_inference_contract,
    build_fit_v3_inference_contract,
    build_runtime_fit_features,
    prepare_fit_training_frame,
    prepare_fit_training_frame_v3,
)
from src.services.image_service import predict_image
from src.services.fit_service import _predict_with_artifacts, predict_fit
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
    assert FIT_MODEL_PATH.parent.name == "fit_active"
    assert FIT_METADATA_PATH.name == "metadata.json"
    assert IMAGE_MODEL_PATH.is_absolute()
    assert load_fit_model() is None
    assert load_fit_artifacts() is None
    assert load_image_model() is None


def test_modcloth_v2_feature_contract_excludes_incoherent_placeholders():
    assert "weight_kg" not in DEFAULT_FEATURE_COLUMNS
    assert "usual_size" not in DEFAULT_FEATURE_COLUMNS
    assert "brand" not in DEFAULT_FEATURE_COLUMNS
    assert "color" not in DEFAULT_FEATURE_COLUMNS
    assert "height_cm_missing" in DEFAULT_FEATURE_COLUMNS


def test_modcloth_category_groups_are_separated():
    assert set(EXPLICIT_CLOTHING_CATEGORIES) == {"tops", "dresses", "bottoms", "outerwear", "wedding"}
    assert set(AMBIGUOUS_COMMERCIAL_CATEGORIES) == {"new", "sale"}
    assert set(EXPLICIT_CLOTHING_CATEGORIES).isdisjoint(AMBIGUOUS_COMMERCIAL_CATEGORIES)


def test_runtime_fit_features_include_missing_measurement_indicator():
    features = build_runtime_fit_features({}, {"item_size": "M", "category": "tops"})
    assert features.loc[0, "height_cm_missing"] == 1
    assert "body_type" in features.columns


def test_inference_contract_reflects_actual_feature_columns():
    contract_without_body_type = build_fit_inference_contract(["height_cm", "height_cm_missing", "item_size_order"])
    assert "body_type" not in contract_without_body_type["user_profile"]

    contract_with_body_type = build_fit_inference_contract(["height_cm", "item_size_order", "body_type"])
    assert "body_type" in contract_with_body_type["user_profile"]


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
    assert features["height_cm_missing"].sum() == 1
    assert diagnostics["feature_columns"] == list(features.columns)
    assert diagnostics["ambiguous_category_row_count"] == 1


def test_modcloth_v3_preprocessing_keeps_only_pre_purchase_features():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "fit": "fit",
                "height": "5ft 5in",
                "size": 8,
                "category": "tops",
                "hips": 38,
                "bra size": 34,
                "cup size": "c",
                "quality": 5,
                "review_text": "post purchase",
                "item_id": 123,
            },
            {
                "fit": "small",
                "height": "7ft 11in",
                "size": 12,
                "category": "new",
                "hips": None,
                "bra size": 36,
                "cup size": "dd/e",
                "quality": 3,
                "review_text": "post purchase",
                "item_id": 456,
            },
            {
                "fit": "large",
                "height": "3ft",
                "size": 20,
                "category": "dresses",
                "hips": 44,
                "bra size": None,
                "cup size": None,
                "quality": 4,
                "review_text": "post purchase",
                "item_id": 789,
            },
        ]
    )

    features, target, diagnostics = prepare_fit_training_frame_v3(frame)

    assert list(features.columns) == V3_FEATURE_COLUMNS
    assert list(target) == ["fit", "small", "large"]
    assert features["height_cm_missing"].sum() == 2
    assert features["hips_missing"].sum() == 1
    assert features["bra_size_missing"].sum() == 1
    assert features["cup_size_missing"].sum() == 1
    assert features.loc[1, "cup_size"] == "dd/e"
    assert "quality" not in features.columns
    assert "review_text" not in features.columns
    assert "item_id" not in features.columns
    assert diagnostics["normalization"]["height_outlier_count"] == 2
    assert diagnostics["ambiguous_category_row_count"] == 1


def test_modcloth_v3_inference_contract_is_experimental_and_excludes_ids():
    contract = build_fit_v3_inference_contract(V3_FEATURE_COLUMNS)

    assert contract["status"] == "experimental_only"
    assert "height_cm" in contract["user_profile"]
    assert "hips" in contract["user_profile"]
    assert "bra_size" in contract["user_profile"]
    assert "cup_size" in contract["user_profile"]
    assert "item_size" in contract["item_features"]
    assert "category" in contract["item_features"]
    assert "item_id" in contract["excluded_fields"]
    assert "user_id" in contract["excluded_fields"]
    assert set(contract["missing_value_indicators"]) == {
        "height_cm_missing",
        "hips_missing",
        "bra_size_missing",
        "cup_size_missing",
    }


def test_fit_metadata_v2_true_but_experimental_is_refused():
    metadata = {
        "model_status": "experimental_only",
        "promotable_to_streamlit": True,
        "class_labels": ["large", "fit", "small"],
    }
    assert is_fit_metadata_promoted(metadata) is False

    artifacts = FitModelArtifacts(
        model=FakeFitModel([[0.1, 0.8, 0.1]]),
        preprocessor=FakePreprocessor(),
        label_encoder=FakeLabelEncoder(),
        metadata=metadata,
    )
    result = _predict_with_artifacts({"height_cm": 175}, {"item_size": "M"}, artifacts)
    assert result["fit_prediction"] == "uncertain"
    assert result["mode"] == "tensorflow"


def test_fit_metadata_missing_or_unreadable_is_refused(tmp_path):
    missing_metadata = tmp_path / "metadata.json"
    assert read_fit_metadata(missing_metadata) is None
    assert is_fit_metadata_promoted(read_fit_metadata(missing_metadata)) is False

    model_path = tmp_path / "fit_model.keras"
    preprocessor_path = tmp_path / "fit_preprocessor.joblib"
    metadata_path = tmp_path / "metadata.json"
    model_path.write_text("not a real keras model", encoding="utf-8")
    preprocessor_path.write_text("not a real preprocessor", encoding="utf-8")
    metadata_path.write_text("{bad json", encoding="utf-8")

    assert load_fit_artifacts(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        metadata_path=metadata_path,
    ) is None


def test_fit_metadata_promoted_true_is_authorized():
    metadata = {
        "model_status": "promoted",
        "promotable_to_streamlit": True,
        "feature_columns": ["height_cm", "height_cm_missing", "item_size_order"],
        "class_labels": ["large", "fit", "small"],
        "abstention_strategy": {"minimum_confidence": 0.60},
    }
    assert is_fit_metadata_promoted(metadata) is True

    artifacts = FitModelArtifacts(
        model=FakeFitModel([[0.05, 0.9, 0.05]]),
        preprocessor=FakePreprocessor(),
        label_encoder=FakeLabelEncoder(),
        metadata=metadata,
    )
    result = _predict_with_artifacts({"height_cm": 175}, {"item_size": "M"}, artifacts)
    assert result["fit_prediction"] == "fit"
    assert result["confidence"] == 0.9
    assert result["mode"] == "tensorflow"


def test_fit_promoted_low_confidence_returns_uncertain():
    metadata = {
        "model_status": "promoted",
        "promotable_to_streamlit": True,
        "feature_columns": ["height_cm", "height_cm_missing", "item_size_order"],
        "class_labels": ["large", "fit", "small"],
        "abstention_strategy": {"minimum_confidence": 0.60},
    }
    artifacts = FitModelArtifacts(
        model=FakeFitModel([[0.20, 0.55, 0.25]]),
        preprocessor=FakePreprocessor(),
        label_encoder=FakeLabelEncoder(),
        metadata=metadata,
    )
    result = _predict_with_artifacts({"height_cm": 175}, {"item_size": "M"}, artifacts)
    assert result["fit_prediction"] == "uncertain"
    assert result["raw_fit_prediction"] == "fit"
    assert result["confidence"] == 0.55


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


class FakePreprocessor:
    def transform(self, features):
        return features


class FakeFitModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, transformed, verbose=0):
        import numpy as np

        return np.array(self.probabilities)


class FakeLabelEncoder:
    labels = ["large", "fit", "small"]

    def inverse_transform(self, indexes):
        return [self.labels[index] for index in indexes]
