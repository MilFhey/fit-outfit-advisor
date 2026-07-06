from src.mappings.category_mapping import map_to_common_category
import pytest

from src.mappings.color_mapping import get_compatible_colors
from src.config.paths import (
    FASHION_ACTIVE_DIR,
    FASHION_METADATA_PATH,
    FASHION_MODEL_PATH,
    FASHION_V1_CLASSES_PATH,
    PROJECT_ROOT,
    FIT_MODEL_PATH,
    FIT_METADATA_PATH,
    IMAGE_MODEL_PATH,
)
from src.models.load_fit_model import (
    FitModelArtifacts,
    is_fit_metadata_promoted,
    load_fit_artifacts,
    load_fit_model,
    read_fit_metadata,
)
from src.models.load_image_model import (
    is_image_metadata_promoted,
    load_image_model,
    read_image_metadata,
)
from src.analysis.analyze_fit_v3_abstention import (
    UNCERTAIN_LABEL,
    apply_abstention,
    evaluate_abstention,
    evaluate_thresholds,
    select_threshold,
)
from src.mappings.fashion_v1_mapping import (
    FashionClassConfigError,
    FASHION_CANONICAL_CATEGORIES,
    FASHION_PRODUCT_TYPES_V0,
    load_fashion_v1_class_config,
    map_article_type_to_canonical_category,
    map_article_type_to_product_type,
    map_product_type_to_canonical_category,
    validate_fashion_v1_class_config,
)
from src.preprocessing.image_preprocessing import (
    MOBILENET_V2_ARCHITECTURE,
    SIMPLE_CNN_ARCHITECTURE,
    get_image_preprocessing_mode,
    preprocess_image_for_cnn,
)
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
from src.training.train_fashion_model_v1 import prepare_fashion_v1_training_frame


def test_category_mapping_known_and_unknown():
    assert map_to_common_category("Tshirts") == "top"
    assert map_to_common_category("Jeans") == "bottom"
    assert map_to_common_category("tshirt") == "top"
    assert map_to_common_category("casual_shoes") == "shoes"
    assert map_to_common_category("bag") == "bag"
    assert map_to_common_category("top") == "top"
    assert map_to_common_category("accessory") == "accessory"
    assert map_to_common_category("Unmapped") == "unknown"
    assert map_to_common_category("") == "unknown"


def test_fashion_v1_class_config_is_validated_after_dataset_inspection():
    config = load_fashion_v1_class_config()

    assert FASHION_V1_CLASSES_PATH.name == "fashion_v1_classes.json"
    assert config["target"] == "product_type_v0"
    assert config["source_column"] == "articleType"
    assert config["status"] == "validated_for_training"
    assert config["minimum_readable_images_per_class"] == 450
    assert set(config["product_type_mapping"]) == set(FASHION_PRODUCT_TYPES_V0)
    assert set(config["canonical_mapping"]).issuperset(config["product_type_mapping"])
    assert set(config["canonical_mapping"].values()) <= set(FASHION_CANONICAL_CATEGORIES)
    assert "cap" not in config["product_type_mapping"]
    assert "Caps" not in {
        article_type
        for article_types in config["product_type_mapping"].values()
        for article_type in article_types
    }

    validate_fashion_v1_class_config(config)
    validate_fashion_v1_class_config(config, require_ready=True)


def test_fashion_article_type_to_product_type_then_canonical_mapping():
    config = {
        "target": "product_type_v0",
        "source_column": "articleType",
        "status": "validated_for_training",
        "minimum_readable_images_per_class": 100,
        "product_type_mapping": {
            "tshirt": ["Tshirts"],
            "shirt": ["Shirts"],
            "jeans": ["Jeans"],
            "dress": ["Dresses"],
            "casual_shoes": ["Casual Shoes"],
            "sandals": ["Sandals"],
            "flip_flops": ["Flip Flops"],
            "heels": ["Heels"],
            "flats": ["Flats"],
            "outerwear": ["Jackets"],
            "bag": ["Handbags"],
            "watch": ["Watches"],
            "wallet": ["Wallets"],
            "belt": ["Belts"],
            "jewellery": ["Earrings"],
        },
        "canonical_mapping": {
            "tshirt": "top",
            "shirt": "top",
            "jeans": "bottom",
            "dress": "dress",
            "casual_shoes": "shoes",
            "sandals": "shoes",
            "flip_flops": "shoes",
            "heels": "shoes",
            "flats": "shoes",
            "outerwear": "outerwear",
            "bag": "bag",
            "watch": "accessory",
            "wallet": "accessory",
            "belt": "accessory",
            "jewellery": "accessory",
        },
    }

    validate_fashion_v1_class_config(config, require_ready=True)
    assert map_article_type_to_product_type("Tshirts", config) == "tshirt"
    assert map_article_type_to_product_type("Jeans", config) == "jeans"
    assert map_article_type_to_product_type("Sandals", config) == "sandals"
    assert map_article_type_to_product_type("Flip Flops", config) == "flip_flops"
    assert map_article_type_to_product_type("Wallets", config) == "wallet"
    assert map_article_type_to_product_type("Earrings", config) == "jewellery"
    assert map_article_type_to_canonical_category("Tshirts", config) == "top"
    assert map_article_type_to_canonical_category("Jeans", config) == "bottom"
    assert map_article_type_to_canonical_category("Heels", config) == "shoes"
    assert map_article_type_to_canonical_category("Belts", config) == "accessory"
    assert map_product_type_to_canonical_category("watch", config) == "accessory"
    assert map_article_type_to_canonical_category("Unknown", config) is None


def test_fashion_training_refuses_draft_config():
    config = {
        "target": "product_type_v0",
        "source_column": "articleType",
        "status": "draft_requires_dataset_inspection",
        "minimum_readable_images_per_class": None,
        "product_type_mapping": {"tshirt": ["Tshirts"]},
        "canonical_mapping": {"tshirt": "top"},
    }

    with pytest.raises(FashionClassConfigError):
        prepare_fashion_v1_training_frame(
            metadata_csv=PROJECT_ROOT / "missing_styles.csv",
            image_dir=PROJECT_ROOT / "missing_images",
            class_config=config,
        )


def test_fashion_training_refuses_missing_dataset_after_valid_config():
    config = load_fashion_v1_class_config()

    with pytest.raises(FileNotFoundError):
        prepare_fashion_v1_training_frame(
            metadata_csv=PROJECT_ROOT / "missing_styles.csv",
            image_dir=PROJECT_ROOT / "missing_images",
            class_config=config,
        )


def test_color_mapping_fallback():
    assert "blanc" in get_compatible_colors("noir")
    assert get_compatible_colors("couleur-inconnue") == ["noir", "blanc", "beige"]


def test_category_mapping_preserves_fashion_v1_granular_roles():
    assert map_to_common_category("Flip Flops") == "shoes"
    assert map_to_common_category("Sandals") == "shoes"
    assert map_to_common_category("Wallets") == "accessory"
    assert map_to_common_category("Earrings") == "accessory"
    assert map_to_common_category("Handbags") == "bag"
    assert map_to_common_category("Sweatshirts") == "outerwear"


def test_image_service_simulated_output_keys():
    image_result = predict_image(None)
    assert image_result["product_type"] == "tshirt"
    assert image_result["canonical_category"] == "top"
    assert image_result["predicted_class"] == "tshirt"
    assert image_result["common_category"] == "top"
    assert image_result["model_status"] == "fallback"
    assert image_result["mode"] == "simulation"
    assert {
        "product_type",
        "canonical_category",
        "predicted_class",
        "common_category",
        "confidence",
        "model_status",
        "mode",
    } <= set(image_result)


def test_image_service_real_mode_falls_back_without_model():
    image_result = predict_image(None, use_real_model=True)
    assert image_result["product_type"] == "tshirt"
    assert image_result["mode"] == "simulation"
    assert "fallback_reason" in image_result


def test_image_metadata_fail_closed_and_active_path():
    assert FASHION_ACTIVE_DIR.name == "fashion_active"
    assert FASHION_MODEL_PATH.parent == FASHION_ACTIVE_DIR
    assert IMAGE_MODEL_PATH == FASHION_MODEL_PATH
    assert FASHION_METADATA_PATH.name == "metadata.json"
    assert read_image_metadata() is None
    assert is_image_metadata_promoted(None) is False
    assert is_image_metadata_promoted(
        {"model_status": "experimental_only", "promotable_to_streamlit": True}
    ) is False
    assert is_image_metadata_promoted(
        {"model_status": "promoted", "promotable_to_streamlit": True}
    ) is True


def test_image_preprocessing_modes_are_not_double_normalized():
    from PIL import Image

    assert get_image_preprocessing_mode(SIMPLE_CNN_ARCHITECTURE) == "rescale_1_over_255"
    assert (
        get_image_preprocessing_mode(MOBILENET_V2_ARCHITECTURE)
        == "mobilenet_v2_preprocess_input"
    )

    image = Image.new("RGB", (2, 2), color=(255, 255, 255))
    batch = preprocess_image_for_cnn(image, architecture=SIMPLE_CNN_ARCHITECTURE)
    assert batch.shape == (1, 128, 128, 3)
    assert float(batch.max()) == 1.0


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


def test_abstention_below_threshold_returns_uncertain():
    import numpy as np

    result = apply_abstention(
        np.array([[0.20, 0.55, 0.25]]),
        ["large", "fit", "small"],
        threshold=0.60,
    )

    assert result["raw_predictions"] == ["fit"]
    assert result["predictions"] == [UNCERTAIN_LABEL]
    assert result["confidences"] == [0.55]


def test_abstention_at_threshold_keeps_prediction():
    import numpy as np

    result = apply_abstention(
        np.array([[0.20, 0.60, 0.20]]),
        ["large", "fit", "small"],
        threshold=0.60,
    )

    assert result["raw_predictions"] == ["fit"]
    assert result["predictions"] == ["fit"]
    assert result["confidences"] == [0.60]


def test_abstention_metrics_compute_coverage_and_confusion():
    metrics = evaluate_abstention(
        ["fit", "large", "small", "small"],
        ["fit", UNCERTAIN_LABEL, "large", UNCERTAIN_LABEL],
        ["fit", "large", "small"],
    )

    assert metrics["total_count"] == 4
    assert metrics["covered_count"] == 2
    assert metrics["uncertain_count"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["abstention_rate"] == 0.5
    assert metrics["accuracy_non_abstained"] == 0.5
    assert metrics["confusion_labels"] == ["fit", "large", "small", UNCERTAIN_LABEL]
    assert metrics["abstention_by_true_class"]["large"]["abstention_rate"] == 1.0
    assert metrics["abstention_by_true_class"]["small"]["abstention_rate"] == 0.5


def test_threshold_selection_uses_validation_rows_only():
    import numpy as np

    y_true = ["small", "large", "small", "large", "fit", "fit", "small", "large"]
    probabilities = np.array(
        [
            [0.10, 0.10, 0.80],
            [0.82, 0.10, 0.08],
            [0.20, 0.20, 0.60],
            [0.59, 0.25, 0.16],
            [0.10, 0.70, 0.20],
            [0.10, 0.55, 0.35],
            [0.51, 0.04, 0.45],
            [0.51, 0.24, 0.25],
        ]
    )
    rows = evaluate_thresholds(
        y_true,
        probabilities,
        ["large", "fit", "small"],
        thresholds=[0.50, 0.60],
    )
    selection = select_threshold(
        rows,
        min_coverage=0.25,
        min_precision_small=0.40,
        min_precision_large=0.40,
    )

    assert selection["selected"] is True
    assert selection["selected_threshold"] == 0.60
    assert "validation" in selection["reason"]


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

    assert image_result["product_type"] == "tshirt"
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
