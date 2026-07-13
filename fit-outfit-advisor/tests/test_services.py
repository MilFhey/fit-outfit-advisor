import json

import pytest

from src.mappings.category_mapping import map_to_common_category
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
from src.models.load_outfit_model import is_outfit_metadata_promoted
from src.analysis.analyze_fit_v3_abstention import (
    UNCERTAIN_LABEL,
    apply_abstention,
    evaluate_abstention,
    evaluate_thresholds,
    select_threshold,
)
from src.analysis.analyze_fashion_v1_abstention import (
    UNKNOWN_LABEL,
    apply_image_abstention,
    evaluate_image_abstention,
    evaluate_thresholds as evaluate_image_thresholds,
    select_threshold as select_image_threshold,
)
from src.analysis.analyze_polyvore_v0_schema_mapping import (
    build_schema_mapping_audit,
    detect_product_type,
)
from src.analysis.build_polyvore_v0_cooccurrence_baseline import (
    build_cooccurrence_baseline,
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
from src.mappings.polyvore_mapping import (
    OutfitConfigError,
    build_compatible_role_pair_set,
    load_outfit_v1_config,
    map_polyvore_label_to_fashion,
    validate_outfit_v1_config,
)
from src.preprocessing.image_preprocessing import (
    MOBILENET_V2_ARCHITECTURE,
    SIMPLE_CNN_ARCHITECTURE,
    get_image_preprocessing_mode,
    preprocess_image_for_cnn,
)
from src.preprocessing.outfit_preprocessing import (
    OUTFIT_PAIR_FEATURE_COLUMNS,
    assert_no_positive_pair_leakage,
    build_outfit_feature_frame,
    build_positive_outfit_pairs,
    exact_item_pair_key,
    generate_hard_negative_pairs,
    split_outfits_by_id,
)
from src.preprocessing.outfit_v2_features import (
    as_rgb_image,
    classify_color_family,
    color_harmony_score,
    encoded_image_bytes,
    extract_dominant_rgb,
    preprocess_for_mobilenet_embedding,
    validate_no_forbidden_v2_features,
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
import src.services.image_service as image_service_module
from src.services.fit_service import _predict_with_artifacts, predict_fit
from src.services.outfit_service import recommend_outfit
import src.services.outfit_v2_service as outfit_v2_service_module
from src.services.outfit_v2_service import (
    evaluate_outfit_images,
    recommend_associations_from_image,
)
from src.services.advice_service import generate_advice
from src.training.train_fashion_model_v1 import (
    prepare_fashion_v1_training_frame,
    select_experiment,
)
from src.training.train_outfit_model_v1 import (
    build_outfit_training_splits,
    select_threshold as select_outfit_threshold,
)
from src.training.train_outfit_model_v2 import parse_args as parse_outfit_v2_args


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
    assert "flats" not in config["product_type_mapping"]
    assert "Flats" in config["product_type_mapping"]["dress_shoes"]
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
            "dress_shoes": ["Formal Shoes", "Flats"],
            "sandals": ["Sandals"],
            "flip_flops": ["Flip Flops"],
            "heels": ["Heels"],
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
            "dress_shoes": "shoes",
            "sandals": "shoes",
            "flip_flops": "shoes",
            "heels": "shoes",
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
    assert map_article_type_to_product_type("Flats", config) == "dress_shoes"
    assert map_article_type_to_product_type("Wallets", config) == "wallet"
    assert map_article_type_to_product_type("Earrings", config) == "jewellery"
    assert map_article_type_to_canonical_category("Tshirts", config) == "top"
    assert map_article_type_to_canonical_category("Jeans", config) == "bottom"
    assert map_article_type_to_canonical_category("Heels", config) == "shoes"
    assert map_article_type_to_canonical_category("Flats", config) == "shoes"
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


def test_fashion_validation_selection_prefers_macro_f1_then_balanced_accuracy():
    selected, reason = select_experiment(
        {
            "simple_cnn": {
                "macro_f1": 0.70,
                "balanced_accuracy": 0.72,
                "accuracy": 0.80,
            },
            "mobilenet_v2": {
                "macro_f1": 0.71,
                "balanced_accuracy": 0.70,
                "accuracy": 0.78,
            },
        }
    )

    assert selected == "mobilenet_v2"
    assert "validation only" in reason


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


def test_outfit_v1_config_is_validated_for_baseline_and_aligned_with_fashion_v1():
    config = load_outfit_v1_config()

    assert config["target"] == "outfit_compatibility_v0"
    assert config["taxonomy_source"] == "fashion_v1"
    assert config["status"] == "validated_for_baseline_v0"
    assert config["source_label_column"] == "semantic_category|category_id_name|catgeories"
    assert set(config["allowed_outfit_roles"]) <= set(FASHION_CANONICAL_CATEGORIES)
    assert {"item_id", "outfit_id"} <= set(
        config["feature_policy"]["forbidden_direct_features"]
    )
    assert "item_id" not in config["feature_policy"]["allowed_features"]
    assert "outfit_id" not in config["feature_policy"]["allowed_features"]
    assert len(config["polyvore_label_mapping"]) >= 20

    validate_outfit_v1_config(config)
    validate_outfit_v1_config(config, require_ready=True)
    assert map_polyvore_label_to_fashion("outerwear", config)["product_type_v0"] == "outerwear"
    assert map_polyvore_label_to_fashion("capri cropped pants", config)["product_type_v0"] == "trousers"
    assert map_polyvore_label_to_fashion("converse chuck taylor all star", config)["product_type_v0"] == "sports_shoes"
    assert map_polyvore_label_to_fashion("skirts", config) is None


def test_polyvore_label_maps_to_fashion_v1_taxonomy():
    config = {
        "target": "outfit_compatibility_v0",
        "taxonomy_source": "fashion_v1",
        "status": "validated_for_training",
        "source_label_column": "category",
        "allowed_outfit_roles": ["top", "bottom", "shoes"],
        "compatible_role_pairs": [["top", "bottom"], ["top", "shoes"]],
        "polyvore_label_mapping": {
            "blouse": {
                "product_type_v0": "shirt",
                "canonical_category": "top",
                "outfit_role": "top",
            },
            "denim pants": {
                "product_type_v0": "jeans",
                "canonical_category": "bottom",
                "outfit_role": "bottom",
            },
        },
        "feature_policy": {
            "forbidden_direct_features": ["item_id", "outfit_id"],
            "allowed_features": OUTFIT_PAIR_FEATURE_COLUMNS,
        },
    }

    validate_outfit_v1_config(config, require_ready=True)
    mapped = map_polyvore_label_to_fashion("blouse", config)
    assert mapped == {
        "polyvore_label": "blouse",
        "product_type_v0": "shirt",
        "canonical_category": "top",
        "outfit_role": "top",
    }
    assert map_polyvore_label_to_fashion("unknown label", config) is None


def test_outfit_positive_pairs_and_hard_negatives_exclude_known_positives():
    import pandas as pd

    config = {
        "compatible_role_pairs": [["top", "bottom"], ["top", "shoes"]],
    }
    items = pd.DataFrame(
        [
            {"outfit_id": "o1", "item_id": "shirt-1", "product_type_v0": "shirt", "canonical_category": "top", "outfit_role": "top"},
            {"outfit_id": "o1", "item_id": "jeans-1", "product_type_v0": "jeans", "canonical_category": "bottom", "outfit_role": "bottom"},
            {"outfit_id": "o2", "item_id": "shirt-2", "product_type_v0": "shirt", "canonical_category": "top", "outfit_role": "top"},
            {"outfit_id": "o2", "item_id": "jeans-2", "product_type_v0": "jeans", "canonical_category": "bottom", "outfit_role": "bottom"},
            {"outfit_id": "o3", "item_id": "jeans-3", "product_type_v0": "jeans", "canonical_category": "bottom", "outfit_role": "bottom"},
        ]
    )

    positives = build_positive_outfit_pairs(items, config)
    known_positive_keys = set(positives["pair_key"])
    negatives = generate_hard_negative_pairs(
        positives,
        items,
        positive_pair_keys=known_positive_keys,
        ratio=1.0,
        seed=7,
    )

    assert not positives.empty
    assert not negatives.empty
    assert set(negatives["label"]) == {0}
    assert set(negatives["pair_key"]).isdisjoint(known_positive_keys)
    assert set(negatives["candidate_outfit_role"]).issubset({"bottom", "top"})


def test_outfit_feature_frame_excludes_item_id_and_outfit_id():
    import pandas as pd

    pairs = pd.DataFrame(
        [
            {
                "outfit_id": "o1",
                "input_item_id": "shirt-1",
                "candidate_item_id": "jeans-1",
                "input_product_type": "shirt",
                "input_canonical_category": "top",
                "input_outfit_role": "top",
                "candidate_product_type": "jeans",
                "candidate_canonical_category": "bottom",
                "candidate_outfit_role": "bottom",
                "pair_key": exact_item_pair_key("shirt-1", "jeans-1"),
                "label": 1,
            }
        ]
    )

    features, target = build_outfit_feature_frame(pairs)

    assert list(features.columns) == OUTFIT_PAIR_FEATURE_COLUMNS
    assert "item_id" not in " ".join(features.columns)
    assert "outfit_id" not in features.columns
    assert list(target) == [1]


def test_outfit_split_groups_by_outfit_and_detects_positive_pair_leakage():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"outfit_id": f"o{index}", "pair_key": exact_item_pair_key(f"a{index}", f"b{index}"), "label": 1}
            for index in range(20)
        ]
    )
    train, validation, test = split_outfits_by_id(frame, seed=42)

    assert set(train["outfit_id"]).isdisjoint(set(validation["outfit_id"]))
    assert set(train["outfit_id"]).isdisjoint(set(test["outfit_id"]))
    assert set(validation["outfit_id"]).isdisjoint(set(test["outfit_id"]))
    assert_no_positive_pair_leakage({"train": train, "validation": validation, "test": test})

    leaked_validation = validation.copy()
    leaked_validation.at[leaked_validation.index[0], "pair_key"] = train.iloc[0]["pair_key"]
    with pytest.raises(ValueError):
        assert_no_positive_pair_leakage(
            {"train": train, "validation": leaked_validation, "test": test}
        )


def test_polyvore_schema_mapping_rules_stay_within_fashion_v1():
    fashion_config = load_fashion_v1_class_config()

    blouse = detect_product_type("women blouse", fashion_config)
    assert blouse["status"] == "mapped"
    assert blouse["product_type_v0"] == "shirt"
    assert blouse["outfit_role"] == "top"

    lipstick = detect_product_type("red lipstick", fashion_config)
    assert lipstick["status"] == "excluded"
    assert "outside" in lipstick["reason"]

    assert detect_product_type("outerwear", fashion_config)["product_type_v0"] == "outerwear"
    assert detect_product_type("hoodies", fashion_config)["product_type_v0"] == "outerwear"
    assert detect_product_type("vests", fashion_config)["product_type_v0"] == "outerwear"
    assert detect_product_type("capri cropped pants", fashion_config)["product_type_v0"] == "trousers"


def test_polyvore_schema_mapping_audit_links_raw_items(tmp_path):
    raw_root = tmp_path / "raw_hf_files"
    (raw_root / "disjoint").mkdir(parents=True)
    (raw_root / "nondisjoint").mkdir(parents=True)
    (raw_root / "polyvore_item_metadata.json").write_text(
        json.dumps(
            {
                "item-1": {
                    "semantic_category": "blouse",
                    "category_id": "10",
                    "catgeories": ["Women", "Tops"],
                    "title": "Silk blouse",
                    "url_name": "silk-blouse",
                },
                "item-2": {
                    "semantic_category": "lipstick",
                    "category_id": "99",
                    "catgeories": ["Beauty"],
                    "title": "Matte lipstick",
                    "url_name": "matte-lipstick",
                },
            }
        ),
        encoding="utf-8",
    )
    (raw_root / "categories.csv").write_text(
        "10,1,Blouses\n99,1,Beauty\n",
        encoding="utf-8",
    )
    split_payload = [
        {
            "set_id": "set-1",
            "items": [
                {"item_id": "item-1", "index": 1},
                {"item_id": "item-2", "index": 2},
            ],
        }
    ]
    (raw_root / "disjoint" / "train.json").write_text(
        json.dumps(split_payload),
        encoding="utf-8",
    )
    (raw_root / "disjoint" / "valid.json").write_text(
        "[]",
        encoding="utf-8",
    )
    for relative_path in [
        "disjoint/test.json",
        "nondisjoint/train.json",
        "nondisjoint/valid.json",
        "nondisjoint/test.json",
    ]:
        path = raw_root / relative_path
        path.write_text("[]", encoding="utf-8")

    report = build_schema_mapping_audit(raw_root=raw_root)

    assert report["audit_decision"] == "schema_mapping_ready_for_manual_review"
    assert report["linkage"]["linked_item_count"] == 2
    assert any(
        row["polyvore_label"] == "blouse" and row["product_type_v0"] == "shirt"
        for row in report["mapping_proposal"]
    )
    assert any(row["polyvore_label"] == "lipstick" for row in report["excluded_labels"])


def test_polyvore_cooccurrence_baseline_builds_product_recommendations(tmp_path):
    raw_root = tmp_path / "raw_hf_files"
    (raw_root / "disjoint").mkdir(parents=True)
    (raw_root / "nondisjoint").mkdir(parents=True)
    (raw_root / "categories.csv").write_text(
        "10,1,Tops\n20,1,Jeans\n30,1,Sneakers\n",
        encoding="utf-8",
    )
    (raw_root / "polyvore_item_metadata.json").write_text(
        json.dumps(
            {
                "top-1": {"semantic_category": "tops", "category_id": "10"},
                "jeans-1": {"semantic_category": "jeans", "category_id": "20"},
                "shoes-1": {"semantic_category": "sneakers", "category_id": "30"},
                "top-2": {"semantic_category": "tops", "category_id": "10"},
                "jeans-2": {"semantic_category": "jeans", "category_id": "20"},
            }
        ),
        encoding="utf-8",
    )
    split_payload = [
        {
            "set_id": "set-1",
            "items": [
                {"item_id": "top-1"},
                {"item_id": "jeans-1"},
                {"item_id": "shoes-1"},
            ],
        },
        {
            "set_id": "set-2",
            "items": [
                {"item_id": "top-2"},
                {"item_id": "jeans-2"},
            ],
        },
    ]
    (raw_root / "disjoint" / "train.json").write_text(
        json.dumps(split_payload),
        encoding="utf-8",
    )
    (raw_root / "disjoint" / "valid.json").write_text(
        json.dumps(
            [
                {
                    "set_id": "set-validation-leak",
                    "items": [
                        {"item_id": "top-1"},
                        {"item_id": "jeans-1"},
                    ],
                },
                {
                    "set_id": "set-validation-clean",
                    "items": [
                        {"item_id": "top-2"},
                        {"item_id": "shoes-1"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    for relative_path in [
        "disjoint/test.json",
        "nondisjoint/train.json",
        "nondisjoint/valid.json",
        "nondisjoint/test.json",
    ]:
        (raw_root / relative_path).write_text("[]", encoding="utf-8")

    report = build_cooccurrence_baseline(raw_root=raw_root)

    assert report["baseline_ready"] is True
    assert report["training_executed"] is False
    assert report["tensorflow_used"] is False
    assert report["primary_config"] == "disjoint"
    assert report["primary_training_split"] == "disjoint_train"
    assert report["primary_baseline"] == report["split_baselines"]["disjoint_train"]
    top_recommendations = report["primary_baseline"]["recommendations_by_product_type"]["top"]
    assert top_recommendations[0]["product_type_v0"] == "jeans"
    assert any(row["product_type_v0"] == "sports_shoes" for row in top_recommendations)
    assert report["leakage"]["has_within_config_positive_pair_leakage"] is True
    assert report["leakage"]["has_primary_train_eval_positive_pair_leakage"] is True
    assert report["evaluation_ready_without_leakage"] is False
    assert report["leakage_filtered_evaluation_ready"] is True
    validation_metrics = report["leakage_filtered_evaluation"]["valid"]
    assert validation_metrics["raw_directed_pair_count"] == 4
    assert validation_metrics["filtered_train_overlap_directed_pair_count"] == 2
    assert validation_metrics["evaluable_directed_pair_count"] == 2
    assert validation_metrics["recall_at_k"]["1"] == 0.0
    assert validation_metrics["recall_at_k"]["3"] == 1.0
    assert report["baseline_decision"] == "train_only_baseline_ready_with_leakage_filtered_evaluation"


def test_outfit_tensorflow_training_splits_filter_train_positive_overlap(tmp_path):
    raw_root = tmp_path / "raw_hf_files"
    (raw_root / "disjoint").mkdir(parents=True)
    (raw_root / "nondisjoint").mkdir(parents=True)
    (raw_root / "categories.csv").write_text(
        "10,1,Tops\n20,1,Jeans\n30,1,Sneakers\n",
        encoding="utf-8",
    )

    metadata = {}
    for index in range(1, 7):
        metadata[f"top-{index}"] = {"semantic_category": "tops", "category_id": "10"}
        metadata[f"jeans-{index}"] = {"semantic_category": "jeans", "category_id": "20"}
        metadata[f"shoes-{index}"] = {"semantic_category": "sneakers", "category_id": "30"}
    (raw_root / "polyvore_item_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    (raw_root / "disjoint" / "train.json").write_text(
        json.dumps(
            [
                {
                    "set_id": "train-1",
                    "items": [{"item_id": "top-1"}, {"item_id": "jeans-1"}, {"item_id": "shoes-1"}],
                },
                {
                    "set_id": "train-2",
                    "items": [{"item_id": "top-2"}, {"item_id": "jeans-2"}, {"item_id": "shoes-2"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    (raw_root / "disjoint" / "valid.json").write_text(
        json.dumps(
            [
                {
                    "set_id": "valid-leak",
                    "items": [{"item_id": "top-1"}, {"item_id": "jeans-1"}],
                },
                {
                    "set_id": "valid-clean-1",
                    "items": [{"item_id": "top-3"}, {"item_id": "shoes-3"}],
                },
                {
                    "set_id": "valid-clean-2",
                    "items": [{"item_id": "top-4"}, {"item_id": "jeans-4"}, {"item_id": "shoes-4"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    (raw_root / "disjoint" / "test.json").write_text(
        json.dumps(
            [
                {
                    "set_id": "test-1",
                    "items": [{"item_id": "top-5"}, {"item_id": "jeans-5"}, {"item_id": "shoes-5"}],
                },
                {
                    "set_id": "test-2",
                    "items": [{"item_id": "top-6"}, {"item_id": "jeans-6"}, {"item_id": "shoes-6"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    for relative_path in [
        "nondisjoint/train.json",
        "nondisjoint/valid.json",
        "nondisjoint/test.json",
    ]:
        (raw_root / relative_path).write_text("[]", encoding="utf-8")

    pair_splits, diagnostics = build_outfit_training_splits(raw_root=raw_root)

    leaked_pair_key = exact_item_pair_key("top-1", "jeans-1")
    valid_positive_keys = set(
        pair_splits["valid"].loc[pair_splits["valid"]["label"] == 1, "pair_key"]
    )
    assert leaked_pair_key not in valid_positive_keys
    assert diagnostics["pair_splits"]["filtered_positive_overlap_with_train"]["valid"] == 2
    assert set(pair_splits["train"]["label"]) == {0, 1}
    assert set(pair_splits["valid"]["label"]) == {0, 1}

    features, target = build_outfit_feature_frame(pair_splits["train"])
    assert list(features.columns) == OUTFIT_PAIR_FEATURE_COLUMNS
    assert "item_id" not in " ".join(features.columns)
    assert set(target) == {0, 1}


def test_outfit_threshold_selection_uses_validation_probabilities():
    import numpy as np

    threshold, metrics = select_outfit_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.10, 0.40, 0.55, 0.90]),
        thresholds=[0.50, 0.80],
    )

    assert threshold == 0.5
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


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


def test_image_service_real_mode_falls_back_without_model(monkeypatch):
    monkeypatch.setattr(image_service_module, "load_image_artifacts", lambda: None)

    image_result = predict_image(None, use_real_model=True)
    assert image_result["product_type"] == "tshirt"
    assert image_result["mode"] == "simulation"
    assert "fallback_reason" in image_result


def test_image_metadata_fail_closed_and_active_path(tmp_path):
    assert FASHION_ACTIVE_DIR.name == "fashion_active"
    assert FASHION_MODEL_PATH.parent == FASHION_ACTIVE_DIR
    assert IMAGE_MODEL_PATH == FASHION_MODEL_PATH
    assert FASHION_METADATA_PATH.name == "metadata.json"
    assert read_image_metadata(tmp_path / "missing_metadata.json") is None
    assert is_image_metadata_promoted(None) is False
    assert is_image_metadata_promoted(
        {"model_status": "experimental_only", "promotable_to_streamlit": True}
    ) is False
    assert is_image_metadata_promoted(
        {"model_status": "promoted", "promotable_to_streamlit": True}
    ) is True


def test_image_metadata_reader_accepts_utf8_bom(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        '{"model_status": "promoted", "promotable_to_streamlit": true}',
        encoding="utf-8-sig",
    )

    metadata = read_image_metadata(metadata_path)

    assert metadata is not None
    assert is_image_metadata_promoted(metadata) is True


def test_image_service_promoted_low_confidence_returns_unknown(monkeypatch):
    metadata = {
        "model_status": "promoted",
        "promotable_to_streamlit": True,
        "selected_experiment": "mobilenet_v2",
        "image_size": 224,
        "class_labels": ["dress_shoes", "heels"],
        "canonical_mapping": {"dress_shoes": "shoes", "heels": "shoes"},
        "abstention_strategy": {"minimum_confidence": 0.80},
    }

    monkeypatch.setattr(
        image_service_module,
        "load_image_artifacts",
        lambda: FakeImageArtifacts(
            model=FakeImageModel([[0.76, 0.24]]),
            label_encoder=FakeImageLabelEncoder(["dress_shoes", "heels"]),
            metadata=metadata,
        ),
    )
    monkeypatch.setattr(
        image_service_module,
        "preprocess_image_for_cnn",
        lambda image, image_size, architecture: image,
    )

    result = predict_image(None, use_real_model=True)

    assert result["product_type"] == "unknown"
    assert result["canonical_category"] == "unknown"
    assert result["raw_product_type"] == "dress_shoes"
    assert result["confidence"] == 0.76
    assert result["minimum_confidence"] == 0.80
    assert result["image_size"] == 224


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
    assert load_image_model(
        model_path=PROJECT_ROOT / "missing_fashion_model.keras",
        metadata_path=PROJECT_ROOT / "missing_fashion_metadata.json",
    ) is None


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


def test_fashion_image_abstention_below_threshold_returns_unknown():
    import numpy as np

    result = apply_image_abstention(
        np.array([[0.76, 0.24]]),
        ["dress_shoes", "heels"],
        threshold=0.80,
    )

    assert result["raw_predictions"] == ["dress_shoes"]
    assert result["predictions"] == [UNKNOWN_LABEL]
    assert result["confidences"] == [0.76]


def test_fashion_image_abstention_at_threshold_keeps_prediction():
    import numpy as np

    result = apply_image_abstention(
        np.array([[0.80, 0.20]]),
        ["dress_shoes", "heels"],
        threshold=0.80,
    )

    assert result["raw_predictions"] == ["dress_shoes"]
    assert result["predictions"] == ["dress_shoes"]
    assert result["confidences"] == [0.80]


def test_fashion_image_abstention_metrics_compute_unknown_rate():
    metrics = evaluate_image_abstention(
        ["dress_shoes", "heels", "heels", "dress_shoes"],
        ["dress_shoes", UNKNOWN_LABEL, "heels", UNKNOWN_LABEL],
        ["dress_shoes", "heels"],
    )

    assert metrics["total_count"] == 4
    assert metrics["covered_count"] == 2
    assert metrics["unknown_count"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["unknown_rate"] == 0.5
    assert metrics["accuracy_non_unknown"] == 1.0
    assert metrics["confusion_labels"] == ["dress_shoes", "heels", UNKNOWN_LABEL]
    assert metrics["unknown_by_true_class"]["dress_shoes"]["unknown_rate"] == 0.5
    assert metrics["unknown_by_true_class"]["heels"]["unknown_rate"] == 0.5


def test_fashion_threshold_selection_uses_validation_constraints_only():
    import numpy as np

    y_true = ["dress_shoes", "heels", "dress_shoes", "heels"]
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.45, 0.55],
            [0.52, 0.48],
        ]
    )
    rows = evaluate_image_thresholds(
        y_true,
        probabilities,
        ["dress_shoes", "heels"],
        thresholds=[0.50, 0.80],
    )
    selection = select_image_threshold(
        rows,
        min_coverage=0.50,
        min_macro_f1=0.60,
        min_precision_per_monitored_class=0.90,
        monitored_classes=["dress_shoes", "heels"],
    )

    assert selection["selected"] is True
    assert selection["selected_threshold"] == 0.80
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
    assert outfit_result["recommended_product_types"]
    assert outfit_result["compatible_roles"]
    assert outfit_result["raw_compatibility_score"] == outfit_result["compatibility_score"]
    assert outfit_result["mode"] == "cooccurrence_baseline"
    assert outfit_result["model_status"] == "experimental_only"
    assert outfit_result["recommended_product_types"][:2] == ["casual_shoes", "bag"]
    assert {"shoes", "bag"} <= set(outfit_result["compatible_roles"])
    assert outfit_result["baseline_decision"] == (
        "train_only_baseline_ready_with_leakage_filtered_evaluation"
    )
    assert {
        "input_product_type",
        "recommended_product_types",
        "compatible_roles",
        "raw_compatibility_score",
        "compatible_items",
        "compatible_colors",
        "compatibility_score",
        "reason",
        "mode",
        "model_status",
    } <= set(outfit_result)


def test_outfit_service_falls_back_for_uncovered_product_type():
    outfit_result = recommend_outfit("unknown_product_type", "casual", "noir")

    assert outfit_result["mode"] == "rule_based"
    assert outfit_result["model_status"] == "fallback"
    assert outfit_result["recommended_product_types"] == ["tshirt", "jeans", "casual_shoes"]
    assert "Baseline cooccurrence" in outfit_result["reason"]


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


class FakeImageArtifacts:
    def __init__(self, model, label_encoder, metadata):
        self.model = model
        self.label_encoder = label_encoder
        self.metadata = metadata


class FakeImageModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, batch, verbose=0):
        import numpy as np

        return np.array(self.probabilities)


class FakeImageLabelEncoder:
    def __init__(self, labels):
        self.labels = labels

    def inverse_transform(self, indexes):
        return [self.labels[index] for index in indexes]


def test_outfit_v2_color_features_on_synthetic_images():
    from PIL import Image

    red = Image.new("RGB", (32, 32), color=(220, 20, 30))
    black = Image.new("RGB", (32, 32), color=(5, 5, 5))
    white = Image.new("RGB", (32, 32), color=(250, 250, 250))

    assert classify_color_family(extract_dominant_rgb(red)) == "red"
    assert classify_color_family(extract_dominant_rgb(black)) == "black"
    assert classify_color_family(extract_dominant_rgb(white)) == "white"
    assert color_harmony_score("black", "red") > color_harmony_score("red", "green")


def test_outfit_v2_accepts_huggingface_image_dict_bytes():
    import io
    from PIL import Image

    image = Image.new("RGB", (32, 32), color=(220, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    hf_image = {"bytes": buffer.getvalue(), "path": None}

    rgb_image = as_rgb_image(hf_image)
    image_bytes = encoded_image_bytes(hf_image)
    batch = preprocess_for_mobilenet_embedding(hf_image)

    assert rgb_image.mode == "RGB"
    assert image_bytes.startswith(b"\x89PNG")
    assert classify_color_family(extract_dominant_rgb(hf_image)) == "red"
    assert batch.shape == (1, 224, 224, 3)


def test_outfit_v2_feature_policy_rejects_direct_ids():
    with pytest.raises(ValueError):
        validate_no_forbidden_v2_features(["item_id", "input_product_type"])

    validate_no_forbidden_v2_features(["input_product_type", "color_harmony_score"])


def test_outfit_v2_cli_exposes_embedding_batch_size(monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_outfit_model_v2", "--embedding-batch-size", "96"],
    )
    args = parse_outfit_v2_args()

    assert args.embedding_batch_size == 96


def test_outfit_v2_metadata_promotion_requires_multimodal_v2():
    assert is_outfit_metadata_promoted(
        {
            "version": "outfit_v1",
            "model_status": "promoted",
            "promotable_to_streamlit": True,
            "uses_image_embeddings": True,
            "uses_color_features": True,
        }
    ) is False
    assert is_outfit_metadata_promoted(
        {
            "version": "outfit_v2",
            "model_status": "experimental_only",
            "promotable_to_streamlit": True,
            "uses_image_embeddings": True,
            "uses_color_features": True,
        }
    ) is False
    assert is_outfit_metadata_promoted(
        {
            "version": "outfit_v2",
            "model_status": "promoted",
            "promotable_to_streamlit": True,
            "uses_image_embeddings": True,
            "uses_color_features": True,
        }
    ) is True


def test_outfit_v2_single_image_falls_back_without_promoted_model(monkeypatch):
    monkeypatch.setattr(outfit_v2_service_module, "load_outfit_artifacts", lambda: None)
    monkeypatch.setattr(
        outfit_v2_service_module,
        "predict_image",
        lambda image, use_real_model=True: {
            "product_type": "tshirt",
            "raw_product_type": "tshirt",
            "canonical_category": "top",
            "common_category": "top",
            "confidence": 0.95,
            "mode": "real_model",
        },
    )

    result = recommend_associations_from_image(None, "casual")

    assert result["mode"] == "cooccurrence_baseline"
    assert result["model_status"] == "experimental_only"
    assert result["input_product_type"] == "top"
    assert result["ml_score"] is None
    assert result["cooccurrence_score"] == result["raw_compatibility_score"]
    assert result["detected_item"]["product_type"] == "tshirt"


def test_outfit_v2_multi_image_returns_fallback_shape_without_promoted_model(monkeypatch):
    from PIL import Image

    monkeypatch.setattr(outfit_v2_service_module, "load_outfit_artifacts", lambda: None)
    predictions = iter(
        [
            {
                "product_type": "shirt",
                "canonical_category": "top",
                "common_category": "top",
                "confidence": 0.93,
            },
            {
                "product_type": "jeans",
                "canonical_category": "bottom",
                "common_category": "bottom",
                "confidence": 0.91,
            },
        ]
    )
    monkeypatch.setattr(
        outfit_v2_service_module,
        "predict_image",
        lambda image, use_real_model=True: next(predictions),
    )

    images = [
        Image.new("RGB", (32, 32), color=(15, 15, 15)),
        Image.new("RGB", (32, 32), color=(50, 90, 180)),
    ]
    result = evaluate_outfit_images(images, "casual")

    assert set(result) >= {
        "outfit_score",
        "pair_scores",
        "detected_items",
        "missing_roles",
        "suggested_associations",
        "warnings",
        "mode",
        "model_status",
    }
    assert result["mode"] == "cooccurrence_baseline"
    assert result["model_status"] == "experimental_only"
    assert len(result["detected_items"]) == 2
    assert "shoes" in result["missing_roles"]
