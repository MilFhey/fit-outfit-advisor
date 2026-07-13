from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models"
ENCODERS_DIR = MODELS_DIR / "encoders"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FASHION_V1_CLASSES_PATH = CONFIG_DIR / "fashion_v1_classes.json"
OUTFIT_V1_CONFIG_PATH = CONFIG_DIR / "outfit_v1_config.json"
FASHION_V1_DIR = MODELS_DIR / "fashion_v1"
FASHION_ACTIVE_DIR = MODELS_DIR / "fashion_active"
FASHION_MODEL_PATH = FASHION_ACTIVE_DIR / "fashion_model.keras"
FASHION_LABEL_ENCODER_PATH = FASHION_ACTIVE_DIR / "label_encoder.joblib"
FASHION_METADATA_PATH = FASHION_ACTIVE_DIR / "metadata.json"
IMAGE_MODEL_PATH = FASHION_MODEL_PATH

FIT_V2_DIR = MODELS_DIR / "fit_v2"
FIT_V3_DIR = MODELS_DIR / "fit_v3"
FIT_ACTIVE_DIR = MODELS_DIR / "fit_active"
FIT_MODEL_PATH = FIT_ACTIVE_DIR / "fit_model.keras"
FIT_PREPROCESSOR_PATH = FIT_ACTIVE_DIR / "fit_preprocessor.joblib"
FIT_LABEL_ENCODER_PATH = FIT_ACTIVE_DIR / "fit_label_encoder.joblib"
FIT_METADATA_PATH = FIT_ACTIVE_DIR / "metadata.json"

OUTFIT_V1_DIR = MODELS_DIR / "outfit_v1"
OUTFIT_V2_DIR = MODELS_DIR / "outfit_v2"
OUTFIT_ACTIVE_DIR = MODELS_DIR / "outfit_active"
OUTFIT_MODEL_PATH = OUTFIT_ACTIVE_DIR / "outfit_model.keras"
OUTFIT_PREPROCESSOR_PATH = OUTFIT_ACTIVE_DIR / "outfit_preprocessor.joblib"
OUTFIT_METADATA_PATH = OUTFIT_ACTIVE_DIR / "metadata.json"
OUTFIT_PROTOTYPES_PATH = OUTFIT_ACTIVE_DIR / "product_type_prototypes.json"

DEFAULT_MODCLOTH_DATASET_PATH = RAW_DATA_DIR / "modcloth_final_data.json"
