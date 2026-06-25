from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models"
ENCODERS_DIR = MODELS_DIR / "encoders"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

IMAGE_MODEL_PATH = MODELS_DIR / "fashion_model.keras"
FIT_V2_DIR = MODELS_DIR / "fit_v2"
FIT_MODEL_PATH = FIT_V2_DIR / "fit_model.keras"
FIT_PREPROCESSOR_PATH = FIT_V2_DIR / "fit_preprocessor.joblib"
FIT_LABEL_ENCODER_PATH = FIT_V2_DIR / "fit_label_encoder.joblib"
FIT_METADATA_PATH = FIT_V2_DIR / "metadata.json"

DEFAULT_MODCLOTH_DATASET_PATH = RAW_DATA_DIR / "modcloth_final_data.json"
