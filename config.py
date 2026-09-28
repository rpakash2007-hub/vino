import os
import sys
import tempfile
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

# Detect Vercel / Serverless environment
IS_VERCEL = (
    bool(os.environ.get('VERCEL')) or 
    bool(os.environ.get('VERCEL_ENV')) or
    bool(os.environ.get('AWS_LAMBDA_FUNCTION_NAME')) or
    bool(os.environ.get('LAMBDA_TASK_ROOT'))
)

def _get_writable_dir(preferred_path: Path, fallback_subpath: str) -> Path:
    """Returns preferred path if writable, otherwise falls back to /tmp."""
    if not IS_VERCEL:
        try:
            preferred_path.mkdir(parents=True, exist_ok=True)
            test_file = preferred_path / '.perm_check'
            test_file.touch()
            test_file.unlink()
            return preferred_path
        except Exception:
            pass
    # Fallback to system temp directory (always writable in AWS Lambda / Vercel)
    tmp_path = Path(tempfile.gettempdir()) / 'nephroscan' / fallback_subpath
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path

class Config:
    """Application configuration parameters tuned for local and Vercel environments."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'nephroscan-medical-ai-secure-key-2026')
    
    # Instance directory for local SQLite database
    INSTANCE_DIR = _get_writable_dir(BASE_DIR / 'instance', 'instance')
    
    # Database URL configuration
    _db_url = os.environ.get('DATABASE_URL', '')
    if _db_url:
        # Normalize postgres:// to postgresql:// for SQLAlchemy 1.4/2.0 compatibility
        if _db_url.startswith('postgres://'):
            _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
        SQLALCHEMY_DATABASE_URI = _db_url
    else:
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{INSTANCE_DIR / 'kidney_ai.db'}"
        
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload configuration
    UPLOAD_FOLDER = _get_writable_dir(BASE_DIR / 'uploads', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model storage directory
    MODELS_DIR = BASE_DIR / 'models'
    MODELS_CACHE_DIR = _get_writable_dir(BASE_DIR / 'models', 'models')
    
    SVM_MODEL_PATH = MODELS_DIR / 'svm_model.pkl'
    DT_MODEL_PATH = MODELS_DIR / 'decision_tree_model.pkl'
    SCALER_PATH = MODELS_DIR / 'scaler.pkl'
    CLASS_LABELS_PATH = MODELS_DIR / 'class_labels.pkl'
    METADATA_PATH = MODELS_DIR / 'training_metadata.json'
    
    # Cache model paths in writable location
    SVM_CACHE_PATH = MODELS_CACHE_DIR / 'svm_model.pkl'
    DT_CACHE_PATH = MODELS_CACHE_DIR / 'decision_tree_model.pkl'
    SCALER_CACHE_PATH = MODELS_CACHE_DIR / 'scaler.pkl'
    CLASS_LABELS_CACHE_PATH = MODELS_CACHE_DIR / 'class_labels.pkl'
    METADATA_CACHE_PATH = MODELS_CACHE_DIR / 'training_metadata.json'
    
    # Dataset paths
    DATASET_DIR = BASE_DIR / 'dataset'
    TRAIN_DIR = DATASET_DIR / 'train'
    VAL_DIR = DATASET_DIR / 'validation'
    TEST_DIR = DATASET_DIR / 'test'
    
    # Diagnostic Classes (Strictly the 4 requested classes)
    DIAGNOSTIC_CLASSES = ['NORMAL', 'STONE', 'CYST', 'TUMOR']
    
    # Classification display mapping
    CLASS_DISPLAY_NAMES = {
        'NORMAL': 'NORMAL KIDNEY IMAGE',
        'STONE': 'KIDNEY STONE',
        'CYST': 'KIDNEY CYST',
        'TUMOR': 'KIDNEY TUMOR'
    }
    
    # Low confidence threshold (below 60% flags uncertain/low-confidence warning)
    CONFIDENCE_THRESHOLD = 0.60
