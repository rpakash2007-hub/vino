import os
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Application configuration parameters."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'nephroscan-medical-ai-secure-key-2026')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', f"sqlite:///{BASE_DIR / 'instance' / 'kidney_ai.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload configuration
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model storage directory
    MODELS_DIR = BASE_DIR / 'models'
    SVM_MODEL_PATH = MODELS_DIR / 'svm_model.pkl'
    DT_MODEL_PATH = MODELS_DIR / 'decision_tree_model.pkl'
    SCALER_PATH = MODELS_DIR / 'scaler.pkl'
    CLASS_LABELS_PATH = MODELS_DIR / 'class_labels.pkl'
    METADATA_PATH = MODELS_DIR / 'training_metadata.json'
    
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
    
    # Ensure required runtime folders exist
    os.makedirs(BASE_DIR / 'instance', exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
