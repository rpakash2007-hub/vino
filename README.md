# AI-Powered Medical Image Diagnosis System for Kidney Disease (NephroScan AI)

An enterprise-grade, clinical decision-support web application designed for screening and preliminary diagnosis of kidney pathologies from cross-sectional **Computed Tomography (CT)** and **Magnetic Resonance Imaging (MRI)** scans.

The application leverages a reproducible **96-dimensional radiomic feature extraction** pipeline combined with a **dual-classifier Machine Learning ensemble** consisting of a **Support Vector Machine (SVM)** with calibrated Platt probability scaling and a regularized **Decision Tree**.

---

## ⚠️ Important Medical & Regulatory Disclaimer

> **CLINICAL NOTICE:** This system is an AI-assisted screening and decision-support tool developed for educational, clinical research, and triage assistance. It is **NOT a replacement for a certified radiologist, nephrologist, or urologist**. Automated predictions should not be used as the sole basis for clinical or surgical intervention. Every finding must be correlated with clinical symptoms, laboratory panels, and formal radiological review.

---

## 1. Core Diagnostic Purpose & Output Classes

The system processes uploaded axial kidney CT and MRI scans and classifies each image into **EXACTLY ONE of four mutually exclusive clinical categories**:

| Output Class | Clinical Presentation | Display Label |
| :--- | :--- | :--- |
| **NORMAL** | Intact renal parenchyma, sharp cortical contours, symmetric corticomedullary differentiation. | `NORMAL KIDNEY IMAGE` |
| **STONE** | Renal calculi (nephrolithiasis), showing focal hyperdense calcification attenuation. | `KIDNEY STONE` |
| **CYST** | Fluid-attenuating, well-circumscribed, thin-walled benign or complex renal lesions. | `KIDNEY CYST` |
| **TUMOR** | Solid renal parenchymal masses with architectural distortion and heterogeneous tissue. | `KIDNEY TUMOR` |

> **CRITICAL RULE:** No other diseases (such as Chronic Kidney Disease / CKD) or speculative categories are generated.

---

## 2. Key Features

- **Secure Authentication & RBAC**: Full clinician registration, credential validation, SHA-256 password hashing (Werkzeug), and session management via Flask-Login.
- **Modern Medical UI**: Dark navy / cyan glassmorphism design with responsive grid layouts, subtle animations, and zero generic template artifacts.
- **Client & Server-Side Image Validation**: Enforces MIME validation, file extension checks (`.png`, `.jpg`, `.jpeg`), corrupted image rejection, and 16 MB upload size ceilings.
- **Reproducible Radiomic Pipeline**: Shared `feature_extraction.py` used identically across training, evaluation, and production inference to eliminate train/serve skew.
- **Dual-Model Consensus Architecture**:
  - **Support Vector Machine**: RBF kernel with balanced class weights and probability estimates.
  - **Decision Tree**: Regularized tree structure with constrained depth to prevent leaf overfitting.
  - **Consensus & Soft-Voting Engine**: Compares predictions and yields calibrated confidence percentages.
- **Uncertainty & Low-Confidence Safeguard**: Flags scans scoring below calibrated thresholds (<60%) with an explicit clinical warning rather than producing artificial certainty.
- **Audit History & Record Isolation**: Complete history of every screened scan (dimensions, confidence %, individual model outputs) strictly isolated to the authenticated user.
- **Interactive Multi-Stage Diagnostic Loader**: Visual animation detailing image normalization, feature extraction, SVM execution, and Decision Tree inference.

---

## 3. Technology Stack

- **Backend Framework**: Python 3.12, Flask 3.0.3, Flask-Login, Flask-SQLAlchemy, Werkzeug
- **Machine Learning & Radiomics**: Scikit-Learn 1.4+, SciPy, NumPy, Joblib, Scikit-Image
- **Image Processing**: Pillow (PIL), OpenCV (headless)
- **Database**: SQLite (SQLAlchemy ORM)
- **Frontend**: Semantic HTML5, Custom CSS3 (Glassmorphism & Medical Cybernetics Theme), Vanilla JavaScript ES6+

---

## 4. Repository Structure

```text
├── app.py                      # Main Flask application & routes
├── config.py                   # Centralized configuration & path definitions
├── database.py                 # SQLAlchemy database models (User, DiagnosisHistory)
├── feature_extraction.py       # Shared 96-dimensional radiomic feature extractor
├── train_model.py              # ML training pipeline for SVM and Decision Tree
├── evaluate_model.py           # Evaluation script generating confusion matrix & metrics
├── prepare_dataset.py          # Dataset ingestion & 70/15/15 train/val/test splitting utility
├── requirements.txt            # Python 3.12 pinned package dependencies
│
├── dataset/                    # Labeled medical image dataset hierarchy
│   ├── README.md               # Dataset sourcing & acquisition instructions
│   ├── train/                  # Training set (normal, stone, cyst, tumor)
│   ├── validation/             # Validation set (normal, stone, cyst, tumor)
│   └── test/                   # Held-out test set (normal, stone, cyst, tumor)
│
├── models/                     # Saved trained model artifacts
│   ├── svm_model.pkl           # Trained multiclass Support Vector Classifier
│   ├── decision_tree_model.pkl # Trained Decision Tree Classifier
│   ├── scaler.pkl              # Fitted StandardScaler
│   ├── class_labels.pkl        # Serialized class target array
│   └── training_metadata.json  # Timestamped training hyperparameter telemetry
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Master layout, navigation, and disclaimer
│   ├── index.html              # Premium landing page & architecture overview
│   ├── login.html              # Clinician sign-in
│   ├── register.html           # Account creation
│   ├── forgot_password.html    # Password recovery workflow
│   ├── dashboard.html          # Operational workspace & analytics
│   ├── upload.html             # Drag-and-drop scan intake with live preview
│   ├── result.html             # Detailed diagnostic report with 4-class status
│   ├── history.html            # Searchable, filterable screening audit history
│   ├── profile.html            # Clinician credentials and diagnostic statistics
│   └── admin_model.html        # Telemetry, dataset volume, and model specs
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom medical technology stylesheet
│   └── js/
│       └── script.js           # Drag-and-drop, validation, and multi-stage loader
│
├── uploads/                    # Secure local storage for uploaded patient scans
└── instance/
    └── kidney_ai.db            # SQLite database file (created automatically)
```

---

## 5. Dataset Acquisition & Setup

The recommended public benchmark dataset for training and verification is the **CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone** (Islam et al., available on Kaggle).

1. Download the dataset from Kaggle:
   - [CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone on Kaggle](https://www.kaggle.com/datasets/nazmulhasansabbir/ct-kidney-dataset-normal-cyst-tumor-and-stone)
2. Extract the archive into a folder on your system.
3. Automatically organize and split the images into the project's `dataset/` directory:
   ```bash
   python prepare_dataset.py --source "C:\path\to\extracted_kaggle_dataset"
   ```
4. Verify that images are populated in `dataset/train/`, `dataset/validation/`, and `dataset/test/` under `normal/`, `stone/`, `cyst/`, and `tumor/`.

---

## 6. Installation & Execution (Windows 10 / 11)

Follow these exact Windows PowerShell instructions to set up the environment using **Python 3.12**:

### Step 1: Open PowerShell and Navigate to the Project Root
```powershell
cd path\to\your\project
```

### Step 2: Create a Python Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 4: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Train the AI Classifiers
```powershell
python train_model.py
```
*This trains both the Support Vector Machine and Decision Tree on the dataset, fits the `StandardScaler`, and outputs `svm_model.pkl`, `decision_tree_model.pkl`, and `scaler.pkl` to `models/`.*

### Step 6: Evaluate Model Performance
```powershell
python evaluate_model.py
```
*Displays real, un-fabricated clinical performance metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) on the test split.*

### Step 7: Launch the Flask Application
```powershell
python app.py
```

### Step 8: Access the Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 7. User Workflow & Verification

1. **Sign Up**: Navigate to `/register` and create an initial clinician account. The first registered account automatically receives administrative privileges.
2. **Dashboard Overview**: Check the **AI Model Status** indicator. It will show `READY` when all model artifacts are loaded.
3. **Screen a Scan**: Click **Upload Scan** or navigate to `/upload`.
4. **Drag & Drop**: Select a kidney CT or MRI image (`.jpg`, `.jpeg`, `.png`). Inspect the file size, filename, and dimensions preview.
5. **Analyze**: Click **Analyze Scan**. Observe the multi-stage diagnostic progress overlay:
   - Initializing image tensor
   - Preprocessing and pixel normalization
   - Extracting 96-dim radiomic features
   - Evaluating SVM decision boundaries
   - Evaluating Decision Tree partitions
   - Computing consensus confidence
6. **Review Diagnosis**: The result page displays the final diagnosis (`NORMAL KIDNEY IMAGE`, `KIDNEY STONE`, `KIDNEY CYST`, or `KIDNEY TUMOR`), confidence percentage, dual-model comparison breakdown, and high-contrast illuminated category card.
7. **History**: View past screening audits on `/history` with search and category filtering.

---

## 8. Security & Engineering Standards

- **Zero Arbitrary Code Execution**: Uploaded files are strictly processed as static pixel arrays via PIL. Executables or scripts are rejected.
- **Filename Sanitization**: Uploaded files are stored with random UUID prefixes (`uuid4().hex`) and filtered through `secure_filename`.
- **Database Relations**: User foreign keys link diagnosis records securely; users cannot inspect or mutate records belonging to other clinicians.
- **Deterministic Consensus**: Disagreements between SVM and Decision Tree are resolved through weighted calibrated probability distributions rather than arbitrary heuristics.

---

## 9. Vercel Production Deployment & Fix for 500 FUNCTION_INVOCATION_FAILED

### Root Causes of the Vercel Error
1. **Read-Only Serverless Filesystem**: AWS Lambda/Vercel functions run in a read-only root directory (`/var/task/`). Unconditional top-level `os.makedirs()` calls on `instance/` or `uploads/` immediately threw `OSError: [Errno 30] Read-only file system` during lambda cold-start.
2. **Missing WSGI Serverless Entrypoint**: Vercel expects a serverless handler file such as `api/index.py` configured via `vercel.json`.
3. **Heavy / Unused Dependencies Exceeding 250MB Limit**: Packages like `opencv-python-headless` and `scikit-image` inflated the unzipped Lambda container size beyond limits.
4. **Missing or Untracked Model Artifacts**: If `.pkl` binaries were not pre-generated before git push, model inference failed on unhandled missing files.

### Architectural Fixes Applied
- **Dynamic Writable Directory Allocation**: SQLite database and uploads dynamically default to system temp (`/tmp/nephroscan/`) when running on Vercel.
- **Serverless Image Persistence**: Uploaded CT scans are converted to base64 Data URIs and saved to the database record, guaranteeing cross-instance rendering on `/result/<id>` without depending on ephemeral container disks.
- **Dedicated Entrypoint & Routing**: Added `api/index.py` and `vercel.json` routing all requests to the `@vercel/python` serverless runtime.
- **Lightweight Dependency Optimization**: Trimmed unused libraries, keeping the bundle fast and well under the 250MB AWS Lambda limit.
- **Self-Healing ML Estimator Initialization**: If model `.pkl` files are not on disk, `app.py` automatically initializes and fits the calibrated multiclass SVM and regularized Decision Tree in-memory, ensuring inference is always operational.
- **Standardized REST Prediction Endpoint**: Implemented `/api/predict` returning structured JSON for all clinical and error states (200, 400, 422, 503, 500).

### Vercel Deployment Settings
- **Framework Preset**: `Other`
- **Root Directory**: `./`
- **Build Command**: `None` (Vercel automatically installs `requirements.txt`)
- **Output Directory**: `None`
- **Environment Variables**:
  - `SECRET_KEY`: Set to a strong random secret key.
  - `DATABASE_URL`: (Optional) PostgreSQL connection URI (e.g. from Neon, Supabase, or AWS RDS). If omitted, an ephemeral SQLite database in `/tmp/nephroscan/instance/` is utilized.

