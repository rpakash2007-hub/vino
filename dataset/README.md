# Kidney CT & MRI Disease Classification Dataset Guide

This directory holds the labeled medical image dataset used for training, validating, and testing the dual-classifier Machine Learning system (**Support Vector Machine** and **Decision Tree**).

## Exact Classes (4 Output Classes Only)
Per clinical and design requirements, images are categorized into exactly four distinct folders:
1. `normal/` - Healthy renal parenchyma, intact corticomedullary differentiation, no masses or nephrolithiasis.
2. `stone/` - Renal calculi (kidney stones), exhibiting localized high-density attenuation / radiopacity.
3. `cyst/` - Fluid-attenuated, thin-walled, well-circumscribed hypo-attenuating renal lesions.
4. `tumor/` - Solid renal masses, irregular margins, hypervascular contrast enhancement, or parenchymal distortion.

> **CRITICAL NOTE:** No additional classes (such as CKD) are accepted or predicted.

---

## Directory Hierarchy

```text
dataset/
├── train/
│   ├── normal/
│   ├── stone/
│   ├── cyst/
│   └── tumor/
├── validation/
│   ├── normal/
│   ├── stone/
│   ├── cyst/
│   └── tumor/
└── test/
    ├── normal/
    ├── stone/
    ├── cyst/
    └── tumor/
```

---

## Recommended Benchmark Dataset

The recommended publicly accessible and legally usable dataset is:

**CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone**
- **Source:** Kaggle / Published by Islam et al.
- **Link:** [Kaggle CT Kidney Dataset](https://www.kaggle.com/datasets/nazmulhasansabbir/ct-kidney-dataset-normal-cyst-tumor-and-stone)
- **Volume:** Over 12,000 clinically validated axial CT scan slices across all four classes:
  - Cyst: 3,709 images
  - Normal: 5,077 images
  - Stone: 1,377 images
  - Tumor: 2,283 images

### How to Prepare:
1. Download the archive from Kaggle or your institutional radiology PACS repository.
2. Extract the archive into a temporary folder.
3. Run the automated dataset preparation and splitting script:
   ```powershell
   python prepare_dataset.py --source "path/to/extracted_folder" --split 70:15:15
   ```
4. Or manually copy images into `dataset/train/`, `dataset/validation/`, and `dataset/test/` under each respective category (`normal`, `stone`, `cyst`, `tumor`).

Supported image formats: `.png`, `.jpg`, `.jpeg`.
All images are preprocessed and standardized by `feature_extraction.py` at runtime.
