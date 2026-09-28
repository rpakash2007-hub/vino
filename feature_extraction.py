"""
Feature extraction pipeline for Kidney CT & MRI Image Diagnosis.

CRITICAL ARCHITECTURE REQUIREMENT:
The exact same extract_features function MUST be utilized during:
1. Model training (train_model.py)
2. Model evaluation (evaluate_model.py)
3. Production web prediction (app.py)

This guarantees absolute reproducibility and eliminates training-serving skew.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import skew, kurtosis

# Canonical target resolution for ML feature extraction
TARGET_IMAGE_SIZE = (128, 128)

def preprocess_image(image_input) -> np.ndarray:
    """
    Safely opens, converts, resizes, and normalizes an image input.
    Accepts:
      - filepath string or Path
      - PIL.Image instance
      - numpy ndarray
    Returns:
      - 2D float32 numpy array normalized to [0, 255] with shape (128, 128)
    """
    if isinstance(image_input, (str, bytes)) or hasattr(image_input, 'read'):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Convert to RGB first to standardize format across grayscale / RGBA / palette
    rgb_img = img.convert('RGB')
    
    # Resize to standardized target input size using high-quality Lanczos interpolation
    resized_img = rgb_img.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to grayscale for radiodensity analysis
    gray_img = resized_img.convert('L')
    
    # Return as 2D float32 array
    return np.array(gray_img, dtype=np.float32)


def extract_features(image_input) -> np.ndarray:
    """
    Extracts a deterministic 96-dimensional feature vector from a kidney scan:
    - 32 bins: Normalized Intensity Histogram (radiodensity profile)
    - 11 values: Statistical distribution moments (mean, std, var, skew, kurtosis, percentiles, entropy)
    - 8 values: Concentric radial density bands (center-to-periphery profile)
    - 12 values: Spatial quadrant intensity & gradient variance (detects focal lesions/asymmetry)
    - 32 values: Block-wise Gradient Orientation features (HOG-like structural edge patterns)
    - 1 value: Global edge density ratio
    Total: 96 numerical features.
    """
    gray = preprocess_image(image_input)
    features = []

    # 1. Intensity Histogram (32 bins, normalized)
    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    features.extend(hist.tolist())

    # 2. Statistical Moments
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    var_val = float(np.var(gray))
    skew_val = float(skew(gray.ravel()))
    kurt_val = float(kurtosis(gray.ravel()))
    p10 = float(np.percentile(gray, 10))
    p25 = float(np.percentile(gray, 25))
    p50 = float(np.percentile(gray, 50))
    p75 = float(np.percentile(gray, 75))
    p90 = float(np.percentile(gray, 90))
    
    # Histogram Shannon Entropy
    probs = hist + 1e-12
    entropy = -float(np.sum(probs * np.log2(probs)))
    
    features.extend([mean_val, std_val, var_val, skew_val, kurt_val, p10, p25, p50, p75, p90, entropy])

    # 3. Concentric Radial Density Profile (8 circular zones from organ center)
    h, w = gray.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    radii = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = np.sqrt(cx ** 2 + cy ** 2)
    radial_bins = 8
    
    for i in range(radial_bins):
        r_start = (i / radial_bins) * max_radius
        r_end = ((i + 1) / radial_bins) * max_radius
        mask = (radii >= r_start) & (radii < r_end)
        radial_val = float(np.mean(gray[mask])) if np.any(mask) else 0.0
        features.append(radial_val)

    # 4. Spatial Quadrant Features (4 quadrants x 3 metrics = 12 features)
    # Detects focal lesions (e.g. unilateral mass, cyst in upper pole, calcified stone)
    half_h, half_w = h // 2, w // 2
    quadrants = [
        gray[:half_h, :half_w],   # Top-Left
        gray[:half_h, half_w:],   # Top-Right
        gray[half_h:, :half_w],   # Bottom-Left
        gray[half_h:, half_w:]    # Bottom-Right
    ]
    for q in quadrants:
        features.append(float(np.mean(q)))
        features.append(float(np.std(q)))
        features.append(float(np.percentile(q, 90) - np.percentile(q, 10)))

    # 5. Sobel Edge Gradients & Structural Patterns (32 features)
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    magnitude = np.hypot(sobel_x, sobel_y)
    angles = (np.arctan2(sobel_y, sobel_x) + np.pi) * (180 / np.pi)  # 0 to 360 degrees

    # 4 spatial blocks (2x2 grid), each with 8 gradient orientation bins
    for r_idx in range(2):
        for c_idx in range(2):
            block_mag = magnitude[r_idx * half_h:(r_idx + 1) * half_h, c_idx * half_w:(c_idx + 1) * half_w]
            block_ang = angles[r_idx * half_h:(r_idx + 1) * half_h, c_idx * half_w:(c_idx + 1) * half_w]
            
            # 8 orientation bins across 360 degrees
            bin_hist, _ = np.histogram(block_ang, bins=8, range=(0, 360), weights=block_mag)
            total_weight = float(np.sum(bin_hist)) + 1e-6
            normalized_bins = (bin_hist / total_weight).tolist()
            features.extend(normalized_bins)

    # 6. Global Edge Density Ratio
    edge_threshold = mean_val + 0.5 * std_val
    edge_ratio = float(np.mean(magnitude > edge_threshold))
    features.append(edge_ratio)

    feature_vec = np.array(features, dtype=np.float64)
    # Sanitize any unexpected non-finite values
    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=1.0, neginf=-1.0)
    return feature_vec
