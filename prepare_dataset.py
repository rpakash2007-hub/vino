#!/usr/bin/env python3
"""
Dataset Preparation & Organization Utility for Kidney Disease Medical Images.

Takes an unorganized directory containing class folders (normal, stone, cyst, tumor)
or raw DICOM/PNG files, verifies image integrity, and automatically splits
them into the required train/validation/test structure (e.g. 70% / 15% / 15%).
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image

CLASSES = ['normal', 'stone', 'cyst', 'tumor']
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def prepare_splits(source_dir: Path, target_dir: Path, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    print("=" * 70)
    print(" DATASET PREPARATION & SPLITTING TOOL")
    print(f" Source: {source_dir}")
    print(f" Target: {target_dir}")
    print(f" Split Ratio: {train_ratio*100:.0f}% Train | {val_ratio*100:.0f}% Val | {test_ratio*100:.0f}% Test")
    print("=" * 70)

    if not source_dir.exists():
        print(f"[ERROR] Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Ensure target split folders exist
    for split in ['train', 'validation', 'test']:
        for cls in CLASSES:
            (target_dir / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {cls: {'train': 0, 'val': 0, 'test': 0, 'invalid': 0} for cls in CLASSES}

    for cls in CLASSES:
        # Search for corresponding source class folder
        found_source = None
        for candidate in [cls, cls.upper(), cls.capitalize(), f"kidney_{cls}", f"{cls}_scan"]:
            candidate_path = source_dir / candidate
            if candidate_path.is_dir():
                found_source = candidate_path
                break

        if not found_source:
            print(f"[!] Warning: No matching folder for class '{cls}' in {source_dir}")
            continue

        images = [f for f in found_source.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        random.seed(42)
        random.shuffle(images)

        valid_images = []
        for img in images:
            try:
                with Image.open(img) as im:
                    im.verify()
                valid_images.append(img)
            except Exception:
                summary[cls]['invalid'] += 1

        n_total = len(valid_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_imgs = valid_images[:n_train]
        val_imgs = valid_images[n_train:n_train + n_val]
        test_imgs = valid_images[n_train + n_val:]

        for img in train_imgs:
            shutil.copy2(img, target_dir / 'train' / cls / img.name)
            summary[cls]['train'] += 1

        for img in val_imgs:
            shutil.copy2(img, target_dir / 'validation' / cls / img.name)
            summary[cls]['val'] += 1

        for img in test_imgs:
            shutil.copy2(img, target_dir / 'test' / cls / img.name)
            summary[cls]['test'] += 1

        print(f"  [+] Class '{cls}': {summary[cls]['train']} Train, {summary[cls]['val']} Val, {summary[cls]['test']} Test")

    print("\n[✓] Dataset successfully organized and validated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split raw dataset into train/val/test splits")
    parser.add_argument('--source', type=str, required=True, help="Path to raw dataset folder containing class directories")
    parser.add_argument('--target', type=str, default='dataset', help="Target dataset directory (default: ./dataset)")
    args = parser.parse_args()

    prepare_splits(Path(args.source), Path(args.target))
