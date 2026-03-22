#!/usr/bin/env python3
"""
Quick dataset structure verification script.
Run this to verify your dataset is structured correctly for the cleaner tool.
"""

import sys
from pathlib import Path


def check_dataset_structure(dataset_dir):
    """Check if dataset has the expected structure."""
    dataset_path = Path(dataset_dir)
    
    print("=" * 60)
    print("DATASET STRUCTURE CHECK")
    print("=" * 60)
    print(f"Checking: {dataset_path}")
    print()
    
    if not dataset_path.exists():
        print("❌ ERROR: Directory does not exist!")
        return False
    
    # Check for expected splits
    expected_splits = ['train', 'val', 'test']
    expected_classes = ['cnie_front', 'cnie_back']
    
    found_splits = []
    total_images = 0
    
    for split in expected_splits:
        split_path = dataset_path / split
        if split_path.exists():
            found_splits.append(split)
            print(f"✓ Found split: {split}/")
            
            for cls in expected_classes:
                class_path = split_path / cls
                if class_path.exists():
                    jpg_count = len(list(class_path.glob('*.jpg')))
                    print(f"  ✓ {cls}/: {jpg_count} images")
                    total_images += jpg_count
                else:
                    print(f"  ❌ {cls}/: NOT FOUND")
        else:
            print(f"❌ Missing split: {split}/")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(found_splits) == 0:
        print("❌ NO VALID SPLITS FOUND!")
        print()
        print("You may have selected a subfolder instead of the root.")
        print("Expected structure:")
        print("  dataset_dir/")
        print("    ├── train/")
        print("    │   ├── cnie_front/")
        print("    │   └── cnie_back/")
        print("    ├── val/")
        print("    └── test/")
        return False
    
    print(f"✓ Found {len(found_splits)}/3 splits")
    print(f"✓ Total images: {total_images}")
    print()
    
    if len(found_splits) == 3 and total_images > 0:
        print("✅ Dataset structure is CORRECT!")
        print("You can now run the cleaner tool with this path.")
        return True
    else:
        print("⚠️  Dataset structure is INCOMPLETE")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Check default location
        default_path = Path.home() / "retin-verify" / "training_data" / "v8_stage2_clean"
        if default_path.exists():
            print(f"No path provided. Checking default: {default_path}")
            check_dataset_structure(default_path)
        else:
            print("Usage: python3 check_dataset.py <dataset_directory>")
            print(f"\nDefault path not found: {default_path}")
    else:
        check_dataset_structure(sys.argv[1])
