#!/usr/bin/env python3
"""
Prepare 3-class dataset: cnie_front, cnie_back, no_card
Collects and organizes samples for retraining with "no card" detection.
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from feedback_system import get_feedback_collector


def setup_3class_structure(base_dir: Path):
    """Create directory structure for 3-class dataset."""
    classes = ['cnie_front', 'cnie_back', 'no_card']
    splits = ['train', 'val']
    
    for split in splits:
        for cls in classes:
            (base_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    return base_dir


def collect_existing_feedback(output_dir: Path):
    """Collect existing front/back feedback."""
    collector = get_feedback_collector()
    base = collector.base_dir
    
    print("\n📦 Collecting existing feedback...")
    
    # Copy misclassified front images
    front_src = base / 'misclassified' / 'cnie_front'
    front_count = 0
    if front_src.exists():
        for img in front_src.glob('*.jpg'):
            dst = output_dir / 'train' / 'cnie_front' / img.name
            shutil.copy2(img, dst)
            front_count += 1
    
    # Copy misclassified back images  
    back_src = base / 'misclassified' / 'cnie_back'
    back_count = 0
    if back_src.exists():
        for img in back_src.glob('*.jpg'):
            dst = output_dir / 'train' / 'cnie_back' / img.name
            shutil.copy2(img, dst)
            back_count += 1
    
    print(f"  ✓ cnie_front: {front_count} images")
    print(f"  ✓ cnie_back: {back_count} images")
    
    return {'front': front_count, 'back': back_count}


def create_no_card_collector():
    """Create feedback collector for no_card samples."""
    base_dir = Path.home() / 'retin-verify/apps/classification/feedback_data_3class'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create no_card directory
    (base_dir / 'no_card').mkdir(exist_ok=True)
    
    return base_dir


def count_samples(dataset_dir: Path):
    """Count samples per class."""
    counts = {}
    for split in ['train', 'val']:
        counts[split] = {}
        for cls in ['cnie_front', 'cnie_back', 'no_card']:
            class_dir = dataset_dir / split / cls
            if class_dir.exists():
                counts[split][cls] = len(list(class_dir.glob('*.jpg')))
            else:
                counts[split][cls] = 0
    return counts


def split_to_val(dataset_dir: Path, val_ratio=0.2):
    """Move 20% of training samples to validation."""
    import random
    random.seed(42)
    
    for cls in ['cnie_front', 'cnie_back', 'no_card']:
        train_dir = dataset_dir / 'train' / cls
        val_dir = dataset_dir / 'val' / cls
        
        if not train_dir.exists():
            continue
            
        images = list(train_dir.glob('*.jpg'))
        if len(images) < 5:
            continue
        
        # Calculate how many to move
        n_val = max(1, int(len(images) * val_ratio))
        val_samples = random.sample(images, n_val)
        
        for img in val_samples:
            shutil.move(str(img), str(val_dir / img.name))


def main():
    parser = argparse.ArgumentParser(description='Prepare 3-class dataset')
    parser.add_argument('--output', type=Path, 
                       default=Path.home() / 'retin-verify/apps/classification/dataset_3class',
                       help='Output directory for 3-class dataset')
    parser.add_argument('--split', action='store_true',
                       help='Split 20% to validation set')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  3-CLASS DATASET PREPARATION")
    print("  Classes: cnie_front, cnie_back, no_card")
    print("=" * 70)
    
    # Setup structure
    print("\n📁 Creating directory structure...")
    setup_3class_structure(args.output)
    
    # Collect existing feedback
    counts = collect_existing_feedback(args.output)
    
    # Show current status
    print("\n" + "=" * 70)
    print("  CURRENT STATUS")
    print("=" * 70)
    print(f"\n  cnie_front: {counts['front']} samples ✓")
    print(f"  cnie_back:  {counts['back']} samples ✓")
    print(f"  no_card:    0 samples ⚠️  NEED TO COLLECT")
    
    total = counts['front'] + counts['back']
    print(f"\n  Total (without no_card): {total}")
    
    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print("""
  1. Collect NO CARD samples (target: 40-50 images)
     
     Capture images of:
     • Empty background (no card)
     • Credit cards
     • Driver's licenses
     • Other ID documents
     • Random objects
     • Your hand without card
     • Blurry/unclear images
     
     Save to: feedback_data_3class/no_card/
     
  2. Run this script again:
     python prepare_3class_dataset.py --split
     
  3. Deploy to Colab:
     python deploy_3class.py --host YOUR_HOST
""")
    
    # Check if we have no_card samples
    no_card_dir = Path.home() / 'retin-verify/apps/classification/feedback_data_3class/no_card'
    if no_card_dir.exists():
        no_card_count = len(list(no_card_dir.glob('*.jpg')))
        print(f"\n  ✅ Found {no_card_count} no_card samples!")
        
        # Copy them
        for img in no_card_dir.glob('*.jpg'):
            dst = args.output / 'train' / 'no_card' / img.name
            shutil.copy2(img, dst)
        
        if args.split:
            print("\n  Splitting 20% to validation...")
            split_to_val(args.output)
        
        # Final count
        final_counts = count_samples(args.output)
        print("\n  Final dataset:")
        for split, classes in final_counts.items():
            print(f"    {split}:")
            for cls, count in classes.items():
                print(f"      {cls}: {count}")
        
        total_samples = sum(sum(c.values()) for c in final_counts.values())
        print(f"\n  TOTAL: {total_samples} samples")
        
        if total_samples >= 60:  # 20 per class minimum
            print("\n  🎉 Ready for retraining!")
            print(f"  Dataset location: {args.output}")
        else:
            print(f"\n  ⚠️  Need more samples (have {total_samples}, need 60+)")


if __name__ == '__main__':
    main()
