#!/usr/bin/env python3
"""
Dataset Cleaner CLI Tool
Automatically flags potential mislabels using face and chip detection.
For use on Colab or headless environments.
"""

import os
import sys
import json
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


class ChipDetector:
    """Chip detection using template matching."""
    
    def __init__(self, threshold=0.55):
        self.template = self._create_template()
        self.threshold = threshold
        
    def _create_template(self):
        """Create synthetic CNIE chip template."""
        template = np.ones((80, 60, 3), dtype=np.uint8) * 200
        center = (30, 40)
        axes = (22, 32)
        cv2.ellipse(template, center, axes, 0, 0, 360, (150, 150, 150), 2)
        cv2.ellipse(template, center, axes, 0, 0, 360, (100, 100, 100), -1)
        
        # Globe pattern lines
        for y in range(20, 61, 10):
            cv2.line(template, (10, y), (50, y), (180, 180, 180), 1)
        for x in [15, 30, 45]:
            cv2.line(template, (x, 20), (x, 60), (180, 180, 180), 1)
        
        # Central contact area
        cv2.rectangle(template, (20, 30), (40, 50), (160, 160, 160), -1)
        cv2.rectangle(template, (20, 30), (40, 50), (120, 120, 120), 1)
        
        return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    def detect(self, image_path):
        """Detect chip in image. Returns (detected: bool, confidence: float)."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, 0.0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            max_val = 0.0
            
            # Multi-scale template matching
            for scale in [0.7, 0.85, 1.0, 1.15, 1.3]:
                new_w = int(self.template.shape[1] * scale)
                new_h = int(self.template.shape[0] * scale)
                resized = cv2.resize(self.template, (new_w, new_h))
                
                if gray.shape[0] >= resized.shape[0] and gray.shape[1] >= resized.shape[1]:
                    result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                    _, local_max, _, _ = cv2.minMaxLoc(result)
                    max_val = max(max_val, local_max)
            
            return max_val > self.threshold, max_val
        except Exception as e:
            print(f"Chip detection error: {e}")
            return False, 0.0


class FaceDetector:
    """Face detection using OpenCV Haar cascade."""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.cascade = None
            print(f"Warning: Haar cascade not found at {cascade_path}")
    
    def detect(self, image_path):
        """Detect face in image. Returns (detected: bool, confidence: float, regions: list)."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, 0.0, []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.cascade is None:
                return False, 0.0, []
            
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
            )
            
            detected = len(faces) > 0
            confidence = min(len(faces) * 0.3 + 0.4, 1.0) if detected else 0.0
            
            # Convert numpy array to list properly
            face_list = []
            if detected:
                for face in faces:
                    face_list.append([int(x) for x in face])
            
            return detected, confidence, face_list
        except Exception as e:
            print(f"Face detection error: {e}")
            return False, 0.0, []


def analyze_image(img_path, is_front, face_detector, chip_detector):
    """Analyze a single image and return flag status."""
    result = {
        'path': str(img_path),
        'name': img_path.name,
        'original_label': 'front' if is_front else 'back',
        'split': img_path.parent.parent.name,
        'flagged': False,
        'reason': '',
        'face_detected': False,
        'face_confidence': 0.0,
        'face_regions': [],
        'chip_detected': False,
        'chip_confidence': 0.0,
        'suggested_label': None
    }
    
    # Run both detectors
    face_detected, face_conf, face_regions = face_detector.detect(img_path)
    chip_detected, chip_conf = chip_detector.detect(img_path)
    
    result['face_detected'] = face_detected
    result['face_confidence'] = face_conf
    result['face_regions'] = face_regions
    result['chip_detected'] = chip_detected
    result['chip_confidence'] = chip_conf
    
    # Flag logic
    if is_front:
        # Front should have face
        if not face_detected:
            result['flagged'] = True
            result['reason'] = 'Front image: No face detected'
            # Suggest back if chip detected
            if chip_detected and chip_conf > 0.6:
                result['suggested_label'] = 'back'
    else:
        # Back should have chip
        if not chip_detected:
            result['flagged'] = True
            result['reason'] = 'Back image: No chip detected'
            # Suggest front if face detected
            if face_detected and face_conf > 0.5:
                result['suggested_label'] = 'front'
    
    return result


def scan_dataset(dataset_dir, chip_threshold=0.55):
    """Scan dataset and flag potential mislabels."""
    dataset_path = Path(dataset_dir)
    face_detector = FaceDetector()
    chip_detector = ChipDetector(threshold=chip_threshold)
    
    flagged_images = []
    stats = defaultdict(int)
    
    # Count total images
    total_images = 0
    for split in ['train', 'val', 'test']:
        for class_name in ['cnie_front', 'cnie_back']:
            class_dir = dataset_path / split / class_name
            if class_dir.exists():
                total_images += len(list(class_dir.glob('*.jpg')))
    
    print(f"Scanning {total_images} images...")
    
    # Process images
    for split in ['train', 'val', 'test']:
        for class_name in ['cnie_front', 'cnie_back']:
            class_dir = dataset_path / split / class_name
            if not class_dir.exists():
                continue
            
            is_front = 'front' in class_name
            
            for img_path in tqdm(list(class_dir.glob('*.jpg')), 
                                 desc=f"{split}/{class_name}", leave=False):
                result = analyze_image(img_path, is_front, face_detector, chip_detector)
                stats['total'] += 1
                
                if result['flagged']:
                    flagged_images.append(result)
                    stats['flagged'] += 1
                    if is_front:
                        stats['front_no_face'] += 1
                    else:
                        stats['back_no_chip'] += 1
                else:
                    stats['verified'] += 1
                    if is_front:
                        stats['front_with_face'] += 1
                    else:
                        stats['back_with_chip'] += 1
    
    return flagged_images, dict(stats)


def export_cleaned_dataset(dataset_dir, flagged_images, output_dir, auto_action=None):
    """
    Export cleaned dataset.
    
    Args:
        auto_action: If set, automatically apply this action to all flagged images
                     ('keep', 'exclude', or None for manual)
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    for split in ['train', 'val', 'test']:
        for class_name in ['cnie_front', 'cnie_back']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {'kept': 0, 'relabeled': 0, 'excluded': 0, 'pending': 0}
    
    # Build flagged lookup
    flagged_lookup = {f['path']: f for f in flagged_images}
    
    # Process all images
    for split in ['train', 'val', 'test']:
        for class_name in ['cnie_front', 'cnie_back']:
            class_dir = dataset_path / split / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                img_path_str = str(img_path)
                
                if img_path_str in flagged_lookup:
                    # This image was flagged
                    img_info = flagged_lookup[img_path_str]
                    action = auto_action or img_info.get('action', 'pending')
                    
                    if action == 'exclude':
                        stats['excluded'] += 1
                        continue
                    
                    if action == 'pending':
                        stats['pending'] += 1
                        # Copy to pending folder
                        pending_dir = output_path / 'pending' / class_name
                        pending_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_path, pending_dir / img_path.name)
                        continue
                    
                    if action == 'keep':
                        dest = output_path / split / class_name / img_path.name
                        shutil.copy2(img_path, dest)
                        stats['kept'] += 1
                    elif action.startswith('relabel_to_'):
                        new_label = img_info.get('new_label', 'front')
                        new_class = 'cnie_front' if new_label == 'front' else 'cnie_back'
                        dest = output_path / split / new_class / img_path.name
                        shutil.copy2(img_path, dest)
                        stats['relabeled'] += 1
                else:
                    # Clean image - copy as-is
                    dest = output_path / split / class_name / img_path.name
                    shutil.copy2(img_path, dest)
                    stats['kept'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='CNIE Dataset Cleaner CLI')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--output', help='Output directory for cleaned dataset')
    parser.add_argument('--report', help='Output JSON report file')
    parser.add_argument('--chip-threshold', type=float, default=0.55, 
                        help='Chip detection threshold (default: 0.55)')
    parser.add_argument('--auto-keep', action='store_true',
                        help='Automatically keep all flagged images (skip manual review)')
    parser.add_argument('--auto-exclude', action='store_true',
                        help='Automatically exclude all flagged images')
    
    args = parser.parse_args()
    
    # Scan dataset
    print("="*70)
    print("CNIE Dataset Cleaner")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Chip threshold: {args.chip_threshold}")
    print()
    
    flagged_images, stats = scan_dataset(args.dataset, args.chip_threshold)
    
    # Print statistics
    print("\n" + "="*70)
    print("SCAN RESULTS")
    print("="*70)
    print(f"Total Images: {stats['total']}")
    print(f"Verified Clean: {stats['verified']}")
    print(f"Flagged: {stats['flagged']}")
    print()
    print(f"Front with Face: {stats.get('front_with_face', 0)}")
    print(f"Front without Face: {stats.get('front_no_face', 0)}")
    print(f"Back with Chip: {stats.get('back_with_chip', 0)}")
    print(f"Back without Chip: {stats.get('back_no_chip', 0)}")
    
    # Show flagged samples
    if flagged_images:
        print("\n" + "="*70)
        print(f"FLAGGED IMAGES (showing first 10 of {len(flagged_images)})")
        print("="*70)
        for i, img in enumerate(flagged_images[:10]):
            print(f"{i+1}. {img['name']}")
            print(f"   Original: {img['original_label']} | Reason: {img['reason']}")
            print(f"   Face: {img['face_detected']} ({img['face_confidence']:.2f}) | "
                  f"Chip: {img['chip_detected']} ({img['chip_confidence']:.2f})")
            if img['suggested_label']:
                print(f"   Suggested: {img['suggested_label']}")
            print()
    
    # Save report
    if args.report:
        report = {
            'scan_date': datetime.now().isoformat(),
            'dataset_dir': args.dataset,
            'chip_threshold': args.chip_threshold,
            'statistics': stats,
            'flagged_images': flagged_images
        }
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.report}")
    
    # Export cleaned dataset
    if args.output:
        print("\n" + "="*70)
        print("EXPORTING CLEANED DATASET")
        print("="*70)
        
        auto_action = None
        if args.auto_keep:
            auto_action = 'keep'
            print("Auto-action: Keep all flagged images")
        elif args.auto_exclude:
            auto_action = 'exclude'
            print("Auto-action: Exclude all flagged images")
        else:
            print("Manual review required for flagged images")
            print("(Use --auto-keep or --auto-exclude for automatic handling)")
        
        export_stats = export_cleaned_dataset(
            args.dataset, flagged_images, args.output, auto_action
        )
        
        print(f"\nExport complete: {args.output}")
        print(f"Kept: {export_stats['kept']}")
        print(f"Relabeled: {export_stats['relabeled']}")
        print(f"Excluded: {export_stats['excluded']}")
        print(f"Pending: {export_stats['pending']}")


if __name__ == '__main__':
    main()
