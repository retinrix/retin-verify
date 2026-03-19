#!/usr/bin/env python3
"""
Feedback System for Continuous Model Improvement
Collects misclassified images and triggers retraining.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
from PIL import Image
import io
import base64


class FeedbackCollector:
    """
    Collects user feedback on predictions and manages retraining dataset.
    """
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path.home() / 'retin-verify/apps/classification/feedback_data'
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.misclassified_dir = self.base_dir / 'misclassified'
        self.correct_dir = self.base_dir / 'correct'
        self.low_confidence_dir = self.base_dir / 'low_confidence'
        self.annotations_file = self.base_dir / 'feedback_annotations.json'
        
        for d in [self.misclassified_dir, self.correct_dir, self.low_confidence_dir]:
            d.mkdir(exist_ok=True)
            (d / 'cnie_front').mkdir(exist_ok=True)
            (d / 'cnie_back').mkdir(exist_ok=True)
        
        # Load or create annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict]:
        """Load existing annotations."""
        if self.annotations_file.exists():
            with open(self.annotations_file) as f:
                return json.load(f)
        return []
    
    def _save_annotations(self):
        """Save annotations to file."""
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def _get_image_hash(self, image_data: bytes) -> str:
        """Get unique hash for image."""
        return hashlib.md5(image_data).hexdigest()[:12]
    
    def submit_feedback(
        self,
        image_data: bytes,
        predicted_class: str,
        predicted_confidence: float,
        correct_class: Optional[str] = None,
        is_correct: Optional[bool] = None,
        notes: str = ""
    ) -> Dict:
        """
        Submit feedback for a prediction.
        
        Args:
            image_data: Raw image bytes
            predicted_class: What the model predicted
            predicted_confidence: Model confidence (0-1)
            correct_class: Ground truth class (if is_correct=False)
            is_correct: Whether prediction was correct
            notes: Optional notes from user
            
        Returns:
            Dict with feedback_id and status
        """
        timestamp = datetime.now().isoformat()
        image_hash = self._get_image_hash(image_data)
        feedback_id = f"{timestamp}_{image_hash}"
        
        # Determine category
        if is_correct is True:
            category = 'correct'
            save_dir = self.correct_dir / predicted_class
        elif is_correct is False and correct_class:
            category = 'misclassified'
            save_dir = self.misclassified_dir / correct_class
        elif predicted_confidence < 0.7:  # Low confidence threshold
            category = 'low_confidence'
            save_dir = self.low_confidence_dir / predicted_class
        else:
            category = 'uncertain'
            save_dir = self.low_confidence_dir / predicted_class
        
        # Save image
        image_path = save_dir / f"{feedback_id}.jpg"
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        # Record annotation
        annotation = {
            'feedback_id': feedback_id,
            'timestamp': timestamp,
            'image_hash': image_hash,
            'image_path': str(image_path.relative_to(self.base_dir)),
            'category': category,
            'predicted_class': predicted_class,
            'predicted_confidence': predicted_confidence,
            'correct_class': correct_class,
            'is_correct': is_correct,
            'notes': notes
        }
        
        self.annotations.append(annotation)
        self._save_annotations()
        
        return {
            'feedback_id': feedback_id,
            'status': 'saved',
            'category': category,
            'message': f'Image saved to {category} dataset'
        }
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics."""
        total = len(self.annotations)
        misclassified = len([a for a in self.annotations if a['category'] == 'misclassified'])
        correct = len([a for a in self.annotations if a['category'] == 'correct'])
        low_confidence = len([a for a in self.annotations if a['category'] == 'low_confidence'])
        
        # Count by predicted class
        by_predicted = {}
        for a in self.annotations:
            pc = a['predicted_class']
            by_predicted[pc] = by_predicted.get(pc, 0) + 1
        
        # Count by correct class (for misclassified)
        by_correct = {}
        for a in self.annotations:
            if a['correct_class']:
                cc = a['correct_class']
                by_correct[cc] = by_correct.get(cc, 0) + 1
        
        return {
            'total_feedback': total,
            'misclassified': misclassified,
            'correct_confirmations': correct,
            'low_confidence': low_confidence,
            'by_predicted_class': by_predicted,
            'by_correct_class': by_correct,
            'retraining_recommended': misclassified >= 10  # Threshold for retraining
        }
    
    def prepare_retraining_dataset(self, output_dir: Path = None) -> Path:
        """
        Prepare dataset for retraining with new feedback images.
        
        Returns:
            Path to prepared dataset directory
        """
        if output_dir is None:
            output_dir = self.base_dir / 'retraining_dataset'
        
        output_dir = Path(output_dir)
        
        # Create structure
        for split in ['train', 'val']:
            for cls in ['cnie_front', 'cnie_back']:
                (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
        
        # Copy misclassified images (these are the most valuable)
        misclassified = [a for a in self.annotations if a['category'] == 'misclassified']
        
        # Split: 80% train, 20% val
        train_cutoff = int(len(misclassified) * 0.8)
        
        for i, ann in enumerate(misclassified):
            src = self.base_dir / ann['image_path']
            if src.exists():
                split = 'train' if i < train_cutoff else 'val'
                dst = output_dir / split / ann['correct_class'] / f"feedback_{ann['feedback_id']}.jpg"
                shutil.copy2(src, dst)
        
        # Also copy low confidence images
        low_conf = [a for a in self.annotations if a['category'] == 'low_confidence']
        for i, ann in enumerate(low_conf):
            src = self.base_dir / ann['image_path']
            if src.exists():
                split = 'train' if i % 5 != 0 else 'val'  # 80/20 split
                dst = output_dir / split / ann['predicted_class'] / f"lowconf_{ann['feedback_id']}.jpg"
                shutil.copy2(src, dst)
        
        # Create annotations file
        retraining_info = {
            'created_at': datetime.now().isoformat(),
            'source': 'feedback_collection',
            'total_images': len(misclassified) + len(low_conf),
            'misclassified': len(misclassified),
            'low_confidence': len(low_conf),
            'annotations': misclassified + low_conf
        }
        
        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(retraining_info, f, indent=2)
        
        return output_dir
    
    def export_for_labeling(self, output_file: Path = None) -> Path:
        """
        Export low-confidence predictions for manual labeling.
        Creates a JSON file with base64 images for easy labeling.
        """
        if output_file is None:
            output_file = self.base_dir / 'for_labeling.json'
        
        low_conf = [a for a in self.annotations if a['category'] == 'low_confidence']
        
        labeling_data = []
        for ann in low_conf:
            img_path = self.base_dir / ann['image_path']
            if img_path.exists():
                with open(img_path, 'rb') as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                
                labeling_data.append({
                    'feedback_id': ann['feedback_id'],
                    'predicted_class': ann['predicted_class'],
                    'confidence': ann['predicted_confidence'],
                    'image_base64': img_b64,
                    'needs_label': True
                })
        
        with open(output_file, 'w') as f:
            json.dump(labeling_data, f, indent=2)
        
        return output_file


# Singleton instance
_feedback_collector = None

def get_feedback_collector() -> FeedbackCollector:
    """Get or create feedback collector singleton."""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector
