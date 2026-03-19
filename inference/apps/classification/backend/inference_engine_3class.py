"""
3-Class Classification Inference Engine
Supports: cnie_front, cnie_back, no_card
With threshold adjustment to prevent over-prediction of no_card
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from pathlib import Path
from typing import Union, Dict
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNIEClassifier3Class:
    """3-Class CNIE Classifier: front, back, no_card"""
    
    CLASS_NAMES = ['cnie_front', 'cnie_back', 'no_card']
    DISPLAY_NAMES = {
        'cnie_front': 'CNIE Front',
        'cnie_back': 'CNIE Back', 
        'no_card': 'No CNIE Card'
    }
    
    # Threshold - only classify as no_card if very confident
    NO_CARD_THRESHOLD = 0.70  # 70% minimum for no_card
    
    # Bias to correct model imbalance (model tends to favor back)
    # Only applies when front vs back scores are close (within MARGIN)
    FRONT_BIAS = 0.35  # Add 35% to front score when comparing front vs back
    BIAS_MARGIN = 1.0  # Only apply bias when |front - back| < 1.0 (always)
    
    def __init__(self, model_path: Union[str, Path], device: str = 'auto'):
        self.model_path = Path(model_path)
        self.input_size = 224
        
        # Resolve device
        if device == 'auto':
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] < 7:
                    logger.warning(f"GPU capability sm_{capability[0]}{capability[1]} not compatible. Using CPU.")
                    self.device = torch.device('cpu')
                else:
                    self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing 3-class classifier on device: {self.device}")
        
        self.model = self._build_model()
        self._load_weights()
        self.transform = self._setup_transforms()
        
        logger.info(f"3-class classifier initialized (no_card threshold: {self.NO_CARD_THRESHOLD})")
    
    def _build_model(self):
        """Build 3-class model."""
        model = efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3 classes
        )
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_weights(self):
        """Load model weights."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded 3-class model with val_acc: {checkpoint.get('val_acc', 'unknown')}")
    
    def _setup_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image, return_all_scores: bool = False) -> Dict:
        """Predict class for image with threshold logic."""
        import time
        start = time.time()
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        probs = probabilities[0].cpu().numpy()
        
        # Get top prediction
        predicted_idx = probs.argmax()
        confidence = float(probs[predicted_idx])
        predicted_class = self.CLASS_NAMES[predicted_idx]
        
        # Get raw scores
        front_score = float(probs[0])  # cnie_front
        back_score = float(probs[1])   # cnie_back
        no_card_score = float(probs[2])  # no_card
        
        # Apply conditional front bias to correct model imbalance
        # Only bias when scores are close (within margin) to avoid flipping confident predictions
        score_diff = abs(front_score - back_score)
        if score_diff < self.BIAS_MARGIN:
            # Close call - apply bias to favor front
            biased_front = front_score + self.FRONT_BIAS
        else:
            # Clear winner - don't interfere
            biased_front = front_score
        
        # THRESHOLD LOGIC:
        # Only classify as no_card if confident > threshold
        # Otherwise, choose between front/back (with bias correction)
        if predicted_class == 'no_card' and confidence < self.NO_CARD_THRESHOLD:
            # Not confident enough for no_card, pick best of front/back
            if biased_front > back_score:
                predicted_class = 'cnie_front'
                confidence = front_score
            else:
                predicted_class = 'cnie_back'
                confidence = back_score
        
        # Additional correction: if back was predicted but front is close with bias
        elif predicted_class == 'cnie_back':
            if biased_front > back_score:
                predicted_class = 'cnie_front'
                confidence = front_score
        
        # Recalculate final confidence from the winning class
        if predicted_class == 'cnie_front':
            confidence = front_score
        elif predicted_class == 'cnie_back':
            confidence = back_score
        else:
            confidence = no_card_score
        
        result = {
            'success': True,
            'predicted_class': predicted_class,
            'display_name': self.DISPLAY_NAMES[predicted_class],
            'confidence': confidence,
            'all_scores': {cls: float(prob) for cls, prob in zip(self.CLASS_NAMES, probs)},
            'inference_time_ms': (time.time() - start) * 1000
        }
        
        return result

# Singleton
_classifier = None

def get_3class_classifier(model_path=None):
    global _classifier
    if _classifier is None:
        if model_path is None:
            model_path = Path.home() / 'retin-verify/models/classification/cnie_classifier_3class_v2.pth'
        _classifier = CNIEClassifier3Class(model_path)
    return _classifier
