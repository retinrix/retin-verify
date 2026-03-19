"""
3-Class Classification Inference Engine - v3 (Clean)
No bias corrections - relies on proper training
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

class CNIEClassifier3ClassV3:
    """3-Class CNIE Classifier: front, back, no_card"""
    
    CLASS_NAMES = ['cnie_front', 'cnie_back', 'no_card']
    DISPLAY_NAMES = {
        'cnie_front': 'CNIE Front',
        'cnie_back': 'CNIE Back', 
        'no_card': 'No CNIE Card'
    }
    
    # Threshold for no_card - require high confidence
    NO_CARD_THRESHOLD = 0.70
    
    def __init__(self, model_path: Union[str, Path], device: str = 'auto'):
        self.model_path = Path(model_path)
        self.input_size = 224
        
        # Resolve device
        if device == 'auto':
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] < 7:
                    logger.warning(f"GPU sm_{capability[0]}{capability[1]} not compatible. Using CPU.")
                    self.device = torch.device('cpu')
                else:
                    self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing 3-class v3 classifier on: {self.device}")
        
        self.model = self._build_model()
        self._load_weights()
        self.transform = self._setup_transforms()
        
        logger.info(f"v3 classifier ready (no_card threshold: {self.NO_CARD_THRESHOLD})")
    
    def _build_model(self):
        """Build model with same architecture as training"""
        model = efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_weights(self):
        """Load model weights"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Log metrics if available
        val_acc = checkpoint.get('val_acc', 'unknown')
        balance = checkpoint.get('balance', 'unknown')
        class_acc = checkpoint.get('class_acc', {})
        if class_acc:
            logger.info(f"Loaded v3 model - Val: {val_acc:.1f}%, Balance: {balance:.1f}%")
            logger.info(f"  Class acc: F={class_acc.get(0,0):.1f}% B={class_acc.get(1,0):.1f}% NC={class_acc.get(2,0):.1f}%")
        else:
            logger.info(f"Loaded v3 model - Val acc: {val_acc}")
    
    def _setup_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image, return_all_scores: bool = False) -> Dict:
        """Predict class for image"""
        import time
        start = time.time()
        
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        probs = probabilities[0].cpu().numpy()
        
        # Get scores
        front_score = float(probs[0])
        back_score = float(probs[1])
        no_card_score = float(probs[2])
        
        # Determine prediction
        # If no_card is highest AND above threshold, use it
        # Otherwise pick best of front/back
        if no_card_score > front_score and no_card_score > back_score and no_card_score > self.NO_CARD_THRESHOLD:
            predicted_class = 'no_card'
            confidence = no_card_score
        elif front_score > back_score:
            predicted_class = 'cnie_front'
            confidence = front_score
        else:
            predicted_class = 'cnie_back'
            confidence = back_score
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'display_name': self.DISPLAY_NAMES[predicted_class],
            'confidence': confidence,
            'all_scores': {
                'cnie_front': front_score,
                'cnie_back': back_score,
                'no_card': no_card_score
            },
            'inference_time_ms': (time.time() - start) * 1000
        }

# Singleton
_classifier = None

def get_3class_classifier_v3(model_path=None):
    global _classifier
    if _classifier is None:
        if model_path is None:
            model_path = Path.home() / 'retin-verify/models/classification/cnie_classifier_3class_v3.pth'
        _classifier = CNIEClassifier3ClassV3(model_path)
    return _classifier
