"""
Classification Inference Engine
Shared module for all classification inference applications.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CNIEClassifier:
    """
    CNIE Front/Back Classifier Inference Engine.
    
    Supports both 2-class (cnie_front, cnie_back) and 4-class models:
    - 2-class: ['cnie_front', 'cnie_back']
    - 4-class: ['passport', 'cnie_front', 'cnie_back', 'carte_grise']
    """
    
    # Default 4-class mapping (will be updated based on model)
    CLASS_NAMES = ['passport', 'cnie_front', 'cnie_back', 'carte_grise']
    
    # Display names for UI (will be updated based on model)
    DISPLAY_NAMES = {
        'passport': 'Passport',
        'cnie_front': 'CNIE Front (Carte Nationale - Recto)',
        'cnie_back': 'CNIE Back (Carte Nationale - Verso)',
        'carte_grise': 'Carte Grise (Vehicle Registration)'
    }
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        'high': 0.9,
        'medium': 0.7,
        'low': 0.5
    }
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'auto',
        input_size: int = 224,
        dropout: float = 0.5,
        hidden_dim: int = None  # Auto-detect from checkpoint
    ):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            device: Device to use ('cpu', 'cuda', or 'auto')
            input_size: Input image size (default 224 for EfficientNet-B0)
            dropout: Dropout rate used in training (default 0.5)
            hidden_dim: Hidden layer dimension (auto-detected if None)
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.dropout = dropout
        self._hidden_dim = hidden_dim
        
        # Resolve device
        if device == 'auto':
            # Check if CUDA is available and compatible
            if torch.cuda.is_available():
                # Check GPU capability - sm_50 (GTX 950M) is not compatible with modern PyTorch
                capability = torch.cuda.get_device_capability()
                if capability[0] < 7:  # sm_70 or higher required
                    logger.warning(f"GPU capability sm_{capability[0]}{capability[1]} not compatible with PyTorch. Using CPU.")
                    self.device = torch.device('cpu')
                else:
                    self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing CNIEClassifier on device: {self.device}")
        
        # First, inspect checkpoint to determine model architecture
        self._inspect_checkpoint()
        
        # Build model with correct architecture
        self.model = self._build_model()
        
        # Load weights
        self._load_weights()
        
        # Setup transforms
        self.transform = self._setup_transforms()
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
        
        logger.info("CNIEClassifier initialized successfully")
    
    def _inspect_checkpoint(self):
        """Inspect checkpoint to determine model architecture."""
        logger.info(f"Inspecting checkpoint: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            self.checkpoint_info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'val_acc': checkpoint.get('val_acc', 'unknown'),
                'classes': checkpoint.get('classes', None)
            }
        else:
            state_dict = checkpoint
            self.checkpoint_info = {'epoch': 'unknown', 'val_acc': 'unknown', 'classes': None}
        
        # Detect number of classes from classifier.4.weight shape
        # Shape is (num_classes, hidden_dim)
        if 'classifier.4.weight' in state_dict:
            num_classes = state_dict['classifier.4.weight'].shape[0]
            hidden_dim = state_dict['classifier.4.weight'].shape[1]
        elif 'classifier.1.weight' in state_dict:
            # Different architecture
            num_classes = state_dict['classifier.1.weight'].shape[0]
            hidden_dim = 1280  # Default
        else:
            # Fallback to 4-class
            num_classes = 4
            hidden_dim = self._hidden_dim or 512
        
        self.num_classes = num_classes
        if self._hidden_dim is None:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self._hidden_dim
        
        # Update CLASS_NAMES based on detected classes or checkpoint metadata
        if self.checkpoint_info['classes']:
            self.CLASS_NAMES = self.checkpoint_info['classes']
        elif num_classes == 2:
            self.CLASS_NAMES = ['cnie_front', 'cnie_back']
        else:
            self.CLASS_NAMES = ['passport', 'cnie_front', 'cnie_back', 'carte_grise']
        
        # Update DISPLAY_NAMES
        self.DISPLAY_NAMES = {
            'cnie_front': 'CNIE Front (Carte Nationale - Recto)',
            'cnie_back': 'CNIE Back (Carte Nationale - Verso)',
            'passport': 'Passport',
            'carte_grise': 'Carte Grise (Vehicle Registration)'
        }
        
        logger.info(f"Detected model: {num_classes} classes, hidden_dim={hidden_dim}")
        logger.info(f"Classes: {self.CLASS_NAMES}")
    
    def _build_model(self) -> nn.Module:
        """Build EfficientNet-B0 model with custom classifier head."""
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        
        num_features = model.classifier[1].in_features
        
        # Custom classifier head matching detected architecture
        model.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(num_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, len(self.CLASS_NAMES))
        )
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_weights(self):
        """Load model weights from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint with epoch: {checkpoint.get('epoch', 'unknown')}")
                logger.info(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)
            
            logger.info("Model weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image: Union[np.ndarray, Image.Image, str, Path]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (numpy array, PIL Image, or path)
            
        Returns:
            Preprocessed tensor with batch dimension
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Convert numpy array to PIL
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        return_all_scores: bool = False
    ) -> Union[Tuple[str, float], Dict]:
        """
        Predict class for single image.
        
        Args:
            image: Input image
            return_all_scores: If True, return scores for all classes
            
        Returns:
            If return_all_scores=False: (predicted_class, confidence)
            If return_all_scores=True: Dict with all class scores
        """
        start_time = time.time()
        
        # Preprocess
        tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Get predictions
        probs = probabilities[0].cpu().numpy()
        predicted_idx = np.argmax(probs)
        confidence = float(probs[predicted_idx])
        predicted_class = self.CLASS_NAMES[predicted_idx]
        
        # Check for "no card" - if max confidence is below threshold
        NO_CARD_THRESHOLD = 0.7  # 70% threshold
        if confidence < NO_CARD_THRESHOLD:
            predicted_class = 'no_card'
            confidence = 1.0 - confidence  # Invert confidence for "no card"
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        if return_all_scores:
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_scores': {
                    cls: float(prob) for cls, prob in zip(self.CLASS_NAMES, probs)
                },
                'inference_time_ms': inference_time * 1000
            }
        
        return predicted_class, confidence
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]]
    ) -> List[Dict]:
        """
        Predict classes for batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()
        
        # Preprocess all images
        tensors = []
        for img in images:
            tensor = self.preprocess(img)
            tensors.append(tensor)
        
        # Stack into batch
        batch = torch.cat(tensors, dim=0)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Process results
        results = []
        probs = probabilities.cpu().numpy()
        
        for i, img_probs in enumerate(probs):
            predicted_idx = np.argmax(img_probs)
            results.append({
                'predicted_class': self.CLASS_NAMES[predicted_idx],
                'confidence': float(img_probs[predicted_idx]),
                'all_scores': {
                    cls: float(prob) for cls, prob in zip(self.CLASS_NAMES, img_probs)
                }
            })
        
        batch_time = time.time() - start_time
        logger.info(f"Batch inference: {len(images)} images in {batch_time:.3f}s "
                   f"({len(images)/batch_time:.1f} img/s)")
        
        return results
    
    def get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description."""
        if confidence >= self.CONFIDENCE_THRESHOLDS['high']:
            return 'high'
        elif confidence >= self.CONFIDENCE_THRESHOLDS['medium']:
            return 'medium'
        elif confidence >= self.CONFIDENCE_THRESHOLDS['low']:
            return 'low'
        return 'uncertain'
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics."""
        if not self.inference_times:
            return {'total_inferences': 0}
        
        times = np.array(self.inference_times)
        return {
            'total_inferences': self.total_inferences,
            'mean_time_ms': float(np.mean(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'fps': float(1.0 / np.mean(times))
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times = []
        self.total_inferences = 0


# Try to import OpenCV for camera support
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Camera features disabled.")


class CameraCapture:
    """Camera capture helper for real-time inference."""
    
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (default 0 for built-in webcam)
            width: Capture width
            height: Capture height
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for camera capture")
        
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_capturing = False
    
    def start(self) -> bool:
        """Start camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_capturing = True
        logger.info(f"Camera {self.camera_id} started ({self.width}x{self.height})")
        return True
    
    def read(self) -> Optional[np.ndarray]:
        """Read frame from camera."""
        if not self.is_capturing or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def stop(self):
        """Stop camera capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_capturing = False
        logger.info("Camera stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def get_model_path() -> Path:
    """Get default model path."""
    # Try new 2-class CNIE model first
    new_model_path = Path.home() / 'retin-verify/models/classification/cnie_front_back_real.pth'
    if new_model_path.exists():
        return new_model_path
    
    # Fall back to production model
    production_path = Path.home() / 'retin-verify/models/classification_production/best_model.pth'
    if production_path.exists():
        return production_path
    
    raise FileNotFoundError("No model found. Please train or download a model first.")


if __name__ == '__main__':
    # Test the inference engine
    print("CNIE Classification Inference Engine")
    print("=" * 50)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create dummy model for testing if no model exists
    try:
        model_path = get_model_path()
        print(f"Using model: {model_path}")
    except FileNotFoundError:
        print("No model found, creating dummy model for testing...")
        model_path = Path('/tmp/test_model.pth')
        
        # Create and save dummy model
        classifier = CNIEClassifier(
            model_path='/tmp/dummy.pth',  # Will fail but we catch it
            device='cpu'
        )
        torch.save(classifier.model.state_dict(), model_path)
        print(f"Dummy model saved to: {model_path}")
    
    # Initialize classifier
    classifier = CNIEClassifier(model_path=model_path, device='cpu')
    
    # Test with dummy image
    print("\nTesting with dummy image...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = classifier.predict(dummy_image, return_all_scores=True)
    
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Inference time: {result['inference_time_ms']:.2f} ms")
    print("\nAll class scores:")
    for cls, score in result['all_scores'].items():
        print(f"  {cls}: {score:.4f}")
    
    # Performance stats
    stats = classifier.get_performance_stats()
    print(f"\nPerformance stats:")
    print(f"  Total inferences: {stats['total_inferences']}")
    print(f"  Mean time: {stats['mean_time_ms']:.2f} ms")
    print(f"  FPS: {stats['fps']:.1f}")
