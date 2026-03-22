# v5 Implementation Strategy: Real-Data-Only Training

## Executive Summary

Based on the analysis, the root cause of v4's poor performance is **domain shift from synthetic data**. The solution is to train v5 **exclusively on real images** with enhanced augmentation and improved training configuration.

**Key Insight:** The 2-class real-only model achieved 99-100% accuracy, proving real data alone is sufficient.

---

## Phase 1: Data Preparation (30 minutes)

### 1.1 Collect All Real Images

```bash
# Sources of real images
REAL_DATA_DIR="~/retin-verify/data/processed/classification/dataset_real_only"
mkdir -p $REAL_DATA_DIR/{train,val}/{cnie_front,cnie_back,no_card}

# 1. Original real dataset (219 training + 87 validation)
cp ~/retin-verify/data/processed/classification/dataset_3class/train/*/* $REAL_DATA_DIR/train/
cp ~/retin-verify/data/processed/classification/dataset_3class/val/*/* $REAL_DATA_DIR/val/

# 2. Feedback images - but only NEW uploads, not copies of training data
# (We need to identify truly new images)

# 3. User uploaded documents (if any new ones)
```

### 1.2 Deduplicate and Verify

```python
# Script: deduplicate_dataset.py
import hashlib
from pathlib import Path
from PIL import Image

def get_image_hash(path):
    """Get perceptual hash of image"""
    img = Image.open(path)
    img = img.resize((64, 64)).convert('L')
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = ''.join('1' if p > avg else '0' for p in pixels)
    return hex(int(bits, 2))[2:]

# Remove duplicates
# Ensure no data leakage between train/val
```

### 1.3 Minimum Data Requirements

| Class | Minimum | Target | Status |
|-------|---------|--------|--------|
| Front | 50 | 100+ | Check |
| Back | 50 | 100+ | Check |
| No Card | 50 | 100+ | Check |

**If insufficient:** Collect more via feedback system before training.

---

## Phase 2: Training Configuration (v5_train.py)

### 2.1 Model Architecture Changes

```python
class V5Trainer:
    """v5: Real-data-only with improved configuration"""
    
    # Configuration
    CONFIG = {
        'num_epochs': 150,
        'batch_size': 16,  # Smaller batch for real data
        'early_stop_patience': 20,
        
        # Differential learning rates (CRITICAL FIX)
        'lr_backbone': 1e-5,      # Frozen or very low LR
        'lr_classifier': 1e-3,    # Higher LR for new layers
        
        # Layer freezing strategy
        'freeze_backbone_epochs': 5,  # Warmup: freeze backbone
        'unfreeze_epochs': 10,        # Then unfreeze with low LR
    }
```

### 2.2 Progressive Unfreezing Strategy

```python
def setup_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),  # Wider classifier
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 3)
    )
    
    return model

def get_optimizer(model, epoch):
    """Progressive unfreezing with differential LR"""
    
    if epoch < CONFIG['freeze_backbone_epochs']:
        # Phase 1: Freeze backbone, train classifier only
        for param in model.features.parameters():
            param.requires_grad = False
        
        optimizer = optim.AdamW(
            model.classifier.parameters(),
            lr=CONFIG['lr_classifier']
        )
        
    elif epoch < CONFIG['unfreeze_epochs']:
        # Phase 2: Unfreeze last 50 layers with low LR
        for param in model.features[-50:].parameters():
            param.requires_grad = True
            
        optimizer = optim.AdamW([
            {'params': model.features[-50:].parameters(), 'lr': CONFIG['lr_backbone']},
            {'params': model.classifier.parameters(), 'lr': CONFIG['lr_classifier']}
        ])
        
    else:
        # Phase 3: Unfreeze all with differential LR
        for param in model.parameters():
            param.requires_grad = True
            
        optimizer = optim.AdamW([
            {'params': model.features.parameters(), 'lr': CONFIG['lr_backbone']},
            {'params': model.classifier.parameters(), 'lr': CONFIG['lr_classifier']}
        ])
    
    return optimizer
```

### 2.3 Enhanced Augmentation (Real-World Simulation)

```python
def get_train_transforms():
    """Aggressive augmentation for small real dataset"""
    return transforms.Compose([
        # Geometric
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),  # Increased from 15
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),  # Increased
            scale=(0.85, 1.15),      # Scale jitter
            shear=10                  # Shear augmentation
        ),
        
        # Photometric
        transforms.ColorJitter(
            brightness=0.4,    # Increased
            contrast=0.4,      # Increased
            saturation=0.3,
            hue=0.1
        ),
        
        # Real-world effects (NEW)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        
        transforms.RandomApply([
            AddGaussianNoise(mean=0., std=0.01)  # Camera noise
        ], p=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        
        # Random erasing for occlusion simulation
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

class AddGaussianNoise:
    """Add camera noise"""
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
```

### 2.4 Advanced Regularization

```python
# MixUp / CutMix augmentation
from torchvision.transforms import MixUp

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Weight decay only on classifier
optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': 1e-5, 'weight_decay': 0},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}
])
```

---

## Phase 3: Validation Strategy

### 3.1 Strict Train/Val Split

```python
# Use sklearn for proper stratified split
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure no data leakage
# 80% train / 20% val (stratified by class)
```

### 3.2 Validation Metrics

```python
def validate(model, loader):
    """Comprehensive validation"""
    metrics = {
        'overall_acc': 0,
        'class_acc': {},
        'balance': 0,
        'confusion_matrix': np.zeros((3, 3)),
        'confidence_stats': {'mean': 0, 'std': 0}
    }
    
    # Calculate all metrics
    # Track per-class confidence
    # Flag low-confidence predictions for review
    
    return metrics
```

---

## Phase 4: Training Monitoring

### 4.1 Key Metrics to Track

```python
# Log these every epoch:
metrics = {
    'train_loss': loss,
    'train_acc': acc,
    'val_acc': val_acc,
    'val_balance': balance,
    'val_front_acc': class_acc[0],
    'val_back_acc': class_acc[1],
    'val_nocard_acc': class_acc[2],
    'front_back_confusion': confusion[0,1] + confusion[1,0],
    'learning_rate': current_lr
}
```

### 4.2 Early Stopping Criteria

```python
# Primary: Balance (harmonic mean of class accuracies)
# Secondary: Minimize front-back confusion
if val_balance > best_balance and front_back_confusion < best_confusion:
    save_checkpoint()
```

---

## Phase 5: Deployment & Testing

### 5.1 Confidence Thresholding

```python
def predict_with_rejection(image, threshold=0.7):
    """Reject low-confidence predictions"""
    probs = model(image)
    max_prob = max(probs)
    
    if max_prob < threshold:
        return {
            'predicted_class': 'uncertain',
            'confidence': max_prob,
            'message': 'Confidence too low, please retake photo'
        }
    
    return normal_prediction
```

### 5.2 A/B Testing Protocol

```python
# Run v4 and v5 in parallel
# Log predictions from both
# Compare accuracy on real user uploads
```

---

## Implementation Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data Prep | 30 min | Clean real-only dataset |
| Script Dev | 1 hour | v5_train.py with all fixes |
| Training | 2-3 hours | On Colab GPU |
| Validation | 30 min | Test on held-out real images |
| Deployment | 15 min | Update API server |
| **Total** | **4-5 hours** | v5 model deployed |

---

## Success Criteria

| Metric | v4 | v5 Target |
|--------|-----|-----------|
| Front Acc | 92% | **98%+** |
| Back Acc | 96% | **98%+** |
| No Card Acc | 97% | **98%+** |
| Balance | 95% | **97%+** |
| Front→Back Errors | 135 | **< 20** |
| Back→Front Errors | 120 | **< 20** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Not enough real data | Collect more via feedback UI before training |
| Overfitting | Heavy augmentation + early stopping + dropout |
| Training too long | Start with 50 epochs, evaluate, continue if needed |
| Worse than v4 | Keep v4 as fallback, A/B test before full deployment |

---

## Next Steps

1. **Audit current real image inventory**
2. **Create v5 training script** with above configuration
3. **Deploy to Colab** when ready
4. **Test rigorously** on held-out real images
5. **A/B test** against v4 before full deployment

---

**Decision Point:** If we have < 50 real images per class, we should collect more data before training v5.
