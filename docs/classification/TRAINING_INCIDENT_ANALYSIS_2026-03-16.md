# Training Incident Analysis Report
**Date:** 2026-03-16  
**Incident:** Training deadlock at Epoch 98/100 with severe overfitting  
**Model:** EfficientNet-B0 (Document Classification)

---

## Executive Summary

The training process on Google Colab experienced a **deadlock** at Epoch 98 after running for ~3 hours. The model exhibited **severe overfitting** (0.0000 loss, 100% accuracy) from early epochs. This document analyzes both issues and provides actionable solutions.

---

## Part 1: The Deadlock Analysis

### 1.1 Symptoms Observed

| Symptom | Observation |
|---------|-------------|
| Last log entry | Epoch 98 at 21:11:40 (no completion logged) |
| Process state | `S (sleeping)` - futex_wait_queue |
| GPU utilization | 0% despite 2,804 MB allocated |
| Log file modification | Stopped 2+ hours before process check |
| Stack trace | `[<0>] futex_wait_queue+0xde/0x130` |

### 1.2 Root Cause: DataLoader + Google Drive Interaction

**The smoking gun:** `num_workers=4` in DataLoader combined with Google Drive mounted filesystem.

```python
# From train_cli.py (lines 336-350)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,        # <-- PROBLEM
    pin_memory=True
)
```

#### Why This Causes Deadlock

1. **Multi-process DataLoader workers** spawn child processes to load data in parallel
2. **Google Drive FS** (FUSE-based) has file I/O latency and locking issues
3. **The deadlock occurs when:**
   - Worker processes try to read images from `/content/drive/MyDrive/...`
   - Google Drive's FUSE layer experiences sync delays
   - Workers wait on I/O indefinitely
   - Main process waits for workers via futex (thread synchronization)
   - Result: Complete training hang

#### Technical Explanation

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Process                          │
│  ┌──────────────┐         ┌──────────────────────────┐     │
│  │ Main Thread  │────────▶│  DataLoader (num_workers=4) │     │
│  │              │         │  ┌─────┬─────┬─────┬─────┐  │     │
│  │  Waiting ◄───┼─────────┤  │ W1  │ W2  │ W3  │ W4  │  │     │
│  │  (futex_wait)│         │  │ ▓▓▓ │ ▓▓▓ │ ▓▓▓ │ ▓▓▓ │  │     │
│  └──────────────┘         │  │ I/O │ I/O │ I/O │ I/O │  │     │
│                           │  │WAIT │WAIT │WAIT │WAIT │  │     │
│                           │  └─────┴─────┴─────┴─────┘  │     │
│                           └──────────────────────────┘     │
│                                    │                        │
│                                    ▼                        │
│                           ┌──────────────────┐             │
│                           │ Google Drive FUSE│             │
│                           │ (Blocking I/O)   │             │
│                           └──────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

The `futex_wait_queue` in the stack trace confirms the main thread is waiting for worker synchronization that never completes.

### 1.3 Deadlock Solutions

#### Solution A: Disable Multiprocessing (Immediate Fix)
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,        # Disable multiprocessing
    pin_memory=True if device.type == 'cuda' else False
)
```
**Trade-off:** Slower data loading but guaranteed no deadlock.

#### Solution B: Copy Data to Local SSD (Recommended)
```python
# At training start, copy data to local Colab SSD
import shutil
local_data_dir = Path('/content/local_data')
shutil.copytree(args.data_dir, local_data_dir)
args.data_dir = local_data_dir  # Use local copy for training
```
**Trade-off:** Initial copy time (~1-2 min) but faster I/O throughout training.

#### Solution C: Use Persistent Workers with Timeout
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,        # Reduce workers
    persistent_workers=True,  # Keep workers alive between epochs
    pin_memory=True,
    timeout=60            # Add timeout to detect stuck workers
)
```

#### Solution D: Custom Signal Handler for Graceful Degradation
```python
import signal
import sys

def timeout_handler(signum, frame):
    logger.warning("DataLoader timeout detected - switching to single process")
    # Restart with num_workers=0
    sys.exit(42)  # Special exit code for restart

signal.signal(signal.SIGALRM, timeout_handler)
```

---

## Part 2: Severe Overfitting Analysis

### 2.1 The Evidence

From training logs (Epochs 74-98):
```
Epoch 74: Train Loss: 0.0000, Train Acc: 100.00% | Val Loss: 0.0000, Val Acc: 100.00%
Epoch 90: Train Loss: 0.0000, Train Acc: 100.00% | Val Loss: 0.0000, Val Acc: 100.00%
Epoch 98: ... training hung
```

**Red flags:**
- Loss = 0.0000 (mathematically impossible with real data)
- 100% accuracy on both train AND validation
- No improvement/change across 24+ epochs

### 2.2 Dataset Analysis

| Split | Samples | Classes | Distribution |
|-------|---------|---------|--------------|
| Train | 5,942 | 2 (cnie_front, cnie_back) | 50/50 |
| Val | 744 | 2 (cnie_front, cnie_back) | 50/50 |

**Critical Finding:** The code claims 4 classes (`passport`, `cnie_front`, `cnie_back`, `carte_grise`) but the actual dataset only contains 2 classes (CNIE front and back).

```python
# From train_cli.py (line 325)
class_map = {'passport': 0, 'cnie_front': 1, 'cnie_back': 2, 'carte_grise': 3}
```

This mismatch means the model only sees classes 1 and 2 during training.

### 2.3 Root Causes of Overfitting

#### Cause 1: Dataset Too Small for Model Capacity

**The Math:**
- EfficientNet-B0 parameters: ~5.3 million
- Training samples: 5,942
- Ratio: ~893 parameters per sample

**Rule of thumb:** For deep learning, you need at least 10x-100x more samples than model capacity for good generalization.

**Comparison:**
| Model | Parameters | Recommended Min Samples | Our Dataset | Ratio      |
|-------|----------- |------------------------|-------------|------------ |
| EfficientNet-B0    | 5.3M | 50M+            | 5,942       | ❌ 0.01%   |
| ResNet-18          | 11.7M  | 100M+         | 5,942       | ❌ 0.006%  |
| Custom CNN (<100K) | 100K | 1M+             | 5,942       | ⚠️ 6%      |

#### Cause 2: Insufficient Data Augmentation

Current augmentation (lines 283-289):
```python
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),  # Only horizontal flip!
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**What's missing:**
- Rotation (documents can be at any angle)
- Color jittering (lighting variations)
- Gaussian noise (scanner artifacts)
- Perspective distortion (camera angle variations)
- Random erasing (occlusions)

#### Cause 3: No Regularization Strategy

Current regularization (line 87-88, 135-138):
```python
# Weight decay: 0.01 (moderate)
weight_decay=0.01

# Model has only 0.3 dropout in classifier
nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, num_classes)
)
```

**Missing:**
- Dropout in feature extractor (EfficientNet has none by default)
- Early stopping (training ran 98 epochs unnecessarily)
- Label smoothing
- Mixup/CutMix augmentation
- Gradient clipping

#### Cause 4: Learning Rate Too High for Fine-tuning

```python
# Line 85-86
learning_rate=1e-4  # 0.0001

# Line 132
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
```

When fine-tuning pretrained models, the learning rate should be:
- **Feature extractor:** 1e-5 to 1e-4 (slow updates to preserve pretrained features)
- **Classifier head:** 1e-3 to 1e-2 (faster updates for new task)

Using 1e-4 for ALL parameters causes the pretrained features to be overwritten too quickly.

#### Cause 5: Perfect Validation Scores = Data Leakage Suspicion

**The suspicious pattern:** 100% validation accuracy suggests either:
1. **Train/val overlap:** Same images in both splits
2. **Too easy task:** CNIE front vs back is visually trivial (different layouts)
3. **Validation set too small:** 744 samples not representative

**Visual differences between CNIE front/back:**
```
CNIE Front:                     CNIE Back:
┌─────────────────┐            ┌─────────────────┐
│ [PHOTO]         │            │ [FINGERPRINT]   │
│ Name: XXX       │            │ Father: XXX     │
│ DOB: XX/XX/XXXX │            │ Address: XXX    │
│ ID: XXXXXXXXX   │            │ Blood: X+       │
│ Signature       │            │ [MRZ]           │
└─────────────────┘            └─────────────────┘
```

These are visually distinct - a simple edge detector could classify them.

### 2.4 Why Loss = 0.0000 is Mathematically Suspicious

Cross-entropy loss formula:
```
L = -Σ y_i * log(p_i)
```

For perfect prediction (p = 1.0 for correct class):
```
L = -1.0 * log(1.0) = -1.0 * 0 = 0
```

**However:** With floating-point precision, getting exactly 0.0000 for multiple consecutive epochs requires:
1. Model outputting logits of +∞ (impossible with standard activations)
2. OR numerical underflow in logging
3. OR all samples being classified with 100% confidence (extreme overfitting)

Most likely: The model is outputting extremely large logits (>10), causing softmax to produce ~1.0 probabilities.

---

## Part 3: Comprehensive Solutions

### 3.1 Immediate Fixes for Next Training

#### Fix 1: Correct the DataLoader Configuration
```python
# Option A: No workers (safest for Google Drive)
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=0, pin_memory=False
)

# Option B: Copy to local first (recommended)
import shutil
local_dir = Path('/content/local_data')
if not local_dir.exists():
    shutil.copytree(args.data_dir, local_dir)
    print(f"Data copied to {local_dir}")
args.data_dir = local_dir
```

#### Fix 2: Implement Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# In training loop:
early_stopping = EarlyStopping(patience=5)
...
early_stopping(val_loss)
if early_stopping.early_stop:
    logger.info("Early stopping triggered")
    break
```

#### Fix 3: Add Proper Regularization
```python
# Model with feature extractor dropout
def create_model(base_model: str, num_classes: int):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    
    # Add dropout to feature extractor (requires modifying internal layers)
    # Or use stochastic depth if available
    
    # Classifier with higher dropout
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),  # Increased from 0.3
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model
```

#### Fix 4: Enhanced Data Augmentation
```python
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    
    # Geometric augmentations
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1),  # 10% shift
        scale=(0.9, 1.1),      # 10% zoom
        shear=5                # Slight shear
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Photometric augmentations
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.05
    ),
    
    # Noise and blur (simulates scanner artifacts)
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),  # Occlusion simulation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

#### Fix 5: Use Label Smoothing
```python
# Instead of hard targets [0, 1], use soft targets [0.1, 0.9]
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Why this helps:**
- Prevents model from becoming overconfident
- Softens probability distribution
- Adds implicit regularization

#### Fix 6: Implement Differential Learning Rates
```python
# Split parameters into feature extractor and classifier
feature_params = []
classifier_params = []

for name, param in model.named_parameters():
    if 'classifier' in name:
        classifier_params.append(param)
    else:
        feature_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': feature_params, 'lr': 1e-5},      # 10x lower
    {'params': classifier_params, 'lr': 1e-3}    # 10x higher
], weight_decay=0.01)
```

### 3.2 Long-term Architectural Improvements

#### Option A: Use Smaller Model
```python
# For 6K samples, use a smaller model
model = models.mobilenet_v3_small(pretrained=True)
# Or even custom small CNN
```

#### Option B: Implement Progressive Resizing
```python
# Start with smaller images, gradually increase
# Epoch 1-20:  128x128
# Epoch 21-40: 160x160
# Epoch 41+:   224x224
```

#### Option C: Use Test-Time Augmentation (TTA)
```python
def predict_with_tta(model, image, n_augmentations=5):
    """Average predictions over multiple augmentations"""
    model.eval()
    predictions = []
    
    # Original
    predictions.append(model(image))
    
    # Augmented versions
    with torch.no_grad():
        for _ in range(n_augmentations - 1):
            aug_image = apply_random_augment(image)
            predictions.append(model(aug_image))
    
    return torch.stack(predictions).mean(dim=0)
```

### 3.3 Recommended Training Configuration

```yaml
training:
  epochs: 50
  batch_size: 32
  early_stopping_patience: 5
  
optimizer:
  type: AdamW
  feature_lr: 1e-5      # For pretrained layers
  classifier_lr: 1e-3   # For new layers
  weight_decay: 0.01
  
scheduler:
  type: CosineAnnealingWarmRestarts
  T_0: 10
  T_mult: 2
  
data:
  num_workers: 0        # For Google Drive
  pin_memory: false
  augmentation:
    rotation: 15
    translation: 0.1
    scale: [0.9, 1.1]
    color_jitter: 0.2
    random_erasing: 0.2
    
regularization:
  dropout: 0.5
  label_smoothing: 0.1
  stochastic_depth: 0.1  # If using EfficientNet-V2
```

---

## Part 4: Monitoring and Debugging

### 4.1 Add These Metrics to Detect Issues Early

```python
# Log gradient norms (detect vanishing/exploding gradients)
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        writer.add_scalar(f'grad_norm/{name}', grad_norm, global_step)

# Log weight statistics
def log_weight_stats(model, writer, step):
    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param, step)
        if param.grad is not None:
            writer.add_histogram(f'grads/{name}', param.grad, step)

# Log prediction confidence
def log_confidence(outputs, labels, writer, step):
    probs = F.softmax(outputs, dim=1)
    confidence = probs.max(dim=1).values.mean().item()
    writer.add_scalar('metrics/avg_confidence', confidence, step)
    
    # Track samples with >99% confidence (potential overfitting indicator)
    high_conf = (probs.max(dim=1).values > 0.99).sum().item()
    writer.add_scalar('metrics/high_confidence_samples', high_conf, step)
```

### 4.2 Set Up Alerts for Deadlock Detection

```python
import threading
import time

def watchdog(timeout_seconds=300):
    """Monitor training progress and alert if stuck"""
    last_progress = time.time()
    
    while True:
        time.sleep(60)
        if time.time() - last_progress > timeout_seconds:
            logger.error(f"No progress for {timeout_seconds}s - possible deadlock!")
            # Send alert, save checkpoint, etc.

# Start watchdog in separate thread
watchdog_thread = threading.Thread(target=watchdog, daemon=True)
watchdog_thread.start()

# Update last_progress after each epoch
last_progress = time.time()
```

---

## Part 5: Action Items

### Immediate (Before Next Training)
- [ ] Change `num_workers=0` in DataLoader
- [ ] Add EarlyStopping with patience=5
- [ ] Increase dropout to 0.5
- [ ] Add label smoothing (0.1)
- [ ] Implement differential learning rates
- [ ] Add rotation augmentation (±15°)

### Short-term (This Week)
- [ ] Implement data copy to local SSD
- [ ] Add gradient norm logging
- [ ] Create watchdog for deadlock detection
- [ ] Add prediction confidence monitoring
- [ ] Verify train/val split has no overlap

### Long-term (Next Sprint)
- [ ] Collect more diverse data (different scanners, lighting)
- [ ] Try smaller model (MobileNetV3-Small)
- [ ] Implement Mixup/CutMix augmentation
- [ ] Add test-time augmentation for inference
- [ ] Consider synthetic data generation

---

## Appendix: Why This Matters for Production

### The Overfitting Problem in Document Processing

When a model overfits to training data:

1. **Real-world failure:** Won't work on documents from different scanners, lighting, or angles
2. **Security risk:** Could be fooled by adversarial examples
3. **Maintenance nightmare:** Retraining required for every new data source
4. **User trust:** Inconsistent performance damages credibility

### Example of Overfitting Failure

```
Training Data (CNIE only):
┌─────────────────┐
│ Clean scans     │
│ White background│
│ Perfect lighting│
└─────────────────┘
Model learns: "White background = CNIE"

Real-world Input:
┌─────────────────┐
│ Shadow on doc   │
│ Yellow tint     │
│ Slight rotation │
└─────────────────┘
Model: "This doesn't look like training data!"
Result: Misclassification or low confidence
```

### The Business Impact

| Metric | Overfit Model | Properly Regularized Model |
|--------|--------------|---------------------------|
| Training Accuracy | 100% | 95% |
| Validation Accuracy | 100% (suspicious) | 92% |
| Real-world Accuracy | 60-70% | 88-92% |
| User Complaints | High | Low |
| Maintenance Cost | High (constant retraining) | Low |

---

**Document Version:** 1.0  
**Author:** Kimi Code Analysis  
**Next Review:** After next training run
