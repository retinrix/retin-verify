# v3 Training Approach: From Scratch

## Problem with Current Model (v2)

The v2 model has fundamental confusion between front and back classes:

| Issue | Example |
|-------|---------|
| Front→Back | F=0.20, B=0.79 (needs 60% bias to flip) |
| Front→Back | F=0.11, B=0.88 (needs 80% bias to flip) |
| Back→Front | F=0.69, B=0.28 (raw scores favor front) |

**Root causes:**
1. Model adapted from 2-class base (front vs back originally)
2. Classifier head randomly initialized
3. No proper class balancing during training
4. Weak data augmentation

## v3 Solution: Train From Scratch

### Key Improvements

1. **ImageNet Pre-trained Weights**
   - Use `EfficientNet_B0_Weights.IMAGENET1K_V1`
   - Freeze early layers (0-5), train layers 6-8
   - Proper feature extraction

2. **Strong Data Augmentation**
   - RandomHorizontalFlip (p=0.5)
   - RandomRotation (±15°)
   - ColorJitter (brightness, contrast, saturation)
   - RandomAffine (translation, scaling)

3. **Class Balancing**
   - WeightedRandomSampler for balanced batches
   - Class weights in loss function
   - Ensures all classes represented equally

4. **Better Training Strategy**
   - 50 epochs (vs 30)
   - ReduceLROnPlateau scheduler
   - Different learning rates for backbone vs classifier
   - Optimize for "balance" metric (minimum class accuracy)

### Expected Results

| Metric | v2 (Old) | v3 (Target) |
|--------|----------|-------------|
| Front Acc | 60-80% | 85%+ |
| Back Acc | 80-90% | 85%+ |
| No Card | 100% | 95%+ |
| Balance | 60-70% | 85%+ |

## How to Deploy

### 1. Update HOST in deploy_v3.py
```python
HOST = "your-colab-host.trycloudflare.com"
```

### 2. Run Deployment
```bash
cd ~/retin-verify/apps/classification/colab_retrain/new_training
python3 deploy_v3.py
```

### 3. Monitor Training
```bash
ssh root@your-host "tail -f /content/retin_v3_training/train_v3.log"
```

### 4. Download Model
```bash
scp root@your-host:/content/cnie_classifier_3class_v3.pth \
    ~/retin-verify/models/classification/
```

### 5. Update API Server
Edit `api_server.py` to use new inference engine:
```python
from inference_engine_3class_v3 import get_3class_classifier_v3
```

## Fallback Options

If v3 still has issues, consider:

### Option A: Two-Stage Classifier
1. Stage 1: Card vs No Card (binary)
2. Stage 2: Front vs Back (binary, only if card detected)

### Option B: Larger Model
- Use EfficientNet-B3 or ResNet-50
- More capacity to learn differences

### Option C: More Data
- Collect 200+ samples per class
- Ensure variety in angles, lighting
- Include difficult edge cases

## Files

- `train_from_scratch.py` - Training script
- `deploy_v3.py` - Deployment script
- `inference_engine_3class_v3.py` - Clean inference (no bias)
