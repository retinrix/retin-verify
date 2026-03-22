# V7 Emergency Fix Plan

**Status:** 🔴 CRITICAL - V6 Stage 2 Failure  
**Objective:** Fix front/back classification immediately  
**Timeline:** 2-3 hours

---

## Problem Statement

V6 Stage 2 model has **30.9% accuracy** on feedback images (worse than random). The model is severely biased toward predicting "back" for all inputs.

**Root Cause:** Training data pipeline corruption or model architecture unsuitable for the task.

---

## V7 Solution: Verified Data Pipeline + Simpler Model

### Key Changes from V6

1. **Local Training** (not Colab) - Verify data pipeline
2. **ResNet18** (not EfficientNet-B0) - Better generalization
3. **Verified Labels** - Manual check of 100 random samples
4. **3-Class Single Stage** - Simpler architecture

---

## Implementation Steps

### Phase 1: Data Verification (30 min)

```bash
# 1. Create verified dataset
mkdir -p ~/retin-verify/data/verified/v7/{train,val}/{front,back,no_card}

# 2. Manually verify 50 random samples from each class
# Copy verified samples to v7 dataset
```

**Verification Script:**
```python
# verify_dataset.py - Interactive label verification
# Shows images one by one, user confirms correct label
```

### Phase 2: Local Training (60 min)

**train_v7_simple.py:**
```python
"""
V7: Simple ResNet18 3-class classifier
Verified data pipeline, local training
"""
import torch
import torchvision.models as models

# Use ResNet18 (better generalization than EfficientNet)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes

# Train locally with verified data
# ...
```

### Phase 3: Rigorous Validation (30 min)

```python
# validate_v7.py
# Test on 100% of feedback images
# Require >85% accuracy before deployment
```

### Phase 4: Deployment (15 min)

Update inference engine with V7 model.

---

## Alternative: Quick Hotfix for V6

If we need a quick fix without retraining:

### Option A: Swap Stage 2 Predictions

In `inference_engine_v6_cascade.py`:

```python
# HOTFIX: Swap front/back predictions due to training label swap
if front_prob > FRONT_THRESHOLD:
    predicted_class = 'cnie_back'  # Swapped!
else:
    predicted_class = 'cnie_front'  # Swapped!
```

**Test this hypothesis:**
```bash
# Manually test 10 front images
# If they all predict back -> swap needed
```

### Option B: Confidence Threshold Override

```python
# If model is biased, use threshold to counteract
FRONT_THRESHOLD = 0.10  # Very low threshold to favor front
```

---

## V7 Architecture

```
Input Image (224x224)
    ↓
ResNet18 Backbone (pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
FC Layer (512 → 3)
    ↓
Softmax → [front_prob, back_prob, no_card_prob]
```

**Advantages:**
- Simpler than EfficientNet (less overfitting)
- Proven architecture for small datasets
- Faster inference
- Easier to debug

---

## Data Strategy

### Training Data Mix

| Source | Front | Back | No-Card | Total |
|--------|-------|------|---------|-------|
| Original V6 Stage 1 | 64 | 71 | 84 | 219 |
| Feedback (verified) | 80 | 80 | 50 | 210 |
| **Total** | **144** | **151** | **134** | **429** |

### Augmentation Strategy

```python
transforms = [
    # Geometric
    RandomRotation(15),
    RandomAffine(degrees=0, translate=(0.1, 0.1)),
    
    # Photometric
    ColorJitter(brightness=0.2, contrast=0.2),
    
    # NO horizontal flip (would swap left/right on card)
]
```

---

## Validation Criteria

Before deploying V7, must pass:

| Test | Minimum | Target |
|------|---------|--------|
| Front Accuracy | 85% | 95% |
| Back Accuracy | 85% | 95% |
| No-Card Accuracy | 90% | 98% |
| Feedback Images | 80% | 90% |

---

## Rollback Plan

If V7 also fails:
1. Revert to V5 single-stage model
2. Implement confidence threshold (only accept >0.9)
3. Add "uncertain" category for low confidence
4. Manual review queue for uncertain predictions

---

## Immediate Action Required

**Choose one:**

1. **Test V6 Label Swap Hypothesis** (15 min)
   - Manually check if swapping predictions fixes accuracy
   - If yes, deploy hotfix immediately

2. **Start V7 Training** (2-3 hours)
   - Verified data pipeline
   - ResNet18 architecture
   - Local training with full validation

**Recommendation:** Test hypothesis first (15 min), then decide.

---

## Test Protocol

```bash
# 1. Test hypothesis
cd ~/retin-verify/inference/apps/classification/backend
python3 test_v6_hypothesis.py

# 2. If hypothesis confirmed, apply hotfix
# Edit inference_engine_v6_cascade.py to swap predictions

# 3. Validate hotfix on 50 feedback images
python3 validate_hotfix.py

# 4. Deploy if accuracy > 85%
```

---

## Files to Create

1. `verify_dataset.py` - Interactive label verification
2. `train_v7_simple.py` - ResNet18 training
3. `test_v6_hypothesis.py` - Test label swap hypothesis
4. `validate_hotfix.py` - Post-hotfix validation

---

**Decision Point:** Proceed with hypothesis testing or full V7 retrain?
