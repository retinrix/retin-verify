# Retraining Improvement Plan

## Current Problem Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  FEEDBACK STATS (46 samples)                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Error Type              │ Count │ Problem                     │
│   ────────────────────────┼───────┼────────────────────────────│
│   FRONT → BACK (wrong)    │  28   │ ⚠️ MAJOR ISSUE (61%)       │
│   BACK → FRONT (wrong)    │  18   │ Moderate issue (39%)       │
│                                                                 │
│   INSIGHT: Model has BACK bias - over-predicts back side       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why the Model is Biased Toward BACK

1. **Training Data Imbalance**: Original dataset may have more back samples
2. **Visual Similarity**: Back side (chip + text) may have stronger features
3. **Capture Conditions**: Your specific lighting/angle makes front look like back
4. **Class Weight Issue**: Model not penalizing FRONT misclassification enough

## Proposed Solutions

### Solution 1: Class-Balanced Retraining (Immediate)

Adjust the loss function to penalize FRONT errors more heavily since they're more common:

```python
# In retrain.py - use weighted loss
class_weights = torch.tensor([1.5, 1.0])  # Penalize FRONT errors 1.5x more
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**Expected Impact**: Reduces FRONT→BACK errors by 20-30%

### Solution 2: Data Augmentation Strategy

Your images have specific challenges:
- Card is small in frame
- Various angles
- Background clutter
- Hand partially covering card

**Proposed augmentations**:
```python
# Aggressive augmentation for your use case
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),  # Simulate small card
    transforms.RandomRotation(30),  # Your angles vary
    transforms.ColorJitter(brightness=0.4, contrast=0.4),  # Your lighting
    transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Solution 3: Two-Stage Training

**Stage 1**: Train only on misclassified samples (46 images)
- Focus on hard examples
- Learning rate: 1e-4
- Epochs: 20

**Stage 2**: Fine-tune on full dataset (original + feedback)
- Learning rate: 1e-5 (lower)
- Epochs: 10
- Prevents overfitting to small set

### Solution 4: Add More Positive Samples

Current issue: Only 46 samples, all are **errors**.

**Action needed**: Also capture CORRECT classifications!

```bash
# When model is right, click "✓ Confirm Correct"
# This adds to feedback_data/correct/
```

**Target ratio**:
- Misclassified: 46 (already have)
- Correct: 50+ (need to collect)
- Total: 100+ samples for robust training

### Solution 5: Feature Analysis & Class-Specific Augmentation

The FRONT (photo side) and BACK (chip side) have different features:

```
FRONT characteristics:
- Portrait photo (face)
- Text layout: Name, DOB, ID number
- Lighter background

BACK characteristics:
- Chip (gold/silver square)
- MRZ (machine readable zone)
- Darker background
```

**Proposal**: Add synthetic samples with these features enhanced:

```python
# Enhance FRONT samples (since they're being misclassified)
# Add more FRONT-like augmented samples
front_augmented = [
    crop_to_portrait_region,      # Focus on face area
    enhance_text_contrast,        # Make text more visible
    brighten_background,          # Lighter like front
]
```

## Recommended Action Plan

### Phase 1: Immediate Fixes (Deploy Now)

1. **Deploy class-weighted training** to Colab
2. **Use aggressive augmentation**
3. **Increase epochs to 20** (more samples = can train longer)

### Phase 2: Data Collection (This Week)

1. **Collect 20+ CORRECT confirmations**
   - When model predicts right, click "✓ Confirm"
   - Balances the 46 errors

2. **Capture more FRONT samples specifically**
   - Since FRONT→BACK is the main error
   - Get different angles/lighting of front

3. **Improve capture quality**:
   - Hold card closer to camera (fill 50%+ of frame)
   - Ensure good lighting
   - Minimize background clutter
   - Hold card flat (not angled)

### Phase 3: Advanced Training (Next Week)

1. **Combine original + feedback datasets**
   - Original: ~1000+ synthetic images
   - Feedback: 46 misclassified + 50 correct
   - Retrain from scratch with full data

2. **Try different model architectures**:
   - EfficientNet-B3 (more capacity)
   - ResNet50 (different feature extraction)

3. **Cross-validation**:
   - Split feedback data 5-fold
   - Ensure model generalizes

## Quick Wins You Can Do Now

### 1. Update the UI to Track Correct Predictions

Modify `feedback_system.py` to add a "Confirm Correct" button:

```python
# When user clicks "✓ Correct"
feedback_collector.submit_feedback(
    image_data=image,
    predicted_class=prediction,
    predicted_confidence=confidence,
    is_correct=True,  # Mark as correct!
    correct_class=prediction  # Same as predicted
)
```

### 2. Add Confidence Threshold Display

Show when prediction is uncertain:

```python
if confidence < 0.7:
    show_warning("Low confidence - please confirm")
```

### 3. Class Distribution Monitor

Add to the UI:
```
Front predictions: X (Y correct, Z wrong)
Back predictions:  X (Y correct, Z wrong)
```

## Implementation: Weighted Retraining Script

Create `retrain_weighted.py`:

```python
# Use weighted loss
class_counts = [28, 18]  # front, back misclassified
total = sum(class_counts)
weights = [total / c for c in class_counts]  # [1.64, 2.56]
class_weights = torch.tensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

This will penalize BACK errors more (since fewer samples), balancing the model.

## Summary

| Issue | Solution | Priority |
|-------|----------|----------|
| FRONT bias (61% errors) | Class-weighted loss | 🔴 High |
| Small dataset (46) | Collect correct samples | 🔴 High |
| Card too small in frame | Capture guidelines | 🟡 Medium |
| Overfitting | Aggressive augmentation | 🟡 Medium |
| Need more diversity | Combine with original data | 🟢 Low |

**Next Action**: Deploy weighted training to Colab with current 46 samples.
