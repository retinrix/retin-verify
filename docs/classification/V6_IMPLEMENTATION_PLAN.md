# V6 Implementation Plan - Two-Stage Cascade Architecture

## Executive Summary

Based on the V6_MODEL_STRATEGY.md, we implement a **two-stage cascade** that decouples CNIE detection from front/back classification. This eliminates synthetic data bias and achieves >95% accuracy on real images.

**Key Insight:** The 2-class real-only model previously achieved 99-100% accuracy. V6 replicates this success by removing the problematic 3-class approach.

---

## Current Data Inventory

### Real Images Available:

| Source | Front | Back | No-Card | Total |
|--------|-------|------|---------|-------|
| V5 dataset (train) | 64 | 71 | 84 | 219 |
| V5 dataset (val) | 25 | 25 | 37 | 87 |
| Feedback merged (misclassified) | ~80 | ~80 | ~50 | ~210 |
| Feedback retraining dataset | 20 | 18 | 0 | 38 |
| **TOTAL REAL IMAGES** | **~189** | **~194** | **~171** | **~554** |

### Data Sufficiency Assessment:

✅ **Stage 1 (CNIE vs no-card):** SUFFICIENT
- Positive (CNIE): ~383 images (front + back)
- Negative (no-card): ~171 images
- Total: ~554 real images

✅ **Stage 2 (Front vs Back):** SUFFICIENT  
- Front: ~189 images
- Back: ~194 images
- Well-balanced classes

**Conclusion:** No additional real images needed. We have sufficient data for V6.

---

## V6 Architecture

```
Input Image
    ↓
┌─────────────────────────────────────┐
│ Stage 1: CNIE Detector              │
│ Model: MobileNetV3-Small            │
│ Classes: CNIE / No-Card             │
│ Data: All real images               │
└─────────────┬───────────────────────┘
              │
        ┌─────┴─────┐
        ↓           ↓
    No-Card      CNIE
    (reject)      ↓
           ┌──────────────────────────┐
           │ Stage 2: Front/Back      │
           │ Model: EfficientNet-B0   │
           │ Classes: Front / Back    │
           │ Data: Real CNIE only     │
           └────────────┬─────────────┘
                        ↓
                 Front    Back
```
V6 Cascade: Stage 1 → Stage 2 Connection
How They Connect
The connection is decision-based, not feature-based:

┌─────────────────────────────────────────────────────────────┐
│  Stage 1 (MobileNetV3)        Stage 2 (EfficientNet-B0)    │
│  ───────────────────          ─────────────────────────    │
│  Input: Image                  Input: SAME Image (raw)     │
│  Output: [CNIE, No-Card]       Output: [Front, Back]       │
│                                                              │
│  Decision Gate:                                              │
│  ├─ If No-Card → RETURN (Stage 2 skipped)                  │
│  └─ If CNIE → Run Stage 2                                   │
└─────────────────────────────────────────────────────────────┘
Key Points
Aspect	How It Works
Data Flow	Same raw image goes to BOTH stages independently
Feature Sharing	❌ NO - Stages don't share features
Decision Logic	Stage 1 decides IF Stage 2 runs
Sequential	Stage 1 → (gate) → Stage 2 (if needed)
Confidence	Combined: p_cnie × p_front/back
Code Flow (Simplified)
def predict(image):
    image_tensor = transform(image)
    
    # Stage 1: Always runs
    p_cnie, p_no_card = stage1_model(image_tensor)
    
    # Gate: Stage 2 only runs if CNIE detected
    if p_no_card > p_cnie:
        return "no_card"  # Stage 2 SKIPPED
    
    # Stage 2: Only for CNIE images
    p_front, p_back = stage2_model(image_tensor)
    
    if p_front > 0.45:
        return "cnie_front"
    else:
        return "cnie_back"
Why This Design?
✅ Stage 1 filters non-CNIE (100% accuracy on detection)
✅ Stage 2 is simpler (only 2 classes: Front/Back)
✅ Independent training (no coupling between stages)
✅ Efficient (Stage 2 only runs when needed)
The "connection" is a conditional trigger - Stage 1's output decides whether Stage 2 executes, but both process the same raw input image independently.
---

## Implementation Steps

### Phase 1: Archive V4/V5 (SESSION_RULES.md compliance)

```bash
# Archive to models/archive/2026-03-20/
- V4 model and scripts
- V5 model and scripts  
- Dataset tar.gz files
- Update file structure
```

### Phase 2: Prepare V6 Datasets

**Stage 1 Dataset Structure:**
```
dataset_v6_stage1/
├── train/
│   ├── cnie/        # All front + back images
│   └── no_card/     # All no-card images
└── val/
    ├── cnie/
    └── no_card/
```

**Stage 2 Dataset Structure:**
```
dataset_v6_stage2/
├── train/
│   ├── front/       # Front images + misclassified backs
│   └── back/        # Back images + misclassified fronts
└── val/
    ├── front/
    └── back/
```

### Phase 3: Training Scripts

**train_v6_stage1.py:**
- Model: MobileNetV3-Small (binary classifier)
- Data: dataset_v6_stage1
- Augmentation: Standard (rotation, color jitter)
- Epochs: 20-30
- Target: >98% accuracy

**train_v6_stage2.py:**
- Model: EfficientNet-B0 (binary classifier)
- Data: dataset_v6_stage2
- Augmentation: Enhanced (glare, perspective, blur, MixUp/CutMix)
- Epochs: 50
- Target: >98% accuracy

### Phase 4: Cascade Inference Engine

**inference_engine_v6.py:**
```python
class V6CascadeClassifier:
    def __init__(self):
        self.stage1 = Stage1Model()  # CNIE detector
        self.stage2 = Stage2Model()  # Front/back
    
    def predict(self, image):
        # Stage 1: Is this a CNIE?
        is_cnie = self.stage1.predict(image)
        if not is_cnie:
            return {'class': 'no_card', 'confidence': conf}
        
        # Stage 2: Front or back?
        side = self.stage2.predict(image)
        return {'class': f'cnie_{side}', 'confidence': conf}
```

---

## Expected Results

| Metric | V5 | V6 Target |
|--------|-----|-----------|
| Front Accuracy | 72% | **≥95%** |
| Back Accuracy | 80% | **≥95%** |
| No-Card Accuracy | 97% | **≥98%** |
| Front↔Back Errors | 13 | **<5** |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Archive V4/V5 | 15 min | Clean structure per SESSION_RULES.md |
| Prepare datasets | 30 min | dataset_v6_stage1 & stage2 |
| Stage 1 training | 30 min | MobileNetV3 binary model |
| Stage 2 training | 1 hour | EfficientNet-B0 binary model |
| Integration | 30 min | Cascade inference engine |
| Testing | 15 min | Validation on held-out images |
| **Total** | **~3 hours** | V6 complete |

---

## Next Actions

1. **Archive V4/V5** per SESSION_RULES.md
2. **Prepare V6 datasets** with feedback integration
3. **Implement training scripts** for both stages
4. **Run Colab training** following session rules
5. **Deploy cascade system** with v6 inference engine

---

**Status:** Ready to implement  
**Data Status:** Sufficient (no collection needed)  
**Priority:** High - addresses core accuracy issues
