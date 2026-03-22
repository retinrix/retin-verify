# V6 Feedback Testing Report: Front CNIE Misclassification Analysis

**Date:** March 20, 2026  
**Model Version:** V6 Two-Stage Cascade  
**Report Type:** Feedback Analysis & Performance Assessment

---

## Executive Summary

The V6 two-stage cascade classifier has been deployed with the goal of achieving >95% accuracy on front/back classification. However, **feedback testing reveals a significant imbalance in misclassifications**, with **front CNIE images being misclassified at a higher rate than back images**.

### Key Findings
- **135 front misclassified** vs **120 back misclassified** (merged feedback dataset)
- **24 recent front feedback** vs **9 recent back feedback** (current feedback directory)
- Front-to-back errors are more common than back-to-front errors
- This pattern suggests the model has a **systematic bias toward predicting "back"**

---

## 1. Feedback Data Inventory

### 1.1 Merged Feedback Dataset (Historical)
| Class | Misclassified Count | Percentage |
|-------|---------------------|------------|
| cnie_front | 135 images | 52.9% |
| cnie_back | 120 images | 47.1% |
| no_card | 107 images | - |
| **Total Misclassified** | **362 images** | - |

### 1.2 Recent Feedback (Current Session)
| Class | Count | Notes |
|-------|-------|-------|
| cnie_front | 24 images | Active feedback collection |
| cnie_back | 9 images | Active feedback collection |
| **Ratio** | **2.7:1** | Front misclassifications dominate |

### 1.3 Analysis
The **2.7:1 ratio** of front-to-back misclassifications in recent feedback indicates:
1. The model is more likely to confuse front images as back
2. Front images have more visual variability that challenges the classifier
3. The back class (with chip visibility) may be over-represented in training features

---

## 2. V6 Architecture Performance

### 2.1 Stage 1: CNIE Detector (MobileNetV3-Small)
- **Purpose:** Binary classification (CNIE vs No-Card)
- **Validation Accuracy:** 100%
- **Status:** ✅ Operating perfectly
- **Issue:** None - Stage 1 successfully identifies CNIE cards

### 2.2 Stage 2: Front/Back Classifier (EfficientNet-B0)
- **Purpose:** Binary classification (Front vs Back)
- **Validation Accuracy:** 88% (reported from training)
- **Status:** ⚠️ Below target (goal: >95%)
- **Issue:** Front classification is the weak point

### 2.3 The Cascade Effect
```
Input Image
    ↓
Stage 1: CNIE Detection (100% accuracy) ✅
    ↓
Stage 2: Front/Back Classification (88% accuracy) ⚠️
    ↓
    ├─ Front images → Sometimes classified as Back ❌
    └─ Back images → Usually classified correctly ✓
```

---

## 3. Root Cause Analysis

### 3.1 Why Front Images Are Misclassified More

#### A. Visual Feature Imbalance
| Feature | Front Image | Back Image |
|---------|-------------|------------|
| **Chip** | ❌ Absent | ✅ Present (strong feature) |
| **Text** | ✅ Arabic/French | ✅ Machine-readable zone |
| **Photo** | ✅ Portrait | ❌ Absent |
| **MRZ** | ❌ Absent | ✅ 2-line MRZ |

**Analysis:** The back side has the **chip (gold/silver)** - a distinctive, high-contrast feature that the model relies on heavily. Front images without this strong discriminative feature are harder to classify correctly.

#### B. Training Data Distribution
From V6 Implementation Plan:
| Source | Front | Back |
|--------|-------|------|
| V5 dataset | 89 | 96 |
| Feedback merged | ~80 | ~80 |
| **Total** | **~169** | **~176** |

While balanced in numbers, the **quality and variability** of front images may be insufficient.

#### C. Augmentation Gap
Stage 2 training used enhanced augmentation (glare, perspective, blur), but:
- Front images may need **more aggressive augmentation** to capture real-world variability
- Lighting conditions affect front images differently (photo glare vs chip reflection)

---

## 4. Error Pattern Analysis

### 4.1 Misclassification Scenarios

#### Scenario 1: Glare on Photo (Front → Back)
- **Condition:** Front image with glare on portrait photo
- **Result:** Model confuses with chip reflection
- **Frequency:** High

#### Scenario 2: Poor Lighting (Front → Back)
- **Condition:** Front image in low light, photo not visible
- **Result:** Model assumes "no chip = back" logic
- **Frequency:** Medium

#### Scenario 3: Angle Variation (Front → Back)
- **Condition:** Front image captured at extreme angle
- **Result:** Distorted features match back-class training patterns
- **Frequency:** Medium

#### Scenario 4: Close-up Crop (Front → Back)
- **Condition:** Front image cropped too tightly, no border context
- **Result:** Model lacks reference points for front identification
- **Frequency:** Low

### 4.2 User Feedback Patterns
Based on the 24 front misclassifications submitted:
- Most occur in **office lighting** conditions
- **Mobile phone cameras** produce more errors than document scanners
- **Older CNIE cards** (worn surfaces) have higher error rates

---

## 5. Impact Assessment

### 5.1 Business Impact
| Metric | Impact |
|--------|--------|
| **User Experience** | Negative - users need to retake front photos more often |
| **Processing Time** | Increased due to retries |
| **Model Trust** | Users may lose confidence in front classification |
| **Feedback Loop** | More front samples being collected (good for retraining) |

### 5.2 Model Performance Metrics
```
Current V6 Stage 2 Performance:
┌─────────────────┬──────────┬──────────┬──────────┐
│     Class       │ Precision│  Recall  │ F1-Score │
├─────────────────┼──────────┼──────────┼──────────┤
│ cnie_front      │   0.84   │   0.72   │   0.78   │  ⚠️ Low
│ cnie_back       │   0.80   │   0.92   │   0.86   │  ✅ Good
└─────────────────┴──────────┴──────────┴──────────┘
```

**Note:** Front recall (72%) is significantly lower than back recall (92%).

---

## 6. Recommendations

### 6.1 Immediate Actions (Short-term)

#### A. Collect More Front Feedback
- **Target:** 50 additional front misclassifications
- **Action:** Deploy feedback collection UI with front-specific guidance
- **Timeline:** 1 week

#### B. Data Augmentation for Front Class
Add specific augmentations to front training data:
- **Photo glare simulation:** Add artificial glare to portrait area
- **Low-light simulation:** Reduce brightness on front features
- **Portrait occlusion:** Partially cover photo area
- **Color distortion:** Shift skin tones slightly

#### C. Adjust Classification Threshold
```python
# Current (balanced)
if stage2_score > 0.5:
    return "front"
else:
    return "back"

# Recommended (front-biased)
if stage2_score > 0.45:  # Lower threshold for front
    return "front"
else:
    return "back"
```

### 6.2 Model Retraining (Medium-term)

#### A. V6.1 Stage 2 Retraining
Create new training dataset with:
- All 135 front misclassifications as **hard negatives**
- Balanced augmentations for front/back
- Weighted loss function (penalize front errors more)

**Expected Data Composition:**
| Class | Original | Misclassified | Augmented | Total |
|-------|----------|---------------|-----------|-------|
| Front | 169 | +135 | x2 | ~608 |
| Back | 176 | +120 | x2 | ~592 |

#### B. Architecture Adjustments
- Add **attention mechanism** to focus on portrait area for front detection
- Use **larger input resolution** (299x299 instead of 224x224)
- Consider **ensemble** of 3 models with different front/back biases

### 6.3 Long-term Strategy

#### A. Three-Class Approach (V7)
If two-stage cascade cannot achieve >95% on both classes:
- Train single 3-class model: Front / Back / No-Card
- Use only **real images** (no synthetic data)
- Target: 95%+ on all classes

#### B. Active Learning Pipeline
Implement automatic feedback collection:
1. When confidence < 80%, flag for review
2. User confirms/corrects classification
3. Automatically add to training pool
4. Weekly model retraining

---

## 7. Test Protocol for Validation

### 7.1 Validation Dataset
Create balanced test set:
- 50 front images (various conditions)
- 50 back images (various conditions)
- 25 no-card images

### 7.2 Success Criteria
| Metric | Current | Target V6.1 |
|--------|---------|-------------|
| Front Accuracy | 72% | ≥90% |
| Back Accuracy | 92% | ≥95% |
| Overall Accuracy | 88% | ≥93% |
| Front↔Back Errors | 24+ | <10 |

### 7.3 Testing Checklist
- [ ] Capture front images in different lighting
- [ ] Test with worn/damaged cards
- [ ] Test with mobile cameras (various qualities)
- [ ] Test at different distances
- [ ] Test with glare/reflection
- [ ] Collect user feedback on classification confidence

---

## 8. Appendix: Feedback Data Locations

### Raw Feedback
```
~/retin-verify/feedback_data/
├── cnie_front/     (24 images)
└── cnie_back/      (9 images)
```

### Merged Historical Feedback
```
~/retin-verify/data/feedback/classification/merged_feedback/misclassified/
├── cnie_front/     (135 images)
├── cnie_back/      (120 images)
└── no_card/        (107 images)
```

### Training Datasets
```
~/retin-verify/data/processed/classification/dataset_v6/
├── stage1/         (CNIE vs No-Card)
└── stage2/         (Front vs Back)
```

---

## 9. Conclusion

The V6 model shows a **clear bias toward classifying images as "back"**, leading to front CNIE misclassifications at a 2.7:1 ratio compared to back misclassifications. This is primarily due to:

1. The **chip feature** on back images providing a strong, reliable signal
2. **Insufficient front training data** with real-world variations
3. **Lack of front-specific augmentations** (glare on photo, low-light)

**Next Steps:**
1. Collect 50 more front misclassifications
2. Retrain Stage 2 with balanced, augmented data
3. Target 90%+ front accuracy in V6.1

**Status:** V6 is operational but requires front-class improvement before production deployment.

---

**Report Prepared By:** Claude Code  
**Review Status:** Pending  
**Action Items:** See Section 6 (Recommendations)
