# Retraining Report - March 18, 2026

## Executive Summary

Three retraining cycles completed with incremental improvements in dataset size but revealing fundamental data quantity issues.

---

## Retraining History

### Cycle 1 (v1 Model)
| Metric | Value |
|--------|-------|
| Samples | 10 |
| Val Accuracy | 50% |
| Status | Deployed |

### Cycle 2 (v2 Model)
| Metric | Value |
|--------|-------|
| Samples | 25 |
| Val Accuracy | 44.44% |
| Status | Deployed |

### Cycle 3 (v3 Weighted Model) - CURRENT
| Metric | Value |
|--------|-------|
| Samples | 46 |
| **Best Val Accuracy** | **42.1%** (Epoch 3) |
| **Class Balance** | **26.7%** |
| Training Acc | 94% (overfitting) |
| Status | **Just Deployed** |

---

## Key Finding: OVERFITTING

```
Training Accuracy  ████████████████████ 94%
Validation Accuracy ██░░░░░░░░░░░░░░░░░░ 21% (final)
                    ↑
            Massive gap = OVERFITTING
```

### What This Means
- Model memorized the 36 training images
- Cannot generalize to new images (validation)
- FRONT validation accuracy dropped to 0% by end

### Why It Happened
1. **Too few samples**: 36 train + 19 val = 55 total
2. **Too much diversity**: Your images vary significantly in:
   - Distance from camera (card size in frame)
   - Angle/rotation
   - Lighting conditions
   - Background clutter
3. **No correct samples**: All 46 are errors - no positive examples

---

## Bias Analysis

### Confusion Pattern (46 samples)

```
                 PREDICTED
              FRONT    BACK
Actual FRONT    ✓       28   ← 61% error rate
       BACK    18       ✓    ← 39% error rate
```

**Finding**: Model has BACK bias - over-predicts "back" side

### Root Cause
The model learned that predicting "back" is statistically safer (less penalty in original training), but your feedback data shows FRONT images are being misclassified more.

---

## Recommendations

### Immediate Actions

1. **Test Current Model (v3)**
   - Try at http://127.0.0.1:8000
   - May not be better than v2 due to overfitting
   - If worse, we can rollback to v2

2. **Collect More Data - Priority Actions**
   
   a) **Target 100+ total samples**
      - Currently: 46
      - Need: 54+ more
   
   b) **Include CORRECT predictions**
      - When model is RIGHT, click "✓ Confirm"
      - Target: 50 correct + 50 errors
      - Currently: 0 correct, 46 errors
   
   c) **Standardize capture quality**
      
      | Aspect | Current | Target |
      |--------|---------|--------|
      | Card size | ~10% of frame | 50%+ of frame |
      | Angle | Various | Flat, facing camera |
      | Lighting | Mixed | Bright, even |
      | Background | Room visible | Plain/neutral |
      | Hand position | Covers card | Hold by edges |

### Long-term Strategy

1. **Combine with original synthetic data**
   ```
   Original synthetic: 1000+ images
   Real feedback: 100+ images  
   Retrain from scratch with combined dataset
   ```

2. **Two-stage training process**
   - Stage 1: Train on large synthetic dataset
   - Stage 2: Fine-tune on real feedback
   - This prevents overfitting

3. **Add data augmentation pipeline**
   - Simulate your specific conditions
   - Generate synthetic variations

---

## Rollback Option

If v3 performs worse than v2, rollback:

```bash
cd ~/retin-verify/models/classification

# Restore v2
cp cnie_front_back_real_backup_20260318_204218.pth cnie_front_back_real.pth

# Restart server
pkill -f api_server.py
./start_server.sh
```

---

## Next Steps

1. **Test v3 model** at http://127.0.0.1:8000
2. **Collect 50+ more samples** with better quality
3. **Include correct confirmations** (not just errors)
4. **Retry retraining** when you have 100+ samples

---

## Model Versions Available

```
cnie_front_back_real.pth                    ← CURRENT (v3 Weighted)
cnie_front_back_real_backup_v2_*.pth        ← v2 (44% val)
cnie_front_back_real_backup_20260318_*.pth  ← v1 (50% val)
```

**Recommendation**: Keep v1 as fallback - it had highest validation accuracy despite fewer samples.
