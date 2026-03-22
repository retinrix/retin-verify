# v5 Implementation Strategy - Summary

## Problem Statement (from Analysis)

The v4 model suffers from **domain shift**: it was trained primarily on synthetic data that doesn't match real-world CNIE photos. This causes symmetric confusion between front and back classes (135 front→back, 120 back→front errors).

**Key Evidence:**
- 2-class real-only model achieved 99-100% accuracy
- v4 with synthetic data: 92% front, 96% back accuracy
- Symmetric confusion indicates poor feature separation

---

## Solution: v5 Real-Data-Only Training

### Core Changes

| Aspect | v4 | v5 |
|--------|-----|-----|
| **Training Data** | Real + Synthetic (10K) | **Real Only** (~300) |
| **Learning Rate** | Uniform 1e-4 | **Differential**: backbone 1e-5, classifier 1e-3 |
| **Layer Freezing** | First 100 layers frozen | **Progressive unfreezing** |
| **Augmentation** | Standard | **Enhanced** (rotation 30°, shear, blur, noise) |
| **Loss Function** | CrossEntropy | **Label smoothing** (0.1) |
| **Batch Size** | 32 | **16** (smaller for real data) |

### Progressive Training Phases

```
Phase 1 (Epochs 1-5):    Freeze backbone, train classifier only
Phase 2 (Epochs 6-15):   Unfreeze last 50 layers  
Phase 3 (Epochs 16-150): Full unfreeze with differential LR
```

---

## Files Created

| File | Purpose |
|------|---------|
| `docs/V5_IMPLEMENTATION_STRATEGY.md` | Detailed implementation plan |
| `training/classification/v5_training/train_v5_real_only.py` | v5 training script |
| `docs/V5_IMPLEMENTATION_SUMMARY.md` | This summary |

---

## Prerequisites Before Training

### 1. Real Image Inventory

Count current real images:

```bash
# Check existing real data
echo "Original dataset:"
find ~/retin-verify/data/processed/classification/dataset_3class -name "*.jpg" | wc -l

echo "Feedback images (new):"
find ~/retin-verify/data/feedback/classification/merged_feedback -name "*.jpg" | wc -l
```

**Minimum Required:**
- Front: 50 images (ideally 100+)
- Back: 50 images (ideally 100+)
- No Card: 50 images (ideally 100+)

### 2. Data Preparation

If insufficient data, collect more via feedback system before training.

If sufficient data, prepare dataset:

```bash
# Create real-only dataset
mkdir -p ~/retin-verify/data/processed/classification/dataset_v5/{train,val}/{cnie_front,cnie_back,no_card}

# Copy original real images
cp ~/retin-verify/data/processed/classification/dataset_3class/train/*/* \
   ~/retin-verify/data/processed/classification/dataset_v5/train/

cp ~/retin-verify/data/processed/classification/dataset_3class/val/*/* \
   ~/retin-verify/data/processed/classification/dataset_v5/val/

# Package for upload
cd ~/retin-verify/data/processed/classification
tar -czf dataset_v5_real.tar.gz dataset_v5/
```

---

## Deployment Commands

### Option 1: Using Automation Tools

```bash
# Load automation
source ~/.kimi/autoload_training_automation.sh

# Deploy v5 (after updating script path in automation)
quick_deploy YOUR_HOST.trycloudflare.com
```

### Option 2: Manual Deployment

```bash
HOST="your-host.trycloudflare.com"
PASS="retinrix"
SSHPASS="/tmp/sshpass/usr/bin/sshpass"

# Upload dataset
$SSHPASS -p "$PASS" scp dataset_v5_real.tar.gz root@$HOST:/tmp/

# Upload script
$SSHPASS -p "$PASS" scp train_v5_real_only.py root@$HOST:/tmp/

# Setup and train
$SSHPASS -p "$PASS" ssh root@$HOST "
mkdir -p /content/retin_v5
cd /content/retin_v5
tar -xzf /tmp/dataset_v5_real.tar.gz
cp /tmp/train_v5_real_only.py .
nohup python3 train_v5_real_only.py > train_v5.log 2>&1 &
"
```

---

## Expected Results

| Metric | v4 | v5 Target |
|--------|-----|-----------|
| Front Accuracy | 92% | **98%+** |
| Back Accuracy | 96% | **98%+** |
| No Card Accuracy | 97% | **98%+** |
| Balance | 95.0% | **97%+** |
| Front→Back Errors | 135 | **< 20** |
| Back→Front Errors | 120 | **< 20** |

---

## Timeline

| Phase | Duration |
|-------|----------|
| Data audit | 15 min |
| Data preparation (if needed) | 30 min |
| Deploy to Colab | 15 min |
| Training (150 epochs max) | 2-3 hours |
| Download & test | 15 min |
| **Total** | **3-4 hours** |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Not enough real data | Collect more via feedback UI first |
| Overfitting | Heavy augmentation + early stopping |
| Worse than v4 | Keep v4 as fallback; A/B test |
| Training unstable | Progressive unfreezing prevents this |

---

## Next Steps

1. **Audit current real image count**
2. **If < 50 per class**: Collect more via feedback system
3. **If ≥ 50 per class**: Proceed with v5 training
4. **Deploy to Colab** and monitor
5. **Test on held-out real images**
6. **A/B test v4 vs v5** before full deployment

---

## Key Insight

> The 2-class real-only model achieved 99-100% accuracy. This proves that **real data alone is sufficient** and synthetic data is causing domain shift. v5 eliminates synthetic data entirely.

---

**Ready to proceed? First, audit your real image inventory.**
