# CNIE Classification Session Log
**Date:** 2026-03-19  
**Session Duration:** ~2.5 hours

---

## 🎯 Objectives Completed

1. ✅ Deployed 3-class model v2 to production
2. ✅ Implemented bias correction for front/back imbalance
3. ✅ Created v3 training pipeline (from scratch)

---

## 📊 Current System Status

### Server
- **Status:** Running on http://localhost:8000
- **Model:** cnie_classifier_3class_v2.pth (88.5% val acc)
- **Device:** CPU (GTX 950M incompatible with PyTorch CUDA)

### Performance with 35% Bias
| Class | Accuracy | Notes |
|-------|----------|-------|
| Front | 92% (23/25) | 2 images need >60% bias to correct |
| Back | 68% (17/25) | Tradeoff for front improvement |
| No Card | 100% (10/10) | Excellent |
| **Balance** | **68%** | Min class accuracy |

### Problematic Samples
Two front images consistently misclassified as back:
- `2026-03-18T21:09:15.555377_50332d4d26b7.jpg`: F=0.20, B=0.79
- `2026-03-19T00:15:11.250127_3346ed689a8c.jpg`: F=0.11, B=0.88

---

## 🔧 Configuration Changes Made

### 1. Inference Engine (`backend/inference_engine_3class.py`)
```python
FRONT_BIAS = 0.35      # Added bias toward front class
BIAS_MARGIN = 1.0      # Always apply bias (was 0.25)
NO_CARD_THRESHOLD = 0.70
```

### 2. API Server (`backend/api_server.py`)
- Updated model path to v2: `cnie_classifier_3class_v2.pth`

### 3. Dataset Location
- Training data: `feedback_data_3class/`
- 306 total samples (89 front, 96 back, 121 no_card)

---

## 📁 Files Created/Modified

### Modified Files
```
backend/inference_engine_3class.py    - Added bias correction
backend/api_server.py                 - Updated model path
```

### New Files (v3 Training Pipeline)
```
colab_retrain/new_training/
├── train_from_scratch.py             - Complete v3 training script
├── deploy_v3.py                      - Colab deployment
├── inference_engine_3class_v3.py     - Clean inference (no bias)
└── README_V3.md                      - Documentation
```

---

## 🚀 Next Steps (Pending)

### Option 1: Deploy v3 Training (Recommended)
```bash
cd colab_retrain/new_training
# 1. Update HOST in deploy_v3.py
# 2. python3 deploy_v3.py
# 3. Download model when complete
# 4. Update api_server.py to use v3
```

### Option 2: Two-Stage Classifier
- Stage 1: Card vs No Card (binary)
- Stage 2: Front vs Back (binary, only if card detected)
- More reliable for similar-looking classes

### Option 3: Data Collection
- Collect more front samples (currently 89)
- Target: 150+ per class
- Focus on edge cases and poor lighting

---

## 🐛 Known Issues

1. **GTX 950M CUDA incompatibility** - Forces CPU inference (~90ms vs ~20ms)
2. **Two front images** - Model fundamentally confused, likely need removal
3. **Bias tradeoff** - Improving front hurts back accuracy

---

## 💾 Quick Recovery Commands

```bash
# Check server status
curl http://localhost:8000/health

# Restart server
kill $(lsof -ti:8000)
cd ~/retin-verify/apps/classification/backend
nohup python3 api_server.py > /tmp/api_server.log 2>&1 &

# Test prediction
curl -X POST http://localhost:8000/predict -F "file=@image.jpg"

# View logs
tail -f /tmp/api_server.log
```

---

## 📈 Model History

| Version | Date | Val Acc | Balance | Method |
|---------|------|---------|---------|--------|
| v1 | Mar 18 | ~86% | ~60% | 2-class base + adaptation |
| **v2** | **Mar 18** | **88.5%** | **68%** | **Balanced dataset + 35% bias** |
| v3 | Planned | 90%+ | 85%+ | Train from scratch |

---

## 🔗 Important Paths

```
~/retin-verify/
├── apps/classification/
│   ├── backend/
│   │   ├── api_server.py
│   │   ├── inference_engine_3class.py
│   │   └── inference_engine_3class_v3.py (new)
│   ├── dataset_3class/
│   │   ├── train/
│   │   └── val/
│   ├── feedback_data_3class/
│   └── colab_retrain/
│       ├── deploy_3class.py
│       └── new_training/ (v3 pipeline)
└── models/classification/
    ├── cnie_classifier_3class.pth
    └── cnie_classifier_3class_v2.pth (current)
```

---

## 📞 Session Notes

**Key Decisions:**
1. Bias correction is temporary workaround, not permanent solution
2. v3 training addresses root cause (proper ImageNet pre-training)
3. Two-stage classifier is fallback if v3 fails
4. Dataset has 2-3 mislabeled/confusing samples that may need removal

**User Feedback:**
- "still predicting back when front" → Fixed with 35% bias
- "now front is missclassed by back and no card" → Investigated, root cause is model confusion
- Proposed v3 training approach from scratch

