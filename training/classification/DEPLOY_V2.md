# 3-Class Model v2 Deployment Guide

## Training Results Summary

| Metric | v1 (Old) | **v2 (New)** | Improvement |
|--------|----------|--------------|-------------|
| **Overall Val Accuracy** | ~86% | **88.5%** | ⬆️ +2.5% |
| **Back Accuracy** | ~60% | **84-92%** | ⬆️ **+24-32%** |
| **Balance (min class)** | ~60% | **76.0%** | ⬆️ +16% |
| **Front Accuracy** | 85%+ | 72-80% | Slight drop |
| **No Card Accuracy** | 100% | **100%** | ✅ Maintained |

**Key Wins:**
- ✅ Back classification dramatically improved (60% → 84-92%)
- ✅ Better balanced model across all 3 classes
- ✅ No card detection remains perfect

## Deployment Steps

### 1. Reconnect to Colab Tunnel
```bash
cd ~/retin-verify/apps/classification/colab_retrain
./deploy_3class.py
# Press Ctrl+C after tunnel is established to keep it running
```

### 2. Download the Model
In a new terminal:
```bash
cd ~/retin-verify/apps/classification/colab_retrain
mkdir -p models

# Replace <colab-host> with the actual hostname from deploy_3class.py output
scp root@<colab-host>:/content/retin_retrain_3class/cnie_classifier_3class_v2.pth models/

# Copy to production location
cp models/cnie_classifier_3class_v2.pth ~/retin-verify/models/classification/
```

### 3. Verify Configuration
The following files have been updated to use v2 model:
- `backend/inference_engine_3class.py` (line 139)
- `backend/api_server.py` (line 73)

### 4. Restart API Server
```bash
cd ~/retin-verify/apps/classification/backend
pkill -f "api_server.py"  # Stop old server
python3 api_server.py      # Start with new model
```

### 5. Test the New Model
```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/test_back.jpg"
```

## Training Details

- **Dataset**: 306 samples (89 front, 96 back, 121 no_card)
- **Training**: 30 epochs on Colab GPU
- **Best Epoch**: 25 (balance: 76.0%)
- **Augmentation**: Standard ImageNet transforms
- **Class Weights**: Applied to handle slight imbalance

## Troubleshooting

If model download fails:
1. Check tunnel is active: `ssh root@<host> 'echo OK'`
2. Try downloading via HTTP from Colab notebook directly
3. Model file size: ~17MB

If inference fails:
1. Verify model file exists: `ls -la ~/retin-verify/models/classification/`
2. Check file integrity: The file should be ~17MB
3. Review logs: Check api_server.py output for errors
