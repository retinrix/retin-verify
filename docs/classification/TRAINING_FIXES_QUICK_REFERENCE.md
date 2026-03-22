# Training Fixes Quick Reference

## What Was Fixed

### 1. ✅ Deadlock Prevention
**Problem:** `num_workers=4` + Google Drive = futex deadlock  
**Fix:** Default `num_workers=0`, optional local SSD copy

```bash
# Recommended for Colab with Google Drive
python train_cli.py ... --num-workers 0 --copy-to-local
```

### 2. ✅ Early Stopping
**Problem:** Ran 98 epochs unnecessarily  
**Fix:** Auto-stop when validation loss plateaus

```bash
# Stop if no improvement for 5 epochs
python train_cli.py ... --early-stopping-patience 5
```

### 3. ✅ Enhanced Data Augmentation
**Problem:** Only horizontal flip - insufficient variation  
**Fix:** Rotation, affine transforms, color jitter, blur, random erasing

### 4. ✅ Label Smoothing
**Problem:** 100% confident predictions = overfitting  
**Fix:** Soft targets prevent overconfidence

```bash
python train_cli.py ... --label-smoothing 0.1
```

### 5. ✅ Differential Learning Rates
**Problem:** Same LR overwrote pretrained features  
**Fix:** 10x lower LR for backbone, 10x higher for classifier

```bash
python train_cli.py ... --feature-lr 1e-5 --classifier-lr 1e-3
```

### 6. ✅ Increased Dropout
**Problem:** 0.3 dropout insufficient  
**Fix:** 0.5 dropout + deeper classifier

```bash
python train_cli.py ... --dropout 0.5
```

### 7. ✅ Deadlock Detection
**Problem:** Training hung for 2+ hours undetected  
**Fix:** Watchdog monitors progress every 60s

### 8. ✅ Overfitting Detection
**Problem:** No warnings when model overfits  
**Fix:** Real-time alerts for train/val gap >10%

---

## Usage Examples

### Recommended for Colab (Google Drive)
```bash
python training/classification/train_cli.py \
    --data-dir /content/data \
    --train-annotations /content/data/processed/classification/train.json \
    --val-annotations /content/data/processed/classification/val.json \
    --model-dir /content/drive/MyDrive/retin-verify/models/classification \
    --log-dir /content/drive/MyDrive/retin-verify/logs/classification \
    --epochs 50 \
    --batch-size 32 \
    --device cuda \
    --fp16 \
    --copy-to-local \
    --num-workers 0 \
    --early-stopping-patience 5 \
    --label-smoothing 0.1 \
    --dropout 0.5 \
    --feature-lr 1e-5 \
    --classifier-lr 1e-3
```

### Or use the convenience script
```bash
./scripts/run_colab_training.sh --epochs 50 --batch-size 32
```

### Local Testing (CPU)
```bash
python training/classification/train_cli.py \
    --data-dir data/cnie_dataset_10k \
    --train-annotations data/processed/classification/train.json \
    --val-annotations data/processed/classification/val.json \
    --epochs 5 \
    --batch-size 4 \
    --device cpu \
    --num-workers 0
```

---

## New Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-workers` | 0 | DataLoader workers (0 = disable multiprocessing) |
| `--copy-to-local` | False | Copy data to SSD before training |
| `--local-data-dir` | `/content/local_data` | Local data path |
| `--early-stopping-patience` | 5 | Epochs to wait before stopping (0 = disabled) |
| `--label-smoothing` | 0.1 | Label smoothing factor |
| `--dropout` | 0.5 | Dropout rate in classifier |
| `--feature-lr` | 1e-5 | LR for pretrained backbone |
| `--classifier-lr` | 1e-3 | LR for new classifier head |

---

## Monitoring Improvements

### TensorBoard New Metrics
- `train/confidence` - Average prediction confidence
- `val/confidence` - Validation confidence
- `train/grad_norm` - Gradient norm (detect exploding gradients)
- `train/lr_feature` - Feature extractor learning rate
- `train/lr_classifier` - Classifier learning rate

### Console Warnings
```
⚠️  OVERFITTING DETECTED: Large gap between train and val accuracy!
⚠️  OVERCONFIDENCE: Model is too confident on training data!
⚠️  WATCHDOG ALERT: No progress for 300s! Possible deadlock.
```

---

## Expected Behavior

### Before Fixes (Problematic)
```
Epoch 1:  Train Acc: 95%, Val Acc: 85%
Epoch 10: Train Acc: 100%, Val Acc: 100% ← Suspicious
Epoch 50: Train Acc: 100%, Val Acc: 100% ← Severe overfitting
Epoch 98: [HANG - no output for hours]
```

### After Fixes (Healthy)
```
Epoch 1:  Train Acc: 65%, Val Acc: 60%, Conf: 0.72
Epoch 10: Train Acc: 88%, Val Acc: 85%, Conf: 0.85
Epoch 20: Train Acc: 92%, Val Acc: 89%, Conf: 0.88
Epoch 25: Train Acc: 93%, Val Acc: 89%, Conf: 0.89

EARLY STOPPING triggered at epoch 25
Best val acc: 89.5%
```

---

## Troubleshooting

### Still getting deadlocks?
```bash
# Ensure num_workers=0 and use local copy
python train_cli.py ... --num-workers 0 --copy-to-local
```

### Model still overfitting?
```bash
# Increase regularization
python train_cli.py ... --dropout 0.7 --label-smoothing 0.2

# Or use smaller model
python train_cli.py ... --base-model mobilenet_v3_small
```

### Training too slow?
```bash
# Try with workers (only if NOT using Google Drive)
python train_cli.py ... --num-workers 2
```

### Want to resume training?
```bash
python train_cli.py ... --resume-from models/classification/interrupted_checkpoint.pth
```

---

## Files Changed

| File | Changes |
|------|---------|
| `training/classification/train_cli.py` | All fixes implemented |
| `scripts/run_colab_training.sh` | New convenience script |
| `docs/TRAINING_INCIDENT_ANALYSIS_2026-03-16.md` | Detailed analysis |
| `docs/TRAINING_FIXES_QUICK_REFERENCE.md` | This file |

---

## Next Steps

1. **Test locally first:** Run 5 epochs on CPU to verify script works
2. **Deploy to Colab:** Use the improved script with `--copy-to-local`
3. **Monitor TensorBoard:** Watch for healthy train/val curves
4. **Iterate:** Adjust `--dropout` and `--label-smoothing` if needed
