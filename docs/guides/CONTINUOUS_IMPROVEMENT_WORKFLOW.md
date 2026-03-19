# Continuous Improvement Workflow

**Goal:** Improve model accuracy using real-world feedback from wrong predictions.

---

## The Problem

Your model achieves **90% validation accuracy** but fails on some real-world captures because:

1. **Distribution Shift:** Training data (augmented) ≠ Real captures (lighting, angles, phone cameras)
2. **Edge Cases:** Some lighting/angle combinations weren't in training
3. **Overfitting to Augmentation:** Model learned augmented patterns, not real ones

---

## The Solution: Active Learning Loop

```
┌─────────────────┐
│  Deploy Model   │ ← Current state
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Uploads   │
│  Real Images    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Low Confidence      ┌─────────────────┐
│ Model Predicts  │ ───────────────────────→ │ Flag for Review │
│  with Score     │     (< 70%)             │                 │
└────────┬────────┘                          └─────────────────┘
         │
         │ Correct?
         ▼
┌─────────────────┐
│ User Provides   │ ← Critical feedback
│ Feedback        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Collect Failed  │
│ Captures        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     10+ samples      ┌─────────────────┐
│ Retrain Model   │ ───────────────────→ │ Trigger         │
│ with New Data   │                      │ Retraining      │
└────────┬────────┘                      └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Deploy Improved │
│ Model (v2)      │
└─────────────────┘
```

---

## Implementation (Already Running!)

### 1. Enhanced Web UI with Feedback

**URL:** http://127.0.0.1:8000

**Features:**
- ✅ Confidence visualization (progress bar)
- ✅ Color-coded results:
  - 🟢 **Green (>90%)**: High confidence
  - 🟡 **Yellow (70-90%)**: Medium confidence
  - 🔴 **Red (<70%)**: Low confidence - needs review
- ✅ One-click feedback (✓ Correct / ✗ Wrong)

### 2. Confidence Thresholds

| Confidence | Action | Color |
|------------|--------|-------|
| > 90% | Auto-accept | 🟢 Green |
| 70-90% | Accept with caution | 🟡 Yellow |
| < 70% | Flag for review | 🔴 Red |

### 3. Feedback Collection

When a prediction is wrong:

1. Click **✗ Wrong** button
2. Image is automatically saved to feedback dataset
3. Correct label is recorded
4. Ready for retraining

---

## Step-by-Step Improvement Process

### Phase 1: Collect Feedback (1-2 days of usage)

**Goal:** Collect 10+ misclassified images

```bash
# Check current feedback statistics
curl http://127.0.0.1:8000/stats | python3 -m json.tool
```

**What to look for:**
- Images with **low confidence** (< 70%)
- **Wrong predictions** (user clicks "✗ Wrong")
- Specific patterns (e.g., all failures are dark lighting)

### Phase 2: Analyze Failures

```bash
# View collected feedback
ls -la ~/retin-verify/apps/classification/feedback_data/misclassified/

# Check statistics
cat ~/retin-verify/apps/classification/feedback_data/feedback_annotations.json
```

**Common failure patterns to watch for:**
- Dark/night photos
- Strong shadows
- Glare/reflections
- Partially covered cards
- Wrong orientation

### Phase 3: Retrain Model

**When you have 10+ misclassified samples:**

```bash
# Option 1: Automatic (via API)
curl -X POST http://127.0.0.1:8000/retrain/trigger

# Option 2: Manual (more control)
cd ~/retin-verify/apps/classification
python retrain_with_feedback.py --epochs 10 --lr 5e-5
```

**What happens:**
1. Feedback images are organized into train/val split
2. Model is loaded and fine-tuned with lower learning rate
3. New model saved as `cnie_front_back_real_v2.pth`
4. Validation on feedback data

### Phase 4: Validate & Deploy

```bash
# Test new model
cd ~/retin-verify/apps/classification
python cli.py classify --model ../../models/classification/cnie_front_back_real_v2.pth test_image.jpg

# If better, replace old model
mv models/classification/cnie_front_back_real_v2.pth models/classification/cnie_front_back_real.pth

# Restart server
pkill -f api_server_enhanced
./start_web_ui_enhanced.sh
```

---

## Expected Improvements

| Iteration | Expected Accuracy | Misclassified Needed |
|-----------|------------------|---------------------|
| v1 (current) | 90% | Baseline |
| v2 | 93-95% | 10-20 samples |
| v3 | 96-98% | 20-30 samples |
| v4+ | 98-99% | 30-50 samples |

---

## Best Practices

### 1. Prioritize Edge Cases

Focus on capturing failures that are:
- **Different from training** (new lighting conditions)
- **Common in production** (frequent use cases)
- **Hard examples** (not just random noise)

### 2. Balance Classes

Ensure you collect failures for BOTH classes:
```bash
# Check balance
curl http://127.0.0.1:8000/stats | jq '.feedback_stats.by_correct_class'
```

### 3. Don't Overfit

- **Retrain only when you have 10+ new samples**
- **Use lower learning rate** (5e-5 vs 1e-4)
- **Fewer epochs** (10 vs 15) to prevent overfitting

### 4. Keep History

Always archive old models:
```bash
mv models/classification/cnie_front_back_real.pth \
   models/archive/2026-03-18/cnie_front_back_real_v1.pth
```

---

## Automation Ideas

### Auto-Retraining (Future)

Set up a cron job to check weekly:

```bash
#!/bin/bash
# /home/retinrix/retin-verify/scripts/auto_retrain.sh

STATS=$(curl -s http://127.0.0.1:8000/stats)
RETRAIN=$(echo $STATS | jq '.retraining_recommended')

if [ "$RETRAIN" = "true" ]; then
    echo "$(date): Retraining triggered" >> /var/log/cnie_retrain.log
    cd ~/retin-verify/apps/classification
    python retrain_with_feedback.py
    # Restart server
    pkill -f api_server_enhanced
    ./start_web_ui_enhanced.sh
fi
```

### Confidence Alerting

Monitor for accuracy drops:

```python
# Alert if accuracy drops below threshold
stats = requests.get('http://127.0.0.1:8000/stats').json()
if stats['feedback_stats']['misclassified'] > stats['feedback_stats']['correct_confirmations']:
    send_alert("Model accuracy degrading!")
```

---

## Troubleshooting

### Problem: Model not improving after retraining

**Solutions:**
1. Check if feedback images are actually different from training
2. Increase number of feedback samples (try 20+)
3. Use data augmentation on feedback images too
4. Check if labels are correct

### Problem: Retraining overfits to feedback

**Solutions:**
1. Lower learning rate (try 1e-5)
2. Reduce epochs (try 5)
3. Mix feedback data with original training data
4. Use early stopping

### Problem: Feedback collection too slow

**Solutions:**
1. Lower confidence threshold (70% → 80%)
2. Add explicit "Report Error" button
3. Incentivize users to provide feedback
4. Use active learning: only ask for feedback on uncertain predictions

---

## Summary

| Component | Location | Purpose |
|-----------|----------|---------|
| Enhanced Web UI | `http://127.0.0.1:8000` | User interface with feedback |
| Feedback Collector | `feedback_system.py` | Stores misclassified images |
| Retraining Script | `retrain_with_feedback.py` | Fine-tunes with new data |
| Feedback Data | `feedback_data/` | Organized by category |

**Next Action:**
1. Use the web UI with real captures
2. Click "✗ Wrong" when predictions fail
3. Check `/stats` endpoint for progress
4. When you have 10+ misclassified, run retraining

---

*Continuous improvement is the key to production-grade accuracy!*
