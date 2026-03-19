# 📚 v3 Training with Synthetic + Real Data

## Overview

This approach **combines 16,050 synthetic CNIE images with 306 real photos** to train a more robust classifier.

---

## 📊 Dataset Composition

| Source | Front | Back | No Card | Total |
|--------|-------|------|---------|-------|
| **Real Photos** | 89 | 96 | 121 | **306** |
| **Synthetic Images** | 8,025 | 8,025 | 0 | **16,050** |
| **Combined** | ~8,100 | ~8,100 | 121 | **~16,350** |

### Why Synthetic Data Helps

| Problem with Real-Only | How Synthetic Fixes It |
|------------------------|------------------------|
| Limited variety (306 images) | 16K diverse samples |
| Class imbalance | Can balance front/back perfectly |
| Overfitting to specific backgrounds | Many background variations |
| Overfitting to specific lighting | Controlled synthetic lighting |
| Expensive to collect more | Already generated! |

---

## 🎯 Training Strategy

### Key Principle: **Real Images Are Golden**

While we have 16K synthetic images, we care about **performance on real photos**.

```
Training:    Real + Synthetic (weighted toward real)
Validation:  Real only (measure real-world performance)
Test:        Real only (final evaluation)
```

### Weighting Strategy

```python
# Real images get 2x weight in:
1. Sampling (WeightedRandomSampler)
2. Loss function (higher penalty for real image mistakes)
```

**Example per epoch:**
- Total samples seen: ~5,000
- Real images: ~40% (despite being only 2% of data)
- Synthetic images: ~60% (for variety)

---

## 🔧 Technical Implementation

### 1. Synthetic Sampling (5000 pairs max)

```python
# Random sample to keep training time reasonable
front_dirs = random.sample(front_dirs, 5000)  # 5000 front
back_dirs = random.sample(back_dirs, 5000)    # 5000 back
```

**Why limit?**
- 10,000 images = ~30 min per epoch on T4
- Full 16K = ~45 min per epoch
- Diminishing returns after 5K per class

### 2. WeightedRandomSampler

```python
# Give real images 2x sampling weight
if not is_synthetic:
    weight *= 2.0
```

**Result:**
- Real images: 306 total → appear as ~600 in effective batch
- Synthetic images: 10,000 total → appear as ~4,400 in effective batch

### 3. Class Balancing

```python
# Ensure each batch has balanced classes
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Loss function also uses class weights
criterion = CrossEntropyLoss(weight=class_weights)
```

### 4. Tracking Real vs Synthetic Accuracy

```python
# During training, track accuracy separately
train_real_acc = accuracy on real images only
train_syn_acc = accuracy on synthetic images only
```

**Why?**
- Synthetic accuracy will be high (easy, controlled data)
- Real accuracy is what matters for deployment

---

## 📈 Expected Improvements

| Metric | v2 (Real Only) | v3 (Synthetic + Real) |
|--------|----------------|----------------------|
| **Training samples** | 306 | 10,000+ |
| **Front accuracy** | 60-92% | 90%+ |
| **Back accuracy** | 68-90% | 90%+ |
| **No card accuracy** | 100% | 95%+ |
| **Balance** | 68% | **90%+** |
| **Needs bias?** | Yes (35%) | **No** |
| **Generalization** | Poor | **Good** |

---

## 🚀 Deployment Steps

### Step 1: Update Host

```python
# In deploy_v3_with_synthetic.py
HOST = "your-colab-host.trycloudflare.com"
```

### Step 2: Run Deployment

```bash
cd ~/retin-verify/apps/classification/colab_retrain/new_training
python3 deploy_v3_with_synthetic.py
```

**What happens:**
1. Packages real dataset (306 images) → `real_dataset.tar.gz`
2. Samples synthetic dataset (10K images) → `synthetic_sample.tar.gz`
3. Uploads both to Colab
4. Extracts and sets up directory structure
5. Installs dependencies
6. Starts training in background

### Step 3: Monitor Training

```bash
# Watch live logs
ssh root@your-host "tail -f /content/retin_v3_synthetic/train_v3_synthetic.log"
```

**Expected output:**
```
Loading datasets...
  train: 10306 total images
    Real: 306
    Synthetic: 10000
    cnie_front: 89 real + 5000 synthetic = 5089
    cnie_back: 96 real + 5000 synthetic = 5096
    no_card: 121 real + 0 synthetic = 121
  val: 50 total images
    Real: 50
    Synthetic: 0
...
E01/50 | Train(Real): 45%/52%/48% | Val: 48%/55%/47% | Bal:47%
E02/50 | Train(Real): 55%/62%/58% | Val: 58%/65%/60% | Bal:58%
...
E25/50 | Train(Real): 92%/94%/95% | Val: 90%/91%/93% | Bal:90% <- Saved!
```

### Step 4: Download Model

```bash
# When training completes
scp root@your-host:/content/cnie_classifier_3class_v3_synthetic.pth \
    ~/retin-verify/models/classification/
```

### Step 5: Deploy Locally

```bash
# Update API server to use new model
cd ~/retin-verify/apps/classification/backend

# Edit api_server.py to load v3_synthetic model
# Or create new api_server_v3.py

# Restart server
kill $(lsof -ti:8000)
python3 api_server.py
```

---

## ⚠️ Important Considerations

### 1. Domain Gap

**Risk:** Synthetic and real images look different
- Synthetic: Perfect, clean, uniform
- Real: Lighting variation, camera noise, angles

**Mitigation:** 
- Real images get 2x weight
- Strong augmentation bridges the gap
- Validate only on real images

### 2. No Synthetic No-Card

**Issue:** We only have synthetic front/back, no synthetic no-card

**Why it's OK:**
- No-card is already 100% accurate in v2
- Real no-card samples (121) are sufficient
- No-card is visually distinct (no card = easy)

### 3. Training Time

| Configuration | Time per Epoch | Total Time (50 epochs) |
|---------------|----------------|------------------------|
| Real only (306) | 1 min | ~50 min |
| Synthetic 10K | 8 min | ~6.5 hours |
| Synthetic 5K | 4 min | ~3.5 hours |

**Recommendation:** Use 5K synthetic samples (good balance)

---

## 🔬 Comparison: With vs Without Synthetic

### Without Synthetic (Real Only)
```
Training: 306 images
Augmentation: Essential
Risk: Overfitting
Expected balance: 70-85%
```

### With Synthetic (Proposed)
```
Training: 10,000+ images
Augmentation: Still helpful
Risk: Domain gap
Expected balance: 90%+
```

---

## 📋 File Summary

```
colab_retrain/new_training/
├── train_from_scratch.py           # Real-only version
├── train_with_synthetic.py         # Synthetic + real version ⭐
├── deploy_v3.py                    # Deploy real-only
├── deploy_v3_with_synthetic.py     # Deploy synthetic + real ⭐
├── inference_engine_3class_v3.py   # Clean inference
└── README_V3.md                    # Original docs
```

---

## 🎯 Decision: Which to Use?

| Use Synthetic + Real if... | Use Real Only if... |
|---------------------------|---------------------|
| ✅ You have 4+ hours for training | ✅ Need quick results (< 1 hour) |
| ✅ Want best possible accuracy | ✅ Current 68% balance is acceptable |
| ✅ Colab GPU available | ✅ No GPU time available |
| ✅ Want to eliminate bias hack | ✅ Bias hack is working fine |

---

## Next Steps

1. **Review this document** - understand the tradeoffs
2. **Choose approach:**
   - Option A: Synthetic + Real (recommended, best accuracy)
   - Option B: Real only (faster, good enough)
3. **Provide Colab host** for deployment
4. **Start training**

**Questions before proceeding?**
