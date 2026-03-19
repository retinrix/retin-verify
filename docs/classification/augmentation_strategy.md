# Data Augmentation Strategy: 4 Real ID Cards → Rich Dataset

## Yes, This Works! (With Limitations)

You can create a substantial training dataset from just 4 real ID cards using **data augmentation**. This is a common and effective technique when real data is scarce.

## What Augmentation CAN Do

✅ **Effective augmentations for document classification:**
- Lighting variations (brightness, contrast, gamma)
- Geometric transforms (rotation, perspective, slight zoom)
- Noise and blur (simulating different camera qualities)
- Color shifts (white balance variations)
- Small occlusions (finger partially covering card)

## What Augmentation CANNOT Do

❌ **Cannot create from augmentation:**
- Different card designs (old vs new CNIE)
- Different wear patterns (new vs worn cards)
- Completely different backgrounds
- Multiple people's cards
- Structural changes to the document

## Recommended Augmentation Pipeline

### From 4 Real Cards: Generate 400-800 Images

```
4 Real Cards × 100-200 augmentations each = 400-800 training images
```

### Target Distribution

| Document Type | Real Images | Augmentations | Total |
|--------------|-------------|---------------|-------|
| CNIE Front | 1 | 150 | 151 |
| CNIE Back | 1 | 150 | 151 |
| Passport | 1 | 150 | 151 |
| Carte Grise | 1 | 150 | 151 |
| **Total** | **4** | **600** | **604** |

## Augmentation Categories

### 1. Photometric Augmentations (30-40% of total)
Simulate different lighting conditions:

```python
# Brightness: ±30%
# Contrast: ±30%
# Gamma: 0.7-1.3
# Saturation: ±20%
# Hue shift: ±10 degrees
```

**Purpose:** Match different indoor/outdoor lighting

### 2. Geometric Augmentations (30-40% of total)
Simulate different camera angles and positions:

```python
# Rotation: ±15 degrees (more creates invalid samples)
# Perspective: ±15% warp
# Scale: 0.9x - 1.1x
# Translation: ±10%
# Horizontal flip (only for symmetric docs, not text-heavy)
```

**Purpose:** Different capture angles and distances

### 3. Quality Degradations (20-30% of total)
Simulate different camera qualities:

```python
# Gaussian blur: σ=0.5-2.0
# Motion blur: small kernel
# JPEG compression: quality 60-95
# Gaussian noise: σ=5-25
# Sharpening
```

**Purpose:** Different phones, lighting, compression

### 4. Environmental Augmentations (10-20% of total)
Simulate real-world conditions:

```python
# Random erasing: small occlusions (1-5% of image)
# Shadow simulation
# Vignetting (dark corners)
# Lens distortion
```

**Purpose:** Finger covering part, shadows, lens effects

## Critical: Keep Augmentation Realistic

### ❌ Bad Augmentations (Don't Do This)

```python
# Extreme rotation (>30°) - card doesn't look like card anymore
rotation=90  # Document is sideways - not realistic

# Extreme distortion - destroys text readability
perspective=50%  # Card looks like trapezoid

# Excessive blur - loses all features
blur=10  # Can't read anything

# Wrong color space - documents don't look like this
hue_shift=180  # Blue CNIE - doesn't exist
```

### ✅ Good Augmentations (Do This)

```python
# Slight rotation - realistic hand-held capture
rotation=(-10, 10)

# Natural perspective - realistic angle
perspective=(-0.15, 0.15)

# Slight blur - realistic camera quality
blur=(0, 2)

# Natural lighting variation
brightness=(-0.2, 0.2)
```

## Implementation

I've created an augmentation tool for you:

```bash
python augment_dataset.py \
    --input-dir ./my_4_cards \
    --output-dir ./augmented_dataset \
    --target-per-image 150 \
    --train-val-split 0.8
```

This will:
1. Load your 4 real card images
2. Apply realistic augmentations
3. Generate 150 variations per image
4. Split into train/val sets
5. Save with proper labels

## Validation Strategy

### Hold-Out Test Set (CRITICAL)

**Don't augment everything!** Keep some real images for testing:

```
4 Real Cards
├── 3 cards → Augmented for training (450 images)
└── 1 card  → Reserve for testing (NO augmentation)
```

Or better:
- Take **multiple photos** of each card (5-10 angles)
- Use 80% for augmentation training
- Use 20% raw for testing

## Expected Results

### With 4 Cards + Augmentation (600 images)

| Metric | Synthetic Only | 4 Cards + Augment | 1000 Real |
|--------|----------------|-------------------|-----------|
| Real Image Accuracy | 40-60% | 70-80% | 85-92% |
| Training Time | 5 min | 20-30 min | 2-3 hours |
| Collection Effort | None | 30 min | 1 week |
| **Recommendation** | ❌ Poor | ⚠️ Okay | ✅ Best |

### When This Approach Works

✅ **Good for:**
- Proof of concept
- Limited deployment
- Controlled environment (same lighting/setup)
- Quick iteration

❌ **Not good for:**
- Production systems
- Variable environments
- High accuracy requirements (>85%)
- Regulatory compliance

## Step-by-Step Workflow

### Step 1: Capture Your 4 Cards (15 minutes)

Take **multiple photos** of each card with variations:

```
For each of 4 cards, capture:
├── 2-3 flat, well-lit shots
├── 2-3 angled shots (15-30°)
├── 2-3 different lighting (bright, dim, flash)
└── Total: 8-12 photos per card = 32-48 real images
```

### Step 2: Organize

```
my_4_cards/
├── cnie_front/
│   ├── raw_001.jpg  (flat)
│   ├── raw_002.jpg  (angled)
│   └── raw_003.jpg  (dim light)
├── cnie_back/
├── passport/
└── carte_grise/
```

### Step 3: Augment

```bash
python augment_dataset.py \
    --input-dir ./my_4_cards \
    --output-dir ./augmented_600 \
    --augmentations-per-image 150 \
    --validation-split 0.2
```

### Step 4: Train

```bash
python training/classification/train_cli.py \
    --data-dir ./augmented_600 \
    --epochs 20 \
    --batch-size 16 \
    --device cuda
```

### Step 5: Test on Real Images

Test on **held-out real photos** (not augmented):

```bash
python test_auto.py \
    --input-dir ./my_4_cards/test_set \
    --output-dir ./test_results
```

## Pro Tips

### 1. Use Albumentations Library

Best library for document augmentation:

```python
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
], p=1.0)
```

### 2. Preview Before Training

Always visualize augmentations before training:

```python
# Show 9 random augmentations of one image
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    augmented = transform(image=image)['image']
    ax.imshow(augmented)
    ax.axis('off')
plt.show()
```

### 3. Monitor for Overfitting

Watch validation accuracy during training:
- If train_acc ↑ but val_acc → or ↓ = overfitting
- Reduce augmentation intensity or add more real data

### 4. Progressive Augmentation

Start conservative, increase if needed:

```python
# Stage 1: Conservative (for initial training)
transform_weak = A.Compose([...])  # mild augmentations

# Stage 2: Aggressive (if underfitting)
transform_strong = A.Compose([...])  # stronger augmentations
```

## Summary

**Yes, you can use 4 real cards + augmentation**, but:

✅ **Do:**
- Generate 100-150 augmentations per real image
- Keep augmentations realistic
- Reserve 20% for validation
- Test on held-out real images

❌ **Don't:**
- Expect >80% accuracy on diverse real images
- Use extreme augmentations
- Skip validation on real data
- Use this for production systems

**Best approach:** Start with 4 cards + augmentation, then collect more real data if accuracy isn't sufficient.
