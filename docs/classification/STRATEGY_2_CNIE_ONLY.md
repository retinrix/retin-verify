# Strategy 2: CNIE-Only Front/Back Classification

## Overview

**Target:** Classify ONLY Algerian CNIE cards as Front or Back  
**Approach:** Fine-tune existing synthetic model with 4 real cards  
**Expected Accuracy:** 85-90% on real images  
**Time Required:** 45-60 minutes total

---

## Why This Works Better

### CNIE-Only vs Multi-Document Classification

| Aspect | Multi-Document | CNIE-Only (Strategy 2) |
|--------|----------------|------------------------|
| Classes | passport, cnie_front, cnie_back, carte_grise | **cnie_front, cnie_back** |
| What model learns | Document types + sides | **Layout patterns only** |
| Content matters? | Yes (different documents) | **No (same document type)** |
| Identity matters? | N/A | **No (layout only)** |
| Required data variety | High (different docs, people) | **Low (just 2 cards per side)** |

### Key Insight

**Front/Back classification is about LAYOUT, not CONTENT:**
- Front: Photo on left, name fields, birth date
- Back: Chip visible, MRZ code, different layout

The model doesn't care whose CNIE it is - it only learns the spatial arrangement of features.

---

## Photo Capture Instructions

### Equipment Needed
- Smartphone or webcam
- Good lighting source (window or lamp)
- Flat surface (table/desk)

### Capture Specifications

#### Card Selection
**You need 2 different CNIE cards (yours + one other person's)**

```
Card A: Your CNIE
  ├── Front: 4 photos
  └── Back: 4 photos

Card B: Friend/Family member's CNIE
  ├── Front: 4 photos  
  └── Back: 4 photos

Total: 16 photos (8 per class)
```

#### Photo Types (4 per side per card = 16 total)

For each card side, capture these 4 variations:

**Photo 1: Flat, Perfect Lighting**
- Card flat on table
- Camera directly above (90° angle)
- Bright, even lighting
- Card fills 70-80% of frame
- No shadows

**Photo 2: Angled, Natural Light**
- Card flat on table
- Camera at 30-45° angle (side view)
- Natural window light
- Some shadow visible
- Card fills 60-70% of frame

**Photo 3: Hand-Held, Indoor Light**
- Hold card in hand
- Indoor artificial light
- Slight perspective distortion
- Card fills 50-60% of frame
- May have fingers visible at edges

**Photo 4: Challenging Lighting**
- Any position (flat or angled)
- Dim light OR strong side light
- Possible glare/reflection
- Slightly out of focus (optional)

### Directory Structure

```
my_4_cards_cnie_only/
├── cnie_front/
│   ├── cardA_flat.jpg         # Your CNIE front - flat
│   ├── cardA_angled.jpg       # Your CNIE front - angled
│   ├── cardA_handheld.jpg     # Your CNIE front - hand-held
│   ├── cardA_dim.jpg          # Your CNIE front - dim light
│   ├── cardB_flat.jpg         # Other person's CNIE front - flat
│   ├── cardB_angled.jpg       # Other person's CNIE front - angled
│   ├── cardB_handheld.jpg     # Other person's CNIE front - hand-held
│   └── cardB_dim.jpg          # Other person's CNIE front - dim light
└── cnie_back/
    ├── cardA_flat.jpg         # Your CNIE back - flat
    ├── cardA_angled.jpg       # Your CNIE back - angled
    ├── cardA_handheld.jpg     # Your CNIE back - hand-held
    ├── cardA_dim.jpg          # Your CNIE back - dim light
    ├── cardB_flat.jpg         # Other person's CNIE back - flat
    ├── cardB_angled.jpg       # Other person's CNIE back - angled
    ├── cardB_handheld.jpg     # Other person's CNIE back - hand-held
    └── cardB_dim.jpg          # Other person's CNIE back - dim light
```

### Capture Checklist

For each photo, verify:
- [ ] Card is clearly visible (not obscured)
- [ ] Image is not blurry
- [ ] Card takes up at least 50% of frame
- [ ] Front/back is identifiable by human
- [ ] Resolution >= 640x480

### Common Mistakes to Avoid

**Don't:**
- Use only 1 card (not enough variety)
- Capture all photos from same angle
- Use extreme angles (>60°)
- Cover card with fingers
- Include multiple cards in one photo

**Do:**
- Use 2 different cards (can be same generation)
- Vary lighting conditions
- Include some realistic imperfections
- Keep card as main subject

---

## Fine-Tuning Strategy

### Is It Fine-Tuning or New Training?

**It's FINE-TUNING** because:
1. Base model already trained on 10k synthetic CNIE images
2. We keep the learned features (edge detection, pattern recognition)
3. We adapt the classifier to real-world variations
4. Synthetic + Real combination prevents overfitting

### Structure Compatibility

**Good News:** Structure is compatible!

The existing model already knows:
- What a CNIE looks like (from synthetic data)
- Visual features (edges, corners, text regions)
- Basic front/back differences

We're just adapting it to:
- Real lighting conditions
- Real camera angles
- Real shadows and imperfections

### Two-Class vs Four-Class Model

**Important Decision:**

You have two options:

#### Option A: Two-Class Model (Recommended)
Train a NEW model with only 2 classes:
```python
classes = ['cnie_front', 'cnie_back']
```

**Pros:**
- Simpler problem (binary classification)
- Higher accuracy potential
- Smaller model
- Faster inference

**Cons:**
- Can't classify other documents
- Need separate model for other use cases

#### Option B: Keep Four Classes
Add your real CNIE data to the existing 4-class model.

**Pros:**
- One model for all documents
- Future-proof

**Cons:**
- Harder problem (4-way classification)
- May reduce CNIE-specific accuracy
- Passport/carte_grise remain synthetic-only

**Recommendation:** Start with Option A (2-class) for best results.

---

## Step-by-Step Implementation

### Step 1: Capture Photos (15-20 minutes)

```bash
# Create directory structure
mkdir -p my_4_cards_cnie_only/{cnie_front,cnie_back}

# Capture 16 photos following instructions above
# Save with descriptive names
```

### Step 2: Generate Augmentations (1 minute)

```bash
cd ~/retin-verify/apps/classification
source .venv/bin/activate

# Generate 100 augmentations per real image
# 16 real x 100 aug = 1600 training images
python augment_dataset.py \
    --input-dir ./my_4_cards_cnie_only \
    --output-dir ./cnie_only_augmented \
    --target-per-image 100

# Preview some augmentations
python augment_dataset.py \
    --input-dir ./my_4_cards_cnie_only \
    --preview 9
```

**Expected output:**
```
Train samples: 1280 (16 x 80% x 100)
Val samples: 320 (16 x 20% original, not augmented)
```

### Step 3: Create 2-Class Base Model

```bash
# Create 2-class version of the model
python3 << 'PYTHON_EOF'
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Load base 4-class model
base_path = "./models/classification_production/best_model.pth"
checkpoint = torch.load(base_path, map_location='cpu')

# Create new 2-class model
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# Modify classifier for 2 classes
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)  # 2 classes: front/back
)

# Load feature extractor weights from base model
base_state = checkpoint['model_state_dict']
new_state = model.state_dict()

# Copy feature extractor weights (all except classifier)
for key in base_state:
    if 'classifier' not in key and key in new_state:
        new_state[key] = base_state[key]

model.load_state_dict(new_state)

# Save as 2-class base model
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': ['cnie_front', 'cnie_back'],
    'note': '2-class CNIE front/back base model'
}, './models/cnie_2class_base.pth')

print("Created 2-class base model: ./models/cnie_2class_base.pth")
PYTHON_EOF
```

### Step 4: Fine-Tune Model (15-20 minutes)

```bash
# Fine-tune with augmented real data
python finetune_with_augmentation.py \
    --augmented-dir ./cnie_only_augmented \
    --base-model ./models/cnie_2class_base.pth \
    --output-model ./models/cnie_front_back_real.pth \
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cpu

# If you have GPU, use: --device cuda
```

### Step 5: Test on Real Images

```bash
# Test with a new real capture
python cli.py classify \
    --model ./models/cnie_front_back_real.pth \
    --device cpu \
    ./my_new_capture.jpg

# Batch test
python test_auto.py \
    --model ./models/cnie_front_back_real.pth \
    --input-dir ./test_captures \
    --output-dir ./test_results
```

### Step 6: Export to ONNX (Optional)

```bash
python export_onnx.py \
    --model-path ./models/cnie_front_back_real.pth \
    --output ./models/cnie_front_back_real.onnx
```

---

## Expected Results

### With 16 Real Images + Augmentation

| Metric | Synthetic Only | After Fine-Tuning |
|--------|----------------|-------------------|
| Real Image Accuracy | 50-60% | **85-90%** |
| Confidence | Low/Unstable | **High/Stable** |
| Inference Time | Same | Same |

### Confusion Matrix (Expected)

```
                Predicted
              Front    Back
Actual Front   90%     10%
       Back    8%      92%
```

---

## Troubleshooting

### If accuracy is still low (<70%):

1. **Check capture quality**
   - Are photos blurry?
   - Is card too small in frame?
   - Too extreme angles?

2. **Add more cards**
   - Use 3-4 different CNIE cards instead of 2
   - More variety = better generalization

3. **Increase augmentations**
   - Change `--target-per-image` from 100 to 200
   - More synthetic variety helps

4. **Train longer**
   - Increase epochs from 15 to 25
   - Lower learning rate to 5e-5

### If model overfits (train_acc high, val_acc low):

1. **Reduce augmentation intensity**
   - Some augmentations may be too extreme
   
2. **Add dropout**
   - Increase dropout from 0.5 to 0.7

3. **Use early stopping**
   - Training stops when validation accuracy stops improving

---

## Comparison: Strategy 1 vs Strategy 2

| Aspect | Strategy 1 (4 docs) | Strategy 2 (CNIE only) |
|--------|---------------------|------------------------|
| Classes | 4 | 2 |
| Cards needed | 4 (one per doc) | 2 CNIE cards |
| Photos per card | 3-4 | 4 |
| Total photos | 16 | 16 |
| Augmentations | 150/image | 100/image |
| Training images | 600 | 1600 |
| Expected accuracy | 70-80% | **85-90%** |
| Complexity | Higher | Lower |
| Use case | Multi-document | CNIE-only |

---

## Summary

**For CNIE-only front/back classification:**

1. **Capture:** 16 photos (4 per side × 2 cards)
2. **Augment:** Generate 1600 training images
3. **Fine-tune:** 15 epochs, starting from synthetic base
4. **Result:** 85-90% accuracy on real captures

**Time investment:** 45-60 minutes  
**Hardware:** Any CPU (GPU optional, ~5x faster)

---

## Quick Command Reference

```bash
# Complete workflow:

# 1. Setup
mkdir -p my_4_cards_cnie_only/{cnie_front,cnie_back}
# ... capture 16 photos ...

# 2. Augment
python augment_dataset.py -i ./my_4_cards_cnie_only -o ./cnie_aug -n 100

# 3. Create 2-class base (run Python script from Step 3)

# 4. Fine-tune
python finetune_with_augmentation.py \
    --augmented-dir ./cnie_aug \
    --base-model ./models/cnie_2class_base.pth \
    --output-model ./models/cnie_real.pth \
    --epochs 15

# 5. Test
python cli.py classify --model ./models/cnie_real.pth test.jpg
```
