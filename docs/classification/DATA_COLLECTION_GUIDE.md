# Real CNIE Image Collection & Fine-Tuning Guide

## How Many Images Do You Need?

### Minimum Requirements (Quick Fine-tuning)

| Class | Minimum | Recommended | Optimal |
|-------|---------|-------------|---------|
| CNIE Front | 50-100 | 200-500 | 1000+ |
| CNIE Back | 50-100 | 200-500 | 1000+ |
| Passport | 30-50 | 100-200 | 500+ |
| Carte Grise | 30-50 | 100-200 | 500+ |
| **Total** | **160-300** | **600-1400** | **3000+** |

### Image Distribution Strategy

**Option 1: Balanced Dataset (Recommended)**
```
Equal number per class:
- cnie_front: 25%
- cnie_back: 25%
- passport: 25%
- carte_grise: 25%
```

**Option 2: Weighted by Usage**
```
Based on expected usage frequency:
- cnie_front: 40% (most common)
- cnie_back: 40%
- passport: 15%
- carte_grise: 5%
```

### Image Variation Requirements

For each image, capture variations across these dimensions:

#### 1. Lighting Conditions (Critical)

Detailed instructions for creating each lighting scenario:

**☀️ Bright Natural Light**
- **When:** Midday, near a large window or outdoors in shade
- **How to create:** 
  - Place card on a table within 1-2 meters of a window
  - Avoid direct sunlight hitting the card (causes harsh shadows)
  - Use a white sheet/paper as reflector on the opposite side to fill shadows
- **Best for:** Clean, high-quality training images
- **Target:** 25-30% of your dataset

**💡 Indoor Artificial Light (Warm/Cool)**
- **Warm light (yellow/orange tint):**
  - Use incandescent bulbs or warm LED lights (2700K-3000K color temperature)
  - Typical sources: desk lamps, ceiling lights, bedside lamps
  - Creates a cozy, evening atmosphere appearance
- **Cool light (blue/white tint):**
  - Use daylight LED bulbs or fluorescent lights (5000K-6500K)
  - Typical sources: office lights, bathroom mirrors, ring lights
  - Creates a clinical, daytime appearance
- **How to create:**
  - Turn off natural light sources (close curtains)
  - Use only artificial lighting
  - Position light at 45° angle to create dimension
- **Target:** 25-30% of your dataset

**🌑 Low Light / Shadows**
- **When:** Evening, dimly lit rooms, or intentionally underexposed
- **How to create:**

  - Use only one distant light source
  - Move card to corner of room away from windows
  - Close curtains/blinds completely
  - Use phone in a dark room with minimum brightness
  - Capture at dusk without turning on lights
- **Variations:**
  - Partial shadow: Card half in light, half in dark
  - Spotty shadows: Light through blinds or leaves creating patterns
  - Side shadow: Light from one side only
- **Target:** 15-20% of your dataset

**💫 Backlighting**
- **What:** Strong light source BEHIND the card, card appears darker
- **How to create:**
  - Hold card between camera and window (camera facing window)
  - Use a lamp behind the card
  - Stand with your back to the sun/window and hold card up
- **Expected effect:** Card edges may glow, text becomes harder to read
- **Note:** This is intentionally challenging - the model needs to learn to identify documents even when poorly lit
- **Target:** 5-10% of your dataset

**📸 Flash Photography**
- **How to create:**
  - Enable phone camera flash (forced ON, not auto)

  - Vary distance: 30cm, 60cm, 100cm from card
- **Expected effects:**
  - Harsh shadows behind card
  - Possible glare on holographic elements
  - Overexposed hotspots on shiny surfaces
  - Red-eye effect if photographing people
- **Target:** 5-10% of your dataset

**🌈 Mixed Lighting**
- **What:** Multiple light sources with different color temperatures
- **How to create:**
  - Window light (cool/blue) + indoor lamp (warm/yellow)
  - Two different colored LED strips
  - Sunlight from left + artificial light from right
- **Expected effect:** Card shows color gradient or split tones
- **Target:** 5-10% of your dataset

**🎯 Special Challenging Conditions**
- **Glare/Reflections:** Shine light at extreme angle on holographic card elements
- **Color casts:** Use colored lighting (red/blue/green LED) for artistic effect
- **Sun flare:** Point camera toward sun with card in foreground

**Overall Target:** 20-30% of images with challenging lighting (low light, backlight, harsh flash, or mixed)

#### 2. Camera Angles & Perspective
- [ ] Flat, top-down (0°)
- [ ] Slight angle (15-30°)
- [ ] Moderate angle (30-60°)
- [ ] Extreme angle (>60°)
- [ ] Rotation (0°, 90°, 180°, 270°)

**Target:** Maximum 40% flat images, rest with angles

#### 3. Distance & Framing
- [ ] Close-up (card fills 80%+ of frame)
- [ ] Medium distance (card ~50% of frame)
- [ ] Far distance (card ~30% of frame)
- [ ] Partially visible edges

#### 4. Background Variations
- [ ] Plain desk/table
- [ ] Cluttered background
- [ ] Hand holding card
- [ ] Outdoor scenes
- [ ] Different surface textures

#### 5. Card Conditions
- [ ] New/clean card
- [ ] Worn/aged card
- [ ] Slightly dirty
- [ ] Glare/reflections
- [ ] Covered by plastic sleeve
- [ ] Different card designs (old vs new CNIE)

#### 6. Device Variations
- [ ] Modern smartphone (iPhone 12+, Samsung S21+)
- [ ] Older smartphone
- [ ] Scanner/flatbed
- [ ] Webcam
- [ ] Different resolutions (1080p, 4K, etc.)

### Quick Start: Minimal Viable Dataset

If you need results quickly, start with **300 images**:

```
Minimum 300 Image Collection Plan:
├── cnie_front/
│   ├── 10 flat, perfect lighting
│   ├── 10 angled shots (various angles)
│   ├── 10 challenging lighting
│   ├── 10 different backgrounds
│   └── 10 worn/different conditions
│   └── Total: 50 images minimum
├── cnie_back/
│   └── Same distribution: 50 images
├── passport/
│   └── 30 images (varied conditions)
└── carte_grise/
    └── 30 images (varied conditions)

Plus: 200 more distributed across all classes for variety
```

## Image Naming Convention

While the training pipeline accepts any image filename, we recommend following a consistent naming convention for organization and traceability.

### Recommended Naming Pattern

```
{collection_id}_{photo_type}_{sequence}.jpg
```

| Component | Description | Examples |
|-----------|-------------|----------|
| `collection_id` | Identifier for the card/location | `cardA`, `cardB`, `moms_card`, `dads_card` |
| `photo_type` | Type of photo (see below) | `flat`, `angled`, `handheld`, `dim` |
| `sequence` | Optional sequence number | `01`, `02`, `03` |

### Photo Type Abbreviations

| Abbreviation | Meaning | Use When |
|--------------|---------|----------|
| `flat` | Flat, perfect lighting | Card on table, camera directly above, bright even light |
| `angled` | Angled perspective | Camera at 30-45° angle to the card |
| `handheld` | Hand-held shot | Card held in hand, natural indoor light |
| `dim` | Dim/challenging light | Low light, strong shadows, or difficult conditions |
| `backlit` | Backlighting | Strong light behind card |
| `flash` | Flash photography | Using camera flash |
| `outdoor` | Outdoor setting | Natural outdoor environment |

### Example Filenames

```
# CNIE Front examples
cnie_front/
├── cardA_flat_01.jpg       # Card A, flat on table, perfect light
cnie_front/
├── cardA_angled_01.jpg     # Card A, angled shot
cnie_front/
├── cardA_handheld_01.jpg   # Card A, held in hand
cnie_front/
├── cardA_dim_01.jpg        # Card A, dim lighting

# CNIE Back examples  
cnie_back/
├── cardB_flat_01.jpg       # Card B, flat on table
cnie_back/
├── cardB_flash_01.jpg      # Card B, with flash
cnie_back/
└── cardB_outdoor_01.jpg    # Card B, outdoor setting
```

### For Your Current Task (Strategy 2: 16 Images)

Based on your session, here are the exact filenames to use:

**cnie_front/** directory:
```
cardA_flat.jpg
cardA_angled.jpg
cardA_handheld.jpg
cardA_dim.jpg
cardB_flat.jpg
cardB_angled.jpg
cardB_handheld.jpg
cardB_dim.jpg
```

**cnie_back/** directory:
```
cardA_flat.jpg
cardA_angled.jpg
cardA_handheld.jpg
cardA_dim.jpg
cardB_flat.jpg
cardB_angled.jpg
cardB_handheld.jpg
cardB_dim.jpg
```

### Important Notes

1. **Extensions**: Use `.jpg` (recommended) or `.jpeg`. PNG and BMP are also supported but JPG is preferred for smaller file sizes.

2. **Case sensitivity**: Use lowercase filenames to avoid cross-platform issues.

3. **No spaces**: Use underscores (`_`) instead of spaces in filenames.

4. **Special characters**: Avoid special characters like `!@#$%^&*()` in filenames.

5. **Duplicate names**: Within a single class folder, filenames must be unique.

6. **Metadata in filename**: Keep it simple - the actual metadata (lighting conditions, angles, etc.) will be captured in the annotation JSON file (see below).

## Annotation Format

Each image needs metadata in JSON format:

```json
{
  "image_id": "cnie_front_001",
  "file_name": "cnie_front_001.jpg",
  "document_type": "cnie_front",
  "collection_metadata": {
    "date_collected": "2026-03-17",
    "device": "iPhone14,2",
    "lighting": "indoor_artificial",
    "angle_degrees": 25,
    "distance": "medium",
    "background": "wooden_desk",
    "card_condition": "new",
    "location": "office"
  },
  "quality_score": 4
}
```

## Collection Tools & Workflow

### Mobile App for Collection

Create a simple mobile collection app with these features:

```python
# Pseudo-code for collection app
class CollectionApp:
    def capture_image(self):
        # 1. Show capture guidelines overlay
        # 2. Auto-detect if card is in frame
        # 3. Capture with metadata
        # 4. Immediate quality check
        # 5. Tag with variations
        pass
    
    def quality_check(self, image):
        # Check: Is card visible?
        # Check: Is image blurry?
        # Check: Is lighting adequate?
        return quality_score
```

### Quality Control Checklist

Before adding an image to the dataset:

- [ ] Card is clearly visible (not obscured)
- [ ] Image is not blurry
- [ ] Card takes up at least 30% of frame
- [ ] Document type is identifiable
- [ ] File size > 50KB (not corrupted)
- [ ] Resolution >= 640x480

## Fine-Tuning Strategy

### Phase 1: Domain Adaptation (No labels needed)
If you have unlabeled real images:

```python
# Use self-supervised learning first
# Train on 1000+ unlabeled real images
# Then fine-tune with labeled data
```

### Phase 2: Supervised Fine-tuning (With labels)

**Recommended approach:**

```python
# 1. Freeze backbone, train only classifier (epochs 1-5)
# 2. Unfreeze last 2 blocks of EfficientNet (epochs 6-10)
# 3. Unfreeze all with low LR (epochs 11-15)
# 4. Early stopping based on validation accuracy
```

**Hyperparameters for fine-tuning:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| Initial LR | 1e-4 | Don't destroy pretrained weights |
| Batch Size | 16-32 | Balance speed/stability |
| Epochs | 10-20 | With early stopping |
| Data Augmentation | Heavy | Prevent overfitting to small dataset |
| Label Smoothing | 0.1 | Regularization |
| Dropout | 0.5 | Prevent overfitting |

### Training Schedule

```
Epoch 1-5:  LR=1e-4,  backbone frozen
Epoch 6-10: LR=5e-5,  last 2 blocks unfrozen
Epoch 11-15: LR=1e-5, full model unfrozen
Early stopping patience: 3 epochs
```

## Expected Results

### With 300 images (minimal):
- Synthetic accuracy: 100% → Real accuracy: 70-80%
- Time to collect: 1-2 days
- Training time: 30 minutes

### With 1000 images (recommended):
- Synthetic accuracy: 100% → Real accuracy: 85-92%
- Time to collect: 1 week
- Training time: 1-2 hours

### With 3000+ images (optimal):
- Synthetic accuracy: 100% → Real accuracy: 92-97%
- Time to collect: 2-3 weeks
- Training time: 3-4 hours

## Validation Strategy

**Hold-out test set:**
- 20% of collected images
- Must include all variation types
- Never seen during training

**Real-world validation:**
- Test on 50 completely new real images
- Test in actual usage environment
- Measure end-to-end accuracy

## Cost-Benefit Analysis

| Dataset Size | Collection Time | Accuracy Gain | Recommended? |
|--------------|-----------------|---------------|--------------|
| 300 images | 1-2 days | +70-80% | ⚠️ Quick fix only |
| 1000 images | 1 week | +85-92% | ✅ Best ROI |
| 3000 images | 2-3 weeks | +92-97% | ✅ Production |
| 10000 images | 1-2 months | +95-99% | ⭐ Ultimate |

**Recommendation:** Start with 1000 images for best balance of effort vs. accuracy.
