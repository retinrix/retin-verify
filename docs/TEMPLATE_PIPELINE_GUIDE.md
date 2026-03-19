# Template-based Synthetic Data Generation Pipeline

This guide explains the updated 2D template-based synthetic data generation pipeline for Retin-Verify, which uses real cleaned document templates with OpenCV augmentation.

## Overview

The pipeline generates synthetic Algerian ID documents (Passport, CNIE) using:
- Real cleaned document templates (personal data removed)
- Synthetic identity generation with valid MRZ codes
- 2D image augmentation (perspective transform, lighting, noise)
- Automatic annotation generation

## Quick Start

### 1. Generate Sample Data (Review Mode)

```bash
cd retin-verify
python3 synthetic/scripts/run_template_pipeline.py --sample-only --output-dir data/synthetic_review
```

This generates 5 samples per document type for quality review.

### 2. Generate Full Dataset

```bash
# Generate all document types with default counts
python3 synthetic/scripts/run_template_pipeline.py --output-dir data/synthetic

# Generate specific document type only
python3 synthetic/scripts/run_template_pipeline.py --doc-type passport --num-samples 1000

# Generate with custom output
python3 synthetic/scripts/run_template_pipeline.py --output-dir /path/to/output
```

## Template Files

Templates are stored in `synthetic/templates/real/`:

| Document | Template File | Size |
|----------|--------------|------|
| Passport | `passport_template.jpg` | 1366x1894px |
| CNIE Front | `cnie_front_template.jpg` | 1366x1894px |
| CNIE Back | `cnie_back_template.jpg` | 1366x1894px |

Templates must:
- Have personal data cleaned (names, photos, numbers removed)
- Preserve security patterns, field labels, and document structure
- Be in JPEG or PNG format

## Generated Variations

Each synthetic sample includes:

| Variation Type | Range | Description |
|---------------|-------|-------------|
| **Perspective** | -25° to +25° | Pitch, yaw variations simulating camera angles |
| **Brightness** | ±30 | Random brightness adjustment |
| **Contrast** | 0.8 to 1.2 | Contrast variation |
| **Gamma** | 0.8 to 1.2 | Gamma correction |
| **Blur** | 30% probability | Gaussian blur simulation |
| **Noise** | 30% probability | Random Gaussian noise |

## Output Structure

```
data/synthetic/
├── passport/
│   ├── 000000/
│   │   ├── image.jpg          # Synthetic document image
│   │   └── annotations.json   # Field annotations
│   ├── 000001/
│   └── ...
├── cnie_front/
│   └── ...
├── cnie_back/
│   └── ...
├── exports/
│   ├── annotations_coco.json  # COCO format annotations
│   └── yolo/                  # YOLO format annotations
└── dataset_manifest.json      # Train/val/test splits
```

## Annotation Format

Each `annotations.json` contains:

```json
{
  "sample_id": 0,
  "document_type": "passport",
  "identity": {
    "surname": "BENALI",
    "given_names": "MOHAMED",
    "date_of_birth": "15/03/1985",
    "passport_number": "DZ1234567",
    "mrz": {
      "line1": "P<DZABENALI<<MOHAMED...",
      "line2": "12345678DZA850315..."
    }
  },
  "image_path": "passport/000000/image.jpg",
  "image_size": [1366, 1894],
  "bounding_boxes": [
    {
      "field": "surname",
      "bbox": [450, 1080, 500, 60],
      "text": "BENALI"
    }
  ]
}
```

## Field Positions

The generator overlays text at predefined positions for each document type:

### Passport Fields
- `surname`, `given_names`, `date_of_birth`, `place_of_birth`
- `passport_number`, `date_of_issue`, `date_of_expiry`
- `nationality`, `sex`, `mrz_line1`, `mrz_line2`
- `photo` (synthetic face placeholder)

### CNIE Front Fields
- `national_id`, `surname`, `given_names`
- `date_of_birth`, `place_of_birth`
- `date_of_issue`, `date_of_expiry`, `blood_group`
- `photo` (synthetic face placeholder)

### CNIE Back Fields
- `address`, `father_name`, `mother_name`
- `mrz_line1`, `mrz_line2`

## Customization

### Adjust Field Positions

Edit `DOCUMENT_SPECS` in `synthetic/scripts/template_document_generator.py`:

```python
'passport': {
    'fields': {
        'surname': {
            'rel_bbox': [0.35, 0.57, 0.40, 0.03],  # [x, y, width, height]
            'font_scale': 0.8,
            'color': (60, 60, 80)
        },
        # ... more fields
    }
}
```

Coordinates are relative (0.0 to 1.0) to document dimensions.

### Add Backgrounds

Place background images in `synthetic/backgrounds/`:

```bash
mkdir -p synthetic/backgrounds
cp /path/to/backgrounds/*.jpg synthetic/backgrounds/
```

The generator will randomly composite documents onto backgrounds.

### Adjust Augmentation Parameters

Edit `apply_perspective_transform()`, `apply_lighting_variation()`, etc. in the generator to customize:
- Maximum perspective angles
- Brightness/contrast ranges
- Blur/noise probabilities

## Quality Review

After generating samples, review them visually:

```bash
# List generated samples
ls data/synthetic_review/*/000000/

# Check annotation validity
python3 -c "
import json
with open('data/synthetic_review/passport/000000/annotations.json') as f:
    ann = json.load(f)
    print(f'Fields: {[b[\"field\"] for b in ann[\"bounding_boxes\"]]}')
"
```

## Training Data Preparation

To prepare data for model training:

```bash
# Generate large dataset (e.g., 1000 per type)
python3 synthetic/scripts/run_template_pipeline.py \
    --output-dir data/training \
    --num-samples 1000

# The pipeline automatically creates:
# - COCO format for object detection
# - YOLO format for training
# - Train/val/test splits (80/10/10)
```

## Performance

Generation speed (approximate):
- **Per sample**: ~50-100ms
- **1000 samples**: ~1-2 minutes
- **10000 samples**: ~10-20 minutes

No GPU required - runs on CPU only.

## Troubleshooting

### Text not aligned with fields
- Adjust `rel_bbox` coordinates in `DOCUMENT_SPECS`
- Use image editor to measure field positions on template

### Poor image quality
- Increase `image_quality` in config (default: 95)
- Check template resolution (should be 300+ DPI equivalent)

### Missing fields in annotations
- Ensure field names match between generator and validator
- Check `_validation_errors` in annotation files

## Comparison with Blender Pipeline

| Feature | Template (2D) | Blender (3D) |
|---------|--------------|--------------|
| Speed | Fast (~100ms/sample) | Slow (~5-10s/sample) |
| Realism | Good | Excellent |
| Setup | Simple | Complex |
| Backgrounds | 2D compositing | 3D environments |
| Bending/Warping | Limited | Full 3D deformation |
| GPU Required | No | Optional |

For rapid prototyping and training data generation, the template-based approach is recommended. For maximum realism and advanced augmentations, use the Blender pipeline.

## Next Steps

1. Review generated samples visually
2. Adjust field positions if needed
3. Generate full training dataset
4. Train document classification/detection models
5. Evaluate model performance
