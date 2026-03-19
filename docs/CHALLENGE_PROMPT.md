# Challenge: Position Personal Data on CNIE Image Template

## Problem Statement

We need to accurately position synthetic personal data onto an Algerian CNIE (Carte Nationale d'Identité Électronique) blank template image.

### Current State

- **Template location**: `templates/real/cnie_front_template.jpg` (1366x1894 pixels)
- **Clean template** with Arabic/French labels but NO personal data
- **11 fields** need to be positioned:
  1. National ID (NNI)
  2. Place of Issue
  3. Date of Issue
  4. Date of Expiry
  5. Personal ID (long number)
  6. Surname
  7. Given Names
  8. Sex
  9. Date of Birth
  10. Blood Group
  11. Place of Birth

### Previous Failed Approaches (Archived)

All these approaches have been tried and FAILED:

1. **Pixel Manual Entry** (`get_pixels.py`, `pixel_coordinates.py`) - Too error-prone
2. **Grid Overlay** (`coordinate_overlay.py`, `create_numbered_grid.py`) - Still manual translation
3. **OpenCV Click Measurement** (`measure_cnie_positions.py`) - GUI doesn't work in headless/WSL
4. **Matplotlib Terminal Input** (`measure_cnie_matplotlib.py`) - Too slow, no visual feedback
5. **Web-based Precision Tool** (`measure_engine.js`, `measure_template.html`) - Complex setup, still manual
6. **Auto-Detection** (`auto_positioner.py`) - Positions were messy and incorrect
7. **Interactive Manual Positioner** (`manual_positioner.py`) - Current approach, needs refinement

### Requirements

1. **Headless operation** - Must work in WSL/SSH without X11
2. **Minimal user interaction** - Ideally 1-2 confirmations, not 10+ measurements
3. **Accuracy** - Text must align with Arabic/French labels on template
4. **Repeatability** - Should work for different CNIE templates (same ID-1 format)
5. **Export** - Must generate config for `template_document_generator.py`

### Template Format

- **Standard**: ID-1 (85.6 × 53.98 mm)
- **Image size**: 1366 × 1894 pixels
- **Card in image**: ~1005 × 642 pixels, positioned at approximately (185, 632)

### CNIE Field Layout (Standard)

```
┌─────────────────────────────────────────────────────────────┐
│  [National ID]          [Place]  [Issue Date]              │
│                                                         │
│                                [Expiry Date]             │
│                                                         │
│  [Personal ID Number - Long]                            │
│                                                         │
│  [Given Names]              [Surname]                    │
│                                                         │
│  [Blood]   [Photo]    [Sex]  [DOB]                      │
│                           [Place of Birth]               │
└─────────────────────────────────────────────────────────────┘
```

### Available Tools

- `template_document_generator.py` - Main generator using OpenCV
- `manual_positioner.py` - Last attempt with grid overlay
- `annotation_utils.py` - Annotation format utilities
- `run_template_pipeline.py` - Pipeline runner

### Success Criteria

Generate 3 test samples and verify:
- [ ] All text within card boundaries
- [ ] No overlapping text
- [ ] Fields align with their Arabic/French labels
- [ ] Photo area not obscured
- [ ] Blood group visible on photo

### Current Config Location

Template spec is in `template_document_generator.py` in `DOCUMENT_SPECS['cnie_front']`:
```python
'cnie_front': {
    'template_file': 'cnie_front_template.jpg',
    'aspect_ratio': 85.6/53.98,
    'fields': { ... },
    'photo_bbox': [0.172, 0.3944, 0.2057, 0.142],
}
```

### Challenge

Create a NEW approach that solves the positioning problem with minimal user interaction. Consider:

1. **Template matching** - Detect label positions automatically
2. **OCR-based alignment** - Read Arabic/French labels, position values relative to them
3. **Reference-based** - Use a reference CNIE with known good positions
4. **ML-based detection** - Train on examples (if you have training data)
5. **Smart defaults** - Better ID-1 standard layout with auto-adjustment

### Constraints

- Do NOT use GUI-based tools (no cv2.imshow, no browser)
- Do NOT require 10+ manual measurements
- MUST work headless
- MUST export to `template_document_generator.py` format

### Test Command

```bash
cd retin-verify
python3 synthetic/scripts/run_template_pipeline.py \
    --doc-type cnie_front \
    --num-samples 3 \
    --output-dir data/test_output

# Check results in data/test_output/cnie_front/000000/image.jpg
```

---

## Archived Failed Approaches

All previous attempts are in `synthetic/scripts/archive/`:
- `auto_positioner.py` - Auto-detection (positions were messy)
- `measure_cnie_positions.py` - OpenCV GUI (doesn't work headless)
- `measure_engine.js` - Web tool (too complex)
- `manual_positioner.py` - Current but needs improvement
- And 20+ other failed attempts...

---

**Goal**: Create a working solution that positions personal data accurately on the CNIE template with minimal user interaction (ideally 1 confirmation).
