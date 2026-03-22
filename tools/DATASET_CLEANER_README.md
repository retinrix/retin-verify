# CNIE Dataset Cleaner Tool

## Overview

This tool automatically cleans the CNIE front/back classification dataset by:
1. **Face Detection**: Detects faces in images labeled as "front"
2. **Chip Detection**: Detects chips in images labeled as "back"
3. **Flagging**: Flags potential mislabels for review
4. **Auto-relabeling**: High-confidence cases are automatically relabeled

## Files

- `dataset_cleaner_ui.py` - GUI version (requires display)
- `dataset_cleaner_cli.py` - Command-line version (headless/Colab)

## Quick Start (CLI)

### 1. Scan Dataset

```bash
python3 dataset_cleaner_cli.py \
    --dataset /path/to/v8_stage2_clean \
    --report cleaning_report.json \
    --chip-threshold 0.55
```

### 2. Export Cleaned Dataset

```bash
# Manual review required for flagged images
python3 dataset_cleaner_cli.py \
    --dataset /path/to/v8_stage2_clean \
    --output /path/to/cleaned_dataset \
    --report cleaning_report.json

# Or auto-keep all flagged images
python3 dataset_cleaner_cli.py \
    --dataset /path/to/v8_stage2_clean \
    --output /path/to/cleaned_dataset \
    --auto-keep

# Or auto-exclude all flagged images
python3 dataset_cleaner_cli.py \
    --dataset /path/to/v8_stage2_clean \
    --output /path/to/cleaned_dataset \
    --auto-exclude
```

## GUI Version

```bash
python3 dataset_cleaner_ui.py
```

Features:
- Browse and select dataset
- Visual progress bar during scan
- Image preview with detection overlays
- Flagged images list with reasons
- Action buttons: Keep, Relabel (Front/Back), Exclude
- Export cleaned dataset with metadata

## Flagging Logic

| Original Label | Face Detected | Chip Detected | Action |
|----------------|---------------|---------------|--------|
| Front | ✓ | - | Verified clean |
| Front | ✗ | ✓ (conf>0.6) | **Flagged** - Suggest relabel to Back |
| Front | ✗ | ✗ | **Flagged** - Ambiguous, needs review |
| Back | - | ✓ | Verified clean |
| Back | ✓ (conf>0.5) | ✗ | **Flagged** - Suggest relabel to Front |
| Back | ✗ | ✗ | **Flagged** - Ambiguous, needs review |

## V10 Cleaning Results

### Dataset: v8_stage2_clean

**Statistics:**
- Total Images: 582
- Verified Clean: 486 (83.5%)
- Flagged: 96 (16.5%)
  - Front without Face: 96
  - Back without Chip: 0

**Auto-Cleaning Decisions:**
- Kept (verified clean): 486
- Auto-relabeled to back: 56 (front images with chip detected, confidence > 0.6)
- Excluded (ambiguous): 2
- Kept with manual review needed: 38

**Dataset Shift:**
| Split | Original Front/Back | Cleaned Front/Back |
|-------|---------------------|--------------------|
| Train | 232 / 232 | 184 / 278 |
| Val | 29 / 29 | 24 / 34 |
| Test | 30 / 30 | 25 / 35 |

**Key Finding:** 56 images originally labeled as "front" were actually "back" (had chips but no faces detected).

## Chip Template

The chip detector uses a synthetic template matching the CNIE chip pattern:
- Oval shape with globe-like lines
- Multi-scale template matching (0.7x to 1.3x)
- Threshold: 0.55 (configurable)

## Face Detection

Uses OpenCV Haar Cascade classifier:
- `haarcascade_frontalface_default.xml`
- Min face size: 50x50 pixels
- Scale factor: 1.1
- Min neighbors: 4

## Colab Usage

```python
# Upload dataset_cleaner_cli.py to Colab
# Then run:
!python3 dataset_cleaner_cli.py \
    --dataset /content/v8_stage2_clean \
    --output /content/cleaned_dataset \
    --report /content/cleaning_report.json

# Download report
from google.colab import files
files.download('/content/cleaning_report.json')
```

## Recommendations

1. **Review auto-relabeled images**: Check the 56 images that were relabeled from front to back
2. **Balance the dataset**: After cleaning, the dataset has more back than front images
3. **Collect more front images**: To rebalance the dataset
4. **Retrain model**: Use the cleaned dataset for V11 training

## Next Steps

1. Manually review the 38 images marked for manual review
2. Verify the 56 auto-relabeled images are correct
3. Consider excluding or collecting replacements for the 2 ambiguous images
4. Retrain the model with the cleaned dataset
