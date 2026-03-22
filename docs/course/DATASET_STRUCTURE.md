# Dataset Structure Explanation

This document explains each file in the generated dataset and how they are used in the ML pipeline.

## 📁 Overall Directory Structure

```
data/cnie_dataset_10k/
├── cnie_pairs/                    # Paired CNIE samples (front + back)
│   ├── 000000/
│   │   ├── annotations.json       # Shared identity, MRZ, bounding boxes
│   │   ├── front/
│   │   │   └── image.jpg          # Front side image
│   │   └── back/
│   │       └── image.jpg          # Back side image
│   ├── 000001/
│   └── ... (5,000 pairs total)
├── dataset_manifest.json          # Train/val/test split information
└── exports/
    ├── annotations_coco.json      # COCO format for object detection
    ├── yolo/                      # YOLO format for text detection
    │   ├── 000000_front.txt
    │   ├── 000000_back.txt
    │   └── classes.txt
    └── summary.json               # Generation statistics
```

---

## 📄 File-by-File Explanation

### 1. `cnie_pairs/XXXXXX/annotations.json`

**Purpose**: Master annotation file for each identity pair

**Contains**:
```json
{
  "pair_id": 0,                    // Unique pair identifier
  "identity": {                    // Shared identity data
    "surname": "Hamidi",           // Latin surname
    "surname_ar": "حميدي",         // Arabic surname
    "given_names": "Souad",
    "given_names_ar": "سعاد",
    "sex": "F",
    "national_id": "6782128549",
    "date_of_birth": "24/01/1966",
    "mrz": {                       // MRZ for CNIE back
      "line1": "IDDZA6782128545<<<<<<<<<<<<<<<",
      "line2": "6601247F3101263DZA<<<<<<<<<<4",
      "line3": "HAMIDI<<SOUAD<<<<<<<<<<<<<<<<<"
    }
  },
  "front": {                       // Front side data
    "document_type": "cnie_front",
    "image_path": "cnie_pairs/000000/front/image.jpg",
    "image_size": [1622, 1020],    // Width, Height in pixels
    "bounding_boxes": [...]        // Field locations
  },
  "back": {                        // Back side data
    "document_type": "cnie_back",
    "image_path": "cnie_pairs/000000/back/image.jpg",
    "bounding_boxes": [...]
  }
}
```

**How it's used**:
- **Field Extraction Training**: Train LayoutLMv3 to extract fields using bounding boxes and text
- **OCR Validation**: Ground truth text for measuring OCR accuracy
- **MRZ Validation**: Verify MRZ checksums against generated data
- **Face Recognition**: Link photo to VGGFace2 identity_id

---

### 2. `cnie_pairs/XXXXXX/front/image.jpg`

**Purpose**: The actual image of the CNIE front side

**Characteristics**:
- Resolution: ~1622x1020 pixels (varies by template)
- Format: JPEG with 95% quality
- Contains: Photo, personal info in Arabic/French

**How it's used**:
- **Document Classification**: Train model to recognize CNIE front
- **Text Detection**: Find text regions (DB/MobileNetV3)
- **OCR**: Extract text from detected regions (PaddleOCR/Tesseract)
- **Face Detection/Recognition**: Extract and match face photo

---

### 3. `cnie_pairs/XXXXXX/back/image.jpg`

**Purpose**: The actual image of the CNIE back side

**Characteristics**:
- Resolution: ~1698x1066 pixels (varies by template)
- Format: JPEG with 95% quality
- Contains: MRZ zone, birth year, parent names

**How it's used**:
- **Document Classification**: Train model to recognize CNIE back
- **MRZ Detection**: Specialized detection of MRZ zone
- **MRZ OCR**: OCR-B font recognition for MRZ lines
- **MRZ Validation**: Verify check digits against identity data

---

### 4. `dataset_manifest.json`

**Purpose**: Defines train/validation/test splits for ML training

**Contains**:
```json
{
  "dataset_name": "Retin-Verify CNIE Dataset 10K",
  "created_at": "2026-03-13T23:32:00",
  "total_samples": 5000,
  "splits": {
    "train": {
      "count": 4000,
      "samples": ["000000", "000001", "000002", ...]
    },
    "val": {
      "count": 500,
      "samples": ["4000", "4001", ...]
    },
    "test": {
      "count": 500,
      "samples": ["4500", "4501", ...]
    }
  },
  "samples": [
    {
      "id": "000000",
      "document_type": "cnie_paired",
      "image_path": "cnie_pairs/000000",
      "annotation_path": "cnie_pairs/000000/annotations.json",
      "is_paired": true
    }
  ]
}
```

**How it's used**:
- **Training Scripts**: Load correct split for training/validation
- **Experiment Tracking**: Know which samples were used for what
- **Reproducibility**: Same splits across different training runs
- **Evaluation**: Ensure test set is never seen during training

---

### 5. `exports/annotations_coco.json`

**Purpose**: COCO format for object detection and instance segmentation

**Contains**:
```json
{
  "info": {...},
  "images": [
    {
      "id": 1,
      "file_name": "cnie_pairs/000000/front/image.jpg",
      "width": 1622,
      "height": 1020
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [30, 408, 464, 533],  // [x, y, width, height]
      "area": 247312,
      "attributes": {
        "text": "",
        "confidence": 1.0
      }
    }
  ],
  "categories": [
    {"id": 1, "name": "photo_1", "supercategory": "text_field"},
    {"id": 2, "name": "surname", "supercategory": "text_field"}
  ]
}
```

**How it's used**:
- **Text Detection Training**: Train DB (Differentiable Binarization) model
- **Object Detection**: Detect text regions as objects
- **Tools**: Compatible with Detectron2, MMDetection, etc.
- **Evaluation**: Standard COCO evaluation metrics (AP, AR)

**Training Example**:
```python
# Load COCO annotations for text detection training
coco = COCO('exports/annotations_coco.json')
# Train DB model to detect text regions
```

---

### 6. `exports/yolo/XXXXXX_front.txt` and `XXXXXX_back.txt`

**Purpose**: YOLO format for text detection training

**Contains** (one line per bounding box):
```
0 0.161529 0.661275 0.286067 0.522549
1 0.418619 0.884314 0.054254 0.043137
```

**Format**: `<class_id> <x_center> <y_center> <width> <height>`
- All values normalized to [0, 1]
- x_center, y_center: Center of bounding box
- width, height: Relative to image size

**How it's used**:
- **YOLO Training**: Train YOLOv8 for text detection
- **Fast Inference**: YOLO is faster than DB for production
- **Edge Deployment**: YOLO models are compact

**Training Example**:
```bash
# Train YOLOv8 on the dataset
yolo detect train data=exports/yolo/dataset.yaml model=yolov8n.pt epochs=100
```

---

### 7. `exports/yolo/classes.txt`

**Purpose**: Maps class IDs to field names

**Contains**:
```
photo_1
blood_group
date_of_birth
surname
given_names
mrz_line1
mrz_line2
mrz_line3
...
```

**How it's used**:
- **Model Interpretation**: Convert class IDs back to field names
- **Post-processing**: Map detections to document fields
- **Evaluation**: Calculate per-field accuracy

---

### 8. `exports/summary.json`

**Purpose**: Quick statistics about the generated dataset

**Contains**:
```json
{
  "generated_at": "2026-03-13T23:32:00",
  "total_samples": 10000,
  "by_type": {
    "cnie_front": 5000,
    "cnie_back": 5000
  }
}
```

**How it's used**:
- **Quick Check**: Verify generation completed successfully
- **Documentation**: Dataset statistics for reports
- **Monitoring**: Track dataset size over time

---

## 🎯 How Each Component is Used in the ML Pipeline

### Stage 1: Document Classification
```
Input: image.jpg (front or back)
Output: Document type (cnie_front / cnie_back / passport)

Files used:
- image.jpg (from all folders)
- dataset_manifest.json (for train/val/test splits)
```

**Model**: EfficientNet-B0  
**Training**:
```python
python src/classification/train_classifier.py \
    --data-dir data/cnie_dataset_10k \
    --manifest data/cnie_dataset_10k/dataset_manifest.json
```

---

### Stage 2: Text Detection
```
Input: image.jpg
Output: Bounding boxes around text regions

Files used:
- exports/annotations_coco.json (COCO format)
- OR exports/yolo/*.txt (YOLO format)
```

**Model**: DB (Differentiable Binarization) with MobileNetV3 backbone  
**Training**:
```python
python src/detection/train_detector.py \
    --coco-annotations exports/annotations_coco.json \
    --images-dir data/cnie_dataset_10k
```

---

### Stage 3: OCR (Optical Character Recognition)
```
Input: Detected text regions (cropped from image.jpg)
Output: Text content

Files used:
- cnie_pairs/XXXXXX/annotations.json (ground truth text)
- image.jpg (source images)
```

**Models**: 
- Primary: PaddleOCR (multilingual)
- MRZ: Tesseract with OCR-B font

**Training/Validation**:
```python
# Validate OCR accuracy against ground truth
python src/ocr/validate_ocr.py \
    --annotations cnie_pairs/XXXXXX/annotations.json \
    --images-dir data/cnie_dataset_10k
```

---

### Stage 4: Field Extraction (LayoutLMv3)
```
Input: Full document image + detected text regions
Output: Structured field data (surname, dob, etc.)

Files used:
- cnie_pairs/XXXXXX/annotations.json (field labels + text)
- image.jpg
```

**Model**: LayoutLMv3 ( multimodal: text + layout + image)  
**Training**:
```python
python src/extraction/train_extractor.py \
    --annotations exports/annotations_coco.json \
    --images-dir data/cnie_dataset_10k
```

---

### Stage 5: Validation
```
Input: Extracted fields
Output: Valid/Invalid with confidence scores

Files used:
- cnie_pairs/XXXXXX/annotations.json (ground truth)
  - MRZ lines for checksum validation
  - Cross-field consistency (dob vs expiry)
```

**Validation Rules**:
```python
from src.validation.validators import MRZValidator

validator = MRZValidator()
is_valid = validator.validate_mrz(mrz_line1, mrz_line2, mrz_line3)
```

---

## 📊 Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA GENERATION                            │
│  Template + Identity → image.jpg + annotations.json             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                             │
│  - dataset_manifest.json → Train/Val/Test splits               │
│  - exports/annotations_coco.json → Detection format            │
│  - exports/yolo/ → YOLO format                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML TRAINING                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Classification│  │    Text      │  │    Field     │         │
│  │  (Doc Type)   │  │  Detection   │  │  Extraction  │         │
│  │  EfficientNet │  │   DB/CRNN    │  │  LayoutLMv3  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                           │
│  Input Image → Classify → Detect Text → OCR → Extract Fields   │
│                                          ↓                      │
│                                    Validate MRZ                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💾 Storage Requirements

For 10,000 CNIE samples (5,000 pairs):

| Component | Size | Notes |
|-----------|------|-------|
| Images (front) | ~1.5 GB | 5,000 images × ~300KB |
| Images (back) | ~1.5 GB | 5,000 images × ~300KB |
| Annotations | ~50 MB | JSON files with bounding boxes |
| Exports (COCO/YOLO) | ~20 MB | Converted formats |
| **Total** | **~3 GB** | Plus ~5 GB for temporary files |

---

## 🔍 Quick Verification Commands

```bash
# Check dataset integrity
python3 << 'EOF'
import json
from pathlib import Path

data_dir = Path("data/cnie_dataset_10k")

# Load manifest
with open(data_dir / "dataset_manifest.json") as f:
    manifest = json.load(f)

print(f"Total pairs: {manifest['total_samples']}")
print(f"Train: {manifest['splits']['train']['count']}")
print(f"Val: {manifest['splits']['val']['count']}")
print(f"Test: {manifest['splits']['test']['count']}")

# Verify a sample
with open(data_dir / "cnie_pairs/000000/annotations.json") as f:
    ann = json.load(f)

print(f"\nSample 000000:")
print(f"  Name: {ann['identity']['surname']} {ann['identity']['given_names']}")
print(f"  MRZ Line 1: {ann['identity']['mrz']['line1']}")
print(f"  Front fields: {len(ann['front']['bounding_boxes'])}")
print(f"  Back fields: {len(ann['back']['bounding_boxes'])}")
EOF
```

---

## 🚀 Next Steps

1. **Verify dataset**: Run the verification script above
2. **Start training**: Begin with document classification (fastest)
3. **Iterate**: Add real data if synthetic performance plateaus
4. **Evaluate**: Test on real CNIE scans from ANPDP (with authorization)
