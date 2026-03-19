# Retin-Verify Quick Start Guide

Get started with Retin-Verify IDP System in minutes.

## Prerequisites

- Python 3.8+
- 8GB+ RAM
- Linux/macOS/Windows

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd retin-verify
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-ara \
    libtesseract-dev \
    blender
```

For other systems:
- **macOS**: `brew install tesseract tesseract-lang`
- **Windows**: Download installers from GitHub

## Quick Start

### Option 1: Use the API

Start the API server:

```bash
python -m uvicorn src.api.main:app --reload
```

Access:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/v1/health

Upload a document:

```bash
curl -X POST "http://localhost:8000/v1/extract" \
    -F "file=@/path/to/document.jpg"
```

### Option 2: Use Python API

```python
from src.pipeline import IDPPipeline
from pathlib import Path

# Initialize pipeline
pipeline = IDPPipeline()

# Process document
result = pipeline.process(Path("document.jpg"))

# View results
print(f"Document Type: {result['document_type']}")
print(f"Fields: {result['fields']}")
print(f"Valid: {result['validation']['valid']}")
```

### Option 3: Generate Synthetic Training Data

```bash
# Generate 100 synthetic passport identities
python synthetic/scripts/identity_generator.py

# Full data generation (requires Blender)
python synthetic/scripts/data_acquisition_pipeline.py
```

## Project Structure

```
retin-verify/
├── src/                    # Source code
│   ├── preprocessing/      # Image enhancement
│   ├── classification/     # Document classifier
│   ├── detection/          # Text detection
│   ├── ocr/                # OCR engine
│   ├── extraction/         # Field extraction
│   ├── validation/         # Data validation
│   ├── pipeline.py         # Main pipeline
│   └── api/                # REST API
├── data/                   # Datasets
├── models/                 # Trained models
├── configs/                # Configuration files
├── tests/                  # Unit tests
└── scripts/                # Utility scripts
```

## Training Models

### Document Classification

```bash
python src/classification/train_classifier.py \
    --data-dir data/synthetic/images \
    --train-annotations data/synthetic/train.json \
    --val-annotations data/synthetic/val.json \
    --epochs 50 \
    --batch-size 32
```

### Text Detection

```bash
python src/detection/train_detector.py \
    --train-dir data/synthetic/train \
    --val-dir data/synthetic/val \
    --epochs 100
```

### Field Extraction

```bash
python src/extraction/train_extractor.py \
    --train-file data/synthetic/train_ner.json \
    --val-file data/synthetic/val_ner.json \
    --epochs 20
```

## Testing

Run unit tests:

```bash
# All tests
python -m pytest tests/

# Specific module
python -m pytest tests/test_preprocessing.py

# With coverage
python -m pytest tests/ --cov=src
```

## Common Tasks

### Preprocess an Image

```python
from src.preprocessing.image_enhancement import DocumentPreprocessor
import cv2

preprocessor = DocumentPreprocessor()
image = cv2.imread("document.jpg")
processed = preprocessor.preprocess(image)
cv2.imwrite("processed.jpg", processed)
```

### Validate MRZ

```python
from src.validation.validators import MRZValidator

line1 = "P<DZABENALI<<MOHAMED<<<<<<<<<<<<<<<<<<<<<<<<"
line2 = "12345678<8DZA8503157M2704159<<<<<<<<<<<8"

valid, errors = MRZValidator.validate_full_mrz(line1, line2)
print(f"MRZ Valid: {valid}")
```

### Extract Fields from OCR

```python
from src.extraction.field_extractor import FieldExtractor

extractor = FieldExtractor(use_ml=False)

ocr_results = [
    {'text': 'Nom: BENALI', 'bbox': [100, 100, 150, 30]},
    {'text': 'Prénom: MOHAMED', 'bbox': [100, 150, 180, 30]},
]

fields = extractor._extract_passport_fields(ocr_results, "")
print(fields)
```

## Configuration

Edit `configs/model_configs.yaml` to customize:

- Model architectures
- Training parameters
- Inference settings
- Performance targets

## Troubleshooting

### Tesseract not found

```bash
# Check installation
tesseract --version

# Set path if needed
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
```

### CUDA out of memory

```python
# Use CPU
pipeline = IDPPipeline(use_gpu=False)

# Or reduce batch size in config
```

### Low quality images rejected

```python
# Skip quality check
result = pipeline.process("image.jpg", skip_quality_check=True)
```

## Next Steps

1. **Generate synthetic data**: See `synthetic/README.md`
2. **Train models**: Follow training guides above
3. **Deploy API**: Use Docker or cloud platforms
4. **Customize**: Modify configs and retrain

## Support

- Documentation: See `docs/` directory
- Issues: GitHub Issues
- Examples: `scripts/example_usage.py`
