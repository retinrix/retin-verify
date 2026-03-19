# Retin-Verify Project Assessment
## Intelligent Document Processing (IDP) System for Algerian Identity Documents

---

## 1. EXECUTIVE SUMMARY

### Project Overview
**Retin-Verify** is an AI-powered Intelligent Document Processing system for Algerian identity documents including:
- Biometric Passport (Passeport Biométrique Algérien)
- CNIE (Carte Nationale d'Identité Électronique)
- Carte Grise (Vehicle Registration Certificate)

### System Status
| Parameter | Value |
|-----------|-------|
| **Project Location** | `/home/retinrix/Projects/retin-verify/` |
| **Current Size** | 44 KB (specification document only) |
| **Disk Space Available** | 23 GB |
| **Memory Available** | 4 GB (out of 16 GB) |
| **GPU** | ❌ Not detected |
| **Disk Usage** | 93% (Warning: Critical) |

---

## 2. DISK CAPACITY ANALYSIS

### Current Storage Situation
```
Filesystem: /dev/nvme0n1p5
Total:      341 GB
Used:       301 GB (93%)
Available:  23 GB (7%)
Status:     ⚠️ CRITICAL - Action Required
```

### Project Space Requirements Estimate

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Source Code & Configs | 500 MB | 1 GB | Python, configs, documentation |
| Python Environment | 2 GB | 5 GB | PyTorch, OpenCV, Pillow, etc. |
| Training Datasets | 10 GB | 50+ GB | Synthetic + Real document images |
| Model Checkpoints | 5 GB | 20+ GB | Multiple model versions |
| Blender & Assets | 1 GB | 3 GB | 3D scenes, textures, backgrounds |
| Logs & Cache | 2 GB | 5 GB | Training logs, temp files |
| **TOTAL REQUIRED** | **~21 GB** | **~85+ GB** | |

### ⚠️ CRITICAL FINDING
**Current available space (23 GB) is INSUFFICIENT for full project development.**

- **Minimum viable**: 21 GB (tight fit, no room for growth)
- **Recommended**: 85+ GB for comfortable development with multiple datasets and model checkpoints

---

## 3. HARDWARE CAPABILITY ASSESSMENT

### Memory (RAM)
```
Total:     16 GB
Used:      10 GB
Available: 4 GB
Swap:      2 GB (1.6 GB used)
```

**Assessment**: 
- ✅ Adequate for inference (running trained models)
- ⚠️ Marginal for training smaller models
- ❌ Insufficient for training large deep learning models

### GPU
```
Status: ❌ No NVIDIA GPU detected
```

**Impact**:
- Training will be extremely slow on CPU (10-100x slower)
- Cannot train large models (Vision Transformers, LayoutLMv3)
- Inference will be CPU-based (acceptable for small batches)

---

## 4. PROJECT SPECIFICATIONS SUMMARY

### 4.1 Document Types to Support

| Document | Format | Key Fields | Security Features |
|----------|--------|------------|-------------------|
| **Passport** | ID-3 (125×88mm) | MRZ, Name, DOB, Photo, Expiry | Biometric chip, holograms |
| **CNIE** | ID-1 (85.6×54mm) | ID Number, Address, Blood Group | Electronic chip, laser engraving |
| **Carte Grise** | A4/Card | VIN, Registration, Owner details | Variable formats |

### 4.2 AI Models Required

| Component | Model Options | Size Estimate | Training Data Needed |
|-----------|---------------|---------------|---------------------|
| **Document Classification** | EfficientNet-B3 / ViT | ~100 MB | 10K+ images |
| **Text Detection** | DB (Differentiable Binarization) | ~50 MB | 10K+ annotated images |
| **OCR Engine** | PaddleOCR / Tesseract 5 | ~200 MB | Pre-trained + fine-tuning |
| **Field Extraction (NER)** | LayoutLMv3 | ~500 MB | 5K+ labeled documents |
| **Face Detection** | MTCNN / RetinaFace | ~30 MB | Pre-trained |
| **Face Comparison** | ArcFace | ~100 MB | Pre-trained |

### 4.3 Performance Targets

| Metric | Target | Current Feasibility |
|--------|--------|---------------------|
| Document Classification | > 99.5% | ✅ Achievable |
| Field Extraction | > 98% | ⚠️ Requires GPU for training |
| MRZ CER | < 1% | ✅ Achievable with Tesseract |
| Processing Time | < 3 sec | ⚠️ CPU-only will be slower |
| Throughput | 10+ docs/sec | ❌ Requires GPU |

---

## 5. DATA ACQUISITION STRATEGY

### 5.1 Recommended Approach: Hybrid (Synthetic + Real)

#### Phase 1: Synthetic Data Generation (PRIMARY)
**Tools**: Blender + Python scripting

**Setup Requirements**:
- Blender (free, ~500 MB download)
- Python 3.8+ with packages: `pip install bpy pillow numpy`
- Background image library (~100 images, ~500 MB)
- Document templates (created from clean scans)

**Generation Pipeline**:
```
1. Scan documents at 300+ DPI
2. Remove personal data in GIMP/Photoshop
3. Create Blender 3D scene with document plane
4. Generate synthetic identities (Python script)
5. Randomize: Camera angle (-45° to +45°), Lighting, Backgrounds
6. Render 10,000+ variations
7. Auto-generate perfect annotations (JSON)
```

**Output per Sample**:
- RGB image (PNG/JPEG)
- Bounding boxes (JSON)
- Field labels (JSON)
- Camera pose metadata

#### Phase 2: Real-World Collection (SECONDARY)
**Volume**: 1,000-2,000 real documents for fine-tuning

**Requirements**:
- ANPDP authorization (Algerian data protection)
- Consent forms from data subjects
- Data anonymization pipeline
- Secure storage (encrypted)

### 5.2 Data Volume Estimates

| Phase | Images | Size per Image | Total Size |
|-------|--------|----------------|------------|
| Synthetic Training | 50,000 | ~500 KB | 25 GB |
| Synthetic Validation | 10,000 | ~500 KB | 5 GB |
| Real Fine-tuning | 2,000 | ~2 MB | 4 GB |
| Test Sets | 5,000 | ~500 KB | 2.5 GB |
| **TOTAL** | **67,000** | - | **~37 GB** |

---

## 6. STEP-BY-STEP IMPLEMENTATION PLAN

### Phase 1: Environment Setup (Week 1)

#### Step 1.1: Free Up Disk Space (CRITICAL)
```bash
# Minimum required: 50 GB free space
# Current available: 23 GB
# Action needed: Free up at least 27 GB more

# Check large directories
du -sh ~/* 2>/dev/null | sort -hr | head -20

# Clean package caches
sudo apt-get clean
npm cache clean --force
pip cache purge

# Remove old builds and logs
rm -rf ~/Projects/*/build/
rm -rf ~/Projects/*/dist/
rm -rf ~/.local/share/Trash/files/*
```

#### Step 1.2: Install Core Dependencies
```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv \
    tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara \
    libtesseract-dev \
    blender \
    git git-lfs

# Python virtual environment
python3 -m venv ~/retin-verify-env
source ~/retin-verify-env/bin/activate

# Core ML libraries
pip install torch torchvision --index-url https://download.pytorch.org/cpu/whl/cpu
pip install paddlepaddle paddleocr
pip install transformers layoutlmv3
pip install opencv-python pillow numpy pandas
pip install scikit-learn matplotlib seaborn
pip install fastapi uvicorn
```

**Estimated Install Size**: 8-10 GB

#### Step 1.3: Project Structure Setup
```
retin-verify/
├── docs/                    # Documentation
├── src/                     # Source code
│   ├── preprocessing/       # Image enhancement
│   ├── classification/      # Document type classifier
│   ├── detection/           # Text detection
│   ├── ocr/                 # OCR engine
│   ├── extraction/          # Field extraction
│   ├── validation/          # Data validation
│   └── api/                 # REST API
├── data/                    # Datasets (git-lfs)
│   ├── raw/                 # Original images
│   ├── processed/           # Preprocessed
│   ├── synthetic/           # Blender generated
│   └── annotations/         # Labels
├── models/                  # Trained models
├── synthetic/               # Blender pipeline
│   ├── scenes/              # .blend files
│   ├── templates/           # Clean document images
│   ├── backgrounds/         # Background images
│   └── scripts/             # Generation scripts
├── tests/                   # Unit tests
├── configs/                 # YAML configs
└── notebooks/               # Jupyter experiments
```

### Phase 2: Data Pipeline (Weeks 2-4)

#### Step 2.1: Document Template Creation
1. **Scan documents** at 300-600 DPI (if available)
2. **Clean templates** in GIMP:
   - Remove all personal data (names, photos, numbers)
   - Preserve security patterns and layout
   - Export as transparent PNG
3. **Create variations**:
   - Passport front
   - CNIE front/back
   - Carte Grise front/back

#### Step 2.2: Synthetic Data Generation Setup
```python
# synthetic/generator.py - Key components

import bpy
import json
import random
from pathlib import Path

class DocumentGenerator:
    def __init__(self, template_dir, output_dir):
        self.templates = {
            'passport': 'passport_template.png',
            'cnie_front': 'cnie_front_template.png',
            'cnie_back': 'cnie_back_template.png',
            'carte_grise': 'carte_grise_template.png'
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_synthetic_identity(self):
        """Generate fake Algerian identity data"""
        return {
            'surname': random.choice(['BENALI', 'SAID', 'BOUAZIZ']),
            'given_names': random.choice(['MOHAMED', 'FATIMA', 'KARIM']),
            'date_of_birth': self.random_date(),
            'place_of_birth': random.choice(['ALGER', 'ORAN', 'CONSTANTINE']),
            'document_number': self.generate_document_number(),
            'mrz': self.generate_mrz(),
            'sex': random.choice(['M', 'F']),
            'nationality': 'DZA'
        }
    
    def randomize_scene(self):
        """Randomize camera, lighting, background"""
        # Camera angle: -45° to +45°
        camera = bpy.data.objects['Camera']
        camera.rotation_euler = (
            random.uniform(-0.78, 0.78),  # Pitch
            random.uniform(-0.52, 0.52),  # Yaw
            random.uniform(-0.26, 0.26)   # Roll
        )
        
        # Lighting variation
        light = bpy.data.objects['Light']
        light.data.energy = random.uniform(2, 10)
        light.data.color = random.choice([
            (1, 1, 1),      # White
            (1, 0.95, 0.8), # Warm
            (0.9, 0.95, 1)  # Cool
        ])
        
        # Background
        world = bpy.data.worlds['World']
        # Switch between indoor/outdoor HDRIs
    
    def render_sample(self, identity, sample_id):
        """Render one sample with annotations"""
        # Apply identity data to texture
        self.update_document_texture(identity)
        
        # Randomize scene
        self.randomize_scene()
        
        # Render
        bpy.context.scene.render.filepath = str(
            self.output_dir / f'{sample_id:06d}.png'
        )
        bpy.ops.render.render(write_still=True)
        
        # Save annotations
        annotations = {
            'image_id': sample_id,
            'identity': identity,
            'camera_pose': self.get_camera_pose(),
            'lighting': self.get_lighting_params(),
            'bounding_boxes': self.get_field_bounding_boxes()
        }
        
        with open(self.output_dir / f'{sample_id:06d}.json', 'w') as f:
            json.dump(annotations, f, indent=2)
    
    def generate_dataset(self, num_samples=10000):
        """Generate full dataset"""
        for i in range(num_samples):
            identity = self.generate_synthetic_identity()
            self.render_sample(identity, i)
            
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples} samples")
```

#### Step 2.3: Annotation Schema
```json
{
  "annotation_format": "COCO-like",
  "fields": {
    "document_type": "passport|cni_front|cni_back|carte_grise",
    "image_size": [1920, 1080],
    "fields": [
      {
        "label": "surname",
        "bbox": [x, y, width, height],
        "text": "BENALI",
        "confidence": 1.0
      },
      {
        "label": "mrz_line1",
        "bbox": [x, y, width, height],
        "text": "P<DZABENALI<<MOHAMED<<<<<<<<<<<<<<",
        "confidence": 1.0
      }
    ],
    "document_boundary": {
      "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  }
}
```

### Phase 3: Model Development (Weeks 5-10)

#### Step 3.1: Document Classification Model
```python
# src/classification/model.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = efficientnet_b3(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Classes: passport, cnie_front, cnie_back, carte_grise_front, carte_grise_back
```

#### Step 3.2: Text Detection Model
```python
# src/detection/db_model.py

# Use DB (Differentiable Binarization) from PaddleOCR
# Or implement custom with PyTorch

class DBTextDetector:
    """Text detection using Differentiable Binarization"""
    
    def __init__(self, model_path=None):
        self.model = self.load_model(model_path)
    
    def detect(self, image):
        """Return bounding boxes for text regions"""
        # Preprocess
        # Inference
        # Post-process (DB thresholding)
        # Return: list of polygons
        pass
```

#### Step 3.3: OCR Pipeline
```python
# src/ocr/pipeline.py

from paddleocr import PaddleOCR
import pytesseract

class AlgerianIDOCR:
    def __init__(self):
        # PaddleOCR for general text
        self.paddle = PaddleOCR(
            use_angle_cls=True,
            lang='fr',  # French + Arabic support
            use_gpu=False  # CPU-only for now
        )
        
        # Tesseract for MRZ (OCR-B font)
        self.mrz_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    
    def extract_mrz(self, image, bbox):
        """Extract MRZ with specialized OCR-B handling"""
        mrz_region = image.crop(bbox)
        text = pytesseract.image_to_string(mrz_region, config=self.mrz_config)
        return self.correct_mrz_errors(text)
    
    def correct_mrz_errors(self, text):
        """Correct common OCR errors in MRZ"""
        corrections = {
            '0': 'O',  # In certain positions
            '1': 'I',  # In certain positions
            '8': 'B',  # Common misread
        }
        # Apply position-aware corrections
        return corrected_text
```

#### Step 3.4: Field Extraction (LayoutLMv3)
```python
# src/extraction/layoutlm.py

from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

class FieldExtractor:
    def __init__(self, model_path=None):
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_path or "microsoft/layoutlmv3-base",
            num_labels=len(self.label_list)
        )
        
        self.label_list = [
            'O', 'B-SURNAME', 'I-SURNAME', 'B-GIVEN_NAME', 'I-GIVEN_NAME',
            'B-DOB', 'I-DOB', 'B-DOC_NUMBER', 'I-DOC_NUMBER',
            'B-EXPIRY', 'I-EXPIRY', 'B-NATIONALITY', 'I-NATIONALITY',
            'B-MRZ', 'I-MRZ'
        ]
    
    def extract(self, image, ocr_results):
        """Extract structured fields from OCR text"""
        # Prepare inputs: image + text + bounding boxes
        encoding = self.processor(
            image,
            ocr_results['text'],
            boxes=ocr_results['boxes'],
            return_tensors="pt"
        )
        
        # Inference
        outputs = self.model(**encoding)
        predictions = outputs.logits.argmax(-1)
        
        # Convert predictions to fields
        return self.decode_predictions(predictions, ocr_results)
```

### Phase 4: API Development (Weeks 11-12)

```python
# src/api/main.py

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import numpy as np

app = FastAPI(title="Retin-Verify IDP API")

# Load models (singleton pattern)
classifier = DocumentClassifier()
detector = DBTextDetector()
ocr = AlgerianIDOCR()
extractor = FieldExtractor()

class ExtractionResponse(BaseModel):
    document_type: str
    confidence: float
    extracted_fields: dict
    validation_status: dict
    processing_time_ms: float

@app.post("/v1/extract", response_model=ExtractionResponse)
async def extract_document(
    file: UploadFile = File(...),
    include_face_match: bool = False
):
    """Extract data from uploaded document image"""
    start_time = time.time()
    
    # Read image
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # Pipeline
    doc_type, conf = classifier.predict(image)
    text_regions = detector.detect(image)
    ocr_results = ocr.extract(text_regions)
    fields = extractor.extract(image, ocr_results)
    validation = validate_fields(fields, doc_type)
    
    processing_time = (time.time() - start_time) * 1000
    
    return ExtractionResponse(
        document_type=doc_type,
        confidence=conf,
        extracted_fields=fields,
        validation_status=validation,
        processing_time_ms=processing_time
    )

@app.get("/v1/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}
```

### Phase 5: Testing & Validation (Weeks 13-14)

```python
# tests/test_pipeline.py

import unittest
from src.pipeline import IDPPipeline

class TestIDPPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = IDPPipeline()
    
    def test_passport_classification(self):
        result = self.pipeline.process('tests/data/passport_sample.png')
        self.assertEqual(result['document_type'], 'passport')
        self.assertGreater(result['confidence'], 0.95)
    
    def test_mrz_extraction(self):
        result = self.pipeline.process('tests/data/passport_sample.png')
        self.assertIn('mrz_line1', result['fields'])
        self.assertIn('mrz_line2', result['fields'])
        # Verify checksum
        self.assertTrue(validate_mrz_checksum(result['fields']))
    
    def test_cnie_field_extraction(self):
        result = self.pipeline.process('tests/data/cnie_sample.png')
        self.assertIn('surname', result['fields'])
        self.assertIn('given_names', result['fields'])
        self.assertIn('id_number', result['fields'])
```

---

## 7. RESOURCE REQUIREMENTS SUMMARY

### Hardware Requirements

| Component | Minimum | Recommended | Current Status |
|-----------|---------|-------------|----------------|
| **Disk Space** | 50 GB | 100+ GB | ❌ 23 GB (Insufficient) |
| **RAM** | 8 GB | 16+ GB | ⚠️ 16 GB (Marginal) |
| **GPU** | Not required | NVIDIA GTX 1660+ | ❌ Not available |
| **CPU** | 4 cores | 8+ cores | ? Unknown |

### Software Requirements

| Component | Version | Size |
|-----------|---------|------|
| Python | 3.8+ | - |
| PyTorch (CPU) | Latest | ~2 GB |
| PaddleOCR | Latest | ~500 MB |
| Transformers | Latest | ~2 GB |
| Blender | 3.0+ | ~500 MB |
| Tesseract | 4.1+ | ~100 MB |
| **Total Environment** | - | **~8-10 GB** |

---

## 8. RISK ASSESSMENT

### Critical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Insufficient Disk Space** | High | Critical | Free up space immediately or add external storage |
| **No GPU for Training** | Certain | High | Use cloud GPU (Google Colab, AWS) or CPU with smaller models |
| **Memory Constraints** | Medium | Medium | Use batch processing, smaller batch sizes |
| **Synthetic-Real Domain Gap** | Medium | Medium | Collect real validation data, use domain adaptation |
| **Legal Compliance (ANPDP)** | Medium | High | Consult legal, obtain authorization before real data collection |

---

## 9. RECOMMENDATIONS

### Immediate Actions Required

1. **⚠️ FREE UP DISK SPACE (Priority 1)**
   ```bash
   # Target: At least 50 GB free
   # Current: 23 GB
   # Need: 27+ GB more
   ```

2. **Set Up Cloud GPU Alternative**
   - Google Colab (free tier available)
   - AWS EC2 g4dn.xlarge (~$0.50/hour)
   - Run training jobs in cloud, keep inference local

3. **Start with Synthetic Data Only**
   - Zero legal risk
   - Perfect annotations
   - Can generate unlimited training data

### Recommended Project Phases

| Phase | Duration | Deliverables | Disk Req |
|-------|----------|--------------|----------|
| **1. Setup** | 1 week | Environment, project structure | 10 GB |
| **2. Synthetic Pipeline** | 3 weeks | 10K+ synthetic images, Blender scripts | 15 GB |
| **3. Model Dev (CPU)** | 6 weeks | Classification + OCR models | 20 GB |
| **4. API Development** | 2 weeks | REST API, documentation | 5 GB |
| **5. Testing** | 2 weeks | Test suite, validation | 5 GB |
| **TOTAL** | **14 weeks** | MVP IDP System | **~55 GB** |

---

## 10. CONCLUSION

### Can the project start?

**⚠️ CONDITIONAL - Disk space issue must be resolved first**

| Aspect | Status |
|--------|--------|
| **Disk Space** | ❌ **BLOCKER** - Need 50+ GB, have 23 GB |
| **Memory** | ✅ Sufficient for inference, marginal for training |
| **GPU** | ⚠️ Not available, can use cloud or CPU-only |
| **Specifications** | ✅ Clear and comprehensive |

### Next Steps

1. **Immediate**: Free up 27+ GB disk space
2. **Day 1-7**: Set up development environment
3. **Week 2-4**: Build synthetic data pipeline
4. **Week 5+**: Begin model development

**With disk space resolved, this project is feasible on current hardware using CPU-only training with smaller models, or cloud GPU for larger models.**

---

*Assessment generated: March 2025*
*Project: Retin-Verify - Algerian ID Document Processing System*
