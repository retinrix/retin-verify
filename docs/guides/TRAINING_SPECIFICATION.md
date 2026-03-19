# Retin-Verify Training & Inference Specification

## Executive Summary

This document provides comprehensive specifications for training the Retin-Verify (retin-verif) ID document processing models and deploying them for inference. It covers machine resources, training time estimates, recommended configurations (including Google Colab), updated folder structures, and testing strategies.

---

## 1. Machine Resources & Requirements

### 1.1 Current Local Environment

| Component | Specification | Status |
|-----------|--------------|--------|
| **CPU** | Intel Core i7-6700HQ @ 2.60GHz (4 cores/8 threads) | ✅ Adequate |
| **RAM** | 8 GB DDR4 | ⚠️ Minimal (16GB recommended) |
| **GPU** | NVIDIA GeForce GTX 950M (2GB VRAM) | ⚠️ Limited for LayoutLMv3 |
| **CUDA** | Version 13.0 (Driver 582.28) | ✅ Supported |
| **Storage** | ~50GB free space needed | ⚠️ Check available |
| **OS** | Linux (Ubuntu-based) | ✅ Ideal |

### 1.2 Resource Requirements by Model

| Model | Min VRAM | Recommended VRAM | Min RAM | Training Time (Local) |
|-------|----------|------------------|---------|----------------------|
| **Classification (EfficientNet-B0)** | 2GB | 4GB | 8GB | 2-4 hours |
| **Text Detection (DB/MobileNetV3)** | 4GB | 8GB | 16GB | 8-12 hours |
| **Field Extraction (LayoutLMv3-base)** | 6GB | 12GB | 16GB | 12-24 hours |
| **Full Pipeline Training** | 12GB+ | 16GB+ | 32GB | 24-48 hours |

### 1.3 Google Colab Comparison

| Specification | Local Machine | Google Colab Free | Google Colab Pro | Google Colab Pro+ |
|---------------|---------------|-------------------|------------------|-------------------|
| **GPU** | GTX 950M (2GB) | T4/K80 (12-16GB) | T4/V100 (16GB) | A100 (40GB) |
| **RAM** | 8GB | 12GB | 32GB | 52GB |
| **Disk** | Variable | ~78GB | ~166GB | ~166GB |
| **Runtime** | Unlimited | 12 hours limit | 24 hours limit | 24 hours limit |
| **Cost** | $0 | $0 | $10/month | $50/month |
| **Suitability** | ⚠️ Limited | ✅ Recommended | ✅✅ Best Value | ✅✅✅ Enterprise |

---

## 2. Training Time Estimates

### 2.1 Estimated Training Times (CNIE Dataset 10K)

Dataset split: 8,025 pairs (~16,050 images)
- **Train**: ~6,420 pairs (80%)
- **Validation**: ~800 pairs (10%)
- **Test**: ~805 pairs (10%)

| Model | Local (GTX 950M) | Colab Free (T4) | Colab Pro (V100) | Epochs |
|-------|------------------|-----------------|------------------|--------|
| **Classification** | 3-5 hours | 30-45 min | 15-25 min | 50 |
| **Text Detection** | 12-18 hours* | 2-4 hours | 1-2 hours | 100 |
| **Field Extraction** | 24-36 hours* | 4-6 hours | 2-3 hours | 20 |
| **Total** | 2-3 days* | 6-8 hours | 3-4 hours | - |

*Local training may require gradient accumulation and smaller batch sizes, increasing time.

### 2.2 Memory-Saving Techniques for Local Training

```python
# For Limited VRAM (2-4GB)

# 1. Gradient Accumulation
gradient_accumulation_steps = 4  # Effective batch_size = 4 * actual

# 2. Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 3. Smaller Batch Sizes
batch_size_classification = 8   # Instead of 32
batch_size_extraction = 1       # Instead of 4

# 4. Gradient Checkpointing
model.gradient_checkpointing_enable()

# 5. Freeze Lower Layers (for LayoutLMv3)
for param in model.layoutlmv3.encoder.layer[:6].parameters():
    param.requires_grad = False
```

---

## 3. Recommended Configuration

### 3.1 Decision Matrix

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| **Quick Prototyping** | Google Colab Free | No setup, fast iteration |
| **Full Training** | Google Colab Pro | Faster, longer runtime, reliable |
| **Production Training** | Google Colab Pro+ or Cloud VM (AWS/GCP) | Maximum speed, persistent storage |
| **Edge Deployment** | Local Training + Quantization | Test on target hardware |
| **Continuous Training** | Local Machine (overnight) | No runtime limits, incremental updates |

### 3.2 Recommended: Google Colab Pro

**Why Colab Pro?**
- V100 GPU is ~10x faster than GTX 950M
- 32GB RAM handles full batch sizes
- Can train all models in a single session
- Persistent Google Drive integration
- $10/month is cost-effective vs. local hardware upgrade

**Setup Instructions:**
```python
# Colab Notebook Setup Cell
!pip install torch torchvision transformers datasets paddleocr pytesseract
!apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone <your-repo-url> /content/retin-verify
%cd /content/retin-verify
```

---

## 4. Updated Training Folder Structure

### 4.1 Proposed New Structure

```
retin-verify/
├── 📁 data/
│   ├── cnie_dataset_10k/           # Original dataset
│   ├── processed/                  # Preprocessed data (NEW)
│   │   ├── classification/         # Ready for classification training
│   │   ├── detection/              # Ready for detection training
│   │   └── extraction/             # LayoutLMv3 formatted data
│   └── splits/                     # Train/val/test splits manifest
│
├── 📁 training/                    # NEW: Training workspace
│   ├── 📁 classification/
│   │   ├── train.py                # Training script
│   │   ├── configs/
│   │   │   └── efficientnet_b0.yaml
│   │   ├── logs/                   # TensorBoard logs
│   │   └── checkpoints/            # Model checkpoints
│   │
│   ├── 📁 detection/
│   │   ├── train.py
│   │   ├── configs/
│   │   │   └── db_mobilenetv3.yaml
│   │   ├── logs/
│   │   └── checkpoints/
│   │
│   ├── 📁 extraction/
│   │   ├── train.py
│   │   ├── configs/
│   │   │   └── layoutlmv3_base.yaml
│   │   ├── logs/
│   │   └── checkpoints/
│   │
│   └── 📁 utils/                   # Shared training utilities
│       ├── data_loaders.py
│       ├── metrics.py
│       ├── callbacks.py
│       └── visualization.py
│
├── 📁 models/                      # Production models
│   ├── classification/
│   │   └── efficientnet_b0_final.pth
│   ├── detection/
│   │   └── db_mobilenetv3_final.pth
│   ├── extraction/
│   │   └── layoutlmv3_base_final/
│   └── exported/                   # ONNX, OpenVINO formats
│       ├── classification.onnx
│       ├── detection.onnx
│       └── extraction.onnx
│
├── 📁 inference/                   # NEW: Inference engine
│   ├── __init__.py
│   ├── pipeline.py                 # Optimized inference pipeline
│   ├── batch_processor.py          # Batch processing
│   ├── model_manager.py            # Model loading/caching
│   └── optimizers/
│       ├── onnx_runtime.py
│       ├── openvino_runtime.py
│       └── tensorrt_runtime.py
│
├── 📁 tests/                       # Expanded tests
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── training/                   # Training tests (NEW)
│   ├── inference/                  # Inference tests (NEW)
│   └── data/                       # Test fixtures
│
└── 📁 notebooks/                   # NEW: Jupyter notebooks
    ├── 01_data_exploration.ipynb
    ├── 02_classification_training.ipynb
    ├── 03_detection_training.ipynb
    ├── 04_extraction_training.ipynb
    └── 05_inference_demo.ipynb
```

### 4.2 Migration from Current Structure

```bash
# Create new directories
mkdir -p training/{classification,detection,extraction,utils}/{configs,logs,checkpoints}
mkdir -p models/{classification,detection,extraction,exported}
mkdir -p inference/optimizers
mkdir -p tests/{unit,integration,training,inference,data}
mkdir -p notebooks

# Move existing training scripts
mv src/classification/train_classifier.py training/classification/train.py
mv src/detection/train_detector.py training/detection/train.py
mv src/extraction/train_extractor.py training/extraction/train.py

# Create symlinks for backward compatibility
ln -s ../training/classification/train.py src/classification/train_classifier.py
ln -s ../training/detection/train.py src/detection/train_detector.py
ln -s ../training/extraction/train.py src/extraction/train_extractor.py
```

---

## 5. Inference Architecture

### 5.1 Inference Pipeline Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                      INFERENCE PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Input   │───▶│ Document │───▶│  Text    │───▶│   OCR    │      │
│  │  Image   │    │ Classify │    │ Detect   │    │  Engine  │      │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘      │
│                                                       │              │
│                                                       ▼              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Output  │◄───│ Validate │◄───│  Field   │◄───│  Text    │      │
│  │  JSON    │    │   MRZ    │    │ Extract  │    │  Tokens  │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Model Formats for Different Deployments

| Deployment Target | Format | Latency | Setup Complexity |
|-------------------|--------|---------|------------------|
| **Development** | PyTorch (.pth) | Baseline | Low |
| **API Server (GPU)** | PyTorch + CUDA | 100ms | Low |
| **API Server (CPU)** | ONNX Runtime | 200ms | Medium |
| **Edge/Embedded** | OpenVINO (INT8) | 500ms | High |
| **Mobile** | ONNX Mobile / TFLite | 1s+ | High |

### 5.3 Inference Optimization Strategy

```python
# inference/model_manager.py

class ModelManager:
    """Manages model loading, caching, and optimization."""
    
    def __init__(self, optimization_level: str = "balanced"):
        """
        Args:
            optimization_level: "speed", "balanced", or "memory"
        """
        self.cache = {}
        self.optimization_level = optimization_level
    
    def load_model(self, model_type: str, path: Path, device: str = "auto"):
        """Load model with optimizations."""
        
        cache_key = f"{model_type}_{path}_{device}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load based on optimization level
        if self.optimization_level == "speed":
            model = self._load_optimized(path, "tensorrt" if device == "cuda" else "openvino")
        elif self.optimization_level == "memory":
            model = self._load_quantized(path, "int8")
        else:  # balanced
            model = self._load_onnx(path)
        
        self.cache[cache_key] = model
        return model
    
    def _load_onnx(self, path: Path):
        import onnxruntime as ort
        session = ort.InferenceSession(
            str(path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        return session
```

---

## 6. Testing Strategy

### 6.1 Test Categories

| Test Type | Purpose | Location | Frequency |
|-----------|---------|----------|-----------|
| **Unit Tests** | Test individual functions | `tests/unit/` | Every commit |
| **Integration Tests** | Test component interactions | `tests/integration/` | Every PR |
| **Training Tests** | Validate training pipeline | `tests/training/` | Before training |
| **Inference Tests** | Validate inference pipeline | `tests/inference/` | Before deployment |
| **Performance Tests** | Benchmark latency/accuracy | `tests/performance/` | Weekly |

### 6.2 Training Test Suite

```python
# tests/training/test_classification_training.py

import pytest
import torch
from training.classification.train import ClassificationTrainer

class TestClassificationTraining:
    """Tests for classification training pipeline."""
    
    def test_data_loading(self):
        """Test that data loads correctly."""
        # Verify dataset loads without errors
        pass
    
    def test_model_creation(self):
        """Test model initialization."""
        trainer = ClassificationTrainer(num_classes=4)
        assert trainer.model is not None
    
    def test_single_batch_training(self):
        """Test training on single batch (smoke test)."""
        # Run one forward/backward pass
        pass
    
    def test_checkpoint_save_load(self):
        """Test checkpoint persistence."""
        # Save and load checkpoint
        pass
    
    def test_convergence(self):
        """Test model converges on toy dataset."""
        # Train on 10 samples, verify loss decreases
        pass
```

### 6.3 Inference Test Suite

```python
# tests/inference/test_pipeline.py

import pytest
from inference.pipeline import InferencePipeline

class TestInferencePipeline:
    """Tests for inference pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        return InferencePipeline(
            optimization_level="balanced",
            device="cpu"
        )
    
    def test_document_classification(self, pipeline):
        """Test document type classification."""
        result = pipeline.classify("test_cnie_front.jpg")
        assert result["document_type"] in ["cnie_front", "cnie_back", "passport"]
        assert result["confidence"] > 0.5
    
    def test_field_extraction(self, pipeline):
        """Test field extraction accuracy."""
        result = pipeline.process("test_cnie_front.jpg")
        assert "surname" in result["fields"]
        assert "given_names" in result["fields"]
    
    def test_mrz_validation(self, pipeline):
        """Test MRZ extraction and validation."""
        result = pipeline.process("test_cnie_back.jpg")
        assert result["mrz"]["valid"] is True
    
    def test_latency_requirements(self, pipeline):
        """Test inference meets latency targets."""
        import time
        start = time.time()
        pipeline.process("test_cnie_front.jpg")
        latency = (time.time() - start) * 1000
        assert latency < 3000  # 3 seconds max
```

### 6.4 Continuous Integration Workflow

```yaml
# .github/workflows/train_and_test.yml

name: Train and Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest tests/unit/
  
  training-smoke-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Training smoke tests
        run: pytest tests/training/ -k "smoke"
  
  inference-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Download test models
        run: ./scripts/download_test_models.sh
      - name: Run inference tests
        run: pytest tests/inference/
```

---

## 7. Training Configuration Files

### 7.1 Classification Config

```yaml
# training/classification/configs/efficientnet_b0.yaml

model:
  name: "efficientnet_b0"
  num_classes: 4
  pretrained: true
  dropout: 0.3

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 5
  
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  input_size: 224
  augment: true
  
augmentation:
  random_crop: true
  random_flip: true
  random_rotation: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.1
    hue: 0.05

checkpointing:
  save_every: 10
  keep_best: true
  keep_last: true
  output_dir: "training/classification/checkpoints"

logging:
  use_tensorboard: true
  log_every: 10
  log_dir: "training/classification/logs"
```

### 7.2 Extraction Config

```yaml
# training/extraction/configs/layoutlmv3_base.yaml

model:
  name: "microsoft/layoutlmv3-base"
  num_labels: 25  # BIO labels
  max_seq_length: 512
  
training:
  batch_size: 4
  gradient_accumulation_steps: 2  # Effective batch size = 8
  epochs: 20
  learning_rate: 0.00005
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Memory optimization
  fp16: true
  gradient_checkpointing: true
  
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  max_seq_length: 512
  
checkpointing:
  save_every: 500
  save_total_limit: 3
  output_dir: "training/extraction/checkpoints"
  
logging:
  use_tensorboard: true
  log_every: 10
  eval_every: 100
  log_dir: "training/extraction/logs"
```

---

## 8. Quick Start Commands

### 8.1 Data Preparation

```bash
# 1. Prepare dataset splits
python scripts/prepare_dataset.py \
    --data-dir data/cnie_dataset_10k \
    --output-dir data/processed \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1

# 2. Prepare LayoutLMv3 format data
python scripts/prepare_layoutlmv3_data.py \
    --annotations data/cnie_dataset_10k/exports/annotations_coco.json \
    --output data/processed/extraction
```

### 8.2 Training Commands

```bash
# Classification Training
python training/classification/train.py \
    --config training/classification/configs/efficientnet_b0.yaml \
    --data-dir data/cnie_dataset_10k \
    --output-dir models/classification

# Detection Training (using PaddleOCR tools)
python training/detection/train.py \
    --config training/detection/configs/db_mobilenetv3.yaml \
    --train-dir data/processed/detection/train \
    --val-dir data/processed/detection/val

# Extraction Training
python training/extraction/train.py \
    --config training/extraction/configs/layoutlmv3_base.yaml \
    --train-file data/processed/extraction/train.json \
    --val-file data/processed/extraction/val.json \
    --output-dir models/extraction
```

### 8.3 Inference Commands

```bash
# Single document
python inference/pipeline.py \
    --image path/to/document.jpg \
    --model-dir models \
    --output result.json

# Batch processing
python inference/batch_processor.py \
    --input-dir path/to/documents/ \
    --output-dir path/to/results/ \
    --model-dir models \
    --workers 4

# Export to ONNX
python scripts/export_models.py \
    --input-dir models \
    --output-dir models/exported \
    --format onnx
```

---

## 9. Monitoring & Logging

### 9.1 TensorBoard Setup

```bash
# Start TensorBoard
tensorboard --logdir training/

# View at: http://localhost:6006
```

### 9.2 Metrics to Track

| Model | Key Metrics | Target Values |
|-------|-------------|---------------|
| **Classification** | Accuracy, F1-Score, Confusion Matrix | >99% accuracy |
| **Detection** | Precision, Recall, F1, IoU | F1 >0.90 |
| **Extraction** | Token F1, Entity F1, Exact Match | F1 >0.95 |
| **End-to-End** | Field Accuracy, Processing Time | >95% fields correct, <3s |

---

## 10. Deployment Checklist

- [ ] All models trained and validated
- [ ] Models exported to ONNX format
- [ ] Inference tests passing
- [ ] Latency requirements met
- [ ] API endpoints tested
- [ ] Docker image built
- [ ] Documentation updated
- [ ] Monitoring configured

---

## Appendix A: Google Colab Training Notebook

See `notebooks/02_classification_training.ipynb` for complete example.

```python
# Minimal Colab training example
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q transformers datasets accelerate

# Training
from training.extraction.train import FieldExtractionTrainer

trainer = FieldExtractionTrainer(
    model_dir='/content/drive/MyDrive/retin-verify/models/extraction',
    batch_size=4,
    num_epochs=20
)

trainer.train(
    train_file='/content/drive/MyDrive/retin-verify/data/processed/train.json',
    val_file='/content/drive/MyDrive/retin-verify/data/processed/val.json'
)
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-14  
**Author**: Retin-Verify Team
