# RetinVerify

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-Proprietary-orange)

**AI-Powered Identity Document Verification System**

RetinVerify is a comprehensive machine learning pipeline for automated identity document classification, detection, and information extraction. Built with PyTorch and optimized for CNIE (Carte Nationale d'Identité Electronique) documents.

---

## 🎯 Features

### Classification
- **3-Class Classification**: Front, Back, No Card detection
- **Real-time Inference**: Optimized for production deployment
- **Feedback Loop**: Collect and retrain on misclassified samples
- **Synthetic Data Support**: Train with 16K+ synthetic samples

### Detection (Planned)
- Text region detection
- Field localization
- Photo extraction

### Extraction (Planned)
- MRZ (Machine Readable Zone) parsing
- Field extraction (name, ID, date)
- OCR integration

---

## 📁 Project Structure

```
retin-verify/
├── apps/                    # Production runtime applications
│   └── classification/      # Classification API + Web UI
├── inference/               # Standalone inference apps
│   └── apps/
├── training/                # Training scripts & configs
│   ├── classification/
│   ├── detection/
│   └── extraction/
├── models/                  # Trained model artifacts
│   ├── classification/
│   ├── detection/
│   └── extraction/
├── data/                    # Data storage
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── feedback/
├── docs/                    # Documentation
│   ├── classification/
│   ├── guides/
│   └── deployment/
├── src/                     # Reusable source code
├── tests/                   # Test suites
└── synthetic/               # Synthetic data generation
```

**📖 See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for complete details.**

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/retinrix/retin-verify.git
cd retin-verify

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r apps/classification/requirements.txt
```

### Running the Classification API

```bash
cd apps/classification

# Start server
./start_server.sh

# Or manually
python backend/api_server.py
```

API will be available at `http://localhost:8000`

---

## 📊 Classification Usage

### Web Interface

Open `http://localhost:8000` in your browser for the 4-panel UI:

1. **Camera Capture** - Start camera, capture image
2. **Classification Results** - View prediction with confidence
3. **Model Info** - Current model status
4. **Recent Captures** - History with feedback option

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Predict
# POST /predict with image file
curl -X POST -F "file=@card.jpg" http://localhost:8000/predict

# Response
{
  "predicted_class": "cnie_front",
  "confidence": 0.92,
  "all_scores": {
    "cnie_front": 0.92,
    "cnie_back": 0.05,
    "no_card": 0.03
  },
  "inference_time_ms": 45
}
```

---

## 🏋️ Training

### With Synthetic + Real Data (Recommended)

```bash
cd training/classification

# Deploy to Colab
python colab_retrain/new_training/deploy_v3_with_synthetic.py

# Monitor training
ssh root@your-colab-host "tail -f /content/retin_v3_synthetic/train_v3_synthetic.log"

# Download model
scp root@your-colab-host:/content/cnie_classifier_3class_v3_synthetic.pth \
    ../../models/classification/
```

**📖 See [docs/classification/V3_SYNTHETIC_INTEGRATION.md](docs/classification/V3_SYNTHETIC_INTEGRATION.md)**

### Local Training

```bash
cd training/classification/scripts
python train_from_scratch.py
```

---

## 📚 Documentation

| Topic | Location |
|-------|----------|
| Project Structure | [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) |
| Organization Rules | [.github/ORGANIZATION_RULES.md](.github/ORGANIZATION_RULES.md) |
| Classification v3 Training | [docs/classification/V3_SYNTHETIC_INTEGRATION.md](docs/classification/V3_SYNTHETIC_INTEGRATION.md) |
| Data Collection | [docs/classification/DATA_COLLECTION_GUIDE.md](docs/classification/DATA_COLLECTION_GUIDE.md) |
| Deployment Guides | [docs/deployment/](docs/deployment/) |
| Workflow Guides | [docs/guides/](docs/guides/) |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/unit/
pytest tests/inference/
pytest tests/integration/
```

---

## 🔧 Configuration

Configuration files are in `/configs/`:

```yaml
# configs/classification/efficientnet_b0.yaml
model:
  name: "efficientnet_b0"
  num_classes: 3
  
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

---

## 📦 Models

Current models available in `/models/classification/`:

| Model | Classes | Accuracy | Status |
|-------|---------|----------|--------|
| cnie_classifier_3class_v2.pth | 3 | 88.5% | Production |
| cnie_front_back_real.pth | 2 | 90% | Backup |

---

## 🗺️ Roadmap

### Phase 1: Classification ✅
- [x] 3-class classifier (front/back/no_card)
- [x] Web UI with camera capture
- [x] Feedback collection system
- [x] Synthetic data integration

### Phase 2: Detection 🚧
- [ ] Text region detection
- [ ] Field localization
- [ ] Photo extraction

### Phase 3: Extraction 📋
- [ ] MRZ parsing
- [ ] Field extraction
- [ ] OCR integration

### Phase 4: Production 🚀
- [ ] Model optimization (ONNX, TensorRT)
- [ ] Edge deployment
- [ ] Mobile SDK

---

## 🤝 Contributing

1. Follow [ORGANIZATION_RULES.md](.github/ORGANIZATION_RULES.md)
2. Create feature branch: `git checkout -b feature/name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/name`
5. Submit pull request

---

## 📝 License

Proprietary - All rights reserved.

---

## 🙏 Acknowledgments

- PyTorch team for the excellent ML framework
- EfficientNet paper authors
- Synthetic data generation pipeline

---

## 📞 Support

For questions or issues:
- Check [docs/](docs/) first
- Review [ORGANIZATION_RULES.md](.github/ORGANIZATION_RULES.md)
- Contact team lead

---

**Built with ❤️ for secure identity verification.**
