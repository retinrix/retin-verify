# RetinVerify Project Structure

This document describes the standardized project organization for the RetinVerify identity document verification system.

## 📁 Directory Structure

```
retin-verify/
├── .github/                    # GitHub configuration
│   ├── workflows/              # CI/CD workflows
│   └── ORGANIZATION_RULES.md   # Project organization standards
├── .vscode/                    # VS Code settings
├── apps/                       # Production runtime applications
│   └── classification/         # Classification app (frontend + backend)
├── configs/                    # Configuration files
│   ├── classification/
│   ├── detection/
│   └── extraction/
├── data/                       # Data storage
│   ├── raw/                    # Raw input data
│   ├── processed/              # Processed datasets
│   │   └── classification/
│   ├── synthetic/              # Synthetic generated data
│   ├── annotations/            # Manual annotations
│   └── feedback/               # User feedback data
│       └── classification/
├── docs/                       # Documentation
│   ├── classification/         # Classification-specific docs
│   ├── detection/              # Detection-specific docs
│   ├── extraction/             # Extraction-specific docs
│   ├── ocr/                    # OCR-specific docs
│   ├── synthetic/              # Synthetic data docs
│   ├── deployment/             # Deployment guides
│   ├── api/                    # API documentation
│   └── guides/                 # General guides & workflows
├── inference/                  # Inference runtime
│   ├── apps/                   # Standalone inference apps
│   │   ├── classification/
│   │   ├── detection/
│   │   ├── extraction/
│   │   └── ocr/
│   └── optimizers/             # Inference optimizers
├── models/                     # Trained models
│   ├── classification/         # Classification models
│   ├── detection/              # Detection models
│   ├── extraction/             # Extraction models
│   └── archive/                # Archived old models
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Utility scripts
│   ├── setup/                  # Setup scripts
│   ├── deployment/             # Deployment scripts
│   └── maintenance/            # Maintenance scripts
├── src/                        # Source code
│   ├── api/
│   ├── classification/
│   ├── detection/
│   ├── extraction/
│   ├── ocr/
│   ├── preprocessing/
│   └── validation/
├── synthetic/                  # Synthetic data generation
│   ├── backgrounds/
│   ├── configs/
│   ├── fonts/
│   ├── output/
│   ├── scenes/
│   ├── scripts/
│   └── templates/
├── tests/                      # Test suites
│   ├── data/
│   ├── inference/
│   ├── integration/
│   ├── training/
│   └── unit/
└── training/                   # Training scripts & configs
    ├── classification/
    │   ├── configs/
    │   └── scripts/
    ├── detection/
    │   ├── configs/
    │   └── scripts/
    └── extraction/
        ├── configs/
        └── scripts/
```

## 📂 Directory Purposes

### `/apps/`
Production runtime applications. Contains only what's needed to run the application.

**Rule:** Only runtime files (no training scripts, no archived data).

### `/training/`
Training scripts, configurations, and utilities for model training.

**Subdirectories:**
- `classification/` - Card classification training
- `detection/` - Text detection training
- `extraction/` - Information extraction training

### `/inference/`
Standalone inference applications for deployment.

**Subdirectories:**
- `apps/` - Self-contained inference apps per module
- `optimizers/` - Model optimization tools (ONNX, TensorRT, etc.)

### `/data/`
All data organized by processing stage.

**Subdirectories:**
- `raw/` - Original unprocessed data
- `processed/` - Cleaned, split datasets
- `synthetic/` - Generated synthetic data
- `feedback/` - Collected user feedback
- `annotations/` - Manual annotations

### `/models/`
Trained model artifacts organized by task.

**Naming Convention:**
```
{task}_{version}_{date}.pth
Example: cnie_classifier_3class_v3_20260319.pth
```

### `/docs/`
Documentation organized by subject area.

**Subdirectories:**
- `classification/` - Classification-specific docs
- `detection/` - Detection-specific docs
- `extraction/` - Extraction-specific docs
- `synthetic/` - Synthetic data generation docs
- `deployment/` - Deployment guides
- `guides/` - General workflow guides

### `/src/`
Reusable source code modules.

### `/configs/`
YAML/JSON configuration files for models and pipelines.

### `/scripts/`
Utility scripts for setup, deployment, and maintenance.

### `/tests/`
Test suites organized by type.

### `/synthetic/`
Synthetic data generation tools and resources.

## 📝 File Naming Conventions

### Python Scripts
- `train_{task}.py` - Training scripts
- `inference_{task}.py` - Inference scripts
- `evaluate_{task}.py` - Evaluation scripts
- `utils_{module}.py` - Utility modules

### Models
- `{task}_{description}_v{version}_{YYYYMMDD}.pth`
- Example: `cnie_classifier_3class_v3_20260319.pth`

### Documentation
- `README_{topic}.md` - Topic-specific README
- `GUIDE_{workflow}.md` - Workflow guide
- `API_{module}.md` - API documentation

### Data
- `{timestamp}_{uuid}.jpg` - Image files
- `{split}_{class}_{index}.json` - Annotation files

## 🔗 Symbolic Links

The following symlinks maintain backward compatibility:

```bash
apps/classification/dataset_3class -> ../../data/processed/classification/dataset_3class
apps/classification/feedback_data -> ../../data/feedback/classification/feedback_data
apps/classification/feedback_data_3class -> ../../data/feedback/classification/feedback_3class
```

## 🧹 Cleanup Rules

1. **Archive Old Models:** Move models older than 30 days to `/models/archive/`
2. **Clean Temporary Files:** Remove `.pyc`, `__pycache__`, temp files
3. **Organize Feedback:** Move processed feedback to dated subfolders
4. **Archive Old Docs:** Move superseded documentation to archive

## 🚀 Quick Reference

| Task | Location |
|------|----------|
| Start classification API | `apps/classification/` or `inference/apps/classification/` |
| Train new model | `training/classification/scripts/` |
| Add training data | `data/raw/` then process to `data/processed/` |
| View documentation | `docs/` organized by subject |
| Find models | `models/{task}/` |
| Run tests | `tests/` |
