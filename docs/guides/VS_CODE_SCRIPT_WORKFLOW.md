# VS Code Script-Based Workflow (No Notebooks)

## Overview

This guide shows how to run everything from VS Code terminal using Python scripts, with Kimi writing and updating them. No Jupyter notebooks required!

```
┌─────────────────────────────────────────────────────────────────┐
│                    VS CODE SCRIPT WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│  │   You    │────▶│   Kimi   │────▶│  Script  │               │
│  │ (Request)│     │ (Writes) │     │  (.py)   │               │
│  └──────────┘     └──────────┘     └────┬─────┘               │
│                                          │                       │
│                                          ▼                       │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│  │  Results │◀────│  Monitor │◀────│  Execute │               │
│  │ (Metrics)│     │ (Output) │     │ (Run)    │               │
│  └──────────┘     └──────────┘     └──────────┘               │
│                                          ▲                       │
│                                          │                       │
│                              ┌───────────┘                       │
│                              │                                   │
│                         VS Code Terminal                         │
│                      (python train.py ...)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Scripts Instead of Notebooks?

| Feature | Notebooks | Scripts |
|---------|-----------|---------|
| **Version Control** | ❌ Hard to diff | ✅ Git-friendly |
| **Reproducibility** | ❌ Cell execution order issues | ✅ Linear execution |
| **Debugging** | ⚠️ Cell-based | ✅ Standard debugger |
| **IDE Support** | ⚠️ Limited | ✅ Full VS Code features |
| **Automation** | ⚠️ Manual cell running | ✅ Automated pipelines |
| **Production** | ❌ Hard to deploy | ✅ Easy to deploy |

---

## Quick Start: Run Training from VS Code

### Step 1: Kimi Writes the Script

**You:**
```
Create a complete training script for EfficientNet-B0 classification 
that I can run from command line. Include argparse for all parameters.
```

**Kimi generates:** `training/classification/train.py`

### Step 2: You Review in VS Code

```bash
# Check what Kimi created
cat training/classification/train.py

# Run syntax check
python -m py_compile training/classification/train.py
```

### Step 3: Run Locally (CPU Test)

```bash
# Test with small batch locally
python training/classification/train.py \
    --data-dir data/cnie_dataset_10k \
    --train-annotations data/processed/classification/train.json \
    --val-annotations data/processed/classification/val.json \
    --epochs 1 \
    --batch-size 4 \
    --device cpu
```

### Step 4: Run on Colab (GPU)

Option A: Upload script to Colab
```bash
# Copy script to Colab (via GitHub)
git add training/classification/train.py
git commit -m "Add training script"
git push

# On Colab:
!git pull
!python training/classification/train.py --epochs 50 --device cuda
```

Option B: VS Code Remote SSH to Colab (see below)

---

## Kimi Creates: Script Structure

### 1. Main Training Script

```python
#!/usr/bin/env python3
"""
Training script for document classification.
Run from VS Code terminal or Colab.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train classification model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory containing dataset')
    parser.add_argument('--train-annotations', type=Path, required=True,
                        help='Training annotations JSON')
    parser.add_argument('--val-annotations', type=Path, required=True,
                        help='Validation annotations JSON')
    
    # Model arguments
    parser.add_argument('--model-dir', type=Path, default='models/classification',
                        help='Directory to save models')
    parser.add_argument('--base-model', type=str, default='efficientnet_b0',
                        help='Base model architecture')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu/cuda/auto)')
    
    # Logging arguments
    parser.add_argument('--log-dir', type=Path, default='logs/classification',
                        help='Directory for logs')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Document Classification Training")
    logger.info("=" * 60)
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)
    
    # Import here to avoid slow startup for --help
    from training.classification.trainer import ClassificationTrainer
    
    # Initialize trainer
    trainer = ClassificationTrainer(
        model_dir=args.model_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        log_dir=args.log_dir
    )
    
    # Load datasets
    from training.utils.data_loaders import ClassificationDataset
    
    train_dataset = ClassificationDataset(
        args.data_dir,
        args.train_annotations,
        transform=trainer.get_transforms(is_training=True)
    )
    
    val_dataset = ClassificationDataset(
        args.data_dir,
        args.val_annotations,
        transform=trainer.get_transforms(is_training=False)
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Train
    try:
        trainer.train(train_dataset, val_dataset)
        logger.info("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
```

### 2. Kimi Creates: Launcher Script for Colab

```python
#!/usr/bin/env python3
"""
Launcher script for running training on Google Colab from VS Code.
This script sets up the Colab environment and runs training.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def setup_colab():
    """Setup Colab environment."""
    print("🚀 Setting up Colab environment...")
    
    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Install dependencies
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                    'torch', 'torchvision', 'transformers', 'datasets'])
    
    # Setup paths
    data_path = '/content/drive/MyDrive/retin-verify/data/cnie_dataset_10k'
    if not Path(data_path).exists():
        raise RuntimeError(f"Data not found at {data_path}")
    
    # Create symlink
    if not Path('/content/data').exists():
        Path('/content/data').symlink_to(data_path)
    
    print("✅ Colab setup complete")
    return '/content/data'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup-only', action='store_true',
                        help='Only setup, do not run training')
    parser.add_argument('--train-script', type=str,
                        default='training/classification/train.py',
                        help='Path to training script')
    parser.add_argument('--train-args', type=str, default='',
                        help='Additional arguments for training script')
    
    args, unknown = parser.parse_known_args()
    
    # Setup Colab
    data_path = setup_colab()
    
    if args.setup_only:
        print("Setup complete. Exiting.")
        return
    
    # Build command
    cmd = [
        sys.executable,
        args.train_script,
        '--data-dir', data_path,
        '--device', 'cuda'
    ]
    
    # Add additional args
    if args.train_args:
        cmd.extend(args.train_args.split())
    
    cmd.extend(unknown)
    
    # Run training
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
```

---

## Workflow Examples

### Example 1: Local Development (CPU)

**You:**
```bash
# In VS Code terminal
kimi --continue
```

**You to Kimi:**
```
Create a script to prepare the dataset splits for training.
```

**Kimi writes:** `scripts/prepare_data.py`

**You run:**
```bash
# Review the script first
cat scripts/prepare_data.py

# Run it
python scripts/prepare_data.py \
    --data-dir data/cnie_dataset_10k \
    --output-dir data/processed

# Check output
ls data/processed/
```

### Example 2: Training on Colab (GPU)

**You:**
```
Create a complete training script with all best practices:
- Mixed precision training
- Gradient accumulation
- Checkpointing
- TensorBoard logging
- Early stopping
```

**Kimi writes:** `training/extraction/train_layoutlmv3.py`

**You:**
```bash
# Commit and push
git add training/extraction/train_layoutlmv3.py
git commit -m "Add LayoutLMv3 training script"
./scripts/sync_to_colab.sh
```

**On Colab (or via VS Code SSH):**
```bash
# Pull and run
!git pull
!python training/extraction/train_layoutlmv3.py \
    --train-file /content/data/processed/extraction/train.json \
    --val-file /content/data/processed/extraction/val.json \
    --epochs 20 \
    --batch-size 4 \
    --fp16
```

### Example 3: Batch Processing

**You:**
```
Create a script to evaluate all trained models and generate a comparison report.
```

**Kimi writes:** `scripts/evaluate_models.py`

**You run:**
```bash
python scripts/evaluate_models.py \
    --models-dir models/ \
    --test-data data/processed/test.json \
    --output report.html

# View report
open report.html  # macOS
# or
xdg-open report.html  # Linux
```

---

## VS Code Integration

### 1. Launch Configurations

`.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    },
    {
      "name": "Train Classification (Local Test)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/training/classification/train.py",
      "args": [
        "--data-dir", "data/cnie_dataset_10k",
        "--train-annotations", "data/processed/classification/train.json",
        "--val-annotations", "data/processed/classification/val.json",
        "--epochs", "1",
        "--batch-size", "4",
        "--device", "cpu"
      ],
      "console": "integratedTerminal"
    },
    {
      "name": "Train Classification (Colab GPU)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/run_on_colab.py",
      "args": [
        "--train-script", "training/classification/train.py",
        "--train-args", "--epochs 50 --batch-size 32"
      ],
      "console": "integratedTerminal"
    },
    {
      "name": "Prepare Dataset",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/prepare_data.py",
      "args": [
        "--data-dir", "data/cnie_dataset_10k",
        "--output-dir", "data/processed"
      ],
      "console": "integratedTerminal"
    }
  ]
}
```

### 2. VS Code Tasks

`.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Prepare Dataset",
      "type": "shell",
      "command": "python scripts/prepare_data.py --data-dir data/cnie_dataset_10k --output-dir data/processed",
      "group": "build"
    },
    {
      "label": "Train Classification (Test)",
      "type": "shell",
      "command": "python training/classification/train.py --data-dir data/cnie_dataset_10k --train-annotations data/processed/classification/train.json --val-annotations data/processed/classification/val.json --epochs 1 --batch-size 4 --device cpu",
      "group": "build"
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "pytest tests/ -v",
      "group": "test"
    },
    {
      "label": "Sync to Colab",
      "type": "shell",
      "command": "./scripts/sync_to_colab.sh",
      "group": "deploy"
    }
  ]
}
```

**Usage:**
- Press `Ctrl+Shift+P` → "Tasks: Run Task"
- Select task (e.g., "Prepare Dataset")

### 3. Keyboard Shortcuts

`.vscode/keybindings.json`:

```json
[
  {
    "key": "ctrl+shift+t",
    "command": "workbench.action.tasks.runTask",
    "args": "Run Tests"
  },
  {
    "key": "ctrl+shift+s",
    "command": "workbench.action.tasks.runTask",
    "args": "Sync to Colab"
  }
]
```

---

## Running Scripts on Colab from VS Code

### Method 1: VS Code Remote SSH (Recommended)

1. **Setup SSH to Colab** (run in Colab):
```python
!python scripts/setup_colab_ssh.py
```

2. **Connect VS Code:**
   - Press `F1` → "Remote-SSH: Connect to Host"
   - Select `colab-gpu`

3. **Open terminal in VS Code** (now connected to Colab):
```bash
# You're now on Colab's GPU!
python training/classification/train.py --epochs 50 --device cuda
```

### Method 2: Git Sync + Colab Terminal

1. **Local VS Code:**
```bash
# Kimi writes script
# You review and commit
./scripts/sync_to_colab.sh
```

2. **In Colab browser tab:**
```python
!git pull
!python training/classification/train.py --epochs 50
```

### Method 3: VS Code Jupyter Extension (Hybrid)

1. Install VS Code extension: **Jupyter**

2. Create `run_training.py`:
```python
# %% [markdown]
# # Training Script
# Run cells with Shift+Enter

# %%
# Setup
from google.colab import drive
drive.mount('/content/drive')

# %%
# Train
!python training/classification/train.py --epochs 50
```

3. VS Code treats this as notebook cells but it's a .py file!

---

## Best Practices

### 1. Script Organization

```
scripts/
├── prepare_data.py          # Data preparation
├── train_classification.py  # Training entry point
├── train_extraction.py      # Training entry point
├── evaluate.py              # Evaluation
├── export_model.py          # Model export
└── run_on_colab.py          # Colab launcher
```

### 2. Logging Output

Always save logs to file:

```python
import logging
from pathlib import Path
from datetime import datetime

# Create log directory
log_dir = Path('logs') / datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'train.log'),
        logging.StreamHandler()
    ]
)
```

### 3. Progress Tracking

Use tqdm for progress bars:

```python
from tqdm import tqdm

for epoch in tqdm(range(epochs), desc="Training"):
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
        # Training step
        pass
```

### 4. Resume Training

Save checkpoints that can be resumed:

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
```

---

## Complete Example Session

```bash
# 1. Start Kimi in VS Code terminal
$ kimi --continue

# 2. Ask Kimi to create script
# You type: "Create a script to benchmark all models"

# 3. Kimi writes scripts/benchmark_models.py
# You see the file appear in VS Code explorer

# 4. Review the script
$ cat scripts/benchmark_models.py

# 5. Run locally to test
$ python scripts/benchmark_models.py --models-dir models/ --quick-test

# 6. Everything works? Commit and sync
$ git add scripts/benchmark_models.py
$ git commit -m "Add model benchmarking script"
$ ./scripts/sync_to_colab.sh

# 7. Run on Colab (via SSH or browser)
$ python scripts/benchmark_models.py --models-dir /content/drive/MyDrive/retin-verify/models

# 8. View results
$ cat logs/benchmark_results.json
```

---

## Summary

| Task | You Do | Kimi Does | Where |
|------|--------|-----------|-------|
| Write script | Request | ✅ Generate code | VS Code |
| Review code | ✅ Check | - | VS Code |
| Test locally | ✅ Run | - | VS Code terminal |
| Debug | Describe issue | ✅ Fix code | VS Code |
| Run on Colab | ✅ Execute | - | VS Code SSH or terminal |
| Monitor | ✅ Check logs | - | VS Code output |

**No notebooks needed!** Everything runs as standard Python scripts from VS Code terminal.

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-14
