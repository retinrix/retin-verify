# Quick Script Reference

## 🚀 Run Training from VS Code (No Notebooks!)

### Option 1: Universal Launcher (Recommended)

```bash
# Auto-detects environment (local vs Colab) and configures paths

# Local test (CPU, 1 epoch)
python scripts/run_training.py classification --epochs 1

# Local full (auto-detects GPU if available)
python scripts/run_training.py classification --epochs 50 --fp16

# Colab (auto-mounts Drive, uses GPU)
python scripts/run_training.py classification --epochs 50 --fp16

# Extraction model (smaller batch size)
python scripts/run_training.py extraction --epochs 20 --batch-size 4
```

### Option 2: Direct Script (More Control)

```bash
# Local test
python training/classification/train_cli.py \
    --data-dir data/cnie_dataset_10k \
    --train-annotations data/processed/classification/train.json \
    --val-annotations data/processed/classification/val.json \
    --epochs 1 \
    --batch-size 4 \
    --device cpu

# Full training with GPU
python training/classification/train_cli.py \
    --data-dir data/cnie_dataset_10k \
    --epochs 50 \
    --batch-size 32 \
    --fp16 \
    --save-every 5
```

### Option 3: VS Code Tasks (GUI)

Press `Ctrl+Shift+P` → "Tasks: Run Task" → Select:
- `📦 Prepare Dataset`
- `🚀 Train Classification (Test)`
- `🚀 Train Classification (Full - Local GPU)`
- `🧪 Run Tests`
- `🔄 Sync to Colab`

### Option 4: VS Code Launch Configurations

Press `F5` or `Ctrl+Shift+D` → Select from dropdown:
- `Train Classification (Local Test)`
- `Train Classification (Full)`
- `Universal Launcher (Auto-detect)`
- `Prepare Dataset`

---

## 📝 Kimi + VS Code Workflow

### Step 1: Ask Kimi to Create Script

**In VS Code terminal with Kimi:**
```
You: Create a script to evaluate model performance on test set
```

**Kimi writes:** `scripts/evaluate.py`

### Step 2: Review in VS Code

```bash
# View the script
cat scripts/evaluate.py

# Check syntax
python -m py_compile scripts/evaluate.py
```

### Step 3: Run from VS Code Terminal

```bash
# Test locally
python scripts/evaluate.py --model models/best_model.pth --test-data data/processed/test.json

# Works? Commit and sync
./scripts/sync_to_colab.sh
```

### Step 4: Run on Colab

**In Colab (or VS Code SSH to Colab):**
```bash
!git pull
!python scripts/evaluate.py --model /content/drive/MyDrive/retin-verify/models/best_model.pth
```

---

## 🎯 Common Commands

### Data Preparation
```bash
# Prepare all dataset splits
python scripts/prepare_dataset.py \
    --data-dir data/cnie_dataset_10k \
    --output-dir data/processed

# Check output
ls data/processed/
```

### Training
```bash
# Quick test (1 epoch, CPU)
python scripts/run_training.py classification --epochs 1 --batch-size 2

# Full training (50 epochs, GPU, mixed precision)
python scripts/run_training.py classification --epochs 50 --fp16

# Resume from checkpoint
python training/classification/train_cli.py \
    --resume-from models/classification/checkpoint_epoch_20.pth \
    --epochs 50

# Custom learning rate
python scripts/run_training.py classification --epochs 50 --learning-rate 5e-5
```

### Monitoring
```bash
# Start TensorBoard
tensorboard --logdir logs/

# View in browser: http://localhost:6006

# Or use VS Code task
Ctrl+Shift+P → "Tasks: Run Task" → "📊 Launch TensorBoard"
```

### Evaluation
```bash
# Evaluate model
python scripts/evaluate.py \
    --model models/classification/best_model.pth \
    --test-data data/processed/classification/test.json \
    --output results/evaluation.json
```

### Export
```bash
# Export to ONNX
python scripts/export_models.py \
    --input-dir models/classification \
    --output-dir models/exported \
    --format onnx
```

---

## 🔄 Complete Workflow Example

```bash
# 1. Start Kimi in VS Code terminal
kimi --continue

# 2. Ask Kimi to create evaluation script
# You: "Create a script to compare multiple trained models"

# 3. Review Kimi's script
cat scripts/compare_models.py

# 4. Run locally to test
python scripts/compare_models.py --models-dir models/ --output comparison.html

# 5. Commit and push
./scripts/sync_to_colab.sh

# 6. Run on Colab for full comparison
# (SSH to Colab or use Colab terminal)
python scripts/compare_models.py \
    --models-dir /content/drive/MyDrive/retin-verify/models \
    --output /content/drive/MyDrive/retin-verify/comparison.html

# 7. View results
open comparison.html  # macOS
xdg-open comparison.html  # Linux
```

---

## 💡 Kimi Prompts for Scripts

### Creating New Scripts
```
"Create a script to [TASK] with argparse for:
- input path
- output path
- [other parameters]
Include progress bars with tqdm and logging."
```

### Updating Scripts
```
"Update [SCRIPT] to:
1. Add [FEATURE]
2. Fix [ISSUE]
3. Improve error handling"
```

### Debugging
```
"[SCRIPT] is failing with [ERROR].
Here's the traceback: [PASTE]
Fix it and add better error messages."
```

---

## 📁 Script Organization

```
scripts/
├── run_training.py          # Universal launcher
├── prepare_dataset.py       # Data preparation
├── sync_to_colab.sh         # Git sync
├── fix_rclone.sh           # rclone fix
├── evaluate.py             # Model evaluation (Kimi creates)
├── benchmark.py            # Performance benchmark (Kimi creates)
└── export_models.py        # Model export

training/
├── classification/
│   ├── train_cli.py        # Main training script
│   └── configs/
├── extraction/
│   ├── train.py
│   └── configs/
└── utils/
    ├── data_loaders.py
    └── metrics.py
```

---

## 🐛 Troubleshooting

### Script not found
```bash
# Make sure you're in project root
cd /path/to/retin-verify

# Check if script exists
ls -la scripts/run_training.py

# Make executable
chmod +x scripts/run_training.py
```

### Module not found
```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Permission denied
```bash
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

### Data not found
```bash
# Run preparation first
python scripts/prepare_dataset.py --data-dir data/cnie_dataset_10k

# Or check if data exists
ls -la data/cnie_dataset_10k/
```

---

## ⚡ Pro Tips

### 1. Use VS Code Terminal Shortcuts

```json
// .vscode/keybindings.json
[
  {
    "key": "ctrl+shift+t",
    "command": "workbench.action.tasks.runTask",
    "args": "🧪 Run Tests"
  },
  {
    "key": "ctrl+shift+r",
    "command": "workbench.action.tasks.runTask",
    "args": "🚀 Train Classification (Test)"
  }
]
```

### 2. Create Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc
alias rt='python scripts/run_training.py'
alias rt-test='python scripts/run_training.py classification --epochs 1'
alias rt-full='python scripts/run_training.py classification --epochs 50 --fp16'
alias rt-sync='./scripts/sync_to_colab.sh'
```

### 3. Use Makefile

```makefile
# Makefile
.PHONY: test train sync

test:
	pytest tests/ -v

train-test:
	python scripts/run_training.py classification --epochs 1

train:
	python scripts/run_training.py classification --epochs 50 --fp16

sync:
	./scripts/sync_to_colab.sh

prepare:
	python scripts/prepare_dataset.py --data-dir data/cnie_dataset_10k
```

Then run:
```bash
make train-test
make sync
```

---

**Remember:** Kimi writes the scripts, you run them from VS Code terminal!
