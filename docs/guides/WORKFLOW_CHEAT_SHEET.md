# Workflow Cheat Sheet

## Quick Decision Tree

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: What do you want to do?               │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │  Develop  │      │  Train    │      │  Deploy   │
    │  Code     │      │  Models   │      │  Models   │
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          │                  │                  │
          ▼                  ▼                  ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ VS Code +    │   │ Colab Pro    │   │ Download from│
   │ Kimi Agent   │   │ + Notebook   │   │ Drive        │
   │ (Local)      │   │ (Cloud GPU)  │   │ (Manual)     │
   └──────────────┘   └──────────────┘   └──────────────┘
```

---

## Daily Workflow Commands

### 1. Start Development Session

```bash
# Terminal 1: Start Kimi
cd retin-verify
kimi --continue

# Terminal 2: Git status
cd retin-verify
git status
```

### 2. After Kimi Makes Changes

```bash
# Review changes
git diff

# Run tests
pytest tests/unit/ -v

# Sync to Colab
./scripts/sync_to_colab.sh
```

### 3. Start Training on Colab

```python
# In Colab notebook cell 1:
!python scripts/init_colab.py

# Cell 2: Select model
MODEL_TYPE = "classification"  # or "extraction"
EXPERIMENT_NAME = "exp_v1"

# Cell 3: Run training (Kimi-generated)
# ... training code ...

# Cell 4: Keep alive
import time
while True:
    time.sleep(60)
    print("Running...", flush=True)
```

### 4. Download Results

```bash
# Option 1: rclone (fast)
rclone copy gdrive:retin-verify/models ./models --progress

# Option 2: gdown (for single files)
gdown <file_id>
```

---

## Kimi Prompts You Can Use

### Code Development

```
"Create a training script for [MODEL] with:
- Early stopping (patience=10)
- Model checkpointing every epoch
- TensorBoard logging
- Mixed precision training"
```

```
"Refactor [FILE] to:
- Use class-based structure
- Add type hints
- Improve error handling
- Add docstrings"
```

```
"Write unit tests for [MODULE] covering:
- Normal cases
- Edge cases
- Error conditions"
```

### Configuration

```
"Create a YAML config for training [MODEL] with:
- Batch size optimized for [GPU] with [VRAM]GB
- Learning rate schedule with warmup
- Data augmentation for document images"
```

### Debugging

```
"I'm getting [ERROR] when running [SCRIPT].
Here's the stack trace: [PASTE]
Fix the issue and explain what caused it."
```

```
"Training is too slow. 
GPU utilization is only 30%.
Optimize the data loading pipeline."
```

### Documentation

```
"Document the [COMPONENT] in README format:
- Purpose
- Usage examples
- Configuration options
- Common issues"
```

---

## Checklists by Task

### ✅ Setting Up (One-time)

- [ ] Sign up for Colab Pro
- [ ] Create ngrok account
- [ ] Upload dataset to Google Drive
- [ ] Push code to GitHub
- [ ] Test Colab notebook
- [ ] (Optional) Setup VS Code SSH

### 📝 Development Day

- [ ] Open VS Code
- [ ] Start Kimi agent
- [ ] Define today's tasks
- [ ] Review Kimi's code
- [ ] Run local tests
- [ ] Sync to GitHub
- [ ] Commit changes

### 🚀 Training Day

- [ ] Open Google Colab
- [ ] Start GPU runtime
- [ ] Run init script
- [ ] Configure experiment
- [ ] Start training
- [ ] Open TensorBoard
- [ ] Monitor every 2 hours
- [ ] Download model when done

### 📊 Analysis Day

- [ ] Download trained models
- [ ] Run evaluation script
- [ ] Review metrics
- [ ] Generate comparison plots
- [ ] Document results
- [ ] Plan next experiments

---

## Common Issues & Solutions

### Issue: Colab disconnects during training

**You do:**
1. Reconnect to runtime
2. Re-run initialization cell
3. Resume from checkpoint

**Kimi can:**
- Create auto-resume script
- Add checkpoint verification
- Implement notification on completion

### Issue: Out of Memory (OOM)

**You tell Kimi:**
"Training crashes with OOM on [GPU] with [X]GB VRAM"

**Kimi does:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Add mixed precision
4. Implement gradient accumulation

### Issue: Training too slow

**You tell Kimi:**
"GPU utilization is low, training is slow"

**Kimi does:**
1. Optimize data loading (num_workers, pin_memory)
2. Add prefetching
3. Profile bottlenecks
4. Suggest caching to local disk

---

## Time Estimates

| Task | Manual Time | With Kimi | Speedup |
|------|-------------|-----------|---------|
| Write training script | 2-3 hours | 15 min | 8x |
| Debug OOM error | 1-2 hours | 10 min | 6x |
| Create config files | 30 min | 5 min | 6x |
| Write documentation | 1 hour | 10 min | 6x |
| Setup Colab workflow | 4 hours | 30 min | 8x |
| **Total Setup** | **8-11 hours** | **~1 hour** | **9x** |

---

## Quick Reference: Who Does What

| You Do | Kimi Does |
|--------|-----------|
| Type prompts | Write code |
| Review changes | Generate configs |
| Click buttons in UI | Create scripts |
| Make decisions | Write docs |
| Monitor training | Debug errors |
| Download results | Optimize performance |

---

## File Locations

### Local (Your Machine)
```
retin-verify/
├── src/                  # Source code (Kimi edits)
├── tests/                # Tests (Kimi writes)
├── training/             # Training code (Kimi creates)
├── inference/            # Inference code (Kimi creates)
└── docs/                 # Documentation (Kimi writes)
```

### Google Drive
```
My Drive/retin-verify/
├── data/
│   └── cnie_dataset_10k/     # Dataset (You upload)
├── models/                    # Trained models (Auto-saved)
├── checkpoints/               # Checkpoints (Auto-saved)
└── logs/                      # TensorBoard logs (Auto-saved)
```

### GitHub
```
github.com/YOUR_USERNAME/retin-verify/
├── All code synced from local
└── Version history
```

---

## Emergency Commands

```bash
# Colab crashed, need to restart quickly
!python scripts/init_colab.py

# Need to free GPU memory
import torch
torch.cuda.empty_cache()

# Check what's using GPU
!nvidia-smi

# Kill hanging process
!pkill -9 python

# Remount Drive if disconnected
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Quick sync to Colab (when you don't have time for review)
git add . && git commit -m "quick sync" && git push
```

---

## Success Metrics

After setup completion:
- [ ] Can start training with 3 clicks (Colab)
- [ ] Can sync code with 1 command
- [ ] Kimi writes 80% of new code
- [ ] You review 100% of changes
- [ ] Training runs unattended for hours
- [ ] Models auto-save to Drive
- [ ] Can monitor from phone (TensorBoard/Colab mobile)

---

**Print this page and keep it handy!**
