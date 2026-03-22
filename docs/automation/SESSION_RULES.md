# Session Rules & Guidelines

**Document:** Session Management Rules  
**Version:** 1.0  
**Updated:** 2026-03-18

---

## 1. File & Folder Structure Rules

### 1.1 Models Directory Structure

All models MUST follow this structure:

```
models/
├── classification/              # Current classification models
│   ├── *.pth                   # PyTorch models
│   ├── *.onnx                  # ONNX exports
│   └── *.json                  # Metadata/history
├── detection/                   # Detection models
├── extraction/                  # Extraction models
├── exported/                    # Final exports
├── classification_production/   # Production baselines (protected)
└── archive/                     # Historical versions
    └── YYYY-MM-DD/             # Date-stamped archives
```

### 1.2 Cleanup Rules

**After EVERY session:**

1. **Move new models** to appropriate folder (e.g., `classification/`)
2. **Archive old versions** to `archive/YYYY-MM-DD/`
3. **Remove temp files** (*.tar.gz, temp_*, etc.)
4. **Never leave files in models root** - only subdirectories

**Archive Process:**
```bash
# Create date-stamped archive
mkdir -p models/archive/YYYY-MM-DD

# Move old versions
mv models/classification/old_model.pth models/archive/YYYY-MM-DD/

# Remove empty directories
rmdir models/classification_colab 2>/dev/null || true
```

---

## 2. Session State Management

### 2.1 Session State File

Location: `~/.kimi/session_state.json`

**Required Fields:**
```json
{
  "session_id": "unique_identifier",
  "created_at": "ISO timestamp",
  "last_updated": "ISO timestamp",
  "status": "IN_PROGRESS|COMPLETE|ERROR",
  "phase": "current_phase",
  "project": "project_name",
  "current_task": "description",
  "steps": [
    {"step": 1, "name": "...", "status": "pending|in_progress|completed"}
  ],
  "next_action": "what to do next",
  "cleanup_rules_applied": true
}
```

### 2.2 Session Lifecycle

1. **Initialize:** Create session state with all planned steps
2. **Update:** Mark steps complete as progress happens
3. **Complete:** Final update with results, cleanup flag set
4. **Archive:** Move to `sessions/archive/` after completion

---

## 3. Documentation Requirements

### 3.1 Required Documents per Session

| Document | Location | Purpose |
|----------|----------|---------|
| Session Report | `docs/SESSION_REPORT_YYYY-MM-DD.md` | Complete session log |
| Session State | `~/.kimi/session_state.json` | Machine-readable status |
| Code Changes | Git commits | Track all modifications |

### 3.2 Session Report Template

Every session report MUST include:

1. **Executive Summary** - One-paragraph overview
2. **Phase-by-Phase Results** - What was done, metrics
3. **File Structure Changes** - Before/after cleanup
4. **Deliverables** - What was produced
5. **Usage Instructions** - How to use the outputs
6. **Next Steps** - Recommended follow-up actions

---

## 4. Model Management Rules

### 4.1 Model Naming Convention

```
{purpose}_{variant}_{version}.{ext}

Examples:
- cnie_front_back_real.pth          (production model)
- cnie_front_back_v2.pth            (version 2)
- classification_baseline_effnet.pth (baseline)
```

### 4.2 Model Metadata

Every model MUST include metadata:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': ['class1', 'class2'],
    'val_acc': 90.0,
    'epoch': 10,
    'created_at': '2026-03-18T12:00:00',
    'training_data': '7,600 images'
}, 'model.pth')
```

### 4.3 Export Requirements

For every PyTorch model:
1. ✅ Save .pth with metadata
2. ✅ Export to ONNX format
3. ✅ Validate ONNX with onnx.checker
4. ✅ Test inference on sample images

---

## 5. SSH/Colab Workflow Rules

### 5.1 SSH Key Management

- SSH key: `~/.ssh/id_colab` (private), `~/.ssh/id_colab.pub` (public)
- Config: `~/.ssh/config` with `colab-gpu` host entry
- Deploy key at start of each new Colab session

### 5.2 Colab Session Workflow

#### Automated Scripts (Recommended)

Do not edit the .ssh/config file at all.
read from .ssh/config file the hostname of cloudflared : <sshhostname>

Use these generic scripts for deployment and training:

**1. Deploy and Train:**
```bash
# Usage: ./colab_deploy_train.sh <hostname> <dataset_path> [script_path] [remote_dir]

# Example - Stage 2 retraining:
./scripts/colab_deploy_train.sh \
    <sshhostname> \
    ~/retin-verify/training_data/v6_stage2_corrected.tar.gz \
    ~/retin-verify/colab/v6_stage2_retrain_colab.py \
    /content

# Example - Custom training:
./scripts/colab_deploy_train.sh \
    <sshhostname> \
    ~/data/my_dataset.tar.gz \
    ~/scripts/train.py
```

**2. Download and Deploy:**
```bash
# Usage: ./colab_download_model.sh <hostname> <remote_model_path> [local_path] [restart]

# Example - Download and restart:
./scripts/colab_download_model.sh \
    <sshhostname> \
    /content/v6_stage2_corrected_best.pth

# Example - Download only (no restart):
./scripts/colab_download_model.sh \
    <sshhostname> \
    /content/model.pth \
    ~/models/my_model.pth \
    no
```

#### Complete Workflow Example

```
Step 1: Deploy & Start Training
$ ./scripts/colab_deploy_train.sh xxx.trycloudflare.com \
    ~/retin-verify/training_data/v6_stage2_corrected.tar.gz
✅ SSH key deployed
✅ Dataset uploaded (102MB)
✅ Script uploaded
✅ Dataset extracted
✅ Training started

Step 2: Monitor Training
$ ssh root@<sshhostname> 'tail -f /content/training*.log'
Epoch 1/30: Val Acc = 78.5%
Epoch 2/30: Val Acc = 85.2%
...
✅ TRAINING COMPLETE! Best Val Acc = 94.3%

Step 3: Download & Deploy
$ ./scripts/colab_download_model.sh xxx.trycloudflare.com \
    /content/v6_stage2_corrected_best.pth
✅ Model downloaded (17MB)
✅ Old model backed up to archive/2026-03-21/
✅ Training artifacts downloaded
✅ API server restarted
✅ Server healthy on localhost:8000
```

#### Manual Workflow (If Scripts Fail)

```
1. Start Colab with cloudflared tunnel
2. SSH password (first time): retinrix
3. Deploy SSH key:
   cat ~/.ssh/id_colab.pub | ssh root@xxx.trycloudflare.com \
     "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
4. Upload data:
   scp dataset.tar.gz root@xxx.trycloudflare.com:/content/
5. Extract and run:
   ssh root@xxx.trycloudflare.com "cd /content && tar -xzf dataset.tar.gz"
   ssh root@xxx.trycloudflare.com "cd /content && python train.py"
6. Monitor until complete
7. Download model:
   scp root@xxx.trycloudflare.com:/content/model.pth \
     ~/retin-verify/models/classification/
8. Restart local server
```

#### Status Summary Template

```
Step                      Status
1. SSH Config            ✅ Done
2. Deploy SSH Key        ✅ Done  
3. Upload Dataset        ✅ Done (102MB)
4. Upload Script         ✅ Done
5. Extract Dataset       ✅ Done (584 images)
6. Run Training          ✅ RUNNING NOW
7. Monitor               ✅ Auto-monitoring
8. Download              ⏳ Auto-when ready
9. Deploy Model          ⏳ Auto-when ready
10. Restart Server       ⏳ Auto-when ready

Training Progress
Status: RUNNING on Colab GPU (T4)
Epoch 12/30: Val Acc = 92.4% ✅ (Best model saved!)
Front Acc: 94.2% | Back Acc: 90.1%
Dataset: 584 train, 81 val images
Estimated time remaining: ~15 minutes
```

### 5.3 Script Parameters Reference

#### `colab_deploy_train.sh`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `hostname` | ✅ Yes | - | Colab SSH hostname (e.g., xxx.trycloudflare.com) |
| `dataset_path` | ✅ Yes | - | Path to dataset tarball (.tar.gz) |
| `script_path` | ❌ No | `colab/v6_stage2_retrain_colab.py` | Path to training script |
| `remote_dir` | ❌ No | `/content` | Remote directory on Colab |

**Examples:**
```bash
# Minimal - Stage 2 retraining
./scripts/colab_deploy_train.sh \
    xxx.trycloudflare.com \
    ~/retin-verify/training_data/v6_stage2_corrected.tar.gz

# Full parameters - Custom training
./scripts/colab_deploy_train.sh \
    xxx.trycloudflare.com \
    ~/data/custom_dataset.tar.gz \
    ~/scripts/my_training_script.py \
    /content/my_project
```

#### `colab_download_model.sh`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `hostname` | ✅ Yes | - | Colab SSH hostname |
| `remote_model_path` | ✅ Yes | - | Path to model on Colab |
| `local_model_path` | ❌ No | Auto-detected | Where to save locally |
| `restart_server` | ❌ No | `yes` | Whether to restart API server |

**Examples:**
```bash
# Auto-detect local path and restart
./scripts/colab_download_model.sh \
    xxx.trycloudflare.com \
    /content/v6_stage2_corrected_best.pth

# Custom local path, no restart
./scripts/colab_download_model.sh \
    xxx.trycloudflare.com \
    /content/model.pth \
    ~/models/my_experiment.pth \
    no
```

### 5.4 Customization Guide

#### Creating Custom Training Scripts

Your training script should:
1. Accept dataset from a configurable path (use `/content` or argument)
2. Save model with timestamp: `model_$(date +%Y%m%d_%H%M%S).pth`
3. Log to file for monitoring: `> training.log 2>&1`
4. Save best model: `best_model.pth`

**Template:**
```python
#!/usr/bin/env python3
"""Custom training script for Colab."""

import torch
import sys
from pathlib import Path

# Configurable paths
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/content/dataset"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/content"

# Training code here...
# Save model
model_path = f"{OUTPUT_DIR}/best_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")
```

#### Using with Different Projects

These scripts are generic and work with any project:

```bash
# For object detection project
./scripts/colab_deploy_train.sh \
    xxx.trycloudflare.com \
    ~/projects/detection/data/dataset.tar.gz \
    ~/projects/detection/train.py

# For NLP project  
./scripts/colab_deploy_train.sh \
    xxx.trycloudflare.com \
    ~/projects/nlp/data/corpus.tar.gz \
    ~/projects/nlp/train_transformer.py
```

---

## 6. Data Management Rules

### 6.1 Directory Structure

```
apps/classification/
├── my_cards/                   # Raw captured photos
│   ├── cnie_front/
│   └── cnie_back/
├── augmented/                  # Augmented dataset
│   ├── train/
│   ├── val/
│   └── annotations.json
└── cnie_only_augmented/        # Current session data
```

### 6.2 Data Retention

- **Raw photos:** Keep indefinitely (in `my_cards/`)
- **Augmented data:** Can be regenerated, archive if >10GB
- **Annotations:** Keep with model version

---

## 7. Session Checklist

### Before Starting
- [ ] Update session state
- [ ] Review previous session report
- [ ] Check file structure compliance
- [ ] Verify SSH connectivity (if using Colab)

### During Session
- [ ] Update session state after each phase
- [ ] Document decisions and rationale
- [ ] Test intermediate results
- [ ] Commit code changes regularly

### After Completion
- [ ] Finalize session state (status: COMPLETE)
- [ ] Write session report
- [ ] **Apply cleanup rules** (CRITICAL)
- [ ] Move new models to correct folders
- [ ] Archive old versions
- [ ] Remove temp files
- [ ] Verify file structure
- [ ] Update session state with `cleanup_rules_applied: true`

---

## 8. Quick Reference

### Cleanup Commands
```bash
cd ~/retin-verify/models

# Archive old classification models
mkdir -p archive/$(date +%Y-%m-%d)
mv classification/old_*.pth archive/$(date +%Y-%m-%d)/ 2>/dev/null || true

# Clean temp files
rm -f *.tar.gz model_temp*

# Verify structure
find . -maxdepth 2 -type f -name "*.pth" -o -name "*.onnx" | sort
```

### Session State Update
```bash
# Mark session complete
cat > ~/.kimi/session_state.json << 'EOF'
{
  "status": "COMPLETE",
  "phase": "done",
  "cleanup_rules_applied": true,
  ...
}
EOF
```

---

## 9. Violations & Corrections

### Common Violations

| Violation | Correction |
|-----------|------------|
| Files in models/ root | Move to appropriate subfolder |
| Multiple classification_* folders | Archive old, keep one |
| Missing session report | Create retroactively |
| No metadata in model | Reload and re-save with metadata |

### Emergency Cleanup

If structure is severely disorganized:

```bash
# 1. Create backup
cd ~/retin-verify
tar czf models_backup_$(date +%Y%m%d).tar.gz models/

# 2. Create clean structure
mkdir -p models/{classification,detection,extraction,exported,archive}

# 3. Move files to appropriate folders
# (manual review required)
```

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-18 | Initial rules established after Strategy 2 completion |

---

**Remember: Clean structure = Faster development, fewer errors, easier handoffs.**
