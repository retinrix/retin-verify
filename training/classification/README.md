# Hybrid Retraining Workflow

This directory contains scripts for the **Local → Colab → Local** retraining workflow.

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   LOCAL MACHINE │────▶│  GOOGLE COLAB   │────▶│   LOCAL MACHINE │
│                 │     │   (GPU Power)   │     │                 │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ 1. Collect      │     │ 3. Receive      │     │ 5. Download     │
│    feedback     │────▶│    data + model │────▶│    new model    │
│                 │ SSH │                 │ SSH │                 │
│ 2. Deploy to    │     │ 4. Train on GPU │     │ 6. Restart      │
│    Colab        │     │    (10 epochs)  │     │    server       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Prerequisites

### 1. SSH to Colab Setup (Already Done)

You already have the SSH keys configured:
- SSH key: `~/.ssh/id_colab`
- SSH config: `~/.ssh/config`
- cloudflared: `/usr/local/bin/cloudflared`

### 2. Colab Notebook Setup

In your Colab notebook, run these cells to enable SSH:

```python
# Cell 1: Install colab_ssh
!pip install colab_ssh --upgrade

# Cell 2: Start SSH tunnel
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared("temp_password")

# Cell 3: (Optional) Auto-install your public key
from colab_ssh import init_git_cloudflared
init_git_cloudflared("https://github.com/YOUR_USERNAME/YOUR_REPO.git")
```

**Note the hostname** (e.g., `abc123.trycloudflare.com`) - you'll need it for the scripts.

## The Retraining Loop

### Step 1: Collect Feedback Locally

Use the web UI at http://127.0.0.1:8000

- Capture images and classify
- Click "🚩 Flag & Upload" when predictions are wrong
- Continue until you have **10+ flagged images**

Check status:
```bash
python colab_retrain/retrain_manager.py --status
```

### Step 2: Start Colab Session

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Run the SSH setup cells above
4. **Copy the hostname** (e.g., `abc123.trycloudflare.com`)
5. Keep the notebook running!

### Step 3: Deploy and Train

Option A: **Full automated workflow**
```bash
cd retin-verify/apps/classification

# This does everything: deploy → monitor → download → restart
python colab_retrain/retrain_manager.py --host abc123.trycloudflare.com --full
```

Option B: **Step by step**
```bash
# 1. Deploy data to Colab
python colab_retrain/retrain_manager.py --host abc123.trycloudflare.com --deploy

# 2. Monitor training progress
python colab_retrain/retrain_manager.py --host abc123.trycloudflare.com --monitor

# 3. When done, download and restart
python colab_retrain/retrain_manager.py --host abc123.trycloudflare.com --download --restart
```

### Step 4: Test

After restart, test the improved model:
```bash
# Check server health
curl http://127.0.0.1:8000/health

# Open UI and test classification
open http://127.0.0.1:8000
```

### Step 5: Repeat

Continue the loop:
1. Use the system → Collect more feedback
2. When you have 10+ new samples, repeat from Step 2

## Script Reference

### retrain_manager.py
Main orchestrator script.

```bash
# Full workflow
python retrain_manager.py --host HOSTNAME --full

# Individual steps
python retrain_manager.py --host HOSTNAME --deploy
python retrain_manager.py --host HOSTNAME --monitor
python retrain_manager.py --host HOSTNAME --download --restart
python retrain_manager.py --host HOSTNAME --status
```

### deploy_to_colab.py
Deploys feedback data and starts training.

```bash
python deploy_to_colab.py --host abc123.trycloudflare.com

# Check if enough samples
python deploy_to_colab.py --check

# Deploy even with <10 samples
python deploy_to_colab.py --host abc123.trycloudflare.com --force
```

### download_model.py
Downloads the retrained model and deploys locally.

```bash
# Download only
python download_model.py --host abc123.trycloudflare.com

# Download and restart server
python download_model.py --host abc123.trycloudflare.com --restart

# Force download even if training not marked complete
python download_model.py --host abc123.trycloudflare.com --force
```

## What Happens During Training

### On Your Local Machine
- Feedback images are packaged into `retrain_data_TIMESTAMP.tar.gz`
- Base model is copied
- Everything is uploaded to Colab via SCP

### On Colab
- Data extracted to `/content/retin_retrain/`
- Training runs for 10 epochs with augmentation
- Best model saved as `cnie_front_back_real_retrained.pth`
- Completion flag written to `DONE` file
- Training log saved to `training.log`

### Back On Local
- New model downloaded
- Current model backed up with timestamp
- New model replaces the old one
- Server restarted with new model

## Troubleshooting

### SSH Connection Fails
```bash
# Test SSH manually
ssh root@abc123.trycloudflare.com

# If it fails, check:
# 1. Colab notebook is still running
# 2. Hostname is correct
# 3. cloudflared is installed: which cloudflared
```

### Training Fails on Colab
```bash
# Check logs manually
ssh root@abc123.trycloudflare.com 'cat /content/retin_retrain/training.log'

# Check if process is running
ssh root@abc123.trycloudflare.com 'pgrep -f train_on_colab'
```

### Download Fails
```bash
# Check if training completed
ssh root@abc123.trycloudflare.com 'cat /content/retin_retrain/DONE'

# List files
ssh root@abc123.trycloudflare.com 'ls -la /content/retin_retrain/'

# Force download
python download_model.py --host abc123.trycloudflare.com --force
```

### Server Won't Restart
```bash
# Manual restart
pkill -f api_server.py
./start_server.sh

# Check status
curl http://127.0.0.1:8000/health
```

## File Structure

```
colab_retrain/
├── README.md                 # This file
├── retrain_manager.py        # Main orchestrator
├── deploy_to_colab.py        # Deploy data to Colab
└── download_model.py         # Download model from Colab
```

## Feedback Data Structure

```
feedback_data/
├── misclassified/            # Wrong predictions (used for retraining)
│   ├── cnie_front/
│   └── cnie_back/
├── correct/                  # Confirmed correct
├── low_confidence/           # Uncertain predictions
├── feedback_annotations.json # Metadata
└── retraining_dataset/       # Auto-generated for training
    ├── train/
    │   ├── cnie_front/
    │   └── cnie_back/
    └── val/
        ├── cnie_front/
        └── cnie_back/
```

## Model Backup

Each deployment creates a backup:
```
models/classification/
├── cnie_front_back_real.pth                    # Current model
├── cnie_front_back_real_backup_20260318_143022.pth  # Backup 1
└── cnie_front_back_real_backup_20260317_091530.pth  # Backup 2
```

## Summary

| Step | Command | Time |
|------|---------|------|
| Check status | `python retrain_manager.py --status` | 1s |
| Deploy | `python retrain_manager.py --host X --deploy` | 10-30s |
| Train on Colab | Automatic | 5-15 min |
| Download | `python retrain_manager.py --host X --download --restart` | 10-20s |
| **Total** | | **~10-20 min** |

The model improves with each retraining cycle as it learns from real-world feedback!
