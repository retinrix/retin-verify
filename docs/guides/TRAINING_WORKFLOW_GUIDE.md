# Complete Training Workflow Guide

This document describes the end-to-end workflow for training models with local development and remote GPU execution.

## Quick Start (Recommended)

Use the **master workflow script** for all operations:

```bash
# 1. Initialize new Colab session (SSH key + sync + status check)
./scripts/colab_workflow.sh colab-gpu init

# 2. In Colab notebook: Mount Drive
#    from google.colab import drive
#    drive.mount('/content/drive')

# 3. Setup dataset (via SSH)
./scripts/colab_workflow.sh colab-gpu setup

# 4. Prepare dataset (FIRST TIME ONLY - in Colab notebook)
#    %cd /content/retin-verify
#    !python scripts/prepare_dataset.py --data-dir /content/data --output-dir /content/data/processed

# 5. Full workflow: sync → train → export → pull
./scripts/colab_workflow.sh colab-gpu full --epochs 100 --fp16
```

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW (Simplified)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LOCAL MACHINE                        GOOGLE COLAB                          │
│  ┌─────────────────┐                 ┌─────────────────┐                   │
│  │  1. Start Kimi  │                 │  2. Mount Drive │  ← Notebook cell  │
│  │     kimi --cont │                 │     (in browser)│                   │
│  └────────┬────────┘                 └────────┬────────┘                   │
│           │                                    │                             │
│  ┌────────▼────────┐                 ┌────────▼────────┐                   │
│  │  3. One Command │ ──SSH Tunnel──► │  4. Train & Run │                   │
│  │  colab_workflow │                 │     on GPU      │                   │
│  │     .sh full    │ ◄────────────── │                 │                   │
│  └────────┬────────┘                 └────────┬────────┘                   │
│           │                                    │                             │
│  ┌────────▼────────┐                 ┌────────▼────────┐                   │
│  │  5. Results     │                 │  6. Save to     │                   │
│  │     Local       │                 │     Drive       │                   │
│  └─────────────────┘                 └─────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Local Machine
- VS Code with Kimi Code Agent
- SSH client configured
- Project cloned: `git clone <repo>`

### Google Colab
- Colab Pro (recommended for persistent GPU)
- Dataset uploaded to Google Drive at:
  ```
  /MyDrive/retin-verify/data/cnie_dataset_10k/
  ```

---

## Master Workflow Script

### `colab_workflow.sh` - One Script for Everything

```bash
# Initialize new session (run once per new Colab runtime)
./scripts/colab_workflow.sh colab-gpu init

# Full workflow: sync → train → export → pull
./scripts/colab_workflow.sh colab-gpu full --epochs 100 --fp16

# Individual commands
./scripts/colab_workflow.sh colab-gpu sync      # Sync code only
./scripts/colab_workflow.sh colab-gpu setup     # Setup dataset
./scripts/colab_workflow.sh colab-gpu train     # Run training
./scripts/colab_workflow.sh colab-gpu export    # Export artifacts
./scripts/colab_workflow.sh colab-gpu pull      # Pull results
./scripts/colab_workflow.sh colab-gpu status    # Check status
./scripts/colab_workflow.sh colab-gpu shell     # Interactive shell
```

### Full Workflow Options

```bash
# Basic training
./scripts/colab_workflow.sh colab-gpu full --epochs 50

# With all options
./scripts/colab_workflow.sh colab-gpu full \
    --epochs 100 \
    --batch-size 32 \
    --model-type classification \
    --fp16
```

---

## Step-by-Step Guide

### Step 1: Configure SSH (One-time setup)

Create `~/.ssh/config`:

```ssh-config
Host *.trycloudflare.com
    HostName %h
    User root
    Port 22
    ProxyCommand /usr/local/bin/cloudflared access ssh --hostname %h
    IdentityFile ~/.ssh/id_colab
    IdentitiesOnly yes
    
Host colab-gpu
    HostName convertible-michael-assembled-fog.trycloudflare.com
    User root
    Port 22
    ProxyCommand /usr/local/bin/cloudflared access ssh --hostname %h
    IdentityFile ~/.ssh/id_colab
    IdentitiesOnly yes
```

Install cloudflared:
```bash
# macOS
brew install cloudflared

# Linux
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
```

### Step 2: Start Colab & Initialize

**In Colab notebook:**
```python
# Run cloudflared setup (get trycloudflare.com URL)
!pip install colab_ssh -q
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="your_password")
```

**From local terminal:**
```bash
# Initialize (SSH key + sync + status check)
./scripts/colab_workflow.sh colab-gpu init
```

### Step 3: Mount Drive (in Colab notebook)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Setup Dataset

```bash
./scripts/colab_workflow.sh colab-gpu setup
```

### Step 5: Prepare Dataset (First Time Only)

The raw dataset needs to be processed into train/val/test splits for training. This step creates annotation JSON files.

**In Colab notebook cell (recommended):**
```python
%cd /content/retin-verify
!python scripts/prepare_dataset.py \
    --data-dir /content/data \
    --output-dir /content/data/processed
```

**Or via SSH (runs in background to avoid timeout):**
```bash
./scripts/deploy.sh colab-gpu exec \
    "bash /content/retin-verify/scripts/prepare_dataset_bg.sh"

# Check progress:
./scripts/deploy.sh colab-gpu exec "tail -f /tmp/prepare_dataset.log"

# Verify completion:
./scripts/deploy.sh colab-gpu exec \
    "ls -la /content/data/processed/classification/"
```

**What this creates:**
```
/content/data/processed/
├── classification/
│   ├── train.json    # 80% of data (~2,973 samples)
│   ├── val.json      # 10% of data (~372 samples)
│   └── test.json     # 10% of data (~371 samples)
└── extraction/
    ├── train.json
    ├── val.json
    └── test.json
```

### Step 6: Develop Locally

```bash
# Start Kimi
code .
kimi --continue

# Quick local test (CPU)
python training/classification/train.py --epochs 1 --device cpu
```

### Step 7: Train on Colab

```bash
# Full workflow: sync → train → export → pull
./scripts/colab_workflow.sh colab-gpu full --epochs 100 --fp16
```

Or step by step:
```bash
./scripts/colab_workflow.sh colab-gpu sync
./scripts/colab_workflow.sh colab-gpu train --epochs 100
./scripts/colab_workflow.sh colab-gpu export
./scripts/colab_workflow.sh colab-gpu pull
```

**Note:** Training requires the dataset to be prepared first (Step 5).

---

## Behind the Scenes

The `colab_workflow.sh` script uses these lower-level scripts:

| Script | Purpose |
|--------|---------|
| `deploy_ssh_key.sh` | Deploy SSH key for passwordless auth |
| `deploy.sh` | Sync code and execute commands on remote |
| `prepare_dataset.py` | Prepare train/val/test splits from raw data |
| `prepare_dataset_bg.sh` | Run dataset preparation in background (SSH) |
| `export_artifacts.py` | Export models/checkpoints/logs to Drive |

You can still use these directly if needed, but `colab_workflow.sh` handles the orchestration.

---

## Directory Structure on Colab

After setup:
```
/content/
├── retin-verify/          # Code (synced from local)
│   ├── training/
│   ├── configs/
│   └── scripts/
├── data/                  # Symlink to Drive dataset
│   ├── cnie_pairs/        # Raw images
│   └── processed/         # Train/val/test splits (created by prepare_dataset.py)
│       ├── classification/
│       │   ├── train.json
│       │   ├── val.json
│       │   └── test.json
│       └── extraction/
├── models/ → /content/drive/MyDrive/retin-verify/models
├── checkpoints/ → /content/drive/MyDrive/retin-verify/checkpoints
└── logs/ → /content/drive/MyDrive/retin-verify/logs
```

---

## Troubleshooting

### "Drive not mounted" error

```bash
# In Colab notebook cell, run:
from google.colab import drive
drive.mount('/content/drive')

# Then retry:
./scripts/colab_workflow.sh colab-gpu setup
```

### Check what's happening on Colab

```bash
./scripts/colab_workflow.sh colab-gpu status
```

### Dataset not found

```bash
# Check Drive path
./scripts/colab_workflow.sh colab-gpu shell
# Then in shell:
ls /content/drive/MyDrive/retin-verify/data/
```

### "Train annotations not found" error

This means data preparation hasn't been run. You need to create the train/val/test splits:

```bash
# Option 1: In Colab notebook cell (recommended)
%cd /content/retin-verify
!python scripts/prepare_dataset.py \
    --data-dir /content/data \
    --output-dir /content/data/processed

# Option 2: Via SSH (runs in background)
./scripts/deploy.sh colab-gpu exec \
    "bash /content/retin-verify/scripts/prepare_dataset_bg.sh"

# Check if complete:
./scripts/deploy.sh colab-gpu exec \
    "ls /content/data/processed/classification/train.json"
```

### Connection issues

```bash
# Test SSH
ssh colab-gpu "echo 'Connected!'"

# Re-init if needed
./scripts/colab_workflow.sh colab-gpu init
```

---

## Related Documents

- `REMOTE_DEPLOYMENT_WORKFLOW.md` - Deployment architecture details
- `VS_CODE_SCRIPT_WORKFLOW.md` - Script-based development guide

---

**Document Version**: 2.0  
**Last Updated**: 2026-03-15
