# Remote Deployment Workflow (Rsync-Based)

## Overview

This document describes the **rsync-based deployment workflow** for the Retin-Verify project. The workflow is now orchestrated by a single master script: `colab_workflow.sh`.

## Quick Start

```bash
# 0. In Colab notebook - Start SSH and mount Drive:
#     !pip install colab_ssh -q
#     from colab_ssh import launch_ssh_cloudflared
#     launch_ssh_cloudflared(password="your_password")
#     from google.colab import drive
#     drive.mount('/content/drive')
#
#     Copy the trycloudflare.com URL and update ~/.ssh/config

# 1. Deploy SSH key (required for each new Colab session)
./scripts/deploy_ssh_key.sh colab-gpu

# 2. Initialize
./scripts/colab_workflow.sh colab-gpu init

# 3. Setup dataset
./scripts/colab_workflow.sh colab-gpu setup

# 4. Full training workflow (sync → train → export → pull)
./scripts/colab_workflow.sh colab-gpu full --epochs 100
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RSYNC DEPLOYMENT WORKFLOW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   LOCAL MACHINE                          REMOTE SERVER                   │
│  ┌──────────────────┐                   ┌──────────────────┐            │
│  │                  │  1. rsync push   │                  │            │
│  │  colab_workflow  │ ─────────────────►│   GPU Training   │            │
│  │     .sh          │                   │                  │            │
│  │                  │  2. Execute       │                  │            │
│  │  ┌──────────┐    │ ◄──────────────── │  ┌──────────┐    │            │
│  │  │  Code    │    │   (results)       │  │ Training │    │            │
│  │  │ (local)  │    │                   │  │ Script   │    │            │
│  │  └──────────┘    │                   │  └──────────┘    │            │
│  │        ▲         │                   │                  │            │
│  │        │         │                   │  ┌──────────┐    │            │
│  │  ┌─────┴────┐    │  3. rsync pull    │  │  Drive   │    │            │
│  │  │  Results │    │ ◄──────────────── │  │  (Data)  │◄───┘            │
│  │  │ (local)  │    │                   │  └──────────┘                 │
│  │  └──────────┘    │                   │                                 │
│  └──────────────────┘                   └──────────────────┘             │
│                                                                          │
│   Loop: Edit → colab_workflow.sh full → Iterate                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **Local Machine**: VS Code with Kimi Code Agent
2. **Remote Server**: SSH access with GPU (Colab Pro, AWS, GCP, etc.)
3. **Tools**: `rsync`, `ssh`, `cloudflared` installed locally

---

## Master Workflow Script

### `colab_workflow.sh`

The single entry point for all operations.

```bash
./scripts/colab_workflow.sh <host> <command> [options]
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `init` | Initialize: SSH key + sync + status | `colab_workflow.sh colab-gpu init` |
| `sync` | Sync code to remote | `colab_workflow.sh colab-gpu sync` |
| `setup` | Setup dataset (Drive must be mounted) | `colab_workflow.sh colab-gpu setup` |
| `train` | Run training on remote | `colab_workflow.sh colab-gpu train --epochs 50` |
| `export` | Export artifacts to Drive | `colab_workflow.sh colab-gpu export` |
| `pull` | Pull results from remote | `colab_workflow.sh colab-gpu pull` |
| `full` | Full workflow: sync→train→export→pull | `colab_workflow.sh colab-gpu full --epochs 100` |
| `status` | Check Colab status | `colab_workflow.sh colab-gpu status` |
| `shell` | Interactive SSH shell | `colab_workflow.sh colab-gpu shell` |

### Options for `train` and `full`

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs N` | Number of epochs | 50 |
| `--batch-size N` | Batch size | 32 |
| `--model-type` | classification/extraction/detection | classification |
| `--fp16` | Enable mixed precision | false |

---

## Step-by-Step Setup

### 1. Configure SSH

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

# Generate SSH key
ssh-keygen -t ed25519 -f ~/.ssh/id_colab -C "colab"
```

### 2. Initialize New Colab Session

**In Colab notebook:**
```python
# Run cloudflared setup
!pip install colab_ssh -q
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="your_password")
# Copy the trycloudflare.com URL
```

**Update SSH config with new hostname:**

Edit `~/.ssh/config` and update the `HostName` for `colab-gpu`:
```ssh-config
Host colab-gpu
    HostName YOUR_NEW_URL.trycloudflare.com  # <-- Update this!
    User root
    Port 22
    ProxyCommand /usr/local/bin/cloudflared access ssh --hostname %h
    IdentityFile ~/.ssh/id_colab
    IdentitiesOnly yes
```

**From local terminal - Deploy SSH key and initialize:**
```bash
# 1. Deploy SSH public key (required for each new Colab session)
./scripts/deploy_ssh_key.sh colab-gpu
# Enter the password you set in launch_ssh_cloudflared()

# 2. Initialize (sync + status check)
./scripts/colab_workflow.sh colab-gpu init
```

### 3. Mount Drive, Setup Dataset & Prepare Data

**In Colab notebook cell - Mount Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**From local terminal - Setup dataset:**
```bash
./scripts/colab_workflow.sh colab-gpu setup
```

**In Colab notebook cell - Prepare dataset (FIRST TIME ONLY):**
```python
# Create train/val/test splits from raw data
%cd /content/retin-verify
!python scripts/prepare_dataset.py \
    --data-dir /content/data \
    --output-dir /content/data/processed
```

Or via SSH (runs in background):
```bash
./scripts/deploy.sh colab-gpu exec \
    "bash /content/retin-verify/scripts/prepare_dataset_bg.sh"

# Check progress:
./scripts/deploy.sh colab-gpu exec "tail -f /tmp/prepare_dataset.log"
```

### 4. Train

```bash
# Full workflow: sync → train → export → pull
./scripts/colab_workflow.sh colab-gpu full --epochs 100 --fp16
```

---

## Directory Structure

### Local Project
```
retin-verify/
├── training/          # Training code (synced)
├── configs/           # Config files (synced)
├── scripts/           # Helper scripts (synced)
│   ├── colab_workflow.sh    # Master workflow
│   ├── deploy.sh            # Low-level sync/execute
│   ├── deploy_ssh_key.sh    # SSH key deployment
│   └── export_artifacts.py  # Export models/logs
├── outputs/           # Results (pulled from remote)
├── logs/              # Training logs (pulled)
└── models/            # Trained models (pulled)
```

### Remote (Colab)
```
/content/
├── retin-verify/      # Code (synced from local)
│   ├── training/
│   ├── configs/
│   └── scripts/
├── data/              # Symlink to Drive dataset
│   ├── cnie_pairs/    # Raw images
│   └── processed/     # Train/val/test splits (created by prepare_dataset.py)
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

## Comparison with Other Approaches

| Feature | `colab_workflow.sh` | Git Sync | SSH Remote |
|---------|---------------------|----------|------------|
| **Commands** | One script | Multiple | Multiple |
| **Speed** | Fast (delta rsync) | Push+Pull | Network latency |
| **Large files** | Efficient | Slow | Slow |
| **Version control** | No (rsync) | Yes (git) | No |
| **IDE experience** | Local (fast) | Local (fast) | Remote (laggy) |
| **Setup complexity** | Low | Medium | High |

---

## Troubleshooting

### Check Colab Status
```bash
./scripts/colab_workflow.sh colab-gpu status
```

### Test SSH Connection
```bash
ssh colab-gpu "echo 'Connected!'"
```

### Re-initialize
```bash
./scripts/colab_workflow.sh colab-gpu init
```

### Dataset Not Found
```bash
# Check Drive is mounted
./scripts/colab_workflow.sh colab-gpu shell
ls /content/drive/MyDrive/retin-verify/data/
```

---

## Related Documents

- `TRAINING_WORKFLOW_GUIDE.md` - **Complete workflow guide**
- `VS_CODE_SCRIPT_WORKFLOW.md` - Script-based development

### Helper Scripts

| Script | Purpose |
|--------|---------|
| `colab_workflow.sh` | Master workflow orchestrator |
| `deploy.sh` | Low-level rsync and SSH execution |
| `deploy_ssh_key.sh` | Deploy SSH key for passwordless auth |
| `prepare_dataset.py` | Prepare train/val/test splits from raw data |
| `prepare_dataset_bg.sh` | Run dataset preparation in background (SSH) |
| `export_artifacts.py` | Export models/checkpoints/logs to Drive |
| `run_training.py` | Universal training launcher |

---

✅ New Master Script: colab_workflow.sh
Single entry point for all operations:

# Initialize new session
./scripts/colab_workflow.sh colab-gpu init

# Full workflow: sync → train → export → pull
./scripts/colab_workflow.sh colab-gpu full --epochs 100 --fp16

# Individual commands
./scripts/colab_workflow.sh colab-gpu sync      # Deploy code
./scripts/colab_workflow.sh colab-gpu setup     # Setup dataset
./scripts/colab_workflow.sh colab-gpu train     # Train only
./scripts/colab_workflow.sh colab-gpu export    # Export artifacts
./scripts/colab_workflow.sh colab-gpu pull      # Pull results
./scripts/colab_workflow.sh colab-gpu status    # Check status
./scripts/colab_workflow.sh colab-gpu shell     # Interactive shell
📁 Active Scripts (Cleaned Up)
Script	Purpose
colab_workflow.sh	Master workflow orchestrator (NEW)
deploy.sh	Low-level sync and SSH execution
deploy_ssh_key.sh	Deploy SSH key for passwordless auth
export_artifacts.py	Export models/checkpoints to Drive
run_training.py	Universal training launcher
prepare_dataset.py	Local dataset preparation
sync_to_colab.sh	Git-based sync (alternative)
generate_placeholder_templates.py	Template generation
🗄️ Archived Scripts
Moved to scripts/archive/:

setup_colab_dataset.py - Superseded by workflow
setup_colab_dataset_ssh.py - Integrated into workflow
mount_and_setup.sh - Integrated into workflow
init_colab.py - Superseded by colab_workflow.sh init
setup_colab_ssh.py - Superseded by deploy_ssh_key.sh
And others...
📚 Updated Documentation
TRAINING_WORKFLOW_GUIDE.md - Simplified with new workflow
REMOTE_DEPLOYMENT_WORKFLOW.md - Updated for master script
KIMI_CODE.md - Updated quick start
ARCHIVED_SCRIPTS.md - Complete archive documentation

## 🚀 New Simplified Workflow

```bash
# 1. In Colab notebook - Start SSH and mount Drive:
#    !pip install colab_ssh -q
#    from colab_ssh import launch_ssh_cloudflared
#    launch_ssh_cloudflared(password="your_password")
#    from google.colab import drive
#    drive.mount('/content/drive')
#
#    Copy the trycloudflare.com URL and update ~/.ssh/config

# 2. Deploy SSH key (REQUIRED for each new Colab session!)
./scripts/deploy_ssh_key.sh colab-gpu

# 3. Initialize
./scripts/colab_workflow.sh colab-gpu init

# 4. Setup dataset
./scripts/colab_workflow.sh colab-gpu setup

# 5. Develop with Kimi
kimi --continue

# 6. Full training workflow (one command!)
./scripts/colab_workflow.sh colab-gpu full --epochs 100 --fp16
```


------------
**Document Version**: 2.1  
**Last Updated**: 2026-03-16
