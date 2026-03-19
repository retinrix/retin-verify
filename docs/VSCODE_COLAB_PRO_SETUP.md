# VS Code + Colab Pro Hybrid Development Setup

## Overview

This document describes the ideal hybrid setup for Retin-Verify development:
- **Local**: VS Code + Kimi Agent for development
- **Cloud**: Google Colab Pro for GPU training
- **Storage**: Google Drive for dataset and model persistence

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HYBRID DEVELOPMENT ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐          ┌─────────────────────────────────┐  │
│  │   LOCAL MACHINE  │          │      GOOGLE COLAB PRO           │  │
│  │                  │          │                                 │  │
│  │  ┌──────────┐   │  SSH/    │  ┌──────────┐  ┌─────────────┐ │  │
│  │  │ VS Code  │◄──┼──╳───────┼──►│  Jupyter │  │  V100 GPU   │ │  │
│  │  │ + Kimi   │   │  Port    │   │  Server  │  │  32GB RAM   │ │  │
│  │  │  Agent   │   │ Forward  │   └────┬─────┘  └─────────────┘ │  │
│  │  └────┬─────┘   │          │        │                        │  │
│  │       │         │          │   ┌────▼─────┐                  │  │
│  │  Edit │ Code    │          │   │ Training │                  │  │
│  │       │         │          │   │ Scripts  │                  │  │
│  │  └────▼─────┐   │          │   └────┬─────┘                  │  │
│  │  │  Git    │   │          │        │                        │  │
│  │  │  Repo   │   │          └────────┼────────────────────────┘  │
│  │  └────┬─────┘   │                   │                           │
│  │       │         │                   │                           │
│  │       └─────────┼───────────────────┘                           │
│  │                 │                                               │
│  │            ┌────▼─────────────────┐                             │
│  │            │   GOOGLE DRIVE       │                             │
│  │            │  ┌───────────────┐   │                             │
│  │            │  │ cnie_dataset  │   │                             │
│  │            │  │   (10K pairs) │   │                             │
│  │            │  ├───────────────┤   │                             │
│  │            │  │ models/       │   │                             │
│  │            │  │ ├──classification│                             │
│  │            │  │ ├──detection  │   │                             │
│  │            │  │ └──extraction │   │                             │
│  │            │  ├───────────────┤   │                             │
│  │            │  │ checkpoints/  │   │                             │
│  │            │  └───────────────┘   │                             │
│  │            └─────────────────────┘                             │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Google Drive Setup](#google-drive-setup)
3. [Colab Pro Setup](#colab-pro-setup)
4. [VS Code Integration](#vscode-integration)
5. [Kimi Agent Configuration](#kimi-agent-configuration)
6. [Development Workflow](#development-workflow)
7. [Training Workflow](#training-workflow)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts
- [x] Google Account (for Drive + Colab)
- [x] GitHub Account (for code repository)
- [x] Colab Pro Subscription ($10/month)

### Local Software
- [x] VS Code with Python extension
- [x] Git
- [x] Python 3.8+
- [x] SSH client (built-in for macOS/Linux, PuTTY for Windows)

---

## Google Drive Setup

### Step 1: Create Project Structure in Drive

```
My Drive/
└── retin-verify/
    ├── README.md
    ├── data/
    │   └── cnie_dataset_10k/           # Dataset (mounted to Colab)
    │       ├── cnie_pairs/
    │       └── dataset_manifest.json
    ├── models/                          # Trained models
    │   ├── classification/
    │   ├── detection/
    │   └── extraction/
    ├── checkpoints/                     # Training checkpoints
    ├── notebooks/                       # Jupyter notebooks
    ├── configs/                         # Training configs
    └── logs/                           # TensorBoard logs
```

### Step 2: Upload Dataset

**Option A: Browser Upload (Slow for large datasets)**
1. Go to [Google Drive](https://drive.google.com)
2. Create folder: `My Drive/retin-verify/data/`
3. Upload `cnie_dataset_10k` folder

**Option B: rclone (Recommended for large datasets)**

```bash
# Install rclone
# macOS
brew install rclone

# Linux
curl https://rclone.org/install.sh | sudo bash

# Windows - download from https://rclone.org/downloads/

# Configure rclone for Google Drive
rclone config
# Follow prompts to create "gdrive" remote

# Upload dataset
rclone copy data/cnie_dataset_10k gdrive:retin-verify/data/cnie_dataset_10k \
    --progress --transfers 8
```

**Option C: Google Drive for Desktop**
1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
2. Copy dataset to `Google Drive/retin-verify/data/`
3. Let it sync

### Step 3: Verify Upload

```python
# In a Colab notebook, verify access
from google.colab import drive
drive.mount('/content/drive')

import os
print(os.listdir('/content/drive/MyDrive/retin-verify/data'))
```

---

## Colab Pro Setup

### Step 1: Upgrade to Colab Pro

1. Go to [Colab](https://colab.research.google.com)
2. Click "Upgrade" in top right
3. Select "Colab Pro" ($9.99/month)
4. Complete payment

### Step 2: Configure Runtime

```python
# Add to beginning of every notebook

# Check GPU
!nvidia-smi

# Expected output: Tesla V100 or T4 (not K80)

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify mount
!ls -la /content/drive/MyDrive/retin-verify/
```

### Step 3: Install Dependencies on Colab

Create a setup cell at the beginning of notebooks:

```python
%%capture
# Install dependencies
!pip install -q torch torchvision transformers datasets accelerate
!pip install -q paddlepaddle-gpu paddleocr
!pip install -q opencv-python Pillow scikit-image albumentations
!pip install -q tqdm pyyaml python-dateutil
!pip install -q tensorboard wandb

# Install Colab-specific tools
!pip install -q colab_ssh ngrok

# Clone repository if not exists
import os
if not os.path.exists('/content/retin-verify'):
    !git clone https://github.com/YOUR_USERNAME/retin-verify.git /content/retin-verify
    
%cd /content/retin-verify
!pip install -q -e .
```

---

## VS Code Integration

### Method 1: VS Code + Colab via SSH (Recommended)

This allows you to use VS Code's full features while running code on Colab's GPU.

#### Step 1: Set up SSH tunnel to Colab

Create a Colab notebook `setup_ssh.ipynb`:

```python
# Install ngrok
!pip install colab_ssh

from colab_ssh import launch_ssh
import getpass

# Get ngrok token from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok_token = getpass.getpass("Enter ngrok token: ")

# Launch SSH server
launch_ssh(ngrok_token, password="your_ssh_password")

# Keep cell running
import time
while True:
    time.sleep(60)
```

#### Step 2: Configure VS Code Remote-SSH

1. Install VS Code extension: **Remote - SSH**

2. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)

3. Select: `Remote-SSH: Add New SSH Host`

4. Enter SSH connection string from Colab output (looks like):
   ```
   ssh root@0.tcp.ngrok.io -p 12345
   ```

5. Select configuration file: `~/.ssh/config`

6. Edit `~/.ssh/config`:
   ```
   Host colab-gpu
       HostName 0.tcp.ngrok.io
       User root
       Port 12345
       StrictHostKeyChecking no
       UserKnownHostsFile /dev/null
   ```

7. Connect: `Remote-SSH: Connect to Host` → `colab-gpu`

#### Step 3: Open Project on Colab

Once connected:
1. File → Open Folder → `/content/retin-verify`
2. Install Python extension on remote
3. Select Python interpreter: `/usr/bin/python3`

### Method 2: VS Code + Jupyter (Alternative)

Use VS Code to connect directly to Colab's Jupyter server.

#### Step 1: Get Colab Jupyter URL

In a Colab notebook:

```python
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8888)"))
```

#### Step 2: VS Code Jupyter Connection

1. Install VS Code extension: **Jupyter**

2. Create file `.vscode/settings.json`:
   ```json
   {
       "jupyter.jupyterServerType": "existing",
       "jupyter.serverType": "Remote",
       "jupyter.remoteType": "Existing"
   }
   ```

3. Use Command Palette: `Jupyter: Specify local or remote Jupyter server`

4. Enter URL from Step 1

### Method 3: ColabCode (Simplest, Less Features)

```python
# In Colab
!pip install colabcode

from colabcode import ColabCode
ColabCode(port=10000, password="your_password")
```

Then open the provided URL in browser.

---

## Kimi Agent Configuration

### Current Setup Analysis

You currently use Kimi Agent in VS Code locally. For the hybrid setup:

```
LOCAL (VS Code + Kimi Agent)
    ↓
Edit code, write configs, manage project
    ↓
Push to GitHub
    ↓
COLAB PRO (via SSH from VS Code)
    ↓
Pull code, access Drive data, train on GPU
```

### Configuration for Kimi Agent on Hybrid Setup

#### Option A: Kimi edits locally, you manually sync to Colab

Keep Kimi agent running locally on your machine. When Kimi makes changes:

```bash
# After Kimi edits
# 1. Review changes locally
git diff

# 2. Commit and push
git add .
git commit -m "Kimi updates"
git push

# 3. On Colab (via VS Code SSH), pull changes
git pull

# 4. Run training
python training/classification/train.py
```

#### Option B: Kimi on Colab via VS Code Remote (Advanced)

1. Connect VS Code to Colab via SSH (Method 1 above)
2. Kimi agent will automatically work on the remote Colab machine
3. All edits happen directly on Colab with GPU access

**Pros:**
- Kimi can test code immediately on GPU
- No sync delay
- Direct access to training data

**Cons:**
- Requires stable SSH connection
- Colab runtime may disconnect (12-24h limit)

#### Recommended: Option A with Automation

Create helper scripts to minimize sync friction:

```bash
#!/bin/bash
# scripts/sync_to_colab.sh

# Push local changes
git add .
git commit -m "Sync to Colab - $(date)"
git push

echo "Changes pushed. On Colab, run: git pull"
```

```python
# training/utils/colab_sync.py
"""Auto-sync utilities for Colab."""

import subprocess
import os

def sync_from_github():
    """Pull latest changes from GitHub."""
    result = subprocess.run(
        ['git', 'pull'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    return result.returncode == 0

def check_drive_mounted():
    """Verify Google Drive is mounted."""
    return os.path.exists('/content/drive/MyDrive')

def setup_colab_environment():
    """One-click setup for Colab."""
    # Mount drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Sync code
    sync_from_github()
    
    # Verify data
    if not check_drive_mounted():
        raise RuntimeError("Drive not mounted!")
    
    data_path = '/content/drive/MyDrive/retin-verify/data/cnie_dataset_10k'
    if not os.path.exists(data_path):
        raise RuntimeError(f"Data not found at {data_path}")
    
    print("✅ Colab environment ready!")
    print(f"📁 Data: {data_path}")
    print(f"💻 GPU: {subprocess.getoutput('nvidia-smi -L')}")
```

---

## Development Workflow

### Daily Development Loop

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LOCAL: Open VS Code, Kimi Agent ready                    │
│                                                             │
│ 2. LOCAL: Edit code, configs, notebooks                     │
│    - Kimi helps with code changes                           │
│    - Local linting, type checking                           │
│                                                             │
│ 3. LOCAL: Test logic (small scale, CPU)                     │
│    pytest tests/unit/                                       │
│                                                             │
│ 4. LOCAL → GITHUB: Commit & push                            │
│    git push origin main                                     │
│                                                             │
│ 5. COLAB: Pull & train                                      │
│    - Start Colab runtime                                    │
│    - SSH connect from VS Code (optional)                    │
│    - Pull latest code                                       │
│    - Start training                                         │
│                                                             │
│ 6. COLAB → DRIVE: Save checkpoints                          │
│    - Auto-save to Drive every epoch                         │
│    - TensorBoard logs synced                                │
│                                                             │
│ 7. LOCAL: Monitor & analyze                                 │
│    - TensorBoard (local or Colab)                           │
│    - Download best model from Drive                         │
└─────────────────────────────────────────────────────────────┘
```

### Project Initialization Script

Create `scripts/init_colab.py`:

```python
#!/usr/bin/env python3
"""Initialize Colab environment for Retin-Verify."""

import os
import subprocess
import sys

def main():
    print("🚀 Initializing Retin-Verify on Colab Pro...")
    
    # 1. Check GPU
    print("\n1. Checking GPU...")
    gpu_info = subprocess.getoutput('nvidia-smi -L')
    if 'V100' in gpu_info or 'T4' in gpu_info or 'A100' in gpu_info:
        print(f"✅ GPU: {gpu_info}")
    else:
        print("⚠️  Warning: No GPU detected or using K80")
        print("   Go to Runtime → Change runtime type → GPU")
    
    # 2. Mount Drive
    print("\n2. Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    
    # 3. Clone/Update repo
    print("\n3. Setting up code repository...")
    repo_path = '/content/retin-verify'
    if os.path.exists(repo_path):
        os.chdir(repo_path)
        subprocess.run(['git', 'pull'], check=True)
        print("✅ Repository updated")
    else:
        subprocess.run([
            'git', 'clone', 
            'https://github.com/YOUR_USERNAME/retin-verify.git',
            repo_path
        ], check=True)
        os.chdir(repo_path)
        print("✅ Repository cloned")
    
    # 4. Install dependencies
    print("\n4. Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'])
    print("✅ Dependencies installed")
    
    # 5. Verify data
    print("\n5. Verifying dataset...")
    data_path = '/content/drive/MyDrive/retin-verify/data/cnie_dataset_10k'
    if os.path.exists(data_path):
        pairs = len([d for d in os.listdir(data_path + '/cnie_pairs') 
                     if os.path.isdir(data_path + '/cnie_pairs/' + d)])
        print(f"✅ Dataset found: {pairs} pairs")
    else:
        print(f"❌ Dataset not found at {data_path}")
        print("   Upload dataset to Google Drive first")
        return 1
    
    # 6. Setup symlinks for convenience
    print("\n6. Creating symlinks...")
    os.symlink(data_path, '/content/data', ignore_errors=True)
    print("✅ Symlink: /content/data → Drive dataset")
    
    print("\n" + "="*50)
    print("🎉 Colab environment ready!")
    print("="*50)
    print("\nQuick commands:")
    print("  %cd /content/retin-verify")
    print("  python training/classification/train.py --help")
    print("  tensorboard --logdir training/")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
```

---

## Training Workflow

### Training Script for Colab

Create `notebooks/colab_training_template.ipynb`:

```python
# Cell 1: Initialization
%cd /content
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/retin-verify/main/scripts/init_colab.py
!python init_colab.py

# Cell 2: Configuration
EXPERIMENT_NAME = "efficientnet_b0_v1"
MODEL_TYPE = "classification"  # or "extraction", "detection"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Cell 3: Training
%cd /content/retin-verify

from training.classification.train import ClassificationTrainer

# Setup paths
output_dir = f'/content/drive/MyDrive/retin-verify/models/{MODEL_TYPE}/{EXPERIMENT_NAME}'

# Initialize trainer
trainer = ClassificationTrainer(
    model_dir=output_dir,
    num_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    device='cuda'
)

# Create datasets
train_dataset = DocumentDataset(
    '/content/data',
    '/content/data/processed/classification/train.json',
    transform=trainer.get_transforms(is_training=True)
)

val_dataset = DocumentDataset(
    '/content/data',
    '/content/data/processed/classification/val.json',
    transform=trainer.get_transforms(is_training=False)
)

# Train
trainer.train(train_dataset, val_dataset)

# Cell 4: Save to Drive
!cp -r training/{MODEL_TYPE}/checkpoints/* {output_dir}/
print(f"✅ Model saved to: {output_dir}")

# Cell 5: TensorBoard
%load_ext tensorboard
%tensorboard --logdir training/{MODEL_TYPE}/logs
```

### Monitoring Training

**Option 1: TensorBoard in Colab**
```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/retin-verify/logs
```

**Option 2: TensorBoard locally (synced from Drive)**
```bash
# Local terminal
rclone mount gdrive:retin-verify/logs ./logs &
tensorboard --logdir ./logs
```

**Option 3: Weights & Biases (cloud logging)**
```python
# In training script
import wandb
wandb.init(project="retin-verify", name=EXPERIMENT_NAME)
```

---

## Troubleshooting

### Common Issues

#### 1. Colab Disconnects During Training

**Solution:**
- Use `colab_ssh` for persistent connection
- Enable auto-save checkpoints every epoch
- Use smaller epochs with resume capability

```python
# Auto-save callback
from training.utils.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    output_dir='/content/drive/MyDrive/retin-verify/checkpoints',
    save_every=1  # Save every epoch
)
```

#### 2. Drive Mount Fails

```python
# Re-mount with force
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

#### 3. Out of Memory (OOM)

```python
# Reduce batch size
BATCH_SIZE = 4  # or 2, or 1

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

#### 4. SSH Connection Drops

```python
# In Colab, keep-alive script
import time
while True:
    time.sleep(60)
    print("Keeping alive...", flush=True)
```

Or use `screen`/`tmux` on Colab:
```bash
!apt-get install -y screen
!screen -S training -dm python train.py
```

#### 5. Dataset Path Issues

```python
# Always use absolute paths
data_path = '/content/drive/MyDrive/retin-verify/data/cnie_dataset_10k'

# Verify exists
assert os.path.exists(data_path), f"Data not found: {data_path}"
```

### Performance Tips

| Issue | Solution |
|-------|----------|
| Slow data loading | Enable `pin_memory=True`, increase `num_workers` |
| Drive I/O bottleneck | Copy data to local `/content` first |
| Long epoch times | Use mixed precision (fp16) |
| Memory errors | Reduce batch size, enable gradient accumulation |

### Quick Reference Commands

```bash
# Check GPU
!nvidia-smi

# Monitor resources
!watch -n 1 nvidia-smi

# Check disk space
!df -h

# Check RAM
!free -h

# Kill hanging processes
!pkill -f python

# Clear GPU memory
import torch
torch.cuda.empty_cache()
```

---

## Security Considerations

### Protecting Your Data

1. **Don't commit sensitive data to GitHub**
   ```bash
   # Add to .gitignore
   echo "data/" >> .gitignore
   echo "models/" >> .gitignore
   echo "*.json" >> .gitignore  # If contains PII
   ```

2. **Use Drive for datasets and models only**

3. **Keep ngrok tokens secret**
   ```python
   # Use getpass, don't hardcode
   import getpass
   token = getpass.getpass("Ngrok token: ")
   ```

### Access Control

- **Drive**: Private by default
- **Colab**: Your runtime only
- **GitHub**: Private repository recommended

---

## Summary Checklist

### Initial Setup (One-time)

- [ ] Upgrade to Colab Pro
- [ ] Upload dataset to Google Drive
- [ ] Set up ngrok account
- [ ] Configure SSH in VS Code
- [ ] Test SSH connection to Colab
- [ ] Clone repo to Colab
- [ ] Verify Kimi agent works (locally or on Colab)

### Daily Workflow

- [ ] Start Colab runtime
- [ ] (Optional) Connect VS Code via SSH
- [ ] Pull latest code: `git pull`
- [ ] Mount Drive
- [ ] Run training
- [ ] Monitor via TensorBoard
- [ ] Download results from Drive

---

## Resources

- [Colab Pro FAQ](https://colab.research.google.com/notebooks/colab-pro.ipynb)
- [VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)
- [ngrok Documentation](https://ngrok.com/docs)
- [rclone Google Drive](https://rclone.org/drive/)

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-14  
**Author**: Retin-Verify Team
