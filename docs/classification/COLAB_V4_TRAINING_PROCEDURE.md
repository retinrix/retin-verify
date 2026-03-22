# Colab v4 Training Automation Procedure

This document describes the complete automated pipeline for training the v4 CNIE classifier on Google Colab.

## Overview

The v4 training process automates:
1. SSH key deployment to Colab
2. Dataset and script upload
3. Training initiation
4. Progress monitoring
5. Model download
6. Local API (update)
7. Restart server (back end)

## Prerequisites

1. **Colab notebook running** with:
   ```python
   !pip install colab_ssh --quiet
   from colab_ssh import launch_ssh_cloudflared
   launch_ssh_cloudflared(password="retinrix")
   ```

2. **Google Drive mounted**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Synthetic data available** at:
   `MyDrive/retin-verify/data/cnie_dataset_10k`

## Quick Start

### Option 1: Load Automation Tools

```bash
source ~/retin-verify/.kimi/scripts/load_training_automation.sh
```

Then use:
```bash
deploy_v4_training <hostname> [password]
```

### Option 2: Direct Script Execution

```bash
cd ~/retin-verify/.kimi/scripts
./colab_v4_automation.sh <hostname> [password]
```

Example:
```bash
./colab_v4_automation.sh abc123.trycloudflare.com retinrix
```

## Step-by-Step Procedure

### Step 1: Prepare Colab Environment

In your Colab notebook:

```python
# Cell 1: Install and start SSH
!pip install colab_ssh --quiet
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="retinrix")

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Keep alive (run in separate cell)
import time
while True:
    time.sleep(60)
    print("Active", flush=True)
```

**Copy the hostname** (e.g., `abc123.trycloudflare.com`)

### Step 2: Deploy SSH Key

```bash
HOST="abc123.trycloudflare.com"
PASSWORD="retinrix"

# Deploy public key
cat ~/.ssh/id_rsa.pub | sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
    root@$HOST "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### Step 3: Upload Dataset

```bash
# Dataset is pre-packaged at:
LOCAL_DATASET="$HOME/retin-verify/data/processed/classification/dataset_3class_v4.tar.gz"

# Upload to Colab
sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no \
    "$LOCAL_DATASET" root@$HOST:/tmp/
```

### Step 4: Upload Training Script

```bash
LOCAL_SCRIPT="$HOME/retin-verify/training/classification/v4_training/train_v4_enhanced.py"

sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no \
    "$LOCAL_SCRIPT" root@$HOST:/tmp/
```

### Step 5: Setup and Start Training

```bash
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no root@$HOST << 'REMOTECMD'
mkdir -p /content/retin_v4
cd /content/retin_v4
tar -xzf /tmp/dataset_3class_v4.tar.gz
cp /tmp/train_v4_enhanced.py .
nohup python3 train_v4_enhanced.py > train_v4.log 2>&1 &
echo "Training started with PID: $(pgrep -f train_v4_enhanced)"
REMOTECMD
```

### Step 6: Monitor Training

```bash
# Watch live logs
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no \
    root@$HOST 'tail -f /content/retin_v4/train_v4.log'
```

**Expected log output:**
```
E01/100 | Loss: 1.234 | Train: 45.2% (Real: 42.1%) | Val: 48.5% | Balance: 47.3%
E02/100 | Loss: 0.987 | Train: 52.1% (Real: 51.3%) | Val: 55.2% | Balance: 54.1%
...
E50/100 | Loss: 0.234 | Train: 94.2% (Real: 93.1%) | Val: 91.5% | Balance: 90.2%
```

### Step 7: Download Model

When training completes:

```bash
MODEL_NAME="cnie_classifier_3class_v4.pth"
LOCAL_DIR="$HOME/retin-verify/models/classification"

sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no \
    root@$HOST:/content/retin_v4/$MODEL_NAME \
    "$LOCAL_DIR/"
```

### Step 8: Update Local API

```bash
# Kill existing server
pkill -f api_server.py

# Update model path in api_server.py
cd ~/retin-verify/apps/classification/backend
sed -i 's/cnie_classifier_3class_v[0-9]*.pth/cnie_classifier_3class_v4.pth/g' api_server.py

# Restart server
nohup python3 api_server.py > /tmp/api_server.log 2>&1 &

# Verify
curl -s http://localhost:8000/info | python3 -m json.tool
```

### Step 9: Test

Open browser:
- Web UI: `http://localhost:8000`
- Health: `http://localhost:8000/health`

Test with sample images and verify predictions.

## Automation Functions

When you load the automation tools:

```bash
source ~/retin-verify/.kimi/scripts/load_training_automation.sh
```

The following functions become available:

| Function | Description |
|----------|-------------|
| `deploy_v4_training <host>` | Full deployment pipeline |
| `check_training_status <host>` | Check training progress |
| `download_v4_model <host>` | Download completed model |
| `update_local_api_v4` | Update local API to v4 |

## File Locations

| File | Path |
|------|------|
| Automation script | `~/retin-verify/.kimi/scripts/colab_v4_automation.sh` |
| Session loader | `~/retin-verify/.kimi/scripts/load_training_automation.sh` |
| Training script | `~/retin-verify/training/classification/v4_training/train_v4_enhanced.py` |
| Dataset archive | `~/retin-verify/data/processed/classification/dataset_3class_v4.tar.gz` |
| Model output | `~/retin-verify/models/classification/cnie_classifier_3class_v4.pth` |

## Troubleshooting

### Connection Timeout

If SSH connection fails:
1. Verify Colab cell is still running
2. Check hostname hasn't changed
3. Restart tunnel if needed

### Google Drive Not Mounted

```bash
# On Colab, run:
from google.colab import drive
drive.mount('/content/drive')
```

### Training Not Starting

Check logs:
```bash
sshpass -p "$PASSWORD" ssh root@$HOST 'cat /content/retin_v4/train_v4.log'
```

### Model Not Downloading

Verify training completed:
```bash
sshpass -p "$PASSWORD" ssh root@$HOST 'ls -la /content/retin_v4/*.pth'
```

## Expected Timeline

| Step | Duration |
|------|----------|
| Upload dataset | 2-3 min |
| Upload script | < 1 min |
| Training (100 epochs) | 6-8 hours |
| Download model | 1-2 min |
| Update API | < 1 min |
| **Total** | **~8 hours** |

## v4 Improvements

| Metric | v3 | v4 Target |
|--------|-----|-----------|
| Epochs | 50 | 100 |
| Real weight | 2x | 3x |
| Class weights | Equal | Front/Back 1.2x |
| LR scheduler | None | ReduceLROnPlateau |
| Augmentation | Basic | Enhanced |
| Expected balance | 88% | 93%+ |

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-19  
**Author:** Kimi Code CLI
