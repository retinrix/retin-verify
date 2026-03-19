# v3 Training Deployment Guide

## Prerequisites

1. **Colab Pro/GPU Runtime** - Training requires GPU (T4 recommended)
2. **SSH Tunnel** - Cloudflare or ngrok tunnel to Colab
3. **SSH Key** - Configured SSH access to Colab
4. **~5GB Upload** - Real dataset (small) + Synthetic dataset (sampled to 10K)

---

## Step 1: Start Colab and Create Tunnel

### In your Colab notebook:

```python
# Install cloudflared
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# Start tunnel
import subprocess
import threading
import time

def run_tunnel():
    subprocess.run(['./cloudflared', 'tunnel', '--url', 'ssh://localhost:22'])

thread = threading.Thread(target=run_tunnel)
thread.start()
time.sleep(5)
```

**Copy the hostname** (e.g., `trying-rapidly-seal.ngrok-free.app` or `abc123.trycloudflare.com`)

---

## Step 2: Deploy from Local Machine

### Option A: Use the deploy.sh script (Recommended)

```bash
cd ~/retin-verify/training/classification/new_training

# Deploy with your host
./deploy.sh YOUR_HOST_HERE

# Example:
./deploy.sh abc123.trycloudflare.com
```

### Option B: Edit and run Python script

```bash
# Edit the host in the script
nano deploy_v3_with_synthetic.py
# Change: HOST = "YOUR_COLAB_HOST_HERE"
# To:     HOST = "abc123.trycloudflare.com"

# Run deployment
python3 deploy_v3_with_synthetic.py
```

### Option C: Direct command

```bash
cd ~/retin-verify/training/classification/new_training

# Set host and run
HOST="abc123.trycloudflare.com" python3 deploy_v3_with_synthetic.py
```

---

## Step 3: Monitor Training

```bash
# Watch training progress
ssh root@YOUR_HOST "tail -f /content/retin_v3_synthetic/train_v3_synthetic.log"

# Or check periodically
ssh root@YOUR_HOST "tail -50 /content/retin_v3_synthetic/train_v3_synthetic.log"
```

**Expected output:**
```
Loading datasets...
  train: 10306 total images
    Real: 306
    Synthetic: 10000
...
E01/50 | Train(Real): 45%/52%/48% | Val: 48%/55%/47% | Bal:47%
E02/50 | Train(Real): 55%/62%/58% | Val: 58%/65%/60% | Bal:58%
...
E25/50 | Train(Real): 92%/94%/95% | Val: 90%/91%/93% | Bal:90% <- Saved!
```

---

## Step 4: Download Model (When Training Completes)

```bash
# Download best model
scp root@YOUR_HOST:/content/cnie_classifier_3class_v3_synthetic.pth \
    ~/retin-verify/models/classification/

# Download training history
scp root@YOUR_HOST:/content/training_history_v3_synthetic.json \
    ~/retin-verify/models/classification/
```

---

## Troubleshooting

### SSH Connection Failed

```bash
# Test SSH
ssh root@YOUR_HOST "echo 'Hello'"

# If fails, check:
# 1. Is Colab still running?
# 2. Is the tunnel still active?
# 3. Is the hostname correct?
```

### Upload Takes Too Long

The synthetic dataset is large. The script samples 10K images which takes ~5-10 minutes to package and upload.

### Training Crashes

```bash
# Check GPU availability on Colab
ssh root@YOUR_HOST "nvidia-smi"

# Check disk space
ssh root@YOUR_HOST "df -h"

# Restart training if needed
ssh root@YOUR_HOST "cd /content/retin_v3_synthetic && python3 train_with_synthetic.py"
```

---

## Expected Timeline

| Step | Time |
|------|------|
| Package real dataset | 30 seconds |
| Sample & package synthetic | 5-10 minutes |
| Upload to Colab | 10-15 minutes |
| Training (50 epochs) | 3-4 hours |
| **Total** | **~4 hours** |

---

## What Happens During Deployment

1. **Package Real Dataset** (306 images)
   - Located in `data/processed/classification/dataset_3class/`
   - Packaged as `real_dataset.tar.gz`

2. **Sample Synthetic Dataset** (10K from 16K)
   - Located in `data/cnie_dataset_10k/cnie_pairs/`
   - Sampled to keep training time reasonable
   - Packaged as `synthetic_sample.tar.gz`

3. **Upload to Colab**
   - Both archives uploaded via SCP
   - Extracted in `/content/retin_v3_synthetic/`

4. **Start Training**
   - Training runs in background with `nohup`
   - Logs to `train_v3_synthetic.log`
   - Model saved automatically when balance improves

---

## Files Created on Colab

```
/content/retin_v3_synthetic/
├── dataset_3class/              # Real dataset (extracted)
│   ├── train/
│   └── val/
├── synthetic_cnie/              # Synthetic dataset (extracted)
│   ├── 000000/
│   │   ├── front/
│   │   └── back/
│   └── ...
├── train_with_synthetic.py      # Training script
├── train_v3_synthetic.log       # Training log
└── ...

/content/
├── cnie_classifier_3class_v3_synthetic.pth  # Best model
└── training_history_v3_synthetic.json       # Training history
```

---

## Next Steps After Deployment

1. **Monitor** training progress
2. **Download** model when complete
3. **Test** locally with validation set
4. **Deploy** to production if accuracy >90%
5. **Save session**: `python3 ../../../.kimi/session_manager.py save "v3 Training" "in_progress" "Deployed to Colab" "Monitor and download model"`

---

## Quick Reference

```bash
# Deploy
./deploy.sh YOUR_HOST

# Monitor
ssh root@YOUR_HOST "tail -f /content/retin_v3_synthetic/train_v3_synthetic.log"

# Download
scp root@YOUR_HOST:/content/cnie_classifier_3class_v3_synthetic.pth \
    ~/retin-verify/models/classification/

# Test locally
cd ~/retin-verify/apps/classification
python3 backend/api_server.py --model ../../models/classification/cnie_classifier_3class_v3_synthetic.pth
```
