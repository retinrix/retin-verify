# v4 Training Quick Reference

## One-Command Deployment

```bash
source ~/retin-verify/.kimi/scripts/load_training_automation.sh && deploy_v4_training YOUR_HOST.trycloudflare.com
```

## Manual Steps

### 1. On Colab (run these cells):

```python
# Cell 1: SSH
!pip install colab_ssh --quiet
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="retinrix")

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

### 2. On Local Machine:

```bash
HOST="your-host.trycloudflare.com"
PASS="retinrix"

# Upload dataset
sshpass -p "$PASS" scp ~/retin-verify/data/processed/classification/dataset_3class_v4.tar.gz root@$HOST:/tmp/

# Upload script
sshpass -p "$PASS" scp ~/retin-verify/training/classification/v4_training/train_v4_enhanced.py root@$HOST:/tmp/

# Setup and start
sshpass -p "$PASS" ssh root@$HOST "
  mkdir -p /content/retin_v4
  cd /content/retin_v4
  tar -xzf /tmp/dataset_3class_v4.tar.gz
  cp /tmp/train_v4_enhanced.py .
  nohup python3 train_v4_enhanced.py > train_v4.log 2>&1 &
"
```

### 3. Monitor:

```bash
sshpass -p "$PASS" ssh root@$HOST 'tail -f /content/retin_v4/train_v4.log'
```

### 4. Download (when done):

```bash
sshpass -p "$PASS" scp root@$HOST:/content/retin_v4/cnie_classifier_3class_v4.pth \
  ~/retin-verify/models/classification/
```

### 5. Update API:

```bash
pkill -f api_server.py
cd ~/retin-verify/apps/classification/backend
sed -i 's/v[0-9]\.pth/v4.pth/g' api_server.py
nohup python3 api_server.py > /tmp/api_server.log 2>&1 &
```

## Key Files

| Purpose | Path |
|---------|------|
| Automation | `~/.kimi/scripts/colab_v4_automation.sh` |
| Loader | `~/.kimi/scripts/load_training_automation.sh` |
| Full Guide | `~/retin-verify/docs/automation/COLAB_V4_TRAINING_PROCEDURE.md` |
