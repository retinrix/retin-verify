# SSH Setup Quick Start (5 Minutes)

## Goal
Connect VS Code directly to Google Colab's GPU via SSH.

---

## Step 1: Get ngrok Token (2 minutes)

1. Go to https://dashboard.ngrok.com/signup
2. Sign up (free)
3. Go to https://dashboard.ngrok.com/get-started/your-authtoken
4. **Copy your token** (looks like `2Kxzy...abcdef...`)

---

## Step 2: Run Setup in Colab (2 minutes)

Open Colab notebook, create new cell, paste and run:

```python
!pip install colab_ssh -q

from colab_ssh import launch_ssh
import getpass

# Enter your ngrok token (hidden for security)
token = getpass.getpass("ngrok token: ")
password = getpass.getpass("SSH password: ")

# Start SSH server
launch_ssh(token, password=password)

# Keep running
import time
while True:
    time.sleep(60)
    print("Active", flush=True)
```

**Copy the output** (looks like):
```
Successfully running 2.tcp.ngrok.io:12345
```

---

## Step 3: Configure VS Code (1 minute)

### 3a: Install Extension
1. VS Code → Extensions (Ctrl+Shift+X)
2. Search: `Remote - SSH`
3. Click **Install**

### 3b: Add SSH Host
1. Press `F1`
2. Type: `Remote-SSH: Add New SSH Host`
3. Enter: `ssh root@2.tcp.ngrok.io -p 12345`
   - Replace with your host:port from Step 2
4. Select: `~/.ssh/config`

### 3c: Connect
1. Press `F1`
2. Type: `Remote-SSH: Connect to Host`
3. Select: `colab-gpu`
4. Enter password (from Step 2)

---

## Done! 🎉

Bottom-left of VS Code shows: **SSH: colab-gpu**

### Open Project
1. File → Open Folder
2. Type: `/content/retin-verify`
3. Click OK

### Open Terminal
- `` Ctrl+` `` (backtick)
- This terminal runs on Colab GPU!

### Run Training
```bash
cd /content/retin-verify
python training/classification/train_cli.py \
    --data-dir /content/data \
    --epochs 50 \
    --device cuda
```

---

## Keep Alive

⚠️ **Important:** Keep Colab tab open and active!
- Don't close browser
- Don't let computer sleep
- Keep Colab cell running

---

## When Connection Drops

SSH will disconnect after 12-24 hours (Colab limit).

**To reconnect:**
1. Re-run Step 2 in Colab (get new host:port)
2. Update `~/.ssh/config` with new host:port
3. VS Code → Reconnect

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not establish connection" | Re-run Step 2, get new host:port |
| "Permission denied" | Check password, re-run setup |
| "Connection timeout" | Add to config: `ServerAliveInterval 60` |
| VS Code keeps asking for password | Check `~/.ssh/config` has correct host:port |
| Colab cell stopped | Re-run it, connection is lost |

---

## Quick Commands

| Action | Shortcut |
|--------|----------|
| Connect | `F1` → `Remote-SSH: Connect to Host` → `colab-gpu` |
| Open folder | `File` → `Open Folder` → `/content/retin-verify` |
| New terminal | `` Ctrl+` `` |
| Disconnect | `File` → `Close Remote Connection` |

---

## Alternative (No SSH)

Don't want to set up SSH? Use this instead:

```bash
# Local VS Code (edit code)
# ↓
./scripts/sync_to_colab.sh  # Push to GitHub
# ↓
# Colab browser tab (run training)
!git pull
!python training/classification/train_cli.py ...
```

---

**Full guide:** `docs/VS_CODE_COLAB_SSH_SETUP.md`
