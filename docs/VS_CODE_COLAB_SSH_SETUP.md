# VS Code to Colab SSH Connection Guide

Complete step-by-step guide to connect VS Code directly to Google Colab via SSH.

## Architecture

```
┌─────────────────┐      SSH Tunnel       ┌─────────────────┐
│   Your Laptop   │ ◄──────────────────►  │   Google Colab  │
│                 │    (via ngrok, cloudflare)              │                 │
│  ┌───────────┐  │                       │  ┌───────────┐  │
│  │  VS Code  │  │                       │  │  Python   │  │
│  │  (Local)  │  │                       │  │  Training │  │
│  └─────┬─────┘  │                       │  │  (GPU)    │  │
│        │        │                       │  └───────────┘  │
│  ┌─────▼─────┐  │                       │                 │
│  │   SSH     │  │                       │  ┌───────────┐  │
│  │  Client   │──┼───────────────────────┼──►  SSH      │  │
│  └───────────┘  │                       │     Server   │  │
└─────────────────┘                       └─────────────┘   │
                                                   │        │
                                            ┌──────▼────┐   │
                                            │  NVIDIA   │   │
                                            │   V100    │   │
                                            │  (16GB)   │   │
                                            └───────────┘   │
                                                            │
                                            ┌───────────┐   │
                                            │  Google   │   │
                                            │  Drive    │◄──┘
                                            │  (Data)   │
                                            └───────────┘
```

---

## Prerequisites

1. **VS Code** with extensions:
   - Remote - SSH
   - Remote - SSH: Editing Configuration Files

2. **ngrok account** (free):
   - Sign up: https://dashboard.ngrok.com/signup
   - Get authtoken: https://dashboard.ngrok.com/get-started/your-authtoken

3. **Colab Pro** (for stable GPU access)

---

## Step-by-Step Setup

### Step 1: Install VS Code Extensions

1. Open VS Code
2. Press `Ctrl+Shift+X` (Extensions)
3. Search and install:
   - `Remote - SSH` by Microsoft
   - `Remote - SSH: Editing Configuration Files`

### Step 2: Get ngrok Authtoken

1. Go to https://dashboard.ngrok.com/get-started/your-authtoken
2. Sign up / Log in
3. Copy your authtoken (looks like: `2Kxzy...abcdef...`)
4. **Keep it secret!** This is your private token.

### Step 3: Run Setup Script in Colab

Create a new Colab notebook and run:

```python
# Cell 1: Install colab_ssh
!pip install colab_ssh -q

# Cell 2: Run SSH setup
from colab_ssh import launch_ssh
import getpass

# Get your ngrok token (will be hidden for security)
print("Enter your ngrok authtoken (get it from https://dashboard.ngrok.com/get-started/your-authtoken)")
ngrok_token = getpass.getpass("Token: ")

# Launch SSH server
print("\n🔧 Starting SSH server...")
launch_ssh(ngrok_token, password="your_ssh_password")

print("\n✅ SSH server is running!")
print("Keep this cell running to maintain the connection.")
```

**Expected output:**
```
Successfully running 2.tcp.ngrok.io:12345
[Optional] For a better experience, install the pyngrok package.
✅ SSH server is running!
Keep this cell running to maintain the connection.
```

**Important:** 
- Copy the `2.tcp.ngrok.io:12345` part (your host and port)
- Keep this cell running! If it stops, SSH disconnects.

### Step 4: Configure SSH in VS Code

**Option A: Using VS Code GUI**

1. Press `Ctrl+Shift+P`
2. Type: `Remote-SSH: Add New SSH Host`
3. Enter SSH connection string:
   ```
   ssh root@2.tcp.ngrok.io -p 12345
   ```
   (Replace with your actual host and port from Step 3)

4. Select config file: `~/.ssh/config`

5. VS Code will add entry automatically

**Option B: Manual Config File Edit**

Edit `~/.ssh/config` (create if doesn't exist):

```bash
# Linux/Mac
nano ~/.ssh/config

# Windows (PowerShell)
notepad $env:USERPROFILE\.ssh\config
```

Add this content:

```
Host colab-gpu
    HostName 2.tcp.ngrok.io
    User root
    Port 12345
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

**Replace:**
- `2.tcp.ngrok.io` with your host from Step 3
- `12345` with your port from Step 3

### Step 5: Connect VS Code to Colab

1. Press `F1` or `Ctrl+Shift+P`
2. Type: `Remote-SSH: Connect to Host`
3. Select: `colab-gpu`

4. Enter password when prompted:
   ```
   root@2.tcp.ngrok.io's password: your_ssh_password
   ```

5. Wait for connection...

6. **Success!** Bottom-left shows:
   ```
   SSH: colab-gpu
   ```

### Step 6: Open Project Folder

1. File → Open Folder
2. Type: `/content/retin-verify`
3. Click OK

4. VS Code will install Python extension on remote (Colab)
5. Select Python interpreter: `/usr/bin/python3`

---

## Using the Connection

### Open Terminal in VS Code (Connected to Colab)

```bash
# Terminal → New Terminal
# This terminal is now running ON COLAB!

# Check GPU
nvidia-smi

# Check Python
which python
# Output: /usr/bin/python3

# Check if in Colab
ls /content/

# Mount Drive (if not already)
python -c "from google.colab import drive; drive.mount('/content/drive')"
```

### Run Training from VS Code

```bash
# In VS Code terminal (connected to Colab):
cd /content/retin-verify

# Pull latest code
git pull

# Run training
python training/classification/train_cli.py \
    --data-dir /content/data \
    --epochs 50 \
    --batch-size 32 \
    --device cuda \
    --fp16
```

### Edit Files in VS Code

- All edits happen directly on Colab
- Save with `Ctrl+S` → updates file on Colab
- Kimi can edit files and you see changes immediately

---

## Automated Setup Script

I've created an automated script. In Colab:

```python
!wget https://raw.githubusercontent.com/YOUR_USERNAME/retin-verify/main/scripts/setup_colab_ssh.py
!python setup_colab_ssh.py
```

Or use this complete setup:

```python
#!/usr/bin/env python3
"""
Complete Colab SSH setup for VS Code.
Run this in a Colab notebook.
"""

import os
import subprocess
import sys
import time


def main():
    print("=" * 70)
    print("🔧 VS Code SSH Setup for Google Colab")
    print("=" * 70)
    print()
    
    # Check if running in Colab
    if not os.path.exists('/content'):
        print("❌ Error: Not running in Google Colab!")
        print("   Open this script in a Colab notebook.")
        return 1
    
    # Install colab_ssh
    print("📦 Installing colab_ssh...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'colab_ssh'])
    print("✅ colab_ssh installed")
    print()
    
    # Get ngrok token
    print("=" * 70)
    print("🔑 ngrok Authentication")
    print("=" * 70)
    print()
    print("1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("2. Copy your authtoken")
    print()
    
    try:
        import getpass
        ngrok_token = getpass.getpass("Enter ngrok authtoken: ")
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return 1
    
    if not ngrok_token:
        print("❌ Token cannot be empty")
        return 1
    
    # Get SSH password
    print()
    ssh_password = getpass.getpass("Set SSH password (you'll use this in VS Code): ")
    
    if not ssh_password:
        print("❌ Password cannot be empty")
        return 1
    
    # Launch SSH
    print()
    print("=" * 70)
    print("🚀 Starting SSH Server")
    print("=" * 70)
    print()
    
    from colab_ssh import launch_ssh
    
    try:
        launch_ssh(ngrok_token, password=ssh_password)
    except Exception as e:
        print(f"❌ Error starting SSH: {e}")
        return 1
    
    # Instructions
    print()
    print("=" * 70)
    print("✅ SSH Server is Running!")
    print("=" * 70)
    print()
    print("📋 VS Code Configuration:")
    print()
    print("1. Open VS Code")
    print("2. Press Ctrl+Shift+P")
    print("3. Type: Remote-SSH: Add New SSH Host")
    print("4. Enter: ssh root@<HOST> -p <PORT>")
    print("   (Use host and port shown above ⬆️)")
    print("5. Select config file: ~/.ssh/config")
    print("6. Press F1 → Remote-SSH: Connect to Host → colab-gpu")
    print("7. Enter password when prompted")
    print()
    print("⚠️  IMPORTANT:")
    print("   Keep this notebook running!")
    print("   If this cell stops, SSH will disconnect.")
    print()
    
    # Keep alive
    print("⏱️  Keeping connection alive...")
    print("Press Ctrl+C to stop (this will disconnect SSH)")
    print()
    
    try:
        while True:
            time.sleep(60)
            print(f"[{time.strftime('%H:%M:%S')}] Connection active", flush=True)
    except KeyboardInterrupt:
        print("\n👋 SSH server stopped")
        return 0


if __name__ == '__main__':
    sys.exit(main())
```

---

## VS Code Config Templates

### SSH Config Template

Add to `~/.ssh/config`:

```
# Google Colab GPU
Host colab-gpu
    HostName 2.tcp.ngrok.io      # Replace with your ngrok host
    User root
    Port 12345                   # Replace with your ngrok port
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60       # Keep connection alive
    ServerAliveCountMax 10
    
# Optional: Multiple Colab instances
Host colab-gpu-2
    HostName 3.tcp.ngrok.io
    User root
    Port 23456
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

### VS Code Settings

Add to `.vscode/settings.json`:

```json
{
    "remote.SSH.remotePlatform": {
        "colab-gpu": "linux"
    },
    "remote.SSH.useLocalServer": true,
    "remote.SSH.connectTimeout": 60
}
```

---

## Troubleshooting

### Issue: "Could not establish connection"

**Causes & Fixes:**

1. **Colab cell stopped running**
   - Go back to Colab
   - Re-run the SSH setup cell
   - Get new host:port
   - Update `~/.ssh/config`

2. **Wrong password**
   - Check the password you set in the script
   - Password is case-sensitive

3. **Firewall blocking**
   - Try different network
   - Check corporate firewall settings

4. **ngrok rate limit**
   - Free tier: 1 concurrent tunnel
   - Close other ngrok tunnels
   - Upgrade to ngrok paid plan

### Issue: "Connection timed out"

```bash
# Add to SSH config:
ServerAliveInterval 60
ServerAliveCountMax 10
```

### Issue: "Permission denied (publickey,password)"

- Make sure you're using password auth (not key)
- Re-run Colab setup script
- Use the password you set

### Issue: VS Code keeps disconnecting

1. **Keep Colab tab active**
   - Don't let browser sleep
   - Keep Colab tab visible

2. **Increase timeout in VS Code:**
   ```json
   // settings.json
   {
       "remote.SSH.connectTimeout": 120
   }
   ```

3. **Use keep-alive script in VS Code terminal:**
   ```bash
   while true; do echo "Keep alive $(date)"; sleep 60; done
   ```

### Issue: "Python extension not installed"

When you first connect:
1. VS Code will show "Install" button for Python extension
2. Click "Install in SSH: colab-gpu"
3. Wait for installation
4. Select Python interpreter: `/usr/bin/python3`

---

## Best Practices

### 1. Persistent Connection

Keep Colab tab **active and visible**:
- Don't minimize browser
- Don't let computer sleep
- Use browser extension to keep tab awake

### 2. Reconnection Workflow

When connection drops (inevitable after 12-24h):

```bash
# 1. In Colab: Re-run SSH setup cell
# 2. Get new host:port
# 3. Update ~/.ssh/config with new host:port
# 4. In VS Code: Close remote window
# 5. Reconnect: F1 → Remote-SSH: Connect to Host → colab-gpu
```

### 3. Save Work Frequently

```bash
# In VS Code terminal (connected to Colab):
cd /content/retin-verify
git add .
git commit -m "Work in progress"
git push
```

### 4. Monitor Resources

```bash
# In VS Code terminal:
# Watch GPU usage
watch -n 1 nvidia-smi

# Check disk space
df -h

# Check RAM
free -h
```

---

## Security Notes

⚠️ **Important:**

1. **ngrok token is private**
   - Don't commit it to GitHub
   - Don't share it
   - Regenerate if leaked

2. **SSH password**
   - Use strong password
   - Don't use default passwords
   - Change periodically

3. **Colab runtime is temporary**
   - Anyone with ngrok URL could try to connect
   - Use strong passwords
   - Stop runtime when done

4. **Don't expose sensitive data**
   - Colab runtime is shared infrastructure
   - Don't store credentials in Colab
   - Use environment variables or Drive

---

## Alternative: Without SSH

If SSH is too complex, use this simpler workflow:

```bash
# 1. Local VS Code + Kimi (edit code)
# 2. Sync to GitHub
./scripts/sync_to_colab.sh

# 3. Open Colab in browser
# 4. Run:!git pull
# 5. Run training:!python training/classification/train_cli.py ...

# No SSH needed! Just use Colab's web interface
```

---

## Quick Reference

| Action | Command/Shortcut |
|--------|-----------------|
| Connect to Colab | `F1` → `Remote-SSH: Connect to Host` → `colab-gpu` |
| Open folder | `File` → `Open Folder` → `/content/retin-verify` |
| Open terminal | `` Ctrl+` `` or `Terminal` → `New Terminal` |
| Disconnect | `File` → `Close Remote Connection` |
| Reconnect | `F1` → `Remote-SSH: Connect to Host` → `colab-gpu` |

---

## Summary

| Step | Action | Time |
|------|--------|------|
| 1 | Install VS Code extensions | 2 min |
| 2 | Get ngrok token | 5 min |
| 3 | Run setup in Colab | 2 min |
| 4 | Configure SSH | 5 min |
| 5 | Connect VS Code | 2 min |
| **Total** | | **~16 min** |

Once set up, you have:
- Full VS Code features on Colab GPU
- Direct file editing on remote
- Terminal access to GPU
- Kimi working directly on training environment

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-14
