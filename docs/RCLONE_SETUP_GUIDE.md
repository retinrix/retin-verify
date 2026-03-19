# rclone Setup Guide for Google Drive

## Error Explanation

```
CRITICAL: Failed to create file system for "gdrive:retin-verify/data/cnie_dataset_10k": 
didn't find section in config file ("gdrive")
```

**What this means:** rclone doesn't have a remote named "gdrive" configured yet.

**Solution:** Run `rclone config` to create the remote.

---

## Step-by-Step rclone Configuration

### Step 1: Install rclone

```bash
# macOS
brew install rclone

# Linux
curl https://rclone.org/install.sh | sudo bash

# Windows
# Download from https://rclone.org/downloads/
# Or use Chocolatey: choco install rclone
```

### Step 2: Configure Google Drive Remote

```bash
rclone config
```

**Interactive prompts:**

```
No remotes found - make a new one
n) New remote
s) Set configuration password
q) Quit config
n/s/q> n       ← Type 'n' for new remote

name> gdrive   ← Type 'gdrive' (this is the name you'll use)

Option 13: Google Drive
Storage> 13    ← Type '13' for Google Drive

Google Application Client Id - leave blank normally.
client_id>     ← Press Enter (blank)

Google Application Client Secret - leave blank normally.
client_secret> ← Press Enter (blank)

Scope that rclone should use
1) Full access (recommended)
scope> 1       ← Type '1' for full access

Service Account Credentials JSON file path
service_account_file> ← Press Enter (blank)

Edit advanced config? (y/n)
y) Yes
n) No (default)
y/n> n         ← Type 'n'

Remote config
Use auto config?
 * Say Y if not sure
 * Say N if you are working on a remote or headless machine

y) Yes (default)
n) No
y/n> y         ← Type 'y'
```

**Browser will open** - follow these steps:

1. Sign in to your Google account
2. Click "Allow" to grant rclone access
3. Copy the verification code
4. Paste it back in terminal

```
Configure this as a team drive?
y) Yes
n) No (default)
y/n> n         ← Type 'n'

--------------------
[gdrive]
type = drive
scope = drive
--------------------
y) Yes this is OK (default)
e) Edit this remote
d) Delete this remote
y/e/d> y       ← Type 'y' to confirm

current remotes:
Name                 Type
====                 ====
gdrive               drive   ← SUCCESS!

e) Edit existing remote
n) New remote
d) Delete remote
r) Rename remote
c) Copy remote
s) Set configuration password
q) Quit config
q) Quit config
e/n/d/r/c/s/q> q   ← Type 'q' to quit
```

### Step 3: Verify Configuration

```bash
# List remotes
rclone listremotes
# Output: gdrive:

# List Drive root
rclone ls gdrive:

# List your retin-verify folder
rclone ls "gdrive:retin-verify" 2>/dev/null || echo "Folder doesn't exist yet"
```

---

## Creating the Folder Structure

### Option 1: Create via rclone

```bash
# Create folder structure
rclone mkdir "gdrive:retin-verify"
rclone mkdir "gdrive:retin-verify/data"
rclone mkdir "gdrive:retin-verify/models"
rclone mkdir "gdrive:retin-verify/checkpoints"
rclone mkdir "gdrive:retin-verify/logs"

# Verify
rclone ls "gdrive:retin-verify"
```

### Option 2: Create via Google Drive Web

1. Go to https://drive.google.com
2. Create folder: `retin-verify`
3. Create subfolders: `data`, `models`, `checkpoints`, `logs`

### Option 3: Upload from Local

If you already have the folder locally:

```bash
# Create structure and copy in one command
rclone copy ./data/cnie_dataset_10k "gdrive:retin-verify/data/cnie_dataset_10k" \
    --progress \
    --transfers 8 \
    --checksum
```

---

## Uploading the Dataset

### Method 1: rclone copy (Recommended for large datasets)

```bash
# Navigate to your project
cd /path/to/retin-verify

# Upload with progress
rclone copy ./data/cnie_dataset_10k "gdrive:retin-verify/data/cnie_dataset_10k" \
    --progress \
    --transfers 8 \
    --verbose \
    --stats 10s

# Flags explained:
# --progress      : Show real-time progress
# --transfers 8   : Use 8 parallel transfers
# --verbose       : Show detailed info
# --stats 10s     : Update stats every 10 seconds
```

### Method 2: rclone sync (Sync changes only)

```bash
# If you need to update/sync later
rclone sync ./data/cnie_dataset_10k "gdrive:retin-verify/data/cnie_dataset_10k" \
    --progress \
    --transfers 8
```

**WARNING:** `sync` deletes files on destination that don't exist on source. Use `copy` for safety.

### Method 3: Tar and Upload (For very large datasets)

```bash
# Compress first
tar -czvf cnie_dataset_10k.tar.gz ./data/cnie_dataset_10k

# Upload single file
rclone copy cnie_dataset_10k.tar.gz "gdrive:retin-verify/data/" --progress

# On Colab, extract:
# !tar -xzvf /content/drive/MyDrive/retin-verify/data/cnie_dataset_10k.tar.gz
```

---

## Troubleshooting

### Error: "didn't find section in config file"

**Cause:** Remote name doesn't exist in rclone config.

**Fix:**
```bash
# Check available remotes
rclone listremotes

# If empty, create the remote
rclone config

# If you want a different name, update your commands
# Instead of: rclone copy ... gdrive:...
# Use:        rclone copy ... your_remote_name:...
```

### Error: "Failed to create file system for \"gdrive:...\": drive: failed to get Team/Shared Drive info"

**Cause:** Trying to access a team drive that doesn't exist.

**Fix:**
```bash
# List available drives
rclone backend drives gdrive:

# Use the correct drive ID or use root
rclone ls gdrive:  # Root of My Drive
```

### Error: "token expired"

**Fix:**
```bash
# Re-authenticate
rclone config reconnect gdrive:
```

### Error: "rate limit exceeded"

**Fix:**
```bash
# Add delays between operations
rclone copy ./data "gdrive:retin-verify/data" \
    --tpslimit 10 \
    --transfers 4
```

### Upload is very slow

**Solutions:**
```bash
# Increase parallel transfers
rclone copy ./data "gdrive:..." --transfers 16

# Use fast-list (uses more memory)
rclone copy ./data "gdrive:..." --fast-list

# Check your internet speed
speedtest-cli

# Use a different Google Drive API (if you have one)
# Edit: rclone config
# Add your own client_id and client_secret
```

---

## Quick Reference Commands

```bash
# Upload dataset
rclone copy ./data/cnie_dataset_10k "gdrive:retin-verify/data/cnie_dataset_10k" --progress

# Download models
rclone copy "gdrive:retin-verify/models" ./models --progress

# Sync checkpoints (one-way)
rclone sync "gdrive:retin-verify/checkpoints" ./checkpoints --progress

# Check file count
rclone ls "gdrive:retin-verify/data/cnie_dataset_10k/cnie_pairs" | wc -l

# Get total size
rclone size "gdrive:retin-verify/data/cnie_dataset_10k"

# Mount Drive locally (Linux/Mac)
rclone mount gdrive: ~/mnt/gdrive --daemon

# Verify upload integrity
rclone check ./data/cnie_dataset_10k "gdrive:retin-verify/data/cnie_dataset_10k"

# Delete folder
rclone purge "gdrive:retin-verify/old_data"

# List large files
rclone ls "gdrive:retin-verify" --min-size 100M
```

---

## Alternative: Without rclone

If rclone keeps failing, use these alternatives:

### Option 1: Google Drive for Desktop

```bash
# Install from https://www.google.com/drive/download/
# Then copy as normal filesystem
cp -r ./data/cnie_dataset_10k "~/Google Drive/retin-verify/data/"
```

### Option 2: gdown (Python)

```bash
pip install gdown

# Upload via web interface, get shareable link
gdown <file_id> -O ./models/
```

### Option 3: Direct in Colab

```python
# In Colab, upload directly
from google.colab import files
uploaded = files.upload()  # Select files

# Or mount and copy
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/cnie_dataset_10k /content/drive/MyDrive/retin-verify/data/
```

---

## Configuration File Location

If you need to manually edit config:

```bash
# Show config file location
rclone config file

# Typical locations:
# Linux: ~/.config/rclone/rclone.conf
# macOS: ~/.config/rclone/rclone.conf
# Windows: %APPDATA%\rclone\rclone.conf

# Example config content:
cat ~/.config/rclone/rclone.conf

[gdrive]
type = drive
client_id = 
client_secret = 
token = {"access_token":"...","token_type":"Bearer","refresh_token":"...","expiry":"..."}
```

---

## Automated Upload Script

Create `scripts/upload_to_drive.sh`:

```bash
#!/bin/bash

set -e

echo "Uploading dataset to Google Drive..."

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "Error: rclone not installed"
    echo "Install from https://rclone.org/downloads/"
    exit 1
fi

# Check if remote exists
if ! rclone listremotes | grep -q "^gdrive:$"; then
    echo "Error: 'gdrive' remote not configured"
    echo "Run: rclone config"
    exit 1
fi

# Check if data exists
if [ ! -d "./data/cnie_dataset_10k" ]; then
    echo "Error: ./data/cnie_dataset_10k not found"
    exit 1
fi

# Create folder structure
echo "Creating folder structure..."
rclone mkdir "gdrive:retin-verify/data" 2>/dev/null || true

# Upload with progress
echo "Uploading dataset..."
rclone copy ./data/cnie_dataset_10k "gdrive:retin-verify/data/cnie_dataset_10k" \
    --progress \
    --transfers 8 \
    --checksum \
    --verbose

echo "Upload complete!"
echo ""
echo "Verify with:"
echo "  rclone ls \"gdrive:retin-verify/data/cnie_dataset_10k\" | head -20"
```

Make executable:
```bash
chmod +x scripts/upload_to_drive.sh
./scripts/upload_to_drive.sh
```

---

**Need help?** Run `rclone config` again or check https://rclone.org/drive/
