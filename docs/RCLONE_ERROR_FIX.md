# Fix: "didn't find section in config file (\"gdrive\")"

## Quick Fix (30 seconds)

```bash
# Step 1: Configure rclone
rclone config

# Follow interactive prompts:
# - Type 'n' for new remote
# - Name it 'gdrive'
# - Select '13' for Google Drive
# - Press Enter for client_id (blank)
# - Press Enter for client_secret (blank)
# - Select '1' for full access
# - Select 'n' for advanced config
# - Select 'y' for auto config
# - Sign in via browser
# - Paste verification code
# - Select 'n' for team drive
# - Select 'y' to confirm
# - Select 'q' to quit

# Step 2: Verify
rclone listremotes
# Output should show: gdrive:

# Step 3: Test
rclone ls gdrive:
```

## What Went Wrong?

The error:
```
CRITICAL: Failed to create file system for "gdrive:...": 
didn't find section in config file ("gdrive")
```

**Meaning:** You tried to use `gdrive:` as a remote name, but rclone doesn't have any remote configured with that name.

**Common causes:**
1. Never ran `rclone config`
2. Named the remote something else (like `mydrive` instead of `gdrive`)
3. rclone config file was deleted/moved
4. Using a different computer than where rclone was configured

## Solutions

### Solution 1: Create the 'gdrive' Remote (Recommended)

Run the configuration wizard:

```bash
rclone config
```

**Full walkthrough:**

```
$ rclone config
No remotes found - make a new one
n) New remote
s) Set configuration password
q) Quit config
n/s/q> n

name> gdrive

Type of storage to configure.
Choose a number from below, or type in your own value
[snip]
13 / Google Drive
   \ "drive"
[snip]
Storage> 13

Google Application Client Id
Setting your own is recommended.
See https://rclone.org/drive/#making-your-own-client-id for how to create your own.
If you leave this blank, it will use an internal key which is low performance.
client_id> [PRESS ENTER]

Google Application Client Secret
client_secret> [PRESS ENTER]

Scope that rclone should use
Choose a number from below, or type in your own value
 1 / Full access all files, excluding Application Data Folder.
   \ "drive"
[snip]
scope> 1

ID of the root folder
Leave blank normally.
root_folder_id> [PRESS ENTER]

Service Account Credentials JSON file path
Leave blank normally.
service_account_file> [PRESS ENTER]

Edit advanced config? (y/n)
y) Yes
n) No (default)
y/n> n

Remote config
Use auto config?
 * Say Y if not sure
 * Say N if you are working on a remote or headless machine
y) Yes (default)
n) No
y/n> y

[Browser opens - sign in to Google and allow access]

Enter verification code> [PASTE CODE FROM BROWSER]

Configure this as a team drive?
y) Yes
n) No (default)
y/n> n

--------------------
[gdrive]
type = drive
scope = drive
token = {"access_token":"..."}
--------------------
y) Yes this is OK (default)
e) Edit this remote
d) Delete this remote
y/e/d> y

current remotes:
Name                 Type
====                 ====
gdrive               drive

e/n/d/r/c/s/q> q
```

Done! Now `rclone ls gdrive:` should work.

### Solution 2: Use Existing Remote Name

If you already configured rclone with a different name:

```bash
# Check what remotes you have
rclone listremotes

# Example output:
# mydrive:
# backup:

# Use the existing name in your commands
# Instead of: rclone ls gdrive:
# Use:        rclone ls mydrive:

# Update your upload command:
rclone copy ./data/cnie_dataset_10k "mydrive:retin-verify/data/cnie_dataset_10k" --progress
```

### Solution 3: Use Alternative Methods (No rclone)

If rclone is too complex, use these alternatives:

#### Option A: Google Drive for Desktop (Easiest)

1. Download: https://www.google.com/drive/download/
2. Install and sign in
3. Create folder: `~/Google Drive/retin-verify/data/`
4. Copy files:
   ```bash
   cp -r ./data/cnie_dataset_10k ~/Google Drive/retin-verify/data/
   ```
5. Wait for sync (check Drive icon in system tray)

#### Option B: Browser Upload

1. Go to https://drive.google.com
2. Create folder: `retin-verify/data/`
3. Click "+ New" → "Folder upload"
4. Select `cnie_dataset_10k` folder
5. Wait for upload

#### Option C: Direct Colab Upload

In your Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

# Create folder
!mkdir -p /content/drive/MyDrive/retin-verify/data

# Upload via browser
from google.colab import files
print("Upload your cnie_dataset_10k folder as a zip file")
uploaded = files.upload()

# Extract
!unzip cnie_dataset_10k.zip -d /content/drive/MyDrive/retin-verify/data/
```

## Automated Fix Script

Run this to diagnose and fix:

```bash
./scripts/fix_rclone.sh
```

This script will:
1. Check if rclone is installed
2. List configured remotes
3. Check if 'gdrive' exists
4. Test connection
5. Provide specific fix instructions

## Verification

After fixing, verify with:

```bash
# Should show: gdrive:
rclone listremotes

# Should list your Drive files
rclone ls gdrive: | head -10

# Create test folder
rclone mkdir gdrive:retin-verify-test

# Remove test folder
rclone rmdir gdrive:retin-verify-test

# If all above work, you're ready!
```

## Next Steps

Once rclone is working:

```bash
# Create folder structure
rclone mkdir "gdrive:retin-verify"
rclone mkdir "gdrive:retin-verify/data"
rclone mkdir "gdrive:retin-verify/models"
rclone mkdir "gdrive:retin-verify/checkpoints"
rclone mkdir "gdrive:retin-verify/logs"

# Upload dataset
rclone copy ./data/cnie_dataset_10k "gdrive:retin-verify/data/cnie_dataset_10k" \
    --progress \
    --transfers 8

# Verify upload
rclone ls "gdrive:retin-verify/data/cnie_dataset_10k/cnie_pairs" | wc -l
```

## Common Issues

### Issue: "config file not found"

```bash
# Find config file location
rclone config file

# If it doesn't exist, create it
mkdir -p ~/.config/rclone
touch ~/.config/rclone/rclone.conf

# Then run rclone config again
rclone config
```

### Issue: "token expired"

```bash
# Re-authenticate
rclone config reconnect gdrive:
```

### Issue: "rate limit exceeded"

```bash
# Slow down transfers
rclone copy ./data "gdrive:..." --tpslimit 10 --transfers 4
```

## Getting Help

1. **Run diagnostic:** `./scripts/fix_rclone.sh`
2. **Read full guide:** `docs/RCLONE_SETUP_GUIDE.md`
3. **Use alternatives:** `./scripts/upload_with_gdown.py`
4. **Official docs:** https://rclone.org/drive/
5. **Community:** https://forum.rclone.org/

## Summary

| Error | Cause | Fix |
|-------|-------|-----|
| "didn't find section in config file" | No remote named 'gdrive' | Run `rclone config` |
| "config file not found" | First time using rclone | Run `rclone config` |
| "token expired" | Authentication expired | Run `rclone config reconnect gdrive:` |
| "rate limit exceeded" | Too many requests | Add `--tpslimit 10` |

**Easiest fix:** Run `rclone config` and follow the prompts!
