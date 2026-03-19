#!/usr/bin/env python3
"""
Deploy v3 model training to Colab
Train from scratch with ImageNet weights
"""

import subprocess
import sys
from pathlib import Path

# SSH host - update with your Colab tunnel
HOST = "your-colab-host.trycloudflare.com"

def run_cmd(cmd, check=True):
    print(f">>> {cmd[:80]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"Error: {result.stderr}")
    return result

def main():
    print("=" * 60)
    print("Deploy v3 Training (From Scratch)")
    print("=" * 60)
    
    # 1. Package dataset
    print("\n1. Packaging dataset...")
    dataset_dir = Path.home() / "retin-verify/apps/classification/dataset_3class"
    run_cmd(f"cd {dataset_dir} && tar -czf /tmp/dataset_v3.tar.gz train val")
    
    # 2. Upload to Colab
    print("\n2. Uploading to Colab...")
    run_cmd(f"ssh root@{HOST} 'mkdir -p /content/retin_v3_training'")
    run_cmd(f"scp /tmp/dataset_v3.tar.gz root@{HOST}:/content/retin_v3_training/")
    
    # 3. Upload training script
    print("\n3. Uploading training script...")
    script_path = Path.home() / "retin-verify/apps/classification/colab_retrain/new_training/train_from_scratch.py"
    run_cmd(f"scp {script_path} root@{HOST}:/content/retin_v3_training/")
    
    # 4. Extract and run
    print("\n4. Starting training...")
    setup_cmds = """
cd /content/retin_v3_training
tar -xzf dataset_v3.tar.gz
pip install -q torch torchvision pillow 2>/dev/null
python3 train_from_scratch.py 2>&1 | tee train_v3.log
"""
    run_cmd(f"ssh root@{HOST} '{setup_cmds}'", check=False)
    
    print("\n" + "=" * 60)
    print("Training started!")
    print(f"Monitor with: ssh root@{HOST} 'tail -f /content/retin_v3_training/train_v3.log'")
    print("=" * 60)

if __name__ == '__main__':
    main()
