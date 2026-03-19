#!/usr/bin/env python3
"""
Deploy v3 training with Synthetic + Real data to Colab
Uploads both real dataset (306 images) and synthetic dataset (16K images)
"""

import subprocess
import sys
from pathlib import Path

# SSH host - update with your Colab tunnel
HOST = "your-colab-host.trycloudflare.com"

def run_cmd(cmd, check=True):
    """Run shell command"""
    print(f">>> {cmd[:80]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"Error: {result.stderr}")
    else:
        if result.stdout:
            print(result.stdout[:200])
    return result

def main():
    print("=" * 70)
    print("Deploy v3 Training with Synthetic + Real Data")
    print("=" * 70)
    
    # Paths
    home = Path.home()
    real_dataset = home / "retin-verify/apps/classification/dataset_3class"
    synthetic_dataset = home / "retin-verify/data/cnie_dataset_10k/cnie_pairs"
    training_script = home / "retin-verify/apps/classification/colab_retrain/new_training/train_with_synthetic.py"
    
    # 1. Package real dataset
    print("\n1. Packaging REAL dataset (306 images)...")
    run_cmd(f"cd {real_dataset} && tar -czf /tmp/real_dataset.tar.gz train val")
    run_cmd("ls -lh /tmp/real_dataset.tar.gz")
    
    # 2. Package synthetic dataset (sample 5000 per class to keep size manageable)
    print("\n2. Packaging SYNTHETIC dataset (16K images, sampling 10K)...")
    print("   This may take a few minutes...")
    
    # Create a filtered synthetic dataset with structure Colab expects
    filter_script = """
import os
import shutil
import random
from pathlib import Path

src = Path.home() / "retin-verify/data/cnie_dataset_10k/cnie_pairs"
dst = Path("/tmp/synthetic_sample")

# Sample 5000 pairs max (10000 images)
pairs = sorted([p.name for p in src.iterdir() if p.is_dir()])
sampled = random.sample(pairs, min(5000, len(pairs)))

print(f"Sampling {len(sampled)} pairs from {len(pairs)} total...")

for pair_id in sampled:
    # Copy front image
    front_src = src / pair_id / "front" / "image.jpg"
    front_dst = dst / pair_id / "front"
    front_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(front_src, front_dst / "image.jpg")
    
    # Copy back image
    back_src = src / pair_id / "back" / "image.jpg"
    back_dst = dst / pair_id / "back"
    back_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(back_src, back_dst / "image.jpg")

print(f"Done! Sampled to {dst}")
"""
    
    with open("/tmp/filter_synthetic.py", "w") as f:
        f.write(filter_script)
    
    run_cmd("python3 /tmp/filter_synthetic.py")
    
    # Package the sampled synthetic data
    print("\n   Packaging sampled synthetic data...")
    run_cmd("cd /tmp && tar -czf synthetic_sample.tar.gz synthetic_sample/")
    run_cmd("ls -lh /tmp/synthetic_sample.tar.gz")
    
    # 3. Upload to Colab
    print(f"\n3. Uploading to Colab ({HOST})...")
    run_cmd(f"ssh root@{HOST} 'mkdir -p /content/retin_v3_synthetic'")
    
    print("   Uploading real dataset...")
    run_cmd(f"scp /tmp/real_dataset.tar.gz root@{HOST}:/content/retin_v3_synthetic/")
    
    print("   Uploading synthetic dataset...")
    run_cmd(f"scp /tmp/synthetic_sample.tar.gz root@{HOST}:/content/retin_v3_synthetic/")
    
    # 4. Upload training script
    print("\n4. Uploading training script...")
    run_cmd(f"scp {training_script} root@{HOST}:/content/retin_v3_synthetic/")
    
    # 5. Extract and setup
    print("\n5. Setting up on Colab...")
    setup_cmds = """
cd /content/retin_v3_synthetic &&
echo "Extracting real dataset..." &&
tar -xzf real_dataset.tar.gz &&
mv train val dataset_3class/ 2>/dev/null || mkdir -p dataset_3class && mv train val dataset_3class/ &&
echo "Extracting synthetic dataset..." &&
tar -xzf synthetic_sample.tar.gz &&
mv synthetic_sample synthetic_cnie &&
echo "Setup complete!" &&
ls -la
"""
    run_cmd(f"ssh root@{HOST} '{setup_cmds}'")
    
    # 6. Install dependencies
    print("\n6. Installing dependencies...")
    install_cmd = """
pip install -q torch torchvision pillow numpy 2>&1 | tail -5
"""
    run_cmd(f"ssh root@{HOST} '{install_cmd}'")
    
    # 7. Start training
    print("\n" + "=" * 70)
    print("7. Starting training!")
    print("=" * 70)
    
    train_cmd = f"""
cd /content/retin_v3_synthetic &&
nohup python3 train_with_synthetic.py > train_v3_synthetic.log 2>&1 &
echo "Training started in background"
echo "PID: $!"
sleep 2
tail -20 train_v3_synthetic.log
"""
    run_cmd(f"ssh root@{HOST} '{train_cmd}'")
    
    print("\n" + "=" * 70)
    print("Deployment Complete!")
    print("=" * 70)
    print(f"\nTo monitor training:")
    print(f"  ssh root@{HOST} 'tail -f /content/retin_v3_synthetic/train_v3_synthetic.log'")
    print(f"\nTo check progress:")
    print(f"  ssh root@{HOST} 'tail -50 /content/retin_v3_synthetic/train_v3_synthetic.log'")
    print(f"\nWhen done, download model:")
    print(f"  scp root@{HOST}:/content/cnie_classifier_3class_v3_synthetic.pth \\")
    print(f"      ~/retin-verify/models/classification/")
    print("")

if __name__ == '__main__':
    main()
