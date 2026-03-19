#!/usr/bin/env python3
"""
Download retrained model from Colab and deploy locally.

Usage:
    python download_model.py --host abc123.trycloudflare.com
    python download_model.py --host abc123.trycloudflare.com --restart
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import shutil


COLAB_WORKDIR = "/content/retin_retrain"
LOCAL_MODEL_DIR = Path.home() / "retin-verify/models/classification"


def check_training_status(hostname):
    """Check if training is complete on Colab."""
    print("🔍 Checking training status on Colab...")
    
    ssh_cmd = f"ssh root@{hostname}"
    
    try:
        # Check for DONE file
        result = subprocess.run(
            f"{ssh_cmd} 'cat {COLAB_WORKDIR}/DONE 2>/dev/null'",
            shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout:
            print("\n✅ Training completed!")
            print("=" * 60)
            print(result.stdout)
            print("=" * 60)
            return True
        else:
            # Check if training is still running
            result = subprocess.run(
                f"{ssh_cmd} 'pgrep -f train_on_colab.py'",
                shell=True, capture_output=True
            )
            
            if result.returncode == 0:
                print("⏳ Training still in progress...")
                print(f"   Monitor with: ssh root@{hostname} 'tail -f {COLAB_WORKDIR}/training.log'")
            else:
                print("⚠️ Training may have failed or not started.")
                print(f"   Check logs: ssh root@{hostname} 'cat {COLAB_WORKDIR}/training.log'")
            
            return False
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False


def download_model(hostname):
    """Download the retrained model from Colab."""
    print("\n📥 Downloading model from Colab...")
    
    remote_model = f"root@{hostname}:{COLAB_WORKDIR}/cnie_front_back_real_retrained.pth"
    local_temp = Path("/tmp/cnie_front_back_real_retrained.pth")
    
    try:
        subprocess.run(
            f"scp {remote_model} {local_temp}",
            shell=True, check=True
        )
        print(f"✅ Model downloaded to: {local_temp}")
        
        # Also download history
        remote_history = f"root@{hostname}:{COLAB_WORKDIR}/history.json"
        local_history = Path("/tmp/history.json")
        try:
            subprocess.run(
                f"scp {remote_history} {local_history}",
                shell=True, check=True
            )
            print(f"✅ Training history downloaded")
        except:
            pass
        
        return local_temp
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        return None


def backup_current_model():
    """Backup the current model before replacing."""
    current_model = LOCAL_MODEL_DIR / "cnie_front_back_real.pth"
    
    if not current_model.exists():
        print("⚠️ No current model to backup")
        return True
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = LOCAL_MODEL_DIR / f"cnie_front_back_real_backup_{timestamp}.pth"
    
    try:
        shutil.copy2(current_model, backup_path)
        print(f"✅ Current model backed up to: {backup_path.name}")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False


def deploy_model(temp_model_path):
    """Deploy the new model locally."""
    print("\n🚀 Deploying new model...")
    
    target_path = LOCAL_MODEL_DIR / "cnie_front_back_real.pth"
    
    try:
        # Backup first
        if not backup_current_model():
            print("⚠️ Continuing without backup...")
        
        # Move new model to place
        shutil.move(str(temp_model_path), str(target_path))
        print(f"✅ Model deployed to: {target_path}")
        
        # Move history if exists
        temp_history = Path("/tmp/history.json")
        if temp_history.exists():
            history_path = LOCAL_MODEL_DIR / f"retrain_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.move(str(temp_history), str(history_path))
            print(f"✅ History saved to: {history_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False


def restart_local_server():
    """Restart the local classification server."""
    print("\n🔄 Restarting local server...")
    
    try:
        # Kill existing server
        subprocess.run("pkill -f api_server.py", shell=True)
        print("   Stopped existing server")
        
        # Start new server
        start_script = Path(__file__).parent.parent / "start_server.sh"
        if start_script.exists():
            subprocess.Popen(
                ["bash", str(start_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("   Started new server")
            print("   Waiting for server to be ready...")
            
            import time
            time.sleep(5)
            
            # Check health
            result = subprocess.run(
                "curl -s http://127.0.0.1:8000/health",
                shell=True, capture_output=True, text=True
            )
            
            if '"status": "healthy"' in result.stdout:
                print("✅ Server is healthy and running!")
                return True
            else:
                print("⚠️ Server may not be ready yet. Check manually.")
                return False
        else:
            print(f"❌ Start script not found: {start_script}")
            return False
            
    except Exception as e:
        print(f"❌ Restart failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download retrained model from Colab and deploy locally'
    )
    parser.add_argument('--host', required=True, help='Colab hostname')
    parser.add_argument('--restart', action='store_true', 
                       help='Automatically restart local server after deployment')
    parser.add_argument('--force', action='store_true',
                       help='Download even if training not marked complete')
    
    args = parser.parse_args()
    
    # Check training status
    if not args.force:
        if not check_training_status(args.host):
            print("\n❌ Training not complete. Use --force to download anyway.")
            return 1
    
    # Download model
    temp_model = download_model(args.host)
    if not temp_model:
        return 1
    
    # Deploy locally
    if not deploy_model(temp_model):
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Model deployed successfully!")
    print("=" * 60)
    
    # Restart server if requested
    if args.restart:
        if restart_local_server():
            print("\n🎉 Full deployment complete! Ready to test.")
        else:
            print("\n⚠️ Model deployed but server restart may have issues.")
            print("   Manually restart with: ./start_server.sh")
    else:
        print("\n📋 To apply the new model, restart the server:")
        print("   pkill -f api_server.py && ./start_server.sh")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
