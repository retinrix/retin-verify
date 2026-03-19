#!/usr/bin/env python3
"""
Hybrid Retraining Manager
Orchestrates the local→Colab→local retraining workflow.

Usage:
    # Full workflow
    python retrain_manager.py --host abc123.trycloudflare.com --full
    
    # Step by step
    python retrain_manager.py --host abc123.trycloudflare.com --deploy
    python retrain_manager.py --host abc123.trycloudflare.com --download --restart
    
    # Check status
    python retrain_manager.py --host abc123.trycloudflare.com --status
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Import sibling modules
sys.path.insert(0, str(Path(__file__).parent))
import deploy_to_colab
import download_model


def check_colab_connection(hostname):
    """Test SSH connection to Colab."""
    print(f"🔌 Testing connection to {hostname}...")
    try:
        result = subprocess.run(
            f"ssh -o ConnectTimeout=10 root@{hostname} 'echo connected'",
            shell=True, capture_output=True, text=True, timeout=15
        )
        if "connected" in result.stdout:
            print("✅ SSH connection successful")
            return True
        else:
            print("❌ SSH connection failed")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def monitor_training(hostname, poll_interval=30):
    """Monitor training progress on Colab."""
    print("\n📊 Monitoring training progress...")
    print(f"   Polling every {poll_interval}s (Ctrl+C to stop)")
    print("=" * 60)
    
    try:
        while True:
            result = subprocess.run(
                f"ssh root@{hostname} 'cat {download_model.COLAB_WORKDIR}/training.log 2>/dev/null | tail -5'",
                shell=True, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                # Clear previous lines (simple way)
                print("\033[2J\033[H", end="")  # Clear screen
                print("=" * 60)
                print("Training Log (last 5 lines):")
                print("=" * 60)
                print(result.stdout)
                print("=" * 60)
            
            # Check if done
            done_check = subprocess.run(
                f"ssh root@{hostname} 'test -f {download_model.COLAB_WORKDIR}/DONE && echo done'",
                shell=True, capture_output=True, text=True
            )
            
            if "done" in done_check.stdout:
                print("\n✅ Training completed!")
                return True
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoring stopped by user")
        return False


def full_workflow(hostname, auto_restart=True):
    """Execute the full retraining workflow."""
    print("=" * 70)
    print("  HYBRID RETRAINING WORKFLOW")
    print("  Local → Colab → Local")
    print("=" * 70)
    
    # Step 1: Check connection
    if not check_colab_connection(hostname):
        print("\n❌ Cannot connect to Colab. Please:")
        print("   1. Start your Colab notebook with SSH tunnel")
        print("   2. Get the hostname (e.g., abc123.trycloudflare.com)")
        print("   3. Try again")
        return 1
    
    # Step 2: Check feedback status
    ready, stats = deploy_to_colab.check_feedback_status()
    if not ready:
        print(f"\n⚠️ Only {stats['misclassified']}/10 samples collected")
        response = input("   Deploy anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    # Step 3: Deploy to Colab
    print("\n" + "-" * 70)
    print("STEP 1: Deploy to Colab")
    print("-" * 70)
    
    package_path, base_model = deploy_to_colab.prepare_retraining_package()
    if not package_path:
        return 1
    
    if not deploy_to_colab.deploy_to_colab(hostname, package_path, base_model):
        return 1
    
    # Step 4: Monitor training
    print("\n" + "-" * 70)
    print("STEP 2: Training on Colab")
    print("-" * 70)
    
    completed = monitor_training(hostname)
    
    if not completed:
        print("\n⏸️ Training still in progress or monitoring stopped.")
        print("   To resume monitoring/check later:")
        print(f"   python retrain_manager.py --host {hostname} --status")
        print("\n   To download when ready:")
        print(f"   python retrain_manager.py --host {hostname} --download --restart")
        return 0
    
    # Step 5: Download and deploy
    print("\n" + "-" * 70)
    print("STEP 3: Download & Deploy")
    print("-" * 70)
    
    temp_model = download_model.download_model(hostname)
    if not temp_model:
        return 1
    
    if not download_model.deploy_model(temp_model):
        return 1
    
    # Step 6: Restart server
    if auto_restart:
        print("\n" + "-" * 70)
        print("STEP 4: Restart Server")
        print("-" * 70)
        download_model.restart_local_server()
    
    print("\n" + "=" * 70)
    print("  ✅ WORKFLOW COMPLETE!")
    print("=" * 70)
    print("\n📝 Next steps:")
    print("   1. Test the classification at http://127.0.0.1:8000")
    print("   2. Continue collecting feedback for further improvements")
    print("   3. Repeat this workflow when you have 10+ new samples")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Retraining Manager - Local/Colab workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full automated workflow
  python retrain_manager.py --host abc123.trycloudflare.com --full
  
  # Deploy only
  python retrain_manager.py --host abc123.trycloudflare.com --deploy
  
  # Check training status
  python retrain_manager.py --host abc123.trycloudflare.com --status
  
  # Download and restart
  python retrain_manager.py --host abc123.trycloudflare.com --download --restart
        """
    )
    
    parser.add_argument('--host', help='Colab hostname (e.g., abc123.trycloudflare.com)')
    parser.add_argument('--full', action='store_true', 
                       help='Run full workflow: deploy → monitor → download → restart')
    parser.add_argument('--deploy', action='store_true',
                       help='Deploy data to Colab and start training')
    parser.add_argument('--download', action='store_true',
                       help='Download completed model from Colab')
    parser.add_argument('--restart', action='store_true',
                       help='Restart local server after download')
    parser.add_argument('--status', action='store_true',
                       help='Check training status on Colab')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor training progress')
    parser.add_argument('--force', action='store_true',
                       help='Force action even if checks fail')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.full, args.deploy, args.download, args.status, args.monitor]):
        parser.print_help()
        return 1
    
    if args.host is None and not args.status:
        print("❌ --host is required (except for --status when checking local only)")
        return 1
    
    # Execute requested action
    if args.full:
        return full_workflow(args.host, auto_restart=True)
    
    elif args.deploy:
        if not check_colab_connection(args.host):
            return 1
        ready, stats = deploy_to_colab.check_feedback_status()
        if not ready and not args.force:
            print("\n❌ Not enough samples. Use --force to deploy anyway.")
            return 1
        package_path, base_model = deploy_to_colab.prepare_retraining_package()
        if not package_path:
            return 1
        success = deploy_to_colab.deploy_to_colab(args.host, package_path, base_model)
        if success:
            print("\n🚀 To monitor training:")
            print(f"   python retrain_manager.py --host {args.host} --monitor")
        return 0 if success else 1
    
    elif args.download:
        if not check_colab_connection(args.host):
            return 1
        if not args.force and not download_model.check_training_status(args.host):
            print("\n⚠️ Training not complete. Use --force to download anyway.")
            return 1
        temp_model = download_model.download_model(args.host)
        if not temp_model:
            return 1
        if download_model.deploy_model(temp_model) and args.restart:
            download_model.restart_local_server()
        return 0
    
    elif args.status:
        if args.host:
            if not check_colab_connection(args.host):
                return 1
            download_model.check_training_status(args.host)
        else:
            deploy_to_colab.check_feedback_status()
        return 0
    
    elif args.monitor:
        if not check_colab_connection(args.host):
            return 1
        monitor_training(args.host)
        return 0
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
