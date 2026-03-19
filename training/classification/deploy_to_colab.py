#!/usr/bin/env python3
"""
Deploy feedback data to Colab for retraining.
Automates the hybrid retraining workflow.

Usage:
    python deploy_to_colab.py --host abc123.trycloudflare.com
"""

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from feedback_system import get_feedback_collector


REQUIRED_SAMPLES = 10
COLAB_WORKDIR = "/content/retin_retrain"


def check_feedback_status():
    """Check if we have enough feedback to retrain."""
    collector = get_feedback_collector()
    stats = collector.get_statistics()
    
    print("=" * 60)
    print("Feedback Collection Status")
    print("=" * 60)
    print(f"Total feedback:     {stats['total_feedback']}")
    print(f"Misclassified:      {stats['misclassified']} (need {REQUIRED_SAMPLES})")
    print(f"Correct confirmed:  {stats['correct_confirmations']}")
    print(f"Low confidence:     {stats['low_confidence']}")
    print(f"Retraining ready:   {'✅ YES' if stats['retraining_recommended'] else '❌ NO'}")
    print("=" * 60)
    
    return stats['retraining_recommended'], stats


def prepare_retraining_package():
    """Create a package with feedback data for Colab."""
    collector = get_feedback_collector()
    
    # Prepare dataset
    print("\n📦 Preparing retraining dataset...")
    dataset_dir = collector.prepare_retraining_dataset()
    
    # Create tar.gz package
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"retrain_data_{timestamp}.tar.gz"
    package_path = Path(tempfile.gettempdir()) / package_name
    
    with tarfile.open(package_path, "w:gz") as tar:
        tar.add(dataset_dir, arcname="retrain_data")
    
    # Also get base model
    base_model = Path.home() / "retin-verify/models/classification/cnie_front_back_real.pth"
    if not base_model.exists():
        print(f"❌ Base model not found: {base_model}")
        return None, None
    
    print(f"✅ Package created: {package_path}")
    print(f"   Size: {package_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Dataset: {dataset_dir}")
    
    return package_path, base_model


def create_colab_training_script():
    """Create the training script to run on Colab."""
    script = '''#!/usr/bin/env python3
"""
Retraining script for Colab.
Run this on Google Colab with GPU.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

class FeedbackDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['cnie_front', 'cnie_back']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            class_dir = self.data_dir / cls
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, self.class_to_idx[cls]))
        
        print(f"  Loaded {len(self.samples)} images from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_base_model(model_path, device):
    """Load existing model for fine-tuning."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect architecture
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    hidden_dim = 256
    if 'classifier.4.weight' in state_dict:
        hidden_dim = state_dict['classifier.4.weight'].shape[1]
    
    # Build model
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, 2)
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    return model, hidden_dim

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    data_dir = Path("/content/retin_retrain/retrain_data")
    train_dataset = FeedbackDataset(data_dir / 'train', get_transforms(True))
    val_dataset = FeedbackDataset(data_dir / 'val', get_transforms(False))
    
    if len(train_dataset) == 0:
        print("ERROR: No training images found!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Load model
    print("\\nLoading base model...")
    model, hidden_dim = load_base_model("/content/retin_retrain/base_model.pth", device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    
    print("\\n" + "=" * 60)
    print("Starting retraining...")
    print("=" * 60)
    
    best_val_acc = 0
    history = []
    
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        print(f"Epoch {epoch+1}/10 | "
              f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model
            output_path = Path("/content/retin_retrain/cnie_front_back_real_retrained.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'classes': ['cnie_front', 'cnie_back'],
                'retrained_at': datetime.now().isoformat(),
            }, output_path)
            print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")
    
    print("=" * 60)
    print(f"Training complete! Best val accuracy: {best_val_acc:.2f}%")
    
    # Save history
    with open("/content/retin_retrain/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save completion flag
    with open("/content/retin_retrain/DONE", 'w') as f:
        f.write(f"Retraining completed at {datetime.now().isoformat()}\\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\\n")
    
    print(f"\\nModel saved to: {output_path}")
    print("Ready for download!")

if __name__ == '__main__':
    main()
'''
    return script


def deploy_to_colab(hostname, package_path, base_model):
    """Deploy data and scripts to Colab via SSH."""
    print(f"\n🚀 Deploying to Colab: {hostname}")
    print("=" * 60)
    
    # Create training script
    script_content = create_colab_training_script()
    script_path = Path(tempfile.gettempdir()) / "train_on_colab.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    ssh_cmd = f"ssh root@{hostname}"
    
    try:
        # 1. Create working directory on Colab
        print("1. Creating working directory...")
        subprocess.run(
            f"{ssh_cmd} 'mkdir -p {COLAB_WORKDIR}'",
            shell=True, check=True
        )
        
        # 2. Upload package
        print("2. Uploading feedback data...")
        subprocess.run(
            f"scp {package_path} root@{hostname}:{COLAB_WORKDIR}/",
            shell=True, check=True
        )
        
        # 3. Upload base model
        print("3. Uploading base model...")
        subprocess.run(
            f"scp {base_model} root@{hostname}:{COLAB_WORKDIR}/base_model.pth",
            shell=True, check=True
        )
        
        # 4. Upload training script
        print("4. Uploading training script...")
        subprocess.run(
            f"scp {script_path} root@{hostname}:{COLAB_WORKDIR}/",
            shell=True, check=True
        )
        
        # 5. Extract package
        print("5. Extracting data...")
        subprocess.run(
            f"{ssh_cmd} 'cd {COLAB_WORKDIR} && tar -xzf {package_path.name}'",
            shell=True, check=True
        )
        
        # 6. Start training in background
        print("6. Starting training (background)...")
        subprocess.run(
            f"{ssh_cmd} 'cd {COLAB_WORKDIR} && nohup python3 train_on_colab.py > training.log 2>&1 &'",
            shell=True, check=True
        )
        
        print("\n" + "=" * 60)
        print("✅ Deployment complete!")
        print(f"📁 Colab workdir: {COLAB_WORKDIR}")
        print(f"📊 Monitor: ssh root@{hostname} 'tail -f {COLAB_WORKDIR}/training.log'")
        print("=" * 60)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Deployment failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Deploy feedback data to Colab for retraining')
    parser.add_argument('--host', required=True, help='Colab hostname (e.g., abc123.trycloudflare.com)')
    parser.add_argument('--check', action='store_true', help='Only check feedback status, do not deploy')
    parser.add_argument('--force', action='store_true', help='Deploy even if less than 10 samples')
    
    args = parser.parse_args()
    
    # Check feedback status
    ready, stats = check_feedback_status()
    
    if args.check:
        return 0 if ready else 1
    
    if not ready and not args.force:
        print(f"\n❌ Not enough samples for retraining.")
        print(f"   Need {REQUIRED_SAMPLES}, have {stats['misclassified']}")
        print(f"   Use --force to deploy anyway.")
        return 1
    
    # Prepare and deploy
    package_path, base_model = prepare_retraining_package()
    if not package_path:
        return 1
    
    success = deploy_to_colab(args.host, package_path, base_model)
    
    if success:
        print("\n📋 Next steps:")
        print("   1. Monitor training on Colab")
        print("   2. Wait for completion (check for DONE file)")
        print("   3. Download model: python download_model.py --host", args.host)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
