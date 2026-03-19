#!/usr/bin/env python3
"""
Deploy to Colab with WEIGHTED retraining script.
This version uses class weights to address the FRONT bias issue.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Import from sibling module
sys.path.insert(0, str(Path(__file__).parent))
import deploy_to_colab


def create_weighted_training_script():
    """Create the weighted training script for Colab."""
    script = '''#!/usr/bin/env python3
"""
Weighted Retraining Script for Colab
Addresses class imbalance: 28 FRONT errors vs 18 BACK errors
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
    
    def get_class_distribution(self):
        counts = [0, 0]
        for _, label in self.samples:
            counts[label] += 1
        return counts

def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def compute_class_weights(dataset):
    counts = dataset.get_class_distribution()
    total = sum(counts)
    weights = [total / (len(counts) * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class distribution: Front={counts[0]}, Back={counts[1]}")
    print(f"  Class weights: Front={weights[0]:.2f}, Back={weights[1]:.2f}")
    return torch.tensor(weights, dtype=torch.float32)

def load_base_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    hidden_dim = 256
    if 'classifier.4.weight' in state_dict:
        hidden_dim = state_dict['classifier.4.weight'].shape[1]
    
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
    class_correct = [0, 0]
    class_total = [0, 0]
    
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
        
        for i in range(len(labels)):
            class_total[labels[i]] += 1
            if predicted[i] == labels[i]:
                class_correct[labels[i]] += 1
    
    acc = 100. * correct / total
    class_acc = [100. * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    return total_loss / len(loader), acc, class_acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    class_correct = [0, 0]
    class_total = [0, 0]
    confusion = [[0, 0], [0, 0]]
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(len(labels)):
                actual = labels[i].item()
                pred = predicted[i].item()
                class_total[actual] += 1
                confusion[actual][pred] += 1
                if actual == pred:
                    class_correct[actual] += 1
    
    acc = 100. * correct / total
    class_acc = [100. * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    return total_loss / len(loader), acc, class_acc, confusion

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    data_dir = Path("/content/retin_retrain/retrain_data")
    train_dataset = FeedbackDataset(data_dir / 'train', get_transforms(True))
    val_dataset = FeedbackDataset(data_dir / 'val', get_transforms(False))
    
    if len(train_dataset) == 0:
        print("ERROR: No training images!")
        return
    
    print("\\n📊 Computing class weights...")
    class_weights = compute_class_weights(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print("\\nLoading base model...")
    model, _ = load_base_model("/content/retin_retrain/base_model.pth", device)
    
    # WEIGHTED LOSS - key improvement!
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    
    print("\\n" + "=" * 70)
    print("WEIGHTED RETRAINING (20 epochs)")
    print("=" * 70)
    
    best_class_balance = 0
    history = []
    
    for epoch in range(20):
        train_loss, train_acc, train_class_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_class_acc, confusion = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        class_balance = min(val_class_acc)
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_front_acc': val_class_acc[0],
            'val_back_acc': val_class_acc[1],
            'confusion': confusion
        })
        
        print(f"\\nEpoch {epoch+1}/20")
        print(f"  Train: {train_acc:.1f}% [F:{train_class_acc[0]:.1f}%, B:{train_class_acc[1]:.1f}%]")
        print(f"  Val:   {val_acc:.1f}% [F:{val_class_acc[0]:.1f}%, B:{val_class_acc[1]:.1f}%]")
        print(f"  Confusion: F→F:{confusion[0][0]}, F→B:{confusion[0][1]} | B→F:{confusion[1][0]}, B→B:{confusion[1][1]}")
        
        if class_balance > best_class_balance:
            best_class_balance = class_balance
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_front_acc': val_class_acc[0],
                'val_back_acc': val_class_acc[1],
                'class_balance': class_balance,
                'retrained_at': datetime.now().isoformat(),
            }, "/content/retin_retrain/cnie_front_back_real_retrained.pth")
            print(f"  ✓ Saved (balance: {class_balance:.1f}%)")
    
    print("\\n" + "=" * 70)
    print(f"Complete! Best class balance: {best_class_balance:.1f}%")
    
    with open("/content/retin_retrain/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    with open("/content/retin_retrain/DONE", 'w') as f:
        f.write(f"Weighted retraining done at {datetime.now().isoformat()}\\n")
        f.write(f"Best class balance: {best_class_balance:.1f}%\\n")

if __name__ == '__main__':
    main()
'''
    return script


def deploy_weighted(hostname):
    """Deploy with weighted training script."""
    print("🎯 DEPLOYING WEIGHTED RETRAINING")
    print("=" * 60)
    print("This version addresses the FRONT bias issue:")
    print("  • Class-weighted loss (penalizes errors more)")
    print("  • Aggressive augmentation")
    print("  • 20 epochs (more training)")
    print("  • Saves based on class balance, not just accuracy")
    print("=" * 60)
    
    # Prepare package
    package_path, base_model = deploy_to_colab.prepare_retraining_package()
    if not package_path:
        return False
    
    # Create weighted training script
    script_content = create_weighted_training_script()
    script_path = Path("/tmp/train_weighted.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    ssh_cmd = f"ssh root@{hostname}"
    COLAB_WORKDIR = "/content/retin_retrain"
    
    try:
        # Standard deploy steps
        print("\n📤 Deploying to Colab...")
        subprocess.run(f"{ssh_cmd} 'mkdir -p {COLAB_WORKDIR}'", shell=True, check=True)
        subprocess.run(f"scp {package_path} root@{hostname}:{COLAB_WORKDIR}/", shell=True, check=True)
        subprocess.run(f"scp {base_model} root@{hostname}:{COLAB_WORKDIR}/base_model.pth", shell=True, check=True)
        subprocess.run(f"scp {script_path} root@{hostname}:{COLAB_WORKDIR}/train_on_colab.py", shell=True, check=True)
        subprocess.run(f"{ssh_cmd} 'cd {COLAB_WORKDIR} && tar -xzf {package_path.name}'", shell=True, check=True)
        
        # Start weighted training
        print("\n🚀 Starting WEIGHTED training...")
        subprocess.run(
            f"{ssh_cmd} 'cd {COLAB_WORKDIR} && nohup python3 train_on_colab.py > training.log 2>&1 &'",
            shell=True, check=True
        )
        
        print("\n✅ Weighted training started!")
        print(f"   Monitor: ssh root@{hostname} 'tail -f {COLAB_WORKDIR}/training.log'")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Deploy weighted retraining to Colab')
    parser.add_argument('--host', required=True, help='Colab hostname')
    parser.add_argument('--force', action='store_true', help='Deploy even with <10 samples')
    
    args = parser.parse_args()
    
    # Check feedback
    ready, stats = deploy_to_colab.check_feedback_status()
    
    print("\n📊 BIAS ANALYSIS:")
    print(f"   FRONT misclassified as BACK: ~61% of errors")
    print(f"   BACK misclassified as FRONT: ~39% of errors")
    print(f"   → Model has BACK bias (over-predicts back)")
    
    if not ready and not args.force:
        print("\n❌ Not enough samples")
        return 1
    
    if deploy_weighted(args.host):
        print("\n📋 Next:")
        print(f"   python retrain_manager.py --host {args.host} --monitor")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
