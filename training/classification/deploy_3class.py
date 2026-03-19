#!/usr/bin/env python3
"""
Deploy 3-class retraining to Colab
Includes: cnie_front, cnie_back, no_card
"""

import argparse
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import deploy_to_colab

COLAB_WORKDIR = "/content/retin_retrain_3class"


def create_3class_training_script():
    """Create training script for 3 classes."""
    return '''#!/usr/bin/env python3
"""
3-Class Retraining: cnie_front, cnie_back, no_card
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

class ThreeClassDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['cnie_front', 'cnie_back', 'no_card']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            class_dir = self.data_dir / cls
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, self.class_to_idx[cls]))
        
        print(f"  Loaded {len(self.samples)} images from {data_dir}")
        for cls in self.classes:
            count = len([s for s in self.samples if s[1] == self.class_to_idx[cls]])
            print(f"    {cls}: {count}")
    
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
    from collections import Counter
    labels = [s[1] for s in dataset.samples]
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(dataset.classes)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    print(f"  Class weights: {dict(zip(dataset.classes, [f'{w:.2f}' for w in weights]))}")
    return torch.tensor(weights, dtype=torch.float32)

def load_base_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Build 2-class model first
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    
    # Load 2-class weights
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    model.load_state_dict(state_dict)
    
    # Now modify to 3-class
    # Copy old weights and add new output for no_card
    old_classifier = model.classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3)  # 3 classes now
    )
    
    # Copy weights from old classifier
    with torch.no_grad():
        model.classifier[0].weight.copy_(old_classifier[0].weight)
        model.classifier[0].bias.copy_(old_classifier[0].bias)
        model.classifier[1].weight.copy_(old_classifier[1].weight)
        model.classifier[1].bias.copy_(old_classifier[1].bias)
        model.classifier[2].p = old_classifier[2].p
        # Initialize new output layer
        nn.init.xavier_uniform_(model.classifier[3].weight)
        nn.init.zeros_(model.classifier[3].bias)
    
    model.to(device)
    return model

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

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(len(labels)):
                class_total[labels[i]] += 1
                if predicted[i] == labels[i]:
                    class_correct[labels[i]] += 1
    
    acc = 100. * correct / total
    class_acc = [100. * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    return acc, class_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_dir = Path("/content/retin_retrain_3class/retrain_data")
    train_dataset = ThreeClassDataset(data_dir / 'train', get_transforms(True))
    val_dataset = ThreeClassDataset(data_dir / 'val', get_transforms(False))
    
    if len(train_dataset) == 0:
        print("ERROR: No training images!")
        return
    
    print("\\n📊 Computing class weights...")
    class_weights = compute_class_weights(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print("\\nLoading and adapting base model (2→3 classes)...")
    model = load_base_model("/content/retin_retrain_3class/base_model.pth", device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)
    
    print("\\n" + "=" * 70)
    print("3-CLASS RETRAINING (25 epochs)")
    print("=" * 70)
    
    best_acc = 0
    history = []
    
    for epoch in range(25):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_class_acc = validate(model, val_loader, device)
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_class_acc': val_class_acc
        })
        
        print(f"Epoch {epoch+1}/25 | Train: {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% [F:{val_class_acc[0]:.1f}%, B:{val_class_acc[1]:.1f}%, NC:{val_class_acc[2]:.1f}%]")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'classes': ['cnie_front', 'cnie_back', 'no_card'],
                'retrained_at': datetime.now().isoformat(),
            }, "/content/retin_retrain_3class/cnie_classifier_3class.pth")
            print(f"  ✓ Saved (val_acc: {val_acc:.1f}%)")
    
    print("\\n" + "=" * 70)
    print(f"Complete! Best val accuracy: {best_acc:.1f}%")
    print("=" * 70)
    
    with open("/content/retin_retrain_3class/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    with open("/content/retin_retrain_3class/DONE", 'w') as f:
        f.write(f"3-class retraining done at {datetime.now().isoformat()}\\n")
        f.write(f"Best val accuracy: {best_acc:.1f}%\\n")

if __name__ == '__main__':
    main()
'''


def deploy_3class(hostname):
    """Deploy 3-class dataset and training."""
    dataset_dir = Path.home() / 'retin-verify/apps/classification/dataset_3class'
    base_model = Path.home() / 'retin-verify/models/classification/cnie_front_back_real.pth'
    
    if not dataset_dir.exists():
        print("❌ Dataset not found!")
        print("   Run: python prepare_3class_dataset.py --split")
        return False
    
    # Count samples
    counts = {}
    for split in ['train', 'val']:
        counts[split] = {}
        for cls in ['cnie_front', 'cnie_back', 'no_card']:
            class_dir = dataset_dir / split / cls
            counts[split][cls] = len(list(class_dir.glob('*.jpg'))) if class_dir.exists() else 0
    
    print("\n📊 Dataset Statistics:")
    total = 0
    for split, classes in counts.items():
        print(f"  {split}:")
        for cls, count in classes.items():
            print(f"    {cls}: {count}")
            total += count
    
    if total < 60:
        print(f"\n⚠️  Need more samples (have {total}, need 60+)")
        print("   Collect more 'no_card' samples!")
        return False
    
    # Create package
    print("\n📦 Packaging dataset...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_path = Path(tempfile.gettempdir()) / f"3class_data_{timestamp}.tar.gz"
    
    with tarfile.open(package_path, "w:gz") as tar:
        tar.add(dataset_dir, arcname="retrain_data")
    
    # Create training script
    script_content = create_3class_training_script()
    script_path = Path(tempfile.gettempdir()) / "train_3class.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    ssh_cmd = f"ssh root@{hostname}"
    
    try:
        print("\n🚀 Deploying to Colab...")
        subprocess.run(f"{ssh_cmd} 'mkdir -p {COLAB_WORKDIR}'", shell=True, check=True)
        subprocess.run(f"scp {package_path} root@{hostname}:{COLAB_WORKDIR}/", shell=True, check=True)
        subprocess.run(f"scp {base_model} root@{hostname}:{COLAB_WORKDIR}/base_model.pth", shell=True, check=True)
        subprocess.run(f"scp {script_path} root@{hostname}:{COLAB_WORKDIR}/train_on_colab.py", shell=True, check=True)
        subprocess.run(f"{ssh_cmd} 'cd {COLAB_WORKDIR} && tar -xzf {package_path.name}'", shell=True, check=True)
        
        print("\n🚀 Starting 3-class training...")
        subprocess.run(
            f"{ssh_cmd} 'cd {COLAB_WORKDIR} && nohup python3 train_on_colab.py > training.log 2>&1 &'",
            shell=True, check=True
        )
        
        print("\n✅ 3-class training started!")
        print(f"   Monitor: ssh root@{hostname} 'tail -f {COLAB_WORKDIR}/training.log'")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Deploy 3-class retraining to Colab')
    parser.add_argument('--host', required=True, help='Colab hostname')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  3-CLASS RETRAINING DEPLOYMENT")
    print("  Classes: cnie_front, cnie_back, no_card")
    print("=" * 70)
    
    if deploy_3class(args.host):
        print("\n📋 Monitor training:")
        print(f"   ssh root@{args.host} 'tail -f {COLAB_WORKDIR}/training.log'")


if __name__ == '__main__':
    main()
