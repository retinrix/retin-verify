#!/usr/bin/env python3
"""
Weighted Retraining Script for Colab
Addresses class imbalance in feedback data.

Usage on Colab:
    python train_weighted.py
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
import numpy as np

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
        """Get count per class."""
        counts = [0, 0]
        for _, label in self.samples:
            counts[label] += 1
        return counts


def get_transforms(is_training=True):
    """Aggressive augmentation for challenging images."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),  # Simulate small card
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(25),  # Your angles vary
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
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
    """Compute weights inversely proportional to class frequency."""
    counts = dataset.get_class_distribution()
    total = sum(counts)
    # Weight = total / (num_classes * count)
    # This gives minority class higher weight
    weights = [total / (len(counts) * c) if c > 0 else 1.0 for c in counts]
    print(f"  Class distribution: {dict(zip(['front', 'back'], counts))}")
    print(f"  Class weights: {dict(zip(['front', 'back'], [f'{w:.2f}' for w in weights]))}")
    return torch.tensor(weights, dtype=torch.float32)


def load_base_model(model_path, device):
    """Load existing model for fine-tuning."""
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
        
        # Per-class accuracy
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
    confusion = [[0, 0], [0, 0]]  # [actual][predicted]
    
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
    
    # Load data
    data_dir = Path("/content/retin_retrain/retrain_data")
    train_dataset = FeedbackDataset(data_dir / 'train', get_transforms(True))
    val_dataset = FeedbackDataset(data_dir / 'val', get_transforms(False))
    
    if len(train_dataset) == 0:
        print("ERROR: No training images found!")
        return
    
    # Compute class weights based on training distribution
    print("\n📊 Computing class weights...")
    class_weights = compute_class_weights(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Load model
    print("\nLoading base model...")
    model, hidden_dim = load_base_model("/content/retin_retrain/base_model.pth", device)
    
    # Weighted loss - penalize minority class more
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
    
    print("\n" + "=" * 70)
    print("Starting weighted retraining (20 epochs)...")
    print("=" * 70)
    
    best_val_acc = 0
    best_class_balance = 0
    history = []
    
    for epoch in range(20):
        train_loss, train_acc, train_class_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_class_acc, confusion = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        
        # Class balance score: minimum of both class accuracies
        # We want both classes to perform well, not just one
        class_balance = min(val_class_acc)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_front_acc': train_class_acc[0],
            'train_back_acc': train_class_acc[1],
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_front_acc': val_class_acc[0],
            'val_back_acc': val_class_acc[1],
            'confusion': confusion
        })
        
        print(f"\nEpoch {epoch+1}/20 | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}% "
              f"[F:{train_class_acc[0]:.1f}%, B:{train_class_acc[1]:.1f}%]")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.1f}% "
              f"[F:{val_class_acc[0]:.1f}%, B:{val_class_acc[1]:.1f}%]")
        print(f"  Confusion: [[{confusion[0][0]:2d}, {confusion[0][1]:2d}], "
              f"[{confusion[1][0]:2d}, {confusion[1][1]:2d}]]")
        
        # Save based on class balance (not just overall accuracy)
        if class_balance > best_class_balance:
            best_class_balance = class_balance
            output_path = Path("/content/retin_retrain/cnie_front_back_real_retrained.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_front_acc': val_class_acc[0],
                'val_back_acc': val_class_acc[1],
                'class_balance': class_balance,
                'classes': ['cnie_front', 'cnie_back'],
                'retrained_at': datetime.now().isoformat(),
            }, output_path)
            print(f"  -> ✓ Saved best model (class balance: {class_balance:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best class balance: {best_class_balance:.1f}%")
    print("=" * 70)
    
    # Save history
    with open("/content/retin_retrain/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Completion flag
    with open("/content/retin_retrain/DONE", 'w') as f:
        f.write(f"Weighted retraining completed at {datetime.now().isoformat()}\n")
        f.write(f"Best class balance: {best_class_balance:.1f}%\n")
    
    print(f"\nModel saved. Ready for download!")


if __name__ == '__main__':
    main()
