#!/usr/bin/env python3
"""
3-Class CNIE Classifier - Training with Synthetic + Real Data
Combines 16K synthetic images with real photos for robust training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import random

# Configuration
REAL_DATA_DIR = "/content/retin_retrain_3class/dataset_3class"  # Real photos
SYNTHETIC_DATA_DIR = "/content/synthetic_cnie"  # Will be extracted
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
INPUT_SIZE = 224

# Synthetic data sampling ratio (0.5 = equal real/synthetic)
SYNTHETIC_RATIO = 0.5


class MixedCNIEDataset(Dataset):
    """Dataset that mixes real and synthetic images"""
    
    def __init__(self, real_dir, synthetic_dir, split='train', transform=None, 
                 synthetic_ratio=0.5, max_synthetic_per_class=5000):
        self.transform = transform
        self.split = split
        self.synthetic_ratio = synthetic_ratio
        self.classes = ['cnie_front', 'cnie_back', 'no_card']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []  # (path, label, is_synthetic)
        
        # Load real images
        real_dir = Path(real_dir) / split
        for cls in self.classes:
            class_dir = real_dir / cls
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, self.class_to_idx[cls], False))
        
        self.real_count = len(self.samples)
        
        # Load synthetic images (only front and back, only for training)
        if split == 'train' and synthetic_dir:
            synthetic_dir = Path(synthetic_dir)
            
            # Synthetic front images
            front_dirs = list(synthetic_dir.glob("*/front/image.jpg"))
            random.shuffle(front_dirs)
            front_dirs = front_dirs[:max_synthetic_per_class]  # Limit per class
            for img_path in front_dirs:
                self.samples.append((img_path, self.class_to_idx['cnie_front'], True))
            
            # Synthetic back images
            back_dirs = list(synthetic_dir.glob("*/back/image.jpg"))
            random.shuffle(back_dirs)
            back_dirs = back_dirs[:max_synthetic_per_class]
            for img_path in back_dirs:
                self.samples.append((img_path, self.class_to_idx['cnie_back'], True))
            
            # Note: We don't have synthetic no_card - that's OK, real no_card is already 100% accurate
        
        self.synthetic_count = len(self.samples) - self.real_count
        
        print(f"  {split}: {len(self.samples)} total images")
        print(f"    Real: {self.real_count}")
        print(f"    Synthetic: {self.synthetic_count}")
        for i, cls in enumerate(self.classes):
            real_c = sum(1 for _, idx, syn in self.samples if idx == i and not syn)
            syn_c = sum(1 for _, idx, syn in self.samples if idx == i and syn)
            print(f"    {cls}: {real_c} real + {syn_c} synthetic = {real_c + syn_c}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, is_synthetic = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return is_synthetic as extra info for debugging
        return image, label, is_synthetic


def get_data_transforms():
    """Strong augmentation for training"""
    train_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model():
    """Create model with ImageNet pre-trained weights"""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in model.features[:6].parameters():
        param.requires_grad = False
    
    # Replace classifier with deeper network
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 3)  # 3 classes
    )
    
    return model


def compute_class_weights(dataset):
    """Compute class weights with higher weight for real images"""
    labels = [label for _, label, _ in dataset.samples]
    is_synthetic = [syn for _, _, syn in dataset.samples]
    
    class_counts = np.bincount(labels)
    total = len(labels)
    
    # Base class weights
    base_weights = [total / (len(class_counts) * count) for count in class_counts]
    
    # Sample weights - give 2x weight to real images
    sample_weights = []
    for label, syn in zip(labels, is_synthetic):
        weight = base_weights[label]
        if not syn:  # Real image
            weight *= 2.0  # Give real images 2x weight
        sample_weights.append(weight)
    
    return sample_weights, base_weights


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}
    correct_real = {0: 0, 1: 0, 2: 0}
    total_real = {0: 0, 1: 0, 2: 0}
    
    for images, labels, is_syn in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        
        for pred, true, syn in zip(predicted.cpu().numpy(), labels.cpu().numpy(), is_syn.numpy()):
            total[true] += 1
            if pred == true:
                correct[true] += 1
            
            if not syn:  # Track real image accuracy separately
                total_real[true] += 1
                if pred == true:
                    correct_real[true] += 1
    
    avg_loss = total_loss / len(loader)
    acc_per_class = {k: 100 * correct[k] / total[k] if total[k] > 0 else 0 for k in range(3)}
    overall_acc = 100 * sum(correct.values()) / sum(total.values())
    
    # Real image accuracy (what we care about)
    real_acc_per_class = {k: 100 * correct_real[k] / total_real[k] if total_real[k] > 0 else 0 for k in range(3)}
    
    return avg_loss, overall_acc, acc_per_class, real_acc_per_class


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            for pred, true in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                total[true] += 1
                if pred == true:
                    correct[true] += 1
    
    avg_loss = total_loss / len(loader)
    acc_per_class = {k: 100 * correct[k] / total[k] if total[k] > 0 else 0 for k in range(3)}
    overall_acc = 100 * sum(correct.values()) / sum(total.values())
    balance = min(acc_per_class.values())
    
    return avg_loss, overall_acc, acc_per_class, balance


def main():
    print("=" * 70)
    print("3-Class CNIE Training with Synthetic + Real Data")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Data
    print("\nLoading datasets...")
    train_transform, val_transform = get_data_transforms()
    
    # Training set: Real + Synthetic
    train_dataset = MixedCNIEDataset(
        REAL_DATA_DIR, 
        SYNTHETIC_DATA_DIR,
        'train', 
        train_transform,
        synthetic_ratio=SYNTHETIC_RATIO,
        max_synthetic_per_class=5000
    )
    
    # Validation set: Real only (to measure real-world performance)
    val_dataset = MixedCNIEDataset(
        REAL_DATA_DIR,
        None,  # No synthetic for validation
        'val',
        val_transform
    )
    
    # Weighted sampler - prioritize real images
    sample_weights, class_weights = compute_class_weights(train_dataset)
    print(f"\nClass weights: {[f'{w:.2f}' for w in class_weights]}")
    print(f"Real images get 2x weight in sampling")
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Custom collate function to handle is_synthetic flag
    def train_collate(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        is_syn = torch.tensor([item[2] for item in batch])
        return images, labels, is_syn
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        collate_fn=train_collate
    )
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    print("\nCreating model...")
    model = create_model().to(device)
    
    # Class weights for loss function
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer - different LR for frozen vs unfrozen layers
    optimizer = optim.Adam([
        {'params': model.features[6:].parameters(), 'lr': LR * 0.1},
        {'params': model.classifier.parameters(), 'lr': LR}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training
    print(f"\nTraining {EPOCHS} epochs...")
    print(f"{'='*70}")
    best_balance = 0
    best_real_balance = 0
    history = []
    
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_class_acc, train_real_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_class_acc, balance = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(balance)
        
        # Calculate real-image balance (what we actually care about)
        real_balance = min(train_real_acc.values())
        
        print(f"E{epoch+1:02d}/{EPOCHS} | "
              f"Train(Real): {train_real_acc[0]:.0f}%/{train_real_acc[1]:.0f}%/{train_real_acc[2]:.0f}% | "
              f"Val: {val_class_acc[0]:.0f}%/{val_class_acc[1]:.0f}%/{val_class_acc[2]:.0f}% | "
              f"Bal:{balance:.0f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_front': val_class_acc[0],
            'val_back': val_class_acc[1],
            'val_nocard': val_class_acc[2],
            'balance': balance,
            'real_balance': real_balance
        })
        
        # Save best model based on validation balance (all real images)
        if balance > best_balance:
            best_balance = balance
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'balance': balance,
                'class_acc': val_class_acc
            }, '/content/cnie_classifier_3class_v3_synthetic.pth')
            print(f"  -> Saved (val balance: {balance:.1f}%)")
    
    # Save history
    with open('/content/training_history_v3_synthetic.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Best validation balance: {best_balance:.1f}%")
    print(f"Model saved to: /content/cnie_classifier_3class_v3_synthetic.pth")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
