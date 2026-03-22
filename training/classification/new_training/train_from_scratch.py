#!/usr/bin/env python3
"""
3-Class CNIE Classifier - Training from Scratch
Uses ImageNet pre-trained weights, proper augmentation, class balancing
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

# Configuration
DATA_DIR = "/content/retin_v3_synthetic/dataset_3class"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
INPUT_SIZE = 224

class CNIEDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = ['cnie_front', 'cnie_back', 'no_card']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            class_dir = self.data_dir / cls
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, self.class_to_idx[cls]))
        
        print(f"  {split}: {len(self.samples)} images")
        for i, cls in enumerate(self.classes):
            count = sum(1 for _, idx in self.samples if idx == i)
            print(f"    {cls}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

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
    
    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3)  # 3 classes
    )
    
    return model

def compute_class_weights(dataset):
    """Compute class weights for balanced sampling"""
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = [total / (len(class_counts) * count) for count in class_counts]
    sample_weights = [weights[label] for label in labels]
    return sample_weights, weights

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        
        for pred, true in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
            total[true] += 1
            if pred == true:
                correct[true] += 1
    
    avg_loss = total_loss / len(loader)
    acc_per_class = {k: 100 * correct[k] / total[k] if total[k] > 0 else 0 for k in range(3)}
    overall_acc = 100 * sum(correct.values()) / sum(total.values())
    
    return avg_loss, overall_acc, acc_per_class

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
    print("=" * 60)
    print("3-Class CNIE Training from Scratch")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Data
    print("\nLoading datasets...")
    train_transform, val_transform = get_data_transforms()
    train_dataset = CNIEDataset(DATA_DIR, 'train', train_transform)
    val_dataset = CNIEDataset(DATA_DIR, 'val', val_transform)
    
    # Weighted sampler for balanced batches
    sample_weights, class_weights = compute_class_weights(train_dataset)
    print(f"\nClass weights: {[f'{w:.2f}' for w in class_weights]}")
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    print("\nCreating model...")
    model = create_model().to(device)
    
    # Class weights for loss function
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer - different LR for frozen vs unfrozen layers
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    
    optimizer = optim.Adam([
        {'params': model.features[6:].parameters(), 'lr': LR * 0.1},
        {'params': model.classifier.parameters(), 'lr': LR}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training
    print(f"\nTraining {EPOCHS} epochs...")
    best_balance = 0
    history = []
    
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_class_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_class_acc, balance = validate(model, val_loader, criterion, device)
        
        scheduler.step(balance)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {train_acc:.1f}% [F:{train_class_acc[0]:.0f} B:{train_class_acc[1]:.0f} NC:{train_class_acc[2]:.0f}] | "
              f"Val: {val_acc:.1f}% [F:{val_class_acc[0]:.0f} B:{val_class_acc[1]:.0f} NC:{val_class_acc[2]:.0f}] | "
              f"Balance:{balance:.1f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_front': val_class_acc[0],
            'val_back': val_class_acc[1],
            'val_nocard': val_class_acc[2],
            'balance': balance
        })
        
        # Save best model based on balance
        if balance > best_balance:
            best_balance = balance
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'balance': balance,
                'class_acc': val_class_acc
            }, '/content/cnie_classifier_3class_v3.pth')
            print(f"  -> Saved (balance: {balance:.1f}%)")
    
    # Save history
    with open('/content/training_history_v3.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Best balance: {best_balance:.1f}%")
    print(f"Model saved to: /content/cnie_classifier_3class_v3.pth")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
