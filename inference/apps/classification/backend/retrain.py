#!/usr/bin/env python3
"""
Retrain model with collected feedback data.
Implements incremental learning from user corrections.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

from feedback_system import get_feedback_collector


class FeedbackDataset(Dataset):
    """Dataset for retraining with feedback images."""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['cnie_front', 'cnie_back']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Find all images
        self.samples = []
        for cls in self.classes:
            class_dir = self.data_dir / cls
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, self.class_to_idx[cls]))
        
        print(f"  Found {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(is_training=True):
    """Get image transforms with augmentation for training."""
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


def load_model_for_retraining(model_path: Path, device: torch.device):
    """Load existing model for continued training."""
    print(f"Loading base model: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect architecture from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Detect hidden_dim and num_classes
    if 'classifier.4.weight' in state_dict:
        num_classes = state_dict['classifier.4.weight'].shape[0]
        hidden_dim = state_dict['classifier.4.weight'].shape[1]
    else:
        num_classes = 2
        hidden_dim = 256
    
    # Build model
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, num_classes)
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model, hidden_dim


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in dataloader:
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
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def retrain_with_feedback(
    feedback_dataset_dir: Path,
    base_model_path: Path,
    output_model_path: Path,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 5e-5,  # Lower LR for fine-tuning
    device: str = 'auto'
):
    """
    Retrain model with feedback data.
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Device: {device}")
    print(f"Feedback dataset: {feedback_dataset_dir}")
    print(f"Base model: {base_model_path}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = FeedbackDataset(feedback_dataset_dir / 'train', get_transforms(True))
    val_dataset = FeedbackDataset(feedback_dataset_dir / 'val', get_transforms(False))
    
    if len(train_dataset) == 0:
        print("ERROR: No training images found!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load model
    print("\nLoading model...")
    model, hidden_dim = load_model_for_retraining(base_model_path, device)
    
    # Use lower learning rate for fine-tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    print(f"\nStarting retraining for {epochs} epochs...")
    print("=" * 60)
    
    best_val_acc = 0
    history = []
    
    for epoch in range(epochs):
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
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'classes': ['cnie_front', 'cnie_back'],
                'retrained_at': datetime.now().isoformat(),
                'feedback_data': str(feedback_dataset_dir)
            }, output_model_path)
            print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")
    
    print("=" * 60)
    print(f"Retraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_model_path}")
    
    # Save history
    history_path = output_model_path.parent / (output_model_path.stem + "_retrain_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Retrain with feedback data')
    parser.add_argument('--feedback-dir', type=Path, default=None,
                       help='Path to feedback dataset (auto-detected if not provided)')
    parser.add_argument('--base-model', type=Path, 
                       default=Path.home() / 'retin-verify/models/classification/cnie_front_back_real.pth',
                       help='Base model to start from')
    parser.add_argument('--output', type=Path, 
                       default=Path.home() / 'retin-verify/models/classification/cnie_front_back_real_v2.pth',
                       help='Output model path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    
    args = parser.parse_args()
    
    # Auto-detect feedback directory
    if args.feedback_dir is None:
        collector = get_feedback_collector()
        args.feedback_dir = collector.prepare_retraining_dataset()
        print(f"Auto-prepared dataset: {args.feedback_dir}")
    
    retrain_with_feedback(
        feedback_dataset_dir=args.feedback_dir,
        base_model_path=args.base_model,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == '__main__':
    main()
