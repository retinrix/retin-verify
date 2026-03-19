#!/usr/bin/env python3
"""
Fine-tune CNIE classifier with 2 classes (front/back only).
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np


class DocumentDataset(Dataset):
    """Dataset for fine-tuning with 2 classes only."""
    
    def __init__(self, annotations: list, transform=None):
        # Filter for only cnie_front and cnie_back
        self.annotations = [a for a in annotations 
                           if a['document_type'] in ['cnie_front', 'cnie_back']]
        self.transform = transform
        
        # Only 2 classes
        self.classes = ['cnie_front', 'cnie_back']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        img_path = Path(ann['image_path'])
        if not img_path.is_absolute():
            # Relative to augmented dir
            img_path = Path('cnie_only_augmented') / img_path
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        doc_type = ann['document_type']
        label = self.class_to_idx[doc_type]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size=224, is_training=True):
    """Get image transforms."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_2class_model(device: torch.device):
    """Create fresh 2-class model from EfficientNet."""
    print("Creating new 2-class model from EfficientNet-B0")
    
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    
    # Modify classifier for 2 classes
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)  # Only 2 classes
    )
    
    model.to(device)
    return model


def load_and_adapt_model(model_path: Path, device: torch.device):
    """Load 4-class model and adapt to 2 classes."""
    print(f"Loading 4-class model from {model_path}")
    
    # Create 4-class architecture
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4)
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Replace classifier with 2-class version
    print("Adapting to 2 classes...")
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    
    model.to(device)
    return model


def freeze_layers(model: nn.Module, freeze_backbone: bool = True):
    """Freeze/unfreeze model layers."""
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print("Froze backbone layers, training only classifier")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("Unfroze all layers")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def fine_tune(
    augmented_dir: Path,
    output_model: Path,
    base_model_path: Path = None,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = 'auto'
):
    """Fine-tune 2-class model."""
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load annotations
    annotations_path = augmented_dir / "annotations.json"
    with open(annotations_path) as f:
        annotations = json.load(f)
    
    train_annotations = annotations['train']
    val_annotations = annotations['val']
    
    # Create datasets
    train_dataset = DocumentDataset(train_annotations, transform=get_transforms(is_training=True))
    val_dataset = DocumentDataset(val_annotations, transform=get_transforms(is_training=False))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load or create model
    if base_model_path and base_model_path.exists():
        model = load_and_adapt_model(base_model_path, device)
    else:
        model = create_2class_model(device)
    
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting fine-tuning...")
    print("=" * 60)
    
    # Phase 1: Train classifier only
    freeze_layers(model, freeze_backbone=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    
    phase1_epochs = max(1, epochs // 3)
    for epoch in range(phase1_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} [Classifier] | "
              f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Phase 2: Full fine-tuning
    if epochs > phase1_epochs:
        freeze_layers(model, freeze_backbone=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr/10, weight_decay=0.01)
        
        for epoch in range(phase1_epochs, epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} [Full] | "
                  f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
                  f"Val: {val_loss:.4f}/{val_acc:.1f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                }, output_model)
                print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")
    
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_model}")
    
    # Save history
    history_path = output_model.parent / (output_model.stem + "_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune 2-class CNIE model')
    parser.add_argument('--augmented-dir', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, help='4-class base model to adapt')
    parser.add_argument('--output-model', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    fine_tune(
        augmented_dir=args.augmented_dir,
        output_model=args.output_model,
        base_model_path=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )


if __name__ == '__main__':
    main()
