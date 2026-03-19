#!/usr/bin/env python3
"""
Command-line training script for document classification.
Optimized for running from VS Code terminal.

IMPROVEMENTS (2026-03-16):
- Fixed DataLoader deadlock (num_workers=0 for Google Drive)
- Added Early Stopping to prevent unnecessary training
- Enhanced data augmentation (rotation, color jitter, etc.)
- Differential learning rates (feature extractor vs classifier)
- Label smoothing for regularization
- Increased dropout (0.5)
- Monitoring: gradient norms, prediction confidence
- Watchdog for deadlock detection
- Optional: Copy data to local SSD for faster I/O

Usage:
    # Local test (CPU)
    python training/classification/train_cli.py \
        --data-dir data/cnie_dataset_10k \
        --train-annotations data/processed/classification/train.json \
        --val-annotations data/processed/classification/val.json \
        --epochs 1 --batch-size 4 --device cpu

    # Colab training (GPU) - RECOMMENDED
    python training/classification/train_cli.py \
        --data-dir /content/data \
        --train-annotations data/processed/classification/train.json \
        --val-annotations data/processed/classification/val.json \
        --epochs 50 --batch-size 32 --device cuda --fp16 \
        --copy-to-local  # Copy data to SSD first
"""

import argparse
import json
import logging
import sys
import time
import shutil
import signal
import threading
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def setup_logging(log_dir: Path):
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'train_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_acc = 0
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, val_acc, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.getLogger(__name__).info(
                    f'EarlyStopping counter: {self.counter}/{self.patience}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0


class TrainingWatchdog:
    """Watchdog to detect training deadlocks."""
    def __init__(self, timeout_seconds=300):
        self.timeout = timeout_seconds
        self.last_progress = time.time()
        self._stop_event = threading.Event()
        self._thread = None
        
    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1)
            
    def update_progress(self):
        self.last_progress = time.time()
        
    def _monitor(self):
        logger = logging.getLogger(__name__)
        while not self._stop_event.is_set():
            time.sleep(60)
            elapsed = time.time() - self.last_progress
            if elapsed > self.timeout:
                logger.error(f'WATCHDOG ALERT: No progress for {int(elapsed)}s! Possible deadlock.')
                logger.error('Consider: reducing num_workers, copying data to local SSD, or checking GPU.')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train document classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data-dir', type=Path, required=True,
                            help='Directory containing dataset images')
    data_group.add_argument('--train-annotations', type=Path, required=True,
                            help='Training annotations JSON file')
    data_group.add_argument('--val-annotations', type=Path, required=True,
                            help='Validation annotations JSON file')
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--model-dir', type=Path, default='models/classification',
                             help='Directory to save trained models')
    model_group.add_argument('--base-model', type=str, default='efficientnet_b0',
                             choices=['efficientnet_b0', 'resnet50', 'mobilenet_v2'],
                             help='Base model architecture')
    model_group.add_argument('--num-classes', type=int, default=4,
                             help='Number of document classes')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for training')
    train_group.add_argument('--learning-rate', type=float, default=1e-4,
                             help='Learning rate')
    train_group.add_argument('--weight-decay', type=float, default=0.01,
                             help='Weight decay for optimizer')
    train_group.add_argument('--device', type=str, default='auto',
                             choices=['auto', 'cpu', 'cuda'],
                             help='Device to use for training')
    
    # Optimization arguments
    optim_group = parser.add_argument_group('Optimization')
    optim_group.add_argument('--fp16', action='store_true',
                             help='Use mixed precision training (faster on GPU)')
    optim_group.add_argument('--gradient-accumulation', type=int, default=1,
                             help='Gradient accumulation steps (effective batch = batch_size * this)')
    
    # Checkpointing arguments
    checkpoint_group = parser.add_argument_group('Checkpointing')
    checkpoint_group.add_argument('--save-every', type=int, default=10,
                                  help='Save checkpoint every N epochs')
    checkpoint_group.add_argument('--resume-from', type=Path, default=None,
                                  help='Resume training from checkpoint')
    
    # Logging arguments
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--log-dir', type=Path, default='logs/classification',
                           help='Directory for TensorBoard logs')
    log_group.add_argument('--log-every', type=int, default=10,
                           help='Log every N batches')
    log_group.add_argument('--experiment-name', type=str, default=None,
                           help='Experiment name for logging')
    
    # New: Regularization arguments
    reg_group = parser.add_argument_group('Regularization (NEW)')
    reg_group.add_argument('--label-smoothing', type=float, default=0.1,
                           help='Label smoothing factor (0 = disabled)')
    reg_group.add_argument('--dropout', type=float, default=0.5,
                           help='Dropout rate in classifier head')
    reg_group.add_argument('--early-stopping-patience', type=int, default=5,
                           help='Epochs to wait before early stopping (0 = disabled)')
    
    # New: Data arguments
    data_group = parser.add_argument_group('Data Loading (NEW)')
    data_group.add_argument('--num-workers', type=int, default=0,
                           help='DataLoader workers (0 = disable multiprocessing, recommended for Google Drive)')
    data_group.add_argument('--copy-to-local', action='store_true',
                           help='Copy data to local SSD before training (recommended for Google Drive)')
    data_group.add_argument('--local-data-dir', type=Path, default='/content/local_data',
                           help='Local directory for data when using --copy-to-local')
    
    # New: Learning rate arguments
    lr_group = parser.add_argument_group('Learning Rates (NEW)')
    lr_group.add_argument('--feature-lr', type=float, default=1e-5,
                          help='Learning rate for pretrained feature extractor (backbone)')
    lr_group.add_argument('--classifier-lr', type=float, default=1e-3,
                          help='Learning rate for new classifier head')
    
    return parser.parse_args()


def get_device(device_str: str):
    """Get torch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def create_model(base_model: str, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
    """Create model with improved regularization."""
    from torchvision import models
    
    if base_model == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # Replace classifier with deeper network and higher dropout
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),  # 60% of main dropout
            nn.Linear(512, num_classes)
        )
    elif base_model == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        )
    elif base_model == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        )
    elif base_model == 'mobilenet_v3_small':
        # Smaller model for small datasets
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {base_model}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, 
                scaler=None, grad_accumulation=1, log_every=10, writer=None, epoch=0,
                max_grad_norm=1.0):
    """Train for one epoch with improved monitoring."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_confidences = []
    
    from tqdm import tqdm
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accumulation
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / grad_accumulation
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accumulation == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Stats
        total_loss += loss.item() * grad_accumulation
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Track prediction confidence
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            confidences = probs.max(dim=1).values
            all_confidences.extend(confidences.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
        
        # Log to TensorBoard
        if writer is not None and batch_idx % log_every == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/acc', 100.*correct/total, global_step)
            
            # Log average confidence (detect overfitting)
            avg_conf = sum(all_confidences[-100:]) / min(len(all_confidences), 100)
            writer.add_scalar('train/avg_confidence', avg_conf, global_step)
            
            # Log gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar('train/grad_norm', total_norm, global_step)
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100. * correct / total
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    return avg_loss, avg_acc, avg_confidence


def validate(model, dataloader, criterion, device):
    """Validate model with confidence tracking."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_confidences = []
    
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
            
            # Track confidence
            probs = F.softmax(outputs, dim=1)
            confidences = probs.max(dim=1).values
            all_confidences.extend(confidences.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100. * correct / total
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    high_conf_count = sum(1 for c in all_confidences if c > 0.99)
    
    return avg_loss, avg_acc, avg_confidence, high_conf_count


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    # Log startup info
    logger.info("=" * 70)
    logger.info("Document Classification Training")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("=" * 70)
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Model: {args.base_model}")
    logger.info(f"Classes: {args.num_classes}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Mixed precision: {args.fp16}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation}")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"DataLoader workers: {args.num_workers}")
    logger.info(f"Copy to local: {args.copy_to_local}")
    logger.info("=" * 70)
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create model with configurable dropout
    logger.info(f"Creating model: {args.base_model}")
    model = create_model(args.base_model, args.num_classes, dropout=args.dropout)
    model = model.to(device)
    logger.info(f"Dropout rate: {args.dropout}")
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == 'cuda' else None
    
    # Copy data to local SSD if requested (prevents Google Drive deadlock)
    if args.copy_to_local:
        logger.info("Copying data to local SSD...")
        if not args.local_data_dir.exists():
            args.local_data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(args.data_dir, args.local_data_dir, dirs_exist_ok=True)
            logger.info(f"Data copied to {args.local_data_dir}")
        else:
            logger.info(f"Using existing local data at {args.local_data_dir}")
        args.data_dir = args.local_data_dir
    
    # Create datasets with enhanced augmentation
    logger.info("Loading datasets...")
    try:
        from torchvision import datasets, transforms
        
        # Enhanced training augmentation (prevents overfitting)
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            # Geometric augmentations
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.1, 0.1),      # 10% shift
                scale=(0.9, 1.1),          # 10% zoom
                shear=5                     # Slight shear
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            # Photometric augmentations
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),  # Occlusion simulation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load annotations
        with open(args.train_annotations) as f:
            train_ann = json.load(f)
        with open(args.val_annotations) as f:
            val_ann = json.load(f)
        
        logger.info(f"Train samples: {len(train_ann)}")
        logger.info(f"Val samples: {len(val_ann)}")
        
        # Create simple dataset (implement proper one based on your data)
        from torch.utils.data import Dataset
        from PIL import Image
        
        class DocumentDataset(Dataset):
            def __init__(self, annotations, data_dir, transform=None):
                self.annotations = annotations
                self.data_dir = Path(data_dir)
                self.transform = transform
                
            def __len__(self):
                return len(self.annotations)
            
            def __getitem__(self, idx):
                ann = self.annotations[idx]
                img_path = self.data_dir / ann['image_path']
                image = Image.open(img_path).convert('RGB')
                
                # Map document types to class indices
                class_map = {'passport': 0, 'cnie_front': 1, 'cnie_back': 2, 'carte_grise': 3}
                label = class_map.get(ann['document_type'], 0)
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        train_dataset = DocumentDataset(train_ann, args.data_dir, transform_train)
        val_dataset = DocumentDataset(val_ann, args.data_dir, transform_val)
        
        # Use configurable num_workers (default 0 for Google Drive safety)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda' and args.num_workers > 0),
            persistent_workers=(args.num_workers > 0),
            timeout=60 if args.num_workers > 0 else 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda' and args.num_workers > 0),
            persistent_workers=(args.num_workers > 0),
            timeout=60 if args.num_workers > 0 else 0
        )
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # Setup training with label smoothing (prevents overconfidence)
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        logger.info(f"Using label smoothing: {args.label_smoothing}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates (protects pretrained features)
    # Split parameters into feature extractor and classifier
    feature_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            feature_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': feature_params, 'lr': args.feature_lr, 'weight_decay': args.weight_decay},
        {'params': classifier_params, 'lr': args.classifier_lr, 'weight_decay': args.weight_decay}
    ])
    
    logger.info(f"Feature extractor LR: {args.feature_lr}, Classifier LR: {args.classifier_lr}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    
    # Early stopping
    early_stopping = None
    if args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=0.001,
            verbose=True
        )
        logger.info(f"Early stopping enabled (patience: {args.early_stopping_patience})")
    
    # Setup TensorBoard
    experiment_name = args.experiment_name or f"{args.base_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(args.log_dir / experiment_name)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Create output directory
    args.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logger.info("Starting training...")
    start_time = time.time()
    
    # Start watchdog
    watchdog = TrainingWatchdog(timeout_seconds=300)
    watchdog.start()
    
    try:
        for epoch in range(start_epoch, args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, train_conf = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler, args.gradient_accumulation, args.log_every, writer, epoch
            )
            
            # Validate
            val_loss, val_acc, val_conf, high_conf_count = validate(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Log
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Conf: {train_conf:.3f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Conf: {val_conf:.3f}")
            logger.info(f"High confidence predictions (>99%): {high_conf_count}/{len(val_loader.dataset)}")
            logger.info(f"Epoch time: {epoch_time:.1f}s")
            
            # TensorBoard logging
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)
            writer.add_scalar('val/confidence', val_conf, epoch)
            writer.add_scalar('train/confidence', train_conf, epoch)
            writer.add_scalar('train/lr_feature', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('train/lr_classifier', optimizer.param_groups[1]['lr'], epoch)
            
            # Overfitting warning
            if train_acc > 95 and val_acc < train_acc - 10:
                logger.warning("OVERFITTING DETECTED: Large gap between train and val accuracy!")
            if train_conf > 0.99 and val_conf < 0.8:
                logger.warning("OVERCONFIDENCE: Model is too confident on training data!")
            
            # Update watchdog
            watchdog.update_progress()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                checkpoint_path = args.model_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = args.model_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, best_model_path)
                logger.info(f"Saved best model (val_acc: {val_acc:.2f}%)")
            
            # Early stopping check
            if early_stopping:
                early_stopping(val_loss, val_acc, epoch)
                if early_stopping.early_stop:
                    logger.info(f"\n{'='*70}")
                    logger.info(f"EARLY STOPPING triggered at epoch {epoch + 1}")
                    logger.info(f"Best epoch: {early_stopping.best_epoch + 1}")
                    logger.info(f"Best val loss: {early_stopping.best_loss:.4f}")
                    logger.info(f"Best val acc: {early_stopping.best_acc:.2f}%")
                    logger.info(f"{'='*70}\n")
                    break
        
        # Save final model
        final_model_path = args.model_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
        }, final_model_path)
        
        # Save history
        history_path = args.model_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Training complete
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("Training completed!")
        logger.info(f"Total time: {elapsed / 3600:.2f} hours")
        logger.info(f"Best val accuracy: {best_val_acc:.2f}%")
        logger.info(f"Models saved to: {args.model_dir}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        # Save interrupted checkpoint
        interrupt_path = args.model_dir / 'interrupted_checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, interrupt_path)
        logger.info(f"Saved interrupted checkpoint: {interrupt_path}")
        raise
    
    finally:
        watchdog.stop()
        writer.close()
        logger.info("Training watchdog stopped")


if __name__ == '__main__':
    main()
