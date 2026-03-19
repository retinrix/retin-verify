#!/usr/bin/env python3
"""
Data Augmentation Tool for CNIE Classification
Generate rich training dataset from few real images.

Usage:
    # Basic usage - generate 150 augmentations per image
    python augment_dataset.py --input-dir ./my_cards --output-dir ./augmented
    
    # Custom settings
    python augment_dataset.py \
        --input-dir ./my_cards \
        --output-dir ./augmented \
        --target-per-image 200 \
        --train-ratio 0.8 \
        --seed 42
    
    # Preview augmentations (don't save)
    python augment_dataset.py \
        --input-dir ./my_cards \
        --preview 9

Input directory structure:
    my_cards/
    ├── cnie_front/
    │   ├── card1.jpg
    │   └── card2.jpg
    ├── cnie_back/
    │   └── card1.jpg
    ├── passport/
    └── carte_grise/

Output directory structure:
    augmented/
    ├── train/
    │   ├── cnie_front/
    │   │   ├── aug_000.jpg
    │   │   └── ...
    │   ├── cnie_back/
    │   └── ...
    └── val/
        └── ... (same structure)
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Installing...")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Error: PIL required. Install with: pip install Pillow")
    raise


class DocumentAugmenter:
    """
    Realistic augmentations for document images.
    Keeps augmentations within realistic bounds for ID documents.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def augment(self, image: np.ndarray, intensity: str = "medium") -> np.ndarray:
        """
        Apply realistic augmentations to document image.
        
        Args:
            image: Input image (H, W, C) in RGB format
            intensity: 'light', 'medium', or 'strong'
        
        Returns:
            Augmented image
        """
        img = image.copy()
        
        # Define augmentation parameters based on intensity
        if intensity == "light":
            params = {
                'rotation': (-5, 5),
                'scale': (0.95, 1.05),
                'brightness': (0.9, 1.1),
                'contrast': (0.9, 1.1),
                'blur': (0, 1),
                'noise': (0, 10),
            }
        elif intensity == "medium":
            params = {
                'rotation': (-10, 10),
                'scale': (0.9, 1.1),
                'brightness': (0.8, 1.2),
                'contrast': (0.8, 1.2),
                'blur': (0, 2),
                'noise': (5, 20),
            }
        else:  # strong
            params = {
                'rotation': (-15, 15),
                'scale': (0.85, 1.15),
                'brightness': (0.7, 1.3),
                'contrast': (0.7, 1.3),
                'blur': (0, 3),
                'noise': (10, 30),
            }
        
        # Apply augmentations in random order
        augmentations = [
            self._random_rotation,
            self._random_perspective,
            self._random_scale,
            self._random_brightness,
            self._random_contrast,
            self._random_gamma,
            self._random_blur,
            self._random_noise,
        ]
        
        random.shuffle(augmentations)
        
        for aug_func in augmentations:
            if random.random() > 0.3:  # 70% chance to apply each augmentation
                img = aug_func(img, params)
        
        return img
    
    def _random_rotation(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Rotate image within realistic bounds."""
        angle = random.uniform(*params['rotation'])
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    
    def _random_perspective(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Apply slight perspective transform."""
        h, w = img.shape[:2]
        
        # Random perspective corners (max 15% shift)
        max_shift = 0.15
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([
            [random.uniform(0, w*max_shift), random.uniform(0, h*max_shift)],
            [random.uniform(w*(1-max_shift), w), random.uniform(0, h*max_shift)],
            [random.uniform(w*(1-max_shift), w), random.uniform(h*(1-max_shift), h)],
            [random.uniform(0, w*max_shift), random.uniform(h*(1-max_shift), h)]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    
    def _random_scale(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Random scale with padding."""
        scale = random.uniform(*params['scale'])
        h, w = img.shape[:2]
        
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad or crop to original size
        result = np.full_like(img, 128)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        
        y_start = max(0, y_offset)
        y_end = min(h, y_offset + new_h)
        x_start = max(0, x_offset)
        x_end = min(w, x_offset + new_w)
        
        sy_start = max(0, -y_offset)
        sy_end = sy_start + (y_end - y_start)
        sx_start = max(0, -x_offset)
        sx_end = sx_start + (x_end - x_start)
        
        result[y_start:y_end, x_start:x_end] = scaled[sy_start:sy_end, sx_start:sx_end]
        return result
    
    def _random_brightness(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Adjust brightness."""
        factor = random.uniform(*params['brightness'])
        return np.clip(img * factor, 0, 255).astype(np.uint8)
    
    def _random_contrast(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Adjust contrast."""
        factor = random.uniform(*params['contrast'])
        mean = img.mean()
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _random_gamma(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Apply gamma correction."""
        gamma = random.uniform(0.8, 1.2)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    def _random_blur(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Apply Gaussian blur."""
        kernel_size = random.choice([3, 5, 7])
        if kernel_size > params['blur'][1] * 2 + 1:
            return img
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def _random_noise(self, img: np.ndarray, params: Dict) -> np.ndarray:
        """Add Gaussian noise."""
        noise_level = random.uniform(*params['noise'])
        noise = np.random.normal(0, noise_level, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)


def load_images(input_dir: Path) -> Dict[str, List[Tuple[Path, np.ndarray]]]:
    """
    Load all images from input directory.
    
    Returns:
        Dict mapping class names to list of (path, image) tuples
    """
    images_by_class = {}
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images_by_class[class_name] = []
        
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images_by_class[class_name].append((img_path, img))
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
    
    return images_by_class


def create_augmented_dataset(
    input_dir: Path,
    output_dir: Path,
    target_per_image: int = 150,
    train_ratio: float = 0.8,
    seed: int = 42,
    intensities: List[str] = None
):
    """
    Create augmented dataset from real images.
    
    Args:
        input_dir: Directory containing class subdirectories with images
        output_dir: Output directory for augmented dataset
        target_per_image: Number of augmentations to generate per real image
        train_ratio: Ratio of train vs validation split
        seed: Random seed for reproducibility
        intensities: List of augmentation intensities to use
    """
    if intensities is None:
        intensities = ["light"] * 30 + ["medium"] * 50 + ["strong"] * 20
    
    augmenter = DocumentAugmenter(seed=seed)
    
    # Load images
    print(f"Loading images from {input_dir}...")
    images_by_class = load_images(input_dir)
    
    total_real = sum(len(imgs) for imgs in images_by_class.values())
    print(f"Found {total_real} real images across {len(images_by_class)} classes")
    
    for class_name, images in images_by_class.items():
        print(f"  {class_name}: {len(images)} images")
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    for split_dir in [train_dir, val_dir]:
        for class_name in images_by_class.keys():
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Generate augmentations
    print(f"\nGenerating {target_per_image} augmentations per image...")
    
    annotations = {
        "train": [],
        "val": [],
        "metadata": {
            "seed": seed,
            "target_per_image": target_per_image,
            "train_ratio": train_ratio,
            "intensities": intensities
        }
    }
    
    aug_count = 0
    
    for class_name, images in images_by_class.items():
        print(f"\nProcessing {class_name}...")
        
        # Split into train/val
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Generate for train set
        for img_idx, (orig_path, img) in enumerate(train_images):
            for i in range(target_per_image):
                # Cycle through intensities
                intensity = intensities[i % len(intensities)]
                
                # Apply augmentation
                aug_img = augmenter.augment(img, intensity=intensity)
                
                # Save
                aug_filename = f"aug_{img_idx:03d}_{i:04d}.jpg"
                aug_path = train_dir / class_name / aug_filename
                cv2.imwrite(str(aug_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
                annotations["train"].append({
                    "image_path": str(aug_path.relative_to(output_dir)),
                    "document_type": class_name,
                    "original_image": str(orig_path),
                    "augmentation_intensity": intensity,
                    "augmentation_index": i
                })
                
                aug_count += 1
                
                if aug_count % 100 == 0:
                    print(f"  Generated {aug_count} images...", end="\r")
        
        # Copy original images to val set (no augmentation for validation)
        for img_idx, (orig_path, img) in enumerate(val_images):
            val_filename = f"orig_{img_idx:03d}.jpg"
            val_path = val_dir / class_name / val_filename
            cv2.imwrite(str(val_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            annotations["val"].append({
                "image_path": str(val_path.relative_to(output_dir)),
                "document_type": class_name,
                "original_image": str(orig_path)
            })
    
    # Save annotations
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n✅ Generated {aug_count} augmented images")
    print(f"   Train: {len(annotations['train'])} images")
    print(f"   Val: {len(annotations['val'])} images (original, not augmented)")
    print(f"\nDataset saved to: {output_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for class_name in images_by_class.keys():
        train_count = len(list((train_dir / class_name).glob("*.jpg")))
        val_count = len(list((val_dir / class_name).glob("*.jpg")))
        print(f"  {class_name:20s}: {train_count:4d} train, {val_count:3d} val")


def preview_augmentations(input_dir: Path, num_samples: int = 9):
    """Preview augmentations without saving."""
    import matplotlib.pyplot as plt
    
    images_by_class = load_images(input_dir)
    augmenter = DocumentAugmenter(seed=42)
    
    # Get first image from first class
    first_class = list(images_by_class.keys())[0]
    _, img = images_by_class[first_class][0]
    
    # Create preview grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Augmentation Preview - {first_class}", fontsize=16)
    
    intensities = ["light", "medium", "strong"]
    
    for idx, ax in enumerate(axes.flat):
        intensity = intensities[idx // 3]
        aug_img = augmenter.augment(img, intensity=intensity)
        ax.imshow(aug_img)
        ax.set_title(f"{intensity} #{idx % 3 + 1}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_preview.png', dpi=150, bbox_inches='tight')
    print("Preview saved to: augmentation_preview.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Augment document images for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate dataset
    python augment_dataset.py --input-dir ./my_cards --output-dir ./augmented
    
    # More augmentations
    python augment_dataset.py -i ./my_cards -o ./augmented -n 200
    
    # Preview only
    python augment_dataset.py -i ./my_cards --preview 9
        """
    )
    
    parser.add_argument('-i', '--input-dir', type=Path, required=True,
                       help='Input directory with class subdirectories')
    parser.add_argument('-o', '--output-dir', type=Path,
                       help='Output directory for augmented dataset')
    parser.add_argument('-n', '--target-per-image', type=int, default=150,
                       help='Number of augmentations per image (default: 150)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--preview', type=int, metavar='N',
                       help='Preview N augmentations without saving')
    
    args = parser.parse_args()
    
    if args.preview:
        preview_augmentations(args.input_dir, args.preview)
    else:
        if not args.output_dir:
            parser.error("--output-dir is required when not using --preview")
        
        create_augmented_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_per_image=args.target_per_image,
            train_ratio=args.train_ratio,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
