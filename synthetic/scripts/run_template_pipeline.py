"""
Template-based Data Acquisition Pipeline for Retin-Verify
Main runner for 2D template-based synthetic data generation.

This module is aligned with server.py to produce the same output format.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
import sys
from datetime import datetime

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'archive'))  # Add archive for annotation_utils
from template_document_generator import TemplateDocumentGenerator
from annotation_utils import (
    create_dataset_manifest, 
    AnnotationConverter,
    AnnotationValidator
)


class TemplatePipelineConfig:
    """Configuration for template-based data acquisition pipeline."""
    
    DEFAULT_CONFIG = {
        'dataset_name': 'Retin-Verify Template-based Synthetic ID Documents',
        'version': '1.0.0',
        
        # Document types to generate
        'document_types': {
            'passport': {
                'template': 'real/passport_template.jpg',
                'num_samples': 1000,
                'augmentations': {
                    'perspective_angle': [-25, 25],
                    'brightness': [-30, 30],
                    'contrast': [0.8, 1.2],
                    'gamma': [0.8, 1.2],
                    'blur_probability': 0.3,
                    'noise_probability': 0.3,
                }
            },
            'cnie_front': {
                'template': 'real/cnie_front_template.jpg',
                'num_samples': 500,
                'augmentations': {
                    'perspective_angle': [-25, 25],
                    'brightness': [-30, 30],
                    'contrast': [0.8, 1.2],
                    'gamma': [0.8, 1.2],
                }
            },
            'cnie_back': {
                'template': 'real/cnie_back_template.jpg',
                'num_samples': 500,
                'augmentations': {
                    'perspective_angle': [-25, 25],
                    'brightness': [-30, 30],
                    'contrast': [0.8, 1.2],
                    'gamma': [0.8, 1.2],
                }
            },
            'cnie_paired': {
                'template': 'real/cnie_front_template.jpg',  # Front template
                'template_back': 'real/cnie_back_template.jpg',
                'num_samples': 500,
                'augmentations': {
                    'perspective_angle': [-25, 25],
                    'brightness': [-30, 30],
                    'contrast': [0.8, 1.2],
                    'gamma': [0.8, 1.2],
                }
            }
        },
        
        # Output settings
        'output': {
            'base_dir': 'data/synthetic',
            'image_format': 'JPEG',
            'image_quality': 95,
            'save_annotations': True,
            'save_metadata': True
        },
        
        # Dataset splits
        'splits': {
            'train': 0.8,
            'val': 0.1,
            'test': 0.1
        },
        
        # Export formats
        'export_formats': ['coco', 'yolo', 'json'],
        
        # Paths - aligned with server.py
        'paths': {
            'templates_dir': 'synthetic/templates',
            'backgrounds_dir': 'synthetic/backgrounds',
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Load configuration from file or use defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._deep_update(self.config, user_config)
    
    def _deep_update(self, d: Dict, u: Dict):
        """Recursively update dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
    
    def save(self, path: Path):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get config value by dot-notation key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class TemplateDataPipeline:
    """Pipeline for template-based data generation - aligned with server.py behavior."""
    
    def __init__(
        self, 
        config: TemplatePipelineConfig, 
        doc_config_file: Optional[Path] = None,
        config_front: Optional[Path] = None,
        config_back: Optional[Path] = None,
        arabic_font: Optional[Path] = None, 
        face_photos_dir: Optional[Path] = None,
        template_dir: Optional[Path] = None,
        fast_preview: bool = False
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            doc_config_file: Generic document config (backward compatibility)
            config_front: CNIE front config (for paired generation)
            config_back: CNIE back config (for paired generation)
            arabic_font: Path to Arabic font
            face_photos_dir: Path to VGGFace2 dataset
            template_dir: Override template directory
            fast_preview: Skip augmentations for faster generation
        """
        self.config = config
        self.base_dir = Path(config.get('output.base_dir'))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.fast_preview = fast_preview
        
        # Use provided template_dir or get from config
        self.template_dir = Path(template_dir) if template_dir else Path(config.get('paths.templates_dir'))
        
        # Initialize generator with all config options (aligned with server.py)
        self.generator = TemplateDocumentGenerator(
            template_dir=self.template_dir,
            output_dir=self.base_dir,
            config_file=doc_config_file,
            config_front=config_front,
            config_back=config_back,
            backgrounds_dir=Path(config.get('paths.backgrounds_dir')) if config.get('paths.backgrounds_dir') else None,
            arabic_font_path=arabic_font,
            face_photos_dir=face_photos_dir
        )
        
        # Store configs for paired generation
        self.config_front = config_front
        self.config_back = config_back
        
        # Statistics
        self.stats = {
            'generated': 0,
            'failed': 0,
            'by_type': {}
        }
    
    def run(
        self, 
        doc_types: Optional[List[str]] = None, 
        num_samples: Optional[int] = None,
        generate_paired: bool = False
    ):
        """
        Run the complete data generation pipeline.
        
        Args:
            doc_types: List of document types to generate (e.g., ['cnie_front', 'cnie_back'])
            num_samples: Override number of samples from config
            generate_paired: If True, generate paired CNIE front/back samples
        """
        print("=" * 70)
        print("Retin-Verify Template-based Data Generation Pipeline")
        print("=" * 70)
        print(f"Output directory: {self.base_dir}")
        print(f"Template directory: {self.template_dir}")
        print(f"Fast preview mode: {self.fast_preview}")
        print(f"Start time: {datetime.now().isoformat()}")
        print()
        
        # Determine which document types to generate
        doc_type_configs = self.config.config['document_types']
        if doc_types:
            doc_type_configs = {k: v for k, v in doc_type_configs.items() if k in doc_types}
        
        # Phase 1: Generate synthetic documents
        print("Phase 1: Generating synthetic documents from templates...")
        
        # Handle paired generation first if requested
        if generate_paired or 'cnie_paired' in doc_type_configs:
            self._generate_paired_cnie(num_samples)
        
        # Generate individual document types
        for doc_type, spec in doc_type_configs.items():
            if doc_type == 'cnie_paired':
                continue  # Already handled above
            
            n_samples = num_samples if num_samples else spec['num_samples']
            
            # Check if template exists
            if doc_type not in self.generator.templates:
                print(f"⚠️  Template for {doc_type} not found, skipping...")
                continue
            
            print(f"\n📄 Generating {n_samples} samples for {doc_type}...")
            generated_paths = self.generator.generate_dataset(
                doc_type, 
                n_samples, 
                fast_preview=self.fast_preview
            )
            
            self.stats['by_type'][doc_type] = len(generated_paths)
            self.stats['generated'] += len(generated_paths)
        
        # Phase 2: Validate annotations
        print("\nPhase 2: Validating annotations...")
        self._validate_annotations()
        
        # Phase 3: Create dataset splits
        print("\nPhase 3: Creating dataset splits...")
        self._create_splits()
        
        # Phase 4: Export to different formats
        print("\nPhase 4: Exporting to different formats...")
        self._export_formats()
        
        # Print summary
        print("\n" + "=" * 70)
        print("Pipeline Complete!")
        print("=" * 70)
        print(f"Total samples generated: {self.stats['generated']}")
        print(f"Failed: {self.stats['failed']}")
        for doc_type, count in self.stats['by_type'].items():
            print(f"  - {doc_type}: {count}")
        print(f"\nOutput location: {self.base_dir}")
    
    def _generate_paired_cnie(self, num_samples: Optional[int] = None):
        """Generate paired CNIE front/back samples."""
        if 'cnie_front' not in self.generator.templates or 'cnie_back' not in self.generator.templates:
            print("⚠️  Both CNIE front and back templates required for paired generation")
            return
        
        spec = self.config.config['document_types'].get('cnie_paired', {})
        n_samples = num_samples if num_samples else spec.get('num_samples', 500)
        
        print(f"\n📄 Generating {n_samples} paired CNIE samples (front + back)...")
        
        results = self.generator.generate_paired_cnie_dataset(
            num_pairs=n_samples,
            fast_preview=self.fast_preview
        )
        
        self.stats['by_type']['cnie_paired'] = len(results)
        self.stats['generated'] += len(results) * 2  # Count both front and back
    
    def _validate_annotations(self):
        """Validate generated annotations."""
        validator = AnnotationValidator()
        
        total = 0
        valid = 0
        
        # Check both regular samples and paired samples
        for doc_type_dir in self.base_dir.glob('*/'):
            for sample_dir in doc_type_dir.glob('*/'):
                ann_file = sample_dir / 'annotations.json'
                
                if ann_file.exists():
                    try:
                        with open(ann_file) as f:
                            annotation = json.load(f)
                        
                        total += 1
                        is_valid, errors = validator.validate_annotations(annotation)
                        
                        if is_valid:
                            valid += 1
                        else:
                            annotation['_validation_errors'] = errors
                            with open(ann_file, 'w') as f:
                                json.dump(annotation, f, indent=2)
                    except Exception as e:
                        print(f"  ⚠️  Error validating {ann_file}: {e}")
        
        if total > 0:
            print(f"  Validated {total} annotations")
            print(f"  Valid: {valid} ({100*valid/total:.1f}%)")
            print(f"  Invalid: {total - valid}")
        else:
            print("  No annotations to validate")
    
    def _create_splits(self):
        """Create train/val/test splits."""
        # Collect all annotations (including paired samples)
        all_annotations = []
        
        for doc_type_dir in self.base_dir.glob('*/'):
            doc_type = doc_type_dir.name
            
            for sample_dir in doc_type_dir.glob('*/'):
                ann_file = sample_dir / 'annotations.json'
                image_file = sample_dir / 'image.jpg'
                
                # For paired samples, check subdirectories
                if not image_file.exists():
                    front_image = sample_dir / 'front' / 'image.jpg'
                    back_image = sample_dir / 'back' / 'image.jpg'
                    if front_image.exists() and back_image.exists():
                        # This is a paired sample - use the pair annotation
                        if ann_file.exists():
                            try:
                                with open(ann_file) as f:
                                    ann = json.load(f)
                                all_annotations.append({
                                    'id': ann.get('pair_id', sample_dir.name),
                                    'document_type': 'cnie_paired',
                                    'image_path': str(sample_dir),
                                    'annotation_path': str(ann_file),
                                    'is_paired': True
                                })
                            except Exception as e:
                                print(f"  ⚠️  Error reading paired annotation {ann_file}: {e}")
                        continue
                
                if ann_file.exists() and image_file.exists():
                    try:
                        with open(ann_file) as f:
                            ann = json.load(f)
                        all_annotations.append({
                            'id': ann.get('sample_id', sample_dir.name),
                            'document_type': ann.get('document_type', doc_type),
                            'image_path': str(image_file),
                            'annotation_path': str(ann_file)
                        })
                    except Exception as e:
                        print(f"  ⚠️  Error reading annotation {ann_file}: {e}")
        
        if not all_annotations:
            print("  No samples to split")
            return
        
        # Shuffle and split
        random.shuffle(all_annotations)
        
        train_ratio = self.config.get('splits.train', 0.8)
        val_ratio = self.config.get('splits.val', 0.1)
        
        n_total = len(all_annotations)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': all_annotations[:n_train],
            'val': all_annotations[n_train:n_train + n_val],
            'test': all_annotations[n_train + n_val:]
        }
        
        manifest = {
            'dataset_name': self.config.get('dataset_name'),
            'created_at': datetime.now().isoformat(),
            'total_samples': n_total,
            'splits': {
                'train': {'count': len(splits['train']), 'samples': [s['id'] for s in splits['train']]},
                'val': {'count': len(splits['val']), 'samples': [s['id'] for s in splits['val']]},
                'test': {'count': len(splits['test']), 'samples': [s['id'] for s in splits['test']]}
            },
            'samples': all_annotations
        }
        
        manifest_path = self.base_dir / 'dataset_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"  Dataset splits created:")
        print(f"    Train: {len(splits['train'])}")
        print(f"    Val: {len(splits['val'])}")
        print(f"    Test: {len(splits['test'])}")
    
    def _export_formats(self):
        """Export annotations to different formats."""
        # Collect annotations (including paired samples)
        # For paired samples, we need to flatten them into separate entries
        annotations = []
        
        for doc_type_dir in self.base_dir.glob('*/'):
            for sample_dir in doc_type_dir.glob('*/'):
                ann_file = sample_dir / 'annotations.json'
                if ann_file.exists():
                    try:
                        with open(ann_file) as f:
                            ann = json.load(f)
                        
                        # Handle paired annotations - flatten into separate entries
                        if 'front' in ann and 'back' in ann:
                            # Add front as separate entry
                            front_ann = {
                                'sample_id': f"{ann.get('pair_id', '0')}_front",
                                'document_type': ann['front']['document_type'],
                                'identity': ann.get('identity', {}),
                                'image_path': ann['front']['image_path'],
                                'image_size': ann['front']['image_size'],
                                'bounding_boxes': ann['front']['bounding_boxes']
                            }
                            annotations.append(front_ann)
                            
                            # Add back as separate entry
                            back_ann = {
                                'sample_id': f"{ann.get('pair_id', '0')}_back",
                                'document_type': ann['back']['document_type'],
                                'identity': ann.get('identity', {}),
                                'image_path': ann['back']['image_path'],
                                'image_size': ann['back']['image_size'],
                                'bounding_boxes': ann['back']['bounding_boxes']
                            }
                            annotations.append(back_ann)
                        else:
                            annotations.append(ann)
                    except Exception as e:
                        print(f"  ⚠️  Error reading {ann_file}: {e}")
        
        if not annotations:
            print("  No annotations to export")
            return
        
        export_dir = self.base_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        formats = self.config.get('export_formats', [])
        
        if 'coco' in formats:
            print("  Exporting to COCO format...")
            coco_path = export_dir / 'annotations_coco.json'
            try:
                AnnotationConverter.to_coco(annotations, coco_path)
                print(f"    Saved to: {coco_path}")
            except Exception as e:
                print(f"    ⚠️  Error exporting to COCO: {e}")
        
        if 'yolo' in formats:
            print("  Exporting to YOLO format...")
            class_map = {}
            class_id = 0
            for ann in annotations:
                for bbox in ann.get('bounding_boxes', []):
                    field = bbox.get('field')
                    if field not in class_map:
                        class_map[field] = class_id
                        class_id += 1
            
            try:
                yolo_dir = export_dir / 'yolo'
                AnnotationConverter.to_yolo(annotations, yolo_dir, class_map)
                
                with open(yolo_dir / 'classes.txt', 'w') as f:
                    for cls_name, cls_id in sorted(class_map.items(), key=lambda x: x[1]):
                        f.write(f"{cls_name}\n")
                
                print(f"    Saved to: {yolo_dir}")
            except Exception as e:
                print(f"    ⚠️  Error exporting to YOLO: {e}")
        
        # Always export a JSON summary
        print("  Exporting JSON summary...")
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_samples': len(annotations),
            'by_type': {}
        }
        for ann in annotations:
            doc_type = ann.get('document_type', 'unknown')
            if doc_type not in summary['by_type']:
                summary['by_type'][doc_type] = 0
            summary['by_type'][doc_type] += 1
        
        summary_path = export_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"    Saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Retin-Verify Template-based Data Generation Pipeline'
    )
    parser.add_argument(
        '--config', 
        type=Path, 
        default='configs/template_pipeline.json',
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--doc-config',
        type=Path,
        default=None,
        help='Path to document field configuration file (e.g., cnie_front_custom.json)'
    )
    parser.add_argument(
        '--config-front',
        type=Path,
        default=None,
        help='Path to CNIE front config (for paired generation)'
    )
    parser.add_argument(
        '--config-back',
        type=Path,
        default=None,
        help='Path to CNIE back config (for paired generation)'
    )
    parser.add_argument(
        '--init-config',
        action='store_true',
        help='Create default configuration file and exit'
    )
    parser.add_argument(
        '--doc-type',
        type=str,
        choices=['passport', 'cnie_front', 'cnie_back', 'cnie_paired', 'all'],
        default='all',
        help='Document type to generate'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Override number of samples per type'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--template-dir',
        type=Path,
        default=None,
        help='Path to templates directory (aligned with server.py)'
    )
    parser.add_argument(
        '--sample-only',
        action='store_true',
        help='Generate only a small sample (5 per type) for testing'
    )
    parser.add_argument(
        '--arabic-font',
        type=Path,
        default=None,
        help='Path to Arabic TrueType font file for Arabic text rendering'
    )
    parser.add_argument(
        '--face-photos-dir',
        type=Path,
        default=None,
        help='Path to VGGFace2 dataset directory for real face photos'
    )
    parser.add_argument(
        '--fast-preview',
        action='store_true',
        help='Fast preview mode: skip augmentations for quicker generation (aligned with server.py)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect if user passed a doc-config as --config
    if args.config and args.config.exists() and not args.doc_config:
        try:
            with open(args.config, 'r') as f:
                test_config = json.load(f)
            # If config has 'fields' key, it's a doc-config
            if 'fields' in test_config or any(k in ['cnie_front', 'cnie_back', 'passport'] for k in test_config.keys()):
                print(f"⚠️  Detected document config in --config. Using it as --doc-config.")
                print(f"   For future runs, use: --doc-config {args.config}")
                args.doc_config = args.config
                args.config = 'configs/template_pipeline.json'
        except:
            pass
    
    if args.init_config:
        config = TemplatePipelineConfig()
        config.save(args.config)
        print(f"Default configuration saved to: {args.config}")
        return
    
    # Load configuration
    config = TemplatePipelineConfig(args.config)
    
    if args.output_dir:
        config.config['output']['base_dir'] = str(args.output_dir)
    
    # Determine document types
    if args.doc_type == 'all':
        doc_types = None  # Will use all from config
        generate_paired = True
    elif args.doc_type == 'cnie_paired':
        doc_types = None
        generate_paired = True
    else:
        doc_types = [args.doc_type]
        generate_paired = False
    
    # Sample mode
    num_samples = 5 if args.sample_only else args.num_samples
    
    # Run pipeline with all options (aligned with server.py)
    pipeline = TemplateDataPipeline(
        config, 
        doc_config_file=args.doc_config, 
        config_front=args.config_front,
        config_back=args.config_back,
        arabic_font=args.arabic_font,
        face_photos_dir=args.face_photos_dir,
        template_dir=args.template_dir,
        fast_preview=args.fast_preview
    )
    pipeline.run(
        doc_types=doc_types, 
        num_samples=num_samples,
        generate_paired=generate_paired
    )


if __name__ == '__main__':
    main()
