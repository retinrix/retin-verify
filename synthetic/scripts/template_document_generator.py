"""
2D Template-based Document Generator for Retin-Verify
Generates synthetic ID document images using real cleaned templates with OpenCV augmentation.
Now supports Arabic text rendering via PIL with dynamic font sizing and offset fine‑tuning.
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
import argparse

# Add parent directory to path
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from identity_generator import AlgerianIdentityGenerator
except ImportError:
    print("Warning: identity_generator module not found")

# Try to import PIL for Arabic text support
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not installed. Arabic text will not render correctly.")


class TemplateDocumentGenerator:
    """Generate synthetic ID documents using real templates with 2D augmentation."""
    
    # Default document specifications (can be overridden by config file)
    DEFAULT_DOCUMENT_SPECS = {
        'passport': {
            'template_file': 'passport_template.jpg',
            'aspect_ratio': 125/88,
            'fields': {
                'surname': {'rel_bbox': [0.30, 0.615, 0.35, 0.035], 'font_scale': 0.85, 'color': (50, 50, 70)},
                'given_names': {'rel_bbox': [0.30, 0.665, 0.40, 0.035], 'font_scale': 0.80, 'color': (50, 50, 70)},
                'date_of_birth': {'rel_bbox': [0.30, 0.71, 0.25, 0.03], 'font_scale': 0.75, 'color': (50, 50, 70)},
                'place_of_birth': {'rel_bbox': [0.30, 0.75, 0.35, 0.03], 'font_scale': 0.75, 'color': (50, 50, 70)},
                'passport_number': {'rel_bbox': [0.72, 0.585, 0.22, 0.03], 'font_scale': 0.8, 'color': (50, 50, 70)},
                'nationality': {'rel_bbox': [0.72, 0.645, 0.20, 0.025], 'font_scale': 0.75, 'color': (50, 50, 70)},
                'sex': {'rel_bbox': [0.72, 0.685, 0.08, 0.025], 'font_scale': 0.75, 'color': (50, 50, 70)},
                'date_of_issue': {'rel_bbox': [0.72, 0.725, 0.20, 0.025], 'font_scale': 0.75, 'color': (50, 50, 70)},
                'date_of_expiry': {'rel_bbox': [0.72, 0.765, 0.20, 0.025], 'font_scale': 0.75, 'color': (50, 50, 70)},
                'mrz_line1': {'rel_bbox': [0.05, 0.905, 0.90, 0.025], 'font_scale': 0.65, 'font': 'ocr_b', 'color': (40, 40, 40)},
                'mrz_line2': {'rel_bbox': [0.05, 0.935, 0.90, 0.025], 'font_scale': 0.65, 'font': 'ocr_b', 'color': (40, 40, 40)},
            },
            'photo_bbox': [0.03, 0.58, 0.24, 0.26],
            'signature_bbox': [0.32, 0.795, 0.28, 0.07],
        },
        'cnie_back': {
            'template_file': 'cnie_back_template.jpg',
            'aspect_ratio': 85.6/54,
            'fields': {
                'surname': {'rel_bbox': [0.13, 0.06, 0.40, 0.07], 'font_scale': 0.95, 'color': (50, 50, 70), 'anchor': 'lm', 'justification': 'left'},
                'given_names': {'rel_bbox': [0.13, 0.13, 0.50, 0.07], 'font_scale': 0.90, 'color': (50, 50, 70), 'anchor': 'lm', 'justification': 'left'},
                'birth_year': {'rel_bbox': [0.38, 0.28, 0.12, 0.08], 'font_scale': 1.0, 'color': (50, 50, 70), 'anchor': 'cm', 'justification': 'center'},
                'mrz_line1': {'rel_bbox': [0.05, 0.70, 0.90, 0.055], 'font_scale': 0.95, 'font': 'ocr_b', 'color': (40, 40, 40), 'anchor': 'cm', 'justification': 'left'},
                'mrz_line2': {'rel_bbox': [0.05, 0.755, 0.90, 0.055], 'font_scale': 0.95, 'font': 'ocr_b', 'color': (40, 40, 40), 'anchor': 'cm', 'justification': 'left'},
                'mrz_line3': {'rel_bbox': [0.05, 0.81, 0.90, 0.055], 'font_scale': 0.95, 'font': 'ocr_b', 'color': (40, 40, 40), 'anchor': 'cm', 'justification': 'left'},
            },
        }
    }
    
    def __init__(
        self,
        template_dir: Path,
        output_dir: Path,
        config_file: Optional[Path] = None,
        config_front: Optional[Path] = None,
        config_back: Optional[Path] = None,
        backgrounds_dir: Optional[Path] = None,
        seed: Optional[int] = None,
        arabic_font_path: Optional[Path] = None,
        face_photos_dir: Optional[Path] = None
    ):
        """Initialize the generator with optional config file and Arabic font.
        
        Args:
            template_dir: Directory containing template images
            output_dir: Output directory for generated samples
            config_file: Generic config file (backward compatibility)
            config_front: Specific config for CNIE front
            config_back: Specific config for CNIE back
            backgrounds_dir: Directory containing background images
            seed: Random seed for reproducibility
            arabic_font_path: Path to Arabic TrueType font file
            face_photos_dir: Path to VGGFace2 dataset directory
        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.backgrounds_dir = Path(backgrounds_dir) if backgrounds_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize document specs
        self.DOCUMENT_SPECS = self.DEFAULT_DOCUMENT_SPECS.copy()
        
        # Load custom configs if provided
        configs_loaded = []
        
        # Load front config if provided
        if config_front and Path(config_front).exists():
            print(f"📂 Loading FRONT config from: {config_front}")
            self._load_config_for_doc_type(config_front, 'cnie_front')
            configs_loaded.append('front')
        
        # Load back config if provided
        if config_back and Path(config_back).exists():
            print(f"📂 Loading BACK config from: {config_back}")
            self._load_config_for_doc_type(config_back, 'cnie_back')
            configs_loaded.append('back')
        
        # Fallback to generic config for backward compatibility
        if not configs_loaded and config_file and Path(config_file).exists():
            print(f"📂 Loading config from: {config_file}")
            self._load_config(config_file)
            configs_loaded.append('generic')
        
        if configs_loaded:
            print(f"✅ Loaded configurations: {configs_loaded}")
        else:
            print("⚠️ No config file provided or file not found. Using defaults.")
        
        # Initialize identity generator
        self.identity_generator = AlgerianIdentityGenerator(seed=seed)
        
        # Load templates
        self.templates = {}
        self._load_templates()
        
        # Load backgrounds if available
        self.backgrounds = []
        self._load_backgrounds()
        
        # Setup Arabic font
        self.arabic_font = None
        if PIL_AVAILABLE:
            if arabic_font_path is None:
                # Try common locations - check relative to script location
                script_dir = Path(__file__).parent.parent  # synthetic/scripts/ -> synthetic/
                candidates = [
                    script_dir / "fonts" / "ScheherazadeNew-regular.ttf",
                    script_dir / "fonts" / "NotoNaskhArabic-Regular.ttf",
                    Path("synthetic/fonts/ScheherazadeNew-regular.ttf"),
                    Path("synthetic/fonts/NotoNaskhArabic-Regular.ttf"),
                    Path("fonts/ScheherazadeNew-regular.ttf"),
                    Path("fonts/NotoNaskhArabic-Regular.ttf"),
                    Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
                ]
                for candidate in candidates:
                    if candidate.exists():
                        arabic_font_path = candidate
                        break
            if arabic_font_path and Path(arabic_font_path).exists():
                self.arabic_font = str(arabic_font_path)
                print(f"✅ Loaded Arabic font: {arabic_font_path}")
            else:
                print("⚠️ Arabic font not found. Arabic text will not render correctly.")
        else:
            print("⚠️ PIL not installed. Arabic text will not render correctly.")
        
        # Initialize face photo manager if VGGFace2 directory provided
        self.face_manager = None
        self.photo_renderer = None
        if face_photos_dir and face_photos_dir.exists():
            try:
                from face_photo_manager import VGGFace2PhotoManager, PhotoPlaceholderRenderer
                self.face_manager = VGGFace2PhotoManager(face_photos_dir, seed=seed)
                self.photo_renderer = PhotoPlaceholderRenderer(self.face_manager)
                print(f"✅ Initialized VGGFace2 photo manager with {len(self.face_manager.face_index)} faces")
            except Exception as e:
                print(f"⚠️ Failed to initialize face photo manager: {e}")
                self.face_manager = None
                self.photo_renderer = None
    
    def _load_config(self, config_file: Path):
        """Load document specifications from JSON config file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Handle different config formats
            if 'cnie_front' in config and isinstance(config['cnie_front'], dict):
                # Full DOCUMENT_SPECS format with pre-calculated aspect ratio
                for doc_type, spec in config.items():
                    self.DOCUMENT_SPECS[doc_type] = spec
                print(f"📋 Loaded configuration for: {list(config.keys())}")
                
            elif 'fields' in config:
                # GUI tool export format - convert to document spec
                print(f"📋 Detected GUI config format with fields: {list(config.get('fields', {}).keys())}")
                self._convert_gui_config(config)
                
            else:
                # Assume it's a direct field configuration for cnie_front
                spec = {
                    'template_file': 'cnie_front_template.jpg',
                    'aspect_ratio': 1.586,  # Fixed value for ID-1 cards
                    'fields': config.get('fields', {}),
                }
                # Only add photo_bbox if explicitly provided
                if 'photo_bbox' in config and config['photo_bbox'] is not None:
                    spec['photo_bbox'] = config['photo_bbox']
                self.DOCUMENT_SPECS['cnie_front'] = spec
                print("➕ Added 'cnie_front' spec from direct config")
            
            print(f"✅ Loaded configuration from: {config_file}")
            
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            raise

    def _load_config_for_doc_type(self, config_file: Path, doc_type: str):
        """Load config and force it to a specific document type.
        
        This is used for paired generation where we have separate config files
        for front and back.
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if 'fields' not in config:
                print(f"⚠️ Config {config_file} does not contain 'fields'")
                return
            
            # Convert fields to the proper format
            fields = {}
            for field_name, field_data in config['fields'].items():
                fields[field_name] = {
                    'rel_bbox': field_data.get('rel_bbox', [0.1, 0.1, 0.2, 0.03]),
                    'font_scale': field_data.get('font_scale', 0.7),
                    'color': field_data.get('color', [0, 0, 0]),
                    'anchor': field_data.get('anchor', 'lt'),
                    'justification': field_data.get('justification', 'left'),
                    'arabic': field_data.get('arabic', False),
                    'offset_y': field_data.get('offset_y', 0)
                }
            
            # Create spec for the specific doc type
            template_file = f'{doc_type}_template.jpg'
            aspect_ratio = 85.6/54 if doc_type == 'cnie_back' else 85.6/53.98
            
            spec = {
                'template_file': template_file,
                'aspect_ratio': aspect_ratio,
                'fields': fields,
            }
            
            # Add optional fields
            if 'card_region' in config and config['card_region'] is not None:
                spec['card_region'] = config['card_region']
            if 'photo_bbox' in config and config['photo_bbox'] is not None:
                spec['photo_bbox'] = config['photo_bbox']
            if 'photo_placeholders' in config:
                spec['photo_placeholders'] = config['photo_placeholders']
            
            self.DOCUMENT_SPECS[doc_type] = spec
            print(f"✅ Loaded '{doc_type}' config from: {config_file}")
            
        except Exception as e:
            print(f"❌ Error loading config file for {doc_type}: {e}")
            raise

    def _convert_gui_config(self, config: Dict, doc_type: str = 'cnie_front'):
        """Convert GUI tool export format to document spec format."""
        fields = {}
        
        for field_name, field_data in config.get('fields', {}).items():
            # Get offset_y if present
            offset_y = field_data.get('offset_y', 0)
            # Get baselineY (optional, not used by generator)
            # baselineY = field_data.get('baselineY')
            fields[field_name] = {
                'rel_bbox': field_data.get('rel_bbox', [0.1, 0.1, 0.2, 0.03]),
                'font_scale': field_data.get('font_scale', 0.7),
                'color': field_data.get('color', [0, 0, 0]),
                'anchor': field_data.get('anchor', 'lt'),
                'justification': field_data.get('justification', 'left'),
                'arabic': field_data.get('arabic', False),
                'offset_y': offset_y
            }
        
        # Auto-detect doc type based on fields present
        if 'mrz_line3' in fields and 'birth_year' in fields:
            # CNIE Back has mrz_line3 and birth_year
            doc_type = 'cnie_back'
            template_file = 'cnie_back_template.jpg'
            aspect_ratio = 85.6/54
            print(f"🔍 Auto-detected doc_type: {doc_type} (has mrz_line3 + birth_year)")
        elif 'national_id' in fields or 'personal_id' in fields:
            # CNIE Front has national_id or personal_id
            doc_type = 'cnie_front'
            template_file = 'cnie_front_template.jpg'
            aspect_ratio = 85.6/53.98
            print(f"🔍 Auto-detected doc_type: {doc_type} (has national_id/personal_id)")
        else:
            # Default to provided doc_type or cnie_front
            template_file = f'{doc_type}_template.jpg'
            aspect_ratio = 85.6/54 if doc_type == 'cnie_back' else 85.6/53.98
            print(f"🔍 Using default doc_type: {doc_type}")
        
        spec = {
            'template_file': template_file,
            'aspect_ratio': aspect_ratio,
            'fields': fields,
        }
        
        # Add card_region if provided (needed for photo placeholders)
        if 'card_region' in config and config['card_region'] is not None:
            spec['card_region'] = config['card_region']
            print(f"📍 Added card_region: {config['card_region']}")
        
        # Only add photo_bbox if explicitly provided in config
        if 'photo_bbox' in config and config['photo_bbox'] is not None:
            spec['photo_bbox'] = config['photo_bbox']
        
        # Add photo_placeholders if provided
        if 'photo_placeholders' in config:
            spec['photo_placeholders'] = config['photo_placeholders']
            print(f"🖼️ Added {len(config['photo_placeholders'])} photo placeholder(s)")
        
        self.DOCUMENT_SPECS[doc_type] = spec
        print(f"🔄 Converted GUI config to '{doc_type}' spec")
    
    def _load_templates(self):
        """Load document templates."""
        template_real_dir = self.template_dir / 'real'
        print(f"\n🔍 Looking for templates in: {template_real_dir}")
        
        # Try to resolve the path if it doesn't exist as-is
        if not template_real_dir.exists():
            # Try resolving as absolute
            if not template_real_dir.is_absolute():
                # Try relative to current working directory
                cwd_path = Path.cwd() / template_real_dir
                if cwd_path.exists():
                    template_real_dir = cwd_path
                    self.template_dir = cwd_path.parent
                else:
                    # Try relative to script location
                    script_dir = Path(__file__).parent
                    script_path = script_dir / '..' / '..' / template_real_dir
                    script_path = script_path.resolve()
                    if script_path.exists():
                        template_real_dir = script_path
                        self.template_dir = template_real_dir.parent
        
        if not template_real_dir.exists():
            print(f"❌ Template directory does not exist: {template_real_dir}")
            print(f"   Tried: {Path.cwd() / self.template_dir / 'real'}")
            print(f"   Current working directory: {Path.cwd()}")
            return
        
        # First, try to load templates defined in DOCUMENT_SPECS
        for doc_type, spec in self.DOCUMENT_SPECS.items():
            template_file = spec['template_file']
            template_path = template_real_dir / template_file
            print(f"  - {doc_type}: {template_file} -> ", end="")
            
            if template_path.exists():
                img = cv2.imread(str(template_path))
                if img is not None:
                    self.templates[doc_type] = img
                    print(f"✅ Loaded ({img.shape[1]}x{img.shape[0]})")
                else:
                    print(f"❌ Could not load image (OpenCV error)")
            else:
                print(f"❌ File not found")
        
        # Auto-discover additional templates (cnie_front, etc.)
        # Look for templates that have a corresponding config file
        config_extensions = ['_auto_config.json', '_config.json', '_custom.json']
        for config_file in template_real_dir.glob('*_config.json'):
            # Extract doc_type from config filename (e.g., cnie_auto_config.json -> cnie_front)
            config_stem = config_file.stem  # e.g., 'cnie_auto_config'
            
            # Try to determine doc_type from the config content
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # If config has document type keys, use those
                for doc_type in config_data.keys():
                    if doc_type in self.templates:
                        continue  # Already loaded
                    
                    spec = config_data[doc_type]
                    template_file = spec.get('template_file')
                    if template_file:
                        template_path = template_real_dir / template_file
                        print(f"  - {doc_type} (from {config_file.name}): {template_file} -> ", end="")
                        
                        if template_path.exists():
                            img = cv2.imread(str(template_path))
                            if img is not None:
                                self.templates[doc_type] = img
                                # Also update DOCUMENT_SPECS
                                if doc_type not in self.DOCUMENT_SPECS:
                                    self.DOCUMENT_SPECS[doc_type] = spec
                                print(f"✅ Loaded ({img.shape[1]}x{img.shape[0]})")
                            else:
                                print(f"❌ Could not load image (OpenCV error)")
                        else:
                            print(f"❌ File not found")
            except Exception as e:
                print(f"⚠️  Error loading config {config_file}: {e}")
        
        print(f"\n📊 Templates loaded: {list(self.templates.keys())}")
    
    def _load_backgrounds(self):
        """Load background images."""
        if self.backgrounds_dir and self.backgrounds_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                self.backgrounds.extend(list(self.backgrounds_dir.glob(ext)))
            print(f"✅ Loaded {len(self.backgrounds)} backgrounds")
    
    def generate_synthetic_face(self, size: Tuple[int, int], sex: str = 'M') -> np.ndarray:
        """Generate a synthetic face placeholder with more realistic features."""
        h, w = size
        # Create skin-tone base (BGR format)
        base_color = np.array([180, 150, 120], dtype=np.uint8) if sex == 'M' else np.array([200, 170, 150], dtype=np.uint8)
        face = np.ones((h, w, 3), dtype=np.uint8) * base_color
        
        # Add gradient for depth (darker at edges)
        y_grad = np.linspace(0.7, 1.0, h).reshape(-1, 1, 1)
        face = (face * y_grad).astype(np.uint8)
        
        # Add noise for texture
        noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
        face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Draw simple face features (eyes and mouth)
        eye_y = int(h * 0.35)
        eye_x1 = int(w * 0.3)
        eye_x2 = int(w * 0.7)
        eye_size = max(3, int(min(w, h) * 0.08))
        
        # Eyes (dark circles)
        cv2.circle(face, (eye_x1, eye_y), eye_size, (60, 50, 40), -1)
        cv2.circle(face, (eye_x2, eye_y), eye_size, (60, 50, 40), -1)
        
        # Mouth (simple line)
        mouth_y = int(h * 0.7)
        mouth_x1 = int(w * 0.35)
        mouth_x2 = int(w * 0.65)
        cv2.line(face, (mouth_x1, mouth_y), (mouth_x2, mouth_y), (80, 60, 50), max(2, int(h*0.03)))
        
        # Add slight blur to soften
        face = cv2.GaussianBlur(face, (5, 5), 1.5)
        
        return face
    
    def _parse_color(self, color) -> Tuple[int, int, int]:
        """Parse color from various formats to BGR tuple."""
        if isinstance(color, str):
            # Hex color
            color = color.lstrip('#')
            if len(color) == 6:
                r = int(color[0:2], 16)
                g = int(color[2:4], 16)
                b = int(color[4:6], 16)
                return (b, g, r)  # OpenCV uses BGR
        elif isinstance(color, (list, tuple)) and len(color) >= 3:
            return tuple(int(c) for c in color[:3])
        # Default
        return (60, 60, 80)
    
    def _parse_color_rgb(self, color) -> Tuple[int, int, int]:
        """Parse color to RGB tuple (for PIL)."""
        bgr = self._parse_color(color)
        return (bgr[2], bgr[1], bgr[0])  # BGR to RGB
    
    def overlay_text(self, image: np.ndarray, text: str, rel_bbox: List[float], 
                     font_scale: float = 0.7, font_type: str = 'regular',
                     color = (60, 60, 80),
                     anchor: str = 'lt', justification: str = 'left',
                     arabic: bool = False, offset_y: int = 0) -> np.ndarray:
        """
        Overlay text onto image at specified relative bbox.
        If arabic=True and PIL is available, use Arabic rendering.
        """
        if arabic and PIL_AVAILABLE and self.arabic_font:
            return self.overlay_arabic_text(image, text, rel_bbox, font_scale, color, anchor, justification, offset_y)
        
        # Fallback to OpenCV (Latin-only)
        h, w = image.shape[:2]
        
        # Convert relative bbox to absolute
        x = int(rel_bbox[0] * w)
        y = int(rel_bbox[1] * h)
        bw = int(rel_bbox[2] * w)
        bh = int(rel_bbox[3] * h)
        
        # Choose font
        if font_type == 'ocr_b':
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
        
        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Adjust font scale if text is too wide
        if text_w > bw * 0.9:
            font_scale = font_scale * (bw * 0.9) / text_w
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate position based on anchor
        if anchor == 'lt':  # Left-top
            text_x = x
            text_y = y + text_h
        elif anchor == 'lm':  # Left-middle
            text_x = x
            text_y = y + bh//2 + text_h//2
        elif anchor == 'lb':  # Left-bottom
            text_x = x
            text_y = y + bh - baseline
        elif anchor == 'rt':  # Right-top
            text_x = x + bw - text_w
            text_y = y + text_h
        elif anchor == 'rm':  # Right-middle
            text_x = x + bw - text_w
            text_y = y + bh//2 + text_h//2
        elif anchor == 'rb':  # Right-bottom
            text_x = x + bw - text_w
            text_y = y + bh - baseline
        elif anchor == 'ct':  # Center-top
            text_x = x + (bw - text_w) // 2
            text_y = y + text_h
        elif anchor == 'cm':  # Center-middle
            text_x = x + (bw - text_w) // 2
            text_y = y + bh//2 + text_h//2
        elif anchor == 'cb':  # Center-bottom
            text_x = x + (bw - text_w) // 2
            text_y = y + bh - baseline
        else:
            text_x = x
            text_y = y + text_h
        
        # Apply justification
        if justification == 'center':
            text_x = x + (bw - text_w) // 2
        elif justification == 'right':
            text_x = x + bw - text_w
        
        # Apply vertical offset
        text_y += offset_y
        
        # Parse color to BGR tuple
        bgr_color = self._parse_color(color)
        
        # Draw text with slight shadow
        if text.strip():
            shadow_color = (255, 255, 255)
            cv2.putText(image, text, (text_x + 1, text_y + 1), font, font_scale, shadow_color, thickness, cv2.LINE_AA)
            cv2.putText(image, text, (text_x, text_y), font, font_scale, bgr_color, thickness, cv2.LINE_AA)
        
        return image
    
    def overlay_arabic_text(self, image: np.ndarray, text: str, rel_bbox: List[float],
                            font_scale: float = 0.7, color = (60, 60, 80),
                            anchor: str = 'lt', justification: str = 'left',
                            offset_y: int = 0) -> np.ndarray:
        """
        Render Arabic text using PIL and paste onto OpenCV image.
        Font size is dynamically computed from the bounding box height.
        """
        if not PIL_AVAILABLE or not self.arabic_font:
            return self.overlay_text(image, text, rel_bbox, font_scale, 'regular', color, anchor, justification, arabic=False, offset_y=offset_y)

        h, w = image.shape[:2]
        x = int(rel_bbox[0] * w)
        y = int(rel_bbox[1] * h)
        bw = int(rel_bbox[2] * w)
        bh = int(rel_bbox[3] * h)

        # Convert OpenCV BGR to PIL RGB
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Determine font size based on box height and font_scale
        target_height = int(bh * font_scale)  # use full height
        font_size = max(10, target_height)

        try:
            font = ImageFont.truetype(self.arabic_font, font_size)
        except Exception as e:
            print(f"Error loading font: {e}")
            return image

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Adjust if too wide
        if text_w > bw * 0.9:
            new_size = int(font_size * (bw * 0.9) / text_w)
            font = ImageFont.truetype(self.arabic_font, new_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        # Calculate position based on anchor (for PIL, anchor is top-left)
        if anchor == 'lt':
            text_x = x
            text_y = y
        elif anchor == 'lm':
            text_x = x
            text_y = y + (bh - text_h) // 2
        elif anchor == 'lb':
            text_x = x
            text_y = y + bh - text_h
        elif anchor == 'rt':
            text_x = x + bw - text_w
            text_y = y
        elif anchor == 'rm':
            text_x = x + bw - text_w
            text_y = y + (bh - text_h) // 2
        elif anchor == 'rb':
            text_x = x + bw - text_w
            text_y = y + bh - text_h
        elif anchor == 'ct':
            text_x = x + (bw - text_w) // 2
            text_y = y
        elif anchor == 'cm':
            text_x = x + (bw - text_w) // 2
            text_y = y + (bh - text_h) // 2
        elif anchor == 'cb':
            text_x = x + (bw - text_w) // 2
            text_y = y + bh - text_h
        else:
            text_x = x
            text_y = y

        # Apply justification (horizontal only)
        if justification == 'center':
            text_x = x + (bw - text_w) // 2
        elif justification == 'right':
            text_x = x + bw - text_w

        # Apply vertical offset
        text_y += offset_y

        # Parse color to RGB
        rgb_color = self._parse_color_rgb(color)

        # Draw text with optional white shadow
        if text.strip():
            draw.text((text_x + 1, text_y + 1), text, font=font, fill=(255, 255, 255))
            draw.text((text_x, text_y), text, font=font, fill=rgb_color)

        # Convert back to OpenCV BGR
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return image
    
    def apply_perspective_transform(self, image: np.ndarray, 
                                    max_angle: float = 25.0,
                                    fixed_matrix: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply perspective transformation.
        
        Args:
            image: Input image
            max_angle: Maximum rotation angle for random transform
            fixed_matrix: If provided, use this matrix instead of generating random one
            
        Returns:
            Tuple of (transformed_image, transformation_matrix)
        """
        h, w = image.shape[:2]
        
        if fixed_matrix is not None:
            # Use provided fixed matrix (for paired generation)
            M = fixed_matrix
        else:
            # Generate random transformation
            pitch = random.uniform(-max_angle, max_angle)
            yaw = random.uniform(-max_angle * 0.7, max_angle * 0.7)
            roll = random.uniform(-10, 10)
            
            # Calculate transformation matrix
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            # Apply perspective distortion
            dx = int(w * np.tan(np.radians(yaw)) * 0.3)
            dy = int(h * np.tan(np.radians(pitch)) * 0.3)
            dr = int(w * np.tan(np.radians(roll)) * 0.1)
            
            dst_pts = np.float32([
                [max(0, dx + dr), max(0, dy - dr)],
                [min(w, w + dx - dr), max(0, dy + dr)],
                [min(w, w - dx - dr), min(h, h - dy + dr)],
                [max(0, -dx + dr), min(h, h + dy - dr)]
            ])
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        transformed = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, 
                                          borderValue=(255, 255, 255))
        
        return transformed, M
    
    def apply_lighting_variation(self, image: np.ndarray) -> np.ndarray:
        """Apply random lighting variations."""
        # Brightness variation
        brightness = random.uniform(-30, 30)
        image = np.clip(image.astype(np.float32) + brightness, 0, 255).astype(np.uint8)
        
        # Contrast variation
        contrast = random.uniform(0.8, 1.2)
        mean = np.mean(image)
        image = np.clip((image.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # Gamma correction
        gamma = random.uniform(0.8, 1.2)
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
        image = cv2.LUT(image, lookup_table)
        
        return image
    
    def apply_blur_and_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply random blur and noise."""
        # Random blur
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(2, 8), image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def add_background(self, image: np.ndarray, document_bbox: List[int]) -> np.ndarray:
        """Add random background around document."""
        if not self.backgrounds:
            return image
        
        # Load random background
        bg_path = random.choice(self.backgrounds)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            return image
        
        h, w = image.shape[:2]
        bg = cv2.resize(bg, (w, h))
        
        # Create mask for document region
        mask = np.ones((h, w), dtype=np.uint8) * 255
        x, y, bw, bh = document_bbox
        mask[y:y+bh, x:x+bw] = 0
        
        # Blend
        result = image.copy()
        result[mask > 0] = bg[mask > 0]
        
        return result
    
    def _card_relative_to_absolute(self, rel_bbox: List[float], card_region: List[int], img_w: int, img_h: int) -> List[int]:
        """Convert card-relative bbox to absolute pixel coordinates."""
        card_x, card_y, card_w, card_h = card_region
        
        abs_x = int(card_x + rel_bbox[0] * card_w)
        abs_y = int(card_y + rel_bbox[1] * card_h)
        abs_w = int(rel_bbox[2] * card_w)
        abs_h = int(rel_bbox[3] * card_h)
        
        # Clamp to image bounds
        abs_x = max(0, min(abs_x, img_w))
        abs_y = max(0, min(abs_y, img_h))
        abs_w = min(abs_w, img_w - abs_x)
        abs_h = min(abs_h, img_h - abs_y)
        
        return [abs_x, abs_y, abs_w, abs_h]
    
    def render_document(self, doc_type: str, identity: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Render a document with synthetic data."""
        if doc_type not in self.templates:
            raise ValueError(f"Unknown document type: {doc_type}")

        # Get template
        template = self.templates[doc_type].copy()
        h, w = template.shape[:2]

        spec = self.DOCUMENT_SPECS[doc_type]
        annotations = []

        # Get card region if specified (for card-relative coordinates)
        card_region = spec.get('card_region')  # [x, y, w, h] in template pixels

        # Add photo placeholders if configured (new format with VGGFace2 support)
        photo_placeholders = spec.get('photo_placeholders', {})
        if photo_placeholders and card_region:
            print(f"🖼️ Rendering {len(photo_placeholders)} photo placeholder(s)...")
            
            # Pick one face per identity (reused for all placeholders on this document)
            cached_face = None
            cached_identity_id = None
            
            for placeholder_id, placeholder_config in photo_placeholders.items():
                if self.face_manager and self.photo_renderer:
                    # Use real face from VGGFace2
                    try:
                        # Pick a face once and reuse for all placeholders
                        if cached_face is None:
                            cached_face, cached_identity_id = self.face_manager.get_random_face(
                                identity.get('sex', 'M')
                            )
                            print(f"  📸 Selected face identity: {cached_identity_id}")
                        
                        template, photo_ann = self.photo_renderer.render_placeholder_with_face(
                            template,
                            {**placeholder_config, 'id': placeholder_id},
                            card_region,
                            cached_face,
                            cached_identity_id
                        )
                        annotations.append(photo_ann)
                        print(f"  ✅ Rendered '{placeholder_id}' with real face")
                    except Exception as e:
                        print(f"⚠️ Failed to render photo placeholder '{placeholder_id}': {e}")
                        # Fallback to synthetic face
                        template = self._render_synthetic_photo_placeholder(
                            template, placeholder_config, placeholder_id, 
                            card_region, identity, annotations
                        )
                else:
                    # Fallback to synthetic face generation
                    print(f"  🎨 Rendering '{placeholder_id}' with synthetic face (VGGFace2 not available)")
                    template = self._render_synthetic_photo_placeholder(
                        template, placeholder_config, placeholder_id, 
                        card_region, identity, annotations
                    )

        # Add legacy synthetic photo if applicable (backward compatibility)
        elif 'photo_bbox' in spec and spec['photo_bbox'] is not None:
            photo_bbox = spec['photo_bbox']

            if card_region:
                # Card-relative coordinates
                abs_photo_bbox = self._card_relative_to_absolute(photo_bbox, card_region, w, h)
                px, py, pw, ph = abs_photo_bbox
            else:
                # Image-relative coordinates
                px = int(photo_bbox[0] * w)
                py = int(photo_bbox[1] * h)
                pw = int(photo_bbox[2] * w)
                ph = int(photo_bbox[3] * h)

            face = self.generate_synthetic_face((ph, pw), identity.get('sex', 'M'))
            template[py:py+ph, px:px+pw] = face

            annotations.append({
                'field': 'photo',
                'bbox': [px, py, pw, ph],
                'rel_bbox': photo_bbox,
                'card_rel_bbox': photo_bbox if card_region else None,
                'type': 'photo',
                'synthetic': True
            })

        # Overlay text fields
        for field_name, field_spec in spec['fields'].items():
            # MRZ fields (always Latin)
            if field_name.startswith('mrz_line'):
                mrz = identity.get('mrz', {})
                line_key = field_name.replace('mrz_', '')
                text = mrz.get(line_key, '')
                if text:
                    font_type = field_spec.get('font', 'regular')
                    color = field_spec.get('color', (60, 60, 80))
                    anchor = field_spec.get('anchor', 'lt')
                    justification = field_spec.get('justification', 'left')
                    arabic = field_spec.get('arabic', False)
                    offset_y = field_spec.get('offset_y', 0)

                    template = self.overlay_text(
                        template, str(text), field_spec['rel_bbox'],
                        field_spec['font_scale'], font_type, color,
                        anchor, justification, arabic, offset_y
                    )

                    rx, ry, rw, rh = field_spec['rel_bbox']
                    abs_bbox = [int(rx * w), int(ry * h), int(rw * w), int(rh * h)]
                    annotations.append({
                        'field': field_name,
                        'bbox': abs_bbox,
                        'rel_bbox': field_spec['rel_bbox'],
                        'text': str(text)
                    })
            else:
                # Determine if this field should be rendered in Arabic
                arabic_flag = field_spec.get('arabic', False)
                offset_y = field_spec.get('offset_y', 0)

                if arabic_flag:
                    # First try the Arabic version of the field (e.g., surname_ar)
                    value = identity.get(field_name + '_ar', '')
                    # If Arabic version is empty, fallback to French/Latin
                    if not value:
                        value = identity.get(field_name, '')
                else:
                    # For non‑Arabic fields, use the French/Latin version
                    value = identity.get(field_name, '')

                if isinstance(value, datetime):
                    value = value.strftime('%d/%m/%Y')
                
                # Special handling for birth_year field (CNIE back)
                if field_name == 'birth_year' and not value:
                    dob = identity.get('date_of_birth')
                    if isinstance(dob, datetime):
                        value = str(dob.year)
                    else:
                        # Try to parse from string
                        try:
                            value = str(datetime.strptime(dob, '%d/%m/%Y').year)
                        except:
                            value = ''

                if value:
                    font_type = field_spec.get('font', 'regular')
                    color = field_spec.get('color', (60, 60, 80))
                    anchor = field_spec.get('anchor', 'lt')
                    justification = field_spec.get('justification', 'left')

                    # Convert rel_bbox to absolute for overlay (if card_region exists)
                    if card_region:
                        abs_bbox = self._card_relative_to_absolute(
                            field_spec['rel_bbox'], card_region, w, h
                        )
                        overlay_bbox = [abs_bbox[0] / w, abs_bbox[1] / h,
                                       abs_bbox[2] / w, abs_bbox[3] / h]
                    else:
                        overlay_bbox = field_spec['rel_bbox']
                        rx, ry, rw, rh = field_spec['rel_bbox']
                        abs_bbox = [int(rx * w), int(ry * h), int(rw * w), int(rh * h)]

                    template = self.overlay_text(
                        template, str(value), overlay_bbox,
                        field_spec['font_scale'], font_type, color,
                        anchor, justification, arabic_flag, offset_y
                    )

                    ann = {
                        'field': field_name,
                        'bbox': abs_bbox,
                        'rel_bbox': field_spec['rel_bbox'],
                        'text': str(value)
                    }
                    if card_region:
                        ann['card_rel_bbox'] = field_spec['rel_bbox']
                    annotations.append(ann)

        return template, annotations
    
    def _render_synthetic_photo_placeholder(self, template: np.ndarray, 
                                           placeholder_config: Dict, 
                                           placeholder_id: str,
                                           card_region: List[int],
                                           identity: Dict,
                                           annotations: List[Dict]) -> np.ndarray:
        """Render a synthetic face placeholder when VGGFace2 is not available."""
        h, w = template.shape[:2]
        card_x, card_y, card_w, card_h = card_region
        
        rel_bbox = placeholder_config['rel_bbox']
        abs_x = int(card_x + rel_bbox[0] * card_w)
        abs_y = int(card_y + rel_bbox[1] * card_h)
        abs_w = int(rel_bbox[2] * card_w)
        abs_h = int(rel_bbox[3] * card_h)
        
        # Ensure within image bounds
        abs_x = max(0, min(abs_x, w - 1))
        abs_y = max(0, min(abs_y, h - 1))
        abs_w = min(abs_w, w - abs_x)
        abs_h = min(abs_h, h - abs_y)
        
        if abs_w < 10 or abs_h < 10:
            print(f"⚠️ Photo placeholder '{placeholder_id}' too small or out of bounds")
            return template
        
        # Generate synthetic face
        face = self.generate_synthetic_face((abs_h, abs_w), identity.get('sex', 'M'))
        
        # Apply oval mask if needed
        if placeholder_config.get('shape') == 'oval':
            # Create oval mask
            mask = np.zeros((abs_h, abs_w), dtype=np.uint8)
            center = (abs_w // 2, abs_h // 2)
            axes = (abs_w // 2 - 2, abs_h // 2 - 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            face[mask == 0] = [255, 255, 255]
        
        # Place into template
        template[abs_y:abs_y+abs_h, abs_x:abs_x+abs_w] = face
        
        annotations.append({
            'field': placeholder_id,
            'bbox': [abs_x, abs_y, abs_w, abs_h],
            'rel_bbox': rel_bbox,
            'type': 'photo',
            'synthetic': True
        })
        
        return template
    
    def generate_sample(self, doc_type: str, sample_id: int, fast_preview: bool = False) -> Dict:
        """Generate a single synthetic sample with augmentations.
        
        Args:
            doc_type: Type of document to generate
            sample_id: ID for this sample
            fast_preview: If True, skip expensive augmentations for faster preview
        """
        if doc_type == 'carte_grise':
            identity = self.identity_generator.generate_carte_grise_identity()
        else:
            identity = self.identity_generator.generate_identity(doc_type.replace('_front', '').replace('_back', ''))
        
        document, annotations = self.render_document(doc_type, identity)
        
        if fast_preview:
            # Fast path: skip expensive augmentations for preview
            doc_h, doc_w = document.shape[:2]
            transform_matrix = np.eye(3, dtype=np.float32)  # Identity matrix
            transformed_annotations = annotations
        else:
            # Full path: apply all augmentations
            document, transform_matrix = self.apply_perspective_transform(document)
            document = self.apply_lighting_variation(document)
            document = self.apply_blur_and_noise(document)
            
            doc_h, doc_w = document.shape[:2]
            document_bbox = [0, 0, doc_w, doc_h]
            document = self.add_background(document, document_bbox)
            
            transformed_annotations = self._transform_annotations(annotations, transform_matrix, doc_w, doc_h)
        
        sample_dir = self.output_dir / doc_type / f"{sample_id:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        image_path = sample_dir / "image.jpg"
        # Use lower quality for preview to speed up encoding
        jpeg_quality = 85 if fast_preview else 95
        cv2.imwrite(str(image_path), document, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        
        annotation_data = {
            'sample_id': sample_id,
            'document_type': doc_type,
            'identity': self._serialize_identity(identity),
            'image_path': str(image_path.relative_to(self.output_dir)),
            'image_size': [doc_w, doc_h],
            'bounding_boxes': transformed_annotations,
            'transform_matrix': transform_matrix.tolist(),
            'generated_at': datetime.now().isoformat()
        }
        
        annotation_path = sample_dir / "annotations.json"
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        return annotation_data
    
    def _transform_annotations(self, annotations: List[Dict], M: np.ndarray, 
                               img_w: int, img_h: int) -> List[Dict]:
        """Transform annotation bboxes based on perspective matrix."""
        transformed = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
            corners_transformed = cv2.perspectiveTransform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
            new_x = int(np.min(corners_transformed[:, 0]))
            new_y = int(np.min(corners_transformed[:, 1]))
            new_w = int(np.max(corners_transformed[:, 0]) - new_x)
            new_h = int(np.max(corners_transformed[:, 1]) - new_y)
            new_x = max(0, min(new_x, img_w))
            new_y = max(0, min(new_y, img_h))
            new_w = min(new_w, img_w - new_x)
            new_h = min(new_h, img_h - new_y)
            
            # Preserve all annotation fields
            transformed_ann = {
                'field': ann['field'],
                'bbox': [new_x, new_y, new_w, new_h],
                'rel_bbox': ann.get('rel_bbox'),
                'text': ann.get('text', '')
            }
            
            # Preserve optional fields
            if 'type' in ann:
                transformed_ann['type'] = ann['type']
            if 'synthetic' in ann:
                transformed_ann['synthetic'] = ann['synthetic']
            if 'identity_id' in ann:
                transformed_ann['identity_id'] = ann['identity_id']
            if 'card_rel_bbox' in ann:
                transformed_ann['card_rel_bbox'] = ann['card_rel_bbox']
            
            transformed.append(transformed_ann)
        return transformed
    
    def _serialize_identity(self, identity: Dict) -> Dict:
        """Convert identity to JSON-serializable format."""
        result = {}
        for key, value in identity.items():
            if isinstance(value, datetime):
                result[key] = value.strftime('%d/%m/%Y')
            else:
                result[key] = value
        return result
    
    def generate_dataset(self, doc_type: str, num_samples: int, start_id: int = 0, fast_preview: bool = False) -> List[Path]:
        """Generate a complete dataset for a document type."""
        if doc_type not in self.templates:
            print(f"⚠️ No template for {doc_type}, skipping...")
            return []
        
        generated = []
        print(f"📄 Generating {num_samples} samples for {doc_type}...")
        
        for i in range(num_samples):
            sample_id = start_id + i
            try:
                self.generate_sample(doc_type, sample_id, fast_preview=fast_preview)
                generated.append(self.output_dir / doc_type / f"{sample_id:06d}" / "image.jpg")
                if (i + 1) % 10 == 0:
                    print(f"  ✅ Generated {i+1}/{num_samples}")
            except Exception as e:
                print(f"  ❌ Error generating sample {sample_id}: {e}")
        
        return generated
    
    def generate_paired_cnie_sample(self, pair_id: int, fast_preview: bool = False) -> Dict:
        """Generate a paired CNIE front and back sample with the same identity.
        
        Args:
            pair_id: ID for this identity pair
            fast_preview: If True, skip expensive augmentations
            
        Returns:
            Dict with paths to both front and back images and their annotations
        """
        # Generate a single identity for both front and back
        identity = self.identity_generator.generate_identity('cnie')
        
        # Generate CNIE Front
        if 'cnie_front' not in self.templates:
            raise ValueError("CNIE front template not available")
        
        front_doc, front_annotations = self.render_document('cnie_front', identity)
        
        # Generate CNIE Back
        if 'cnie_back' not in self.templates:
            raise ValueError("CNIE back template not available")
            
        back_doc, back_annotations = self.render_document('cnie_back', identity)
        
        # Apply augmentations (same transform for both to maintain consistency)
        if fast_preview:
            transform_matrix = np.eye(3, dtype=np.float32)
            front_transformed = front_doc
            back_transformed = back_doc
            front_ann = front_annotations
            back_ann = back_annotations
        else:
            # Apply same transform to both documents
            front_transformed, transform_matrix = self.apply_perspective_transform(front_doc)
            back_transformed, _ = self.apply_perspective_transform(back_doc, fixed_matrix=transform_matrix)
            
            # Apply other augmentations
            front_transformed = self.apply_lighting_variation(front_transformed)
            front_transformed = self.apply_blur_and_noise(front_transformed)
            
            back_transformed = self.apply_lighting_variation(back_transformed)
            back_transformed = self.apply_blur_and_noise(back_transformed)
            
            # Add backgrounds
            doc_h, doc_w = front_transformed.shape[:2]
            front_transformed = self.add_background(front_transformed, [0, 0, doc_w, doc_h])
            back_transformed = self.add_background(back_transformed, [0, 0, doc_w, doc_h])
            
            # Transform annotations
            front_ann = self._transform_annotations(front_annotations, transform_matrix, doc_w, doc_h)
            back_ann = self._transform_annotations(back_annotations, transform_matrix, doc_w, doc_h)
        
        # Create output directories
        pair_dir = self.output_dir / "cnie_pairs" / f"{pair_id:06d}"
        front_dir = pair_dir / "front"
        back_dir = pair_dir / "back"
        front_dir.mkdir(parents=True, exist_ok=True)
        back_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        jpeg_quality = 85 if fast_preview else 95
        front_path = front_dir / "image.jpg"
        back_path = back_dir / "image.jpg"
        cv2.imwrite(str(front_path), front_transformed, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        cv2.imwrite(str(back_path), back_transformed, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        
        # Create paired annotation
        pair_annotation = {
            'pair_id': pair_id,
            'identity': self._serialize_identity(identity),
            'front': {
                'document_type': 'cnie_front',
                'image_path': str(front_path.relative_to(self.output_dir)),
                'image_size': [front_transformed.shape[1], front_transformed.shape[0]],
                'bounding_boxes': front_ann
            },
            'back': {
                'document_type': 'cnie_back',
                'image_path': str(back_path.relative_to(self.output_dir)),
                'image_size': [back_transformed.shape[1], back_transformed.shape[0]],
                'bounding_boxes': back_ann
            },
            'transform_matrix': transform_matrix.tolist(),
            'generated_at': datetime.now().isoformat()
        }
        
        annotation_path = pair_dir / "annotations.json"
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(pair_annotation, f, indent=2, ensure_ascii=False)
        
        return {
            'pair_id': pair_id,
            'front_image': front_path,
            'back_image': back_path,
            'annotation': annotation_path
        }
    
    def generate_paired_cnie_dataset(self, num_pairs: int, start_id: int = 0, fast_preview: bool = False) -> List[Dict]:
        """Generate a dataset of paired CNIE front/back samples.
        
        Args:
            num_pairs: Number of identity pairs to generate
            start_id: Starting ID for pairs
            fast_preview: If True, skip expensive augmentations
            
        Returns:
            List of generation results
        """
        if 'cnie_front' not in self.templates or 'cnie_back' not in self.templates:
            print("⚠️ Both CNIE front and back templates required for paired generation")
            return []
        
        generated = []
        print(f"📄 Generating {num_pairs} paired CNIE samples (front + back)...")
        
        for i in range(num_pairs):
            pair_id = start_id + i
            try:
                result = self.generate_paired_cnie_sample(pair_id, fast_preview=fast_preview)
                generated.append(result)
                if (i + 1) % 10 == 0:
                    print(f"  ✅ Generated {i+1}/{num_pairs} pairs")
            except Exception as e:
                print(f"  ❌ Error generating pair {pair_id}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Paired dataset generation complete. Output: {self.output_dir}/cnie_pairs")
        return generated


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic ID documents from templates')
    parser.add_argument('--template-dir', type=Path, default=Path('synthetic/templates'),
                       help='Directory containing template images')
    parser.add_argument('--output-dir', type=Path, default=Path('data/synthetic'),
                       help='Output directory for generated samples')
    parser.add_argument('--config', type=Path, default=None,
                       help='JSON configuration file for field positions (generic)')
    parser.add_argument('--config-front', type=Path, default=None,
                       help='JSON configuration file for CNIE front')
    parser.add_argument('--config-back', type=Path, default=None,
                       help='JSON configuration file for CNIE back')
    parser.add_argument('--doc-type', choices=['passport', 'cnie_front', 'cnie_back', 'cnie_paired', 'all'], 
                       default='all', help='Document type to generate')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--backgrounds-dir', type=Path, default=None,
                       help='Directory containing background images')
    parser.add_argument('--arabic-font', type=Path, default=None,
                       help='Path to Arabic TrueType font file')
    parser.add_argument('--face-photos-dir', type=Path, default=None,
                       help='Path to VGGFace2 dataset directory for real face photos')
    parser.add_argument('--fast-preview', action='store_true',
                       help='Fast preview mode: skip augmentations for quicker generation')
    
    args = parser.parse_args()
    
    generator = TemplateDocumentGenerator(
        template_dir=args.template_dir,
        output_dir=args.output_dir,
        config_file=args.config,
        config_front=args.config_front,
        config_back=args.config_back,
        backgrounds_dir=args.backgrounds_dir,
        seed=args.seed,
        arabic_font_path=args.arabic_font,
        face_photos_dir=args.face_photos_dir
    )
    
    if args.doc_type == 'cnie_paired':
        # Generate paired front/back samples
        generator.generate_paired_cnie_dataset(args.num_samples, fast_preview=args.fast_preview)
    elif args.doc_type == 'all':
        # Generate all document types including paired
        for doc_type in ['passport', 'cnie_front', 'cnie_back']:
            generator.generate_dataset(doc_type, args.num_samples, fast_preview=args.fast_preview)
        # Also generate paired CNIE
        generator.generate_paired_cnie_dataset(args.num_samples, fast_preview=args.fast_preview)
    else:
        # Single document type
        generator.generate_dataset(args.doc_type, args.num_samples, fast_preview=args.fast_preview)
    
    print(f"\n✅ Dataset generation complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()