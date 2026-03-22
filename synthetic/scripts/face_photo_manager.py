"""
VGGFace2 Face Photo Manager for Retin-Verify
Handles loading, filtering, and placing real face photos into ID templates.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import json


class VGGFace2PhotoManager:
    """Manages real face photos from VGGFace2 dataset for ID document generation."""
    
    # Class-level cache for face index (shared across instances)
    _cached_index: Optional[List[Dict]] = None
    _cached_path: Optional[Path] = None
    
    def __init__(self, vggface2_dir: Path, seed: Optional[int] = None, use_cache: bool = True):
        """
        Initialize the face photo manager.
        
        Args:
            vggface2_dir: Path to VGGFace2 dataset root (contains train/ and test/ folders)
            seed: Random seed for reproducible face selection
            use_cache: If True, use cached index if available (faster for repeated previews)
        """
        self.vggface2_dir = Path(vggface2_dir)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Index of available faces
        self.face_index: List[Dict] = []
        
        # Use cache if available and paths match
        if use_cache and VGGFace2PhotoManager._cached_index is not None:
            if VGGFace2PhotoManager._cached_path == self.vggface2_dir:
                self.face_index = VGGFace2PhotoManager._cached_index.copy()
                print(f"✅ Using cached index with {len(self.face_index)} faces")
                return
        
        self._build_index()
        
        # Cache the index for future instances
        if use_cache:
            VGGFace2PhotoManager._cached_index = self.face_index.copy()
            VGGFace2PhotoManager._cached_path = self.vggface2_dir
    
    def _build_index(self):
        """Build index of all available face images in VGGFace2 dataset."""
        print(f"🔍 Indexing VGGFace2 dataset at: {self.vggface2_dir}")
        
        # Try standard VGGFace2 structure: vggface2/train/n000001/0001_01.jpg
        found_standard_structure = False
        for split in ['train', 'test']:
            split_dir = self.vggface2_dir / split
            if not split_dir.exists():
                continue
            found_standard_structure = True
                
            for identity_dir in split_dir.iterdir():
                if not identity_dir.is_dir():
                    continue
                
                identity_id = identity_dir.name
                
                # Collect all images for this identity (support both jpg and png)
                for img_path in list(identity_dir.glob('*.jpg')) + list(identity_dir.glob('*.png')):
                    self.face_index.append({
                        'identity_id': identity_id,
                        'image_path': img_path,
                        'split': split
                    })
        
        # If no standard structure found, try flat structure (identity dirs directly in root)
        if not found_standard_structure or len(self.face_index) == 0:
            print(f"   No standard train/test structure found, trying flat structure...")
            for identity_dir in self.vggface2_dir.iterdir():
                if not identity_dir.is_dir():
                    continue
                
                identity_id = identity_dir.name
                
                # Collect all images for this identity (support both jpg and png)
                for img_path in list(identity_dir.glob('*.jpg')) + list(identity_dir.glob('*.png')):
                    self.face_index.append({
                        'identity_id': identity_id,
                        'image_path': img_path,
                        'split': 'unknown'
                    })
        
        print(f"✅ Indexed {len(self.face_index)} face images")
    
    def get_random_face(self, sex: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        Get a random face image from the dataset.
        
        Args:
            sex: Optional filter by sex ('M' or 'F') - requires metadata
            
        Returns:
            Tuple of (face_image, identity_id)
        """
        if not self.face_index:
            raise ValueError("No faces available in index")
        
        # For now, random selection (sex filtering requires additional metadata)
        entry = random.choice(self.face_index)
        
        # Load and preprocess image
        img = cv2.imread(str(entry['image_path']))
        if img is None:
            # Retry with another image if load fails
            return self.get_random_face(sex)
        
        return img, entry['identity_id']
    
    def preprocess_face(self, 
                       face_img: np.ndarray, 
                       target_size: Tuple[int, int],
                       shape: str = 'rect') -> np.ndarray:
        """
        Preprocess face image to fit in template placeholder.
        
        Args:
            face_img: Source face image
            target_size: (width, height) of target area
            shape: 'rect' or 'oval' for masking
            
        Returns:
            Processed face image ready for placement
        """
        target_w, target_h = target_size
        
        # Resize maintaining aspect ratio
        h, w = face_img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create white background canvas
        result = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        
        # Center the resized face
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Apply oval mask if needed
        if shape == 'oval':
            result = self._apply_oval_mask(result)
        
        return result
    
    def _apply_oval_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply oval mask to image for circular photo cutouts."""
        h, w = image.shape[:2]
        
        # Create oval mask
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 2 - 2, h // 2 - 2)  # Slight margin
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Apply mask
        result = image.copy()
        result[mask == 0] = [255, 255, 255]  # White outside oval
        
        return result
    
    def add_border(self, 
                   image: np.ndarray, 
                   border_width: int = 2,
                   border_color: Tuple[int, int, int] = (204, 204, 204)) -> np.ndarray:
        """Add border around photo (simulates ID card photo frame)."""
        return cv2.copyMakeBorder(
            image, 
            border_width, border_width, border_width, border_width,
            cv2.BORDER_CONSTANT,
            value=border_color
        )


class PhotoPlaceholderRenderer:
    """Renders photo placeholders into document templates."""
    
    def __init__(self, face_manager: VGGFace2PhotoManager):
        self.face_manager = face_manager
    
    def render_placeholder_with_face(self,
                                     template: np.ndarray,
                                     placeholder_config: Dict,
                                     card_region: List[int],
                                     face_img: np.ndarray,
                                     identity_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Render a photo placeholder with a pre-loaded face image.
        
        Args:
            template: Document template image
            placeholder_config: Configuration for this placeholder
            card_region: [x, y, w, h] of card region
            face_img: Pre-loaded face image (from VGGFace2)
            identity_id: Identity ID for the face
            
        Returns:
            Tuple of (updated_template, annotation)
        """
        h, w = template.shape[:2]
        card_x, card_y, card_w, card_h = card_region
        
        # Calculate absolute position
        rel_bbox = placeholder_config['rel_bbox']
        abs_x = int(card_x + rel_bbox[0] * card_w)
        abs_y = int(card_y + rel_bbox[1] * card_h)
        abs_w = int(rel_bbox[2] * card_w)
        abs_h = int(rel_bbox[3] * card_h)
        
        # Preprocess face (using the provided face image)
        shape = placeholder_config.get('shape', 'rect')
        processed_face = self.face_manager.preprocess_face(
            face_img, (abs_w, abs_h), shape
        )
        
        # Add border if specified
        border_config = placeholder_config.get('border')
        border_width = 0
        if border_config:
            border_width = border_config.get('width', 2)
            border_color_hex = border_config.get('color', '#cccccc')
            # Convert hex to BGR
            border_color = tuple(int(border_color_hex[i:i+2], 16) 
                                for i in (5, 3, 1))  # BGR order
            processed_face = self.face_manager.add_border(
                processed_face, border_width, border_color
            )
        
        # Place into template with proper bounds checking
        roi_h, roi_w = processed_face.shape[:2]
        
        # Calculate valid ROI region on template
        y_start = max(0, abs_y - border_width)
        x_start = max(0, abs_x - border_width)
        y_end = min(h, y_start + roi_h)
        x_end = min(w, x_start + roi_w)
        
        # Calculate corresponding region in processed_face
        face_y_start = 0 if abs_y >= border_width else border_width - abs_y
        face_x_start = 0 if abs_x >= border_width else border_width - abs_x
        face_y_end = face_y_start + (y_end - y_start)
        face_x_end = face_x_start + (x_end - x_start)
        
        # Only place if there's a valid region
        if y_end > y_start and x_end > x_start and face_y_end > face_y_start and face_x_end > face_x_start:
            template[y_start:y_end, x_start:x_end] = processed_face[face_y_start:face_y_end, face_x_start:face_x_end]
        
        # Create annotation with actual placed position
        annotation = {
            'field': placeholder_config.get('id', 'photo'),
            'bbox': [x_start, y_start, x_end - x_start, y_end - y_start],
            'rel_bbox': rel_bbox,
            'type': 'photo',
            'identity_id': identity_id
        }
        
        return template, annotation
    
    def render_placeholder(self,
                          template: np.ndarray,
                          placeholder_config: Dict,
                          card_region: List[int],
                          sex: str = 'M') -> Tuple[np.ndarray, Dict]:
        """
        Render a photo placeholder with a random real face.
        
        Args:
            template: Document template image
            placeholder_config: Configuration for this placeholder
            card_region: [x, y, w, h] of card region
            sex: Sex for face selection matching
            
        Returns:
            Tuple of (updated_template, annotation)
        """
        # Get random face and delegate to render_placeholder_with_face
        face_img, identity_id = self.face_manager.get_random_face(sex)
        return self.render_placeholder_with_face(
            template, placeholder_config, card_region, face_img, identity_id
        )


# Test function
if __name__ == '__main__':
    import sys
    
    # Example usage
    vggface2_dir = Path('data/vggface2')
    
    if not vggface2_dir.exists():
        print(f"VGGFace2 dataset not found at {vggface2_dir}")
        print("Please download from: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/")
        sys.exit(1)
    
    # Initialize manager
    manager = VGGFace2PhotoManager(vggface2_dir, seed=42)
    
    # Get a random face
    face, identity = manager.get_random_face()
    print(f"Loaded face from identity: {identity}")
    print(f"Face shape: {face.shape}")
    
    # Test preprocessing
    processed = manager.preprocess_face(face, (200, 250), shape='rect')
    print(f"Processed shape: {processed.shape}")
    
    # Save test output
    cv2.imwrite('test_face_processed.jpg', processed)
    print("Saved test_face_processed.jpg")
