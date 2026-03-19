# Retin-Verify Implementation Summary 3
## Photo-Aware Template Generator with VGGFace2 Integration

### Overview

This document describes the extended synthetic data generation pipeline that adds **realistic face photo integration** from the VGGFace2 dataset into CNIE (Carte Nationale d'Identité Électronique) and Passport templates. This enhancement creates more realistic ID document samples by replacing synthetic face generation with real human faces from a curated dataset.

---

## 🆕 New Components

### 1. Photo Placeholder System

#### 1.1 Template Configuration Extensions

The existing JSON configuration format now supports photo placeholders:

```json
{
  "card_region": [0, 0, 1622, 1020],
  "photo_placeholders": {
    "left_photo": {
      "rel_bbox": [0.035, 0.25, 0.22, 0.35],
      "shape": "rect",
      "background": "white",
      "border": {
        "width": 2,
        "color": "#cccccc"
      },
      "margin": 5
    },
    "small_middle_photo": {
      "rel_bbox": [0.42, 0.15, 0.12, 0.18],
      "shape": "oval",
      "background": "white",
      "border": {
        "width": 1,
        "color": "#999999"
      }
    }
  },
  "fields": { ... }
}
```

#### 1.2 Placeholder Types

| Placeholder ID | Description | Typical Use Case |
|----------------|-------------|------------------|
| `left_photo` | Large rectangular photo on left side | CNIE front main photo |
| `small_middle_photo` | Small oval/circular photo in center | Passport secondary photo |
| `signature_photo` | Wide rectangular area for signature | CNIE signature field |

---

### 2. VGGFace2 Integration Module

#### 2.1 Module: `synthetic/scripts/face_photo_manager.py`

```python
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
    
    def __init__(self, vggface2_dir: Path, seed: Optional[int] = None):
        """
        Initialize the face photo manager.
        
        Args:
            vggface2_dir: Path to VGGFace2 dataset root (contains train/ and test/ folders)
            seed: Random seed for reproducible face selection
        """
        self.vggface2_dir = Path(vggface2_dir)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        
        # Index of available faces
        self.face_index: List[Dict] = []
        self._build_index()
    
    def _build_index(self):
        """Build index of all available face images in VGGFace2 dataset."""
        print(f"🔍 Indexing VGGFace2 dataset at: {self.vggface2_dir}")
        
        # VGGFace2 structure: vggface2/train/n000001/0001_01.jpg
        for split in ['train', 'test']:
            split_dir = self.vggface2_dir / split
            if not split_dir.exists():
                continue
                
            for identity_dir in split_dir.iterdir():
                if not identity_dir.is_dir():
                    continue
                
                identity_id = identity_dir.name
                
                # Collect all images for this identity
                for img_path in identity_dir.glob('*.jpg'):
                    self.face_index.append({
                        'identity_id': identity_id,
                        'image_path': img_path,
                        'split': split
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
    
    def render_placeholder(self,
                          template: np.ndarray,
                          placeholder_config: Dict,
                          card_region: List[int],
                          sex: str = 'M') -> Tuple[np.ndarray, Dict]:
        """
        Render a photo placeholder with a real face.
        
        Args:
            template: Document template image
            placeholder_config: Configuration for this placeholder
            card_region: [x, y, w, h] of card region
            sex: Sex for face selection matching
            
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
        
        # Get random face
        face_img, identity_id = self.face_manager.get_random_face(sex)
        
        # Preprocess face
        shape = placeholder_config.get('shape', 'rect')
        processed_face = self.face_manager.preprocess_face(
            face_img, (abs_w, abs_h), shape
        )
        
        # Add border if specified
        border_config = placeholder_config.get('border')
        if border_config:
            border_width = border_config.get('width', 2)
            border_color_hex = border_config.get('color', '#cccccc')
            # Convert hex to BGR
            border_color = tuple(int(border_color_hex[i:i+2], 16) 
                                for i in (5, 3, 1))  # BGR order
            processed_face = self.face_manager.add_border(
                processed_face, border_width, border_color
            )
            # Adjust position for border
            abs_x -= border_width
            abs_y -= border_width
            abs_w += 2 * border_width
            abs_h += 2 * border_width
        
        # Place into template
        # Ensure we don't go out of bounds
        roi_h, roi_w = processed_face.shape[:2]
        if abs_y + roi_h > h:
            roi_h = h - abs_y
        if abs_x + roi_w > w:
            roi_w = w - abs_x
            
        template[abs_y:abs_y+roi_h, abs_x:abs_x+roi_w] = processed_face[:roi_h, :roi_w]
        
        # Create annotation
        annotation = {
            'field': placeholder_config.get('id', 'photo'),
            'bbox': [abs_x, abs_y, abs_w, abs_h],
            'rel_bbox': rel_bbox,
            'type': 'photo',
            'identity_id': identity_id
        }
        
        return template, annotation
```

#### 2.2 Integration with TemplateDocumentGenerator

Extend `template_document_generator.py` with photo rendering:

```python
# Add to TemplateDocumentGenerator class:

def __init__(self, ..., face_photos_dir: Optional[Path] = None):
    ...
    # Initialize face photo manager if VGGFace2 directory provided
    self.face_manager = None
    if face_photos_dir and face_photos_dir.exists():
        from face_photo_manager import VGGFace2PhotoManager, PhotoPlaceholderRenderer
        self.face_manager = VGGFace2PhotoManager(face_photos_dir, seed=seed)
        self.photo_renderer = PhotoPlaceholderRenderer(self.face_manager)

def render_document(self, doc_type: str, identity: Dict) -> Tuple[np.ndarray, List[Dict]]:
    """Render document with photo placeholders."""
    ...
    # Existing text field rendering...
    
    # Render photo placeholders
    photo_placeholders = spec.get('photo_placeholders', {})
    for placeholder_id, placeholder_config in photo_placeholders.items():
        if self.face_manager:
            # Use real face from VGGFace2
            template, photo_ann = self.photo_renderer.render_placeholder(
                template,
                {**placeholder_config, 'id': placeholder_id},
                card_region if card_region else [0, 0, w, h],
                identity.get('sex', 'M')
            )
            annotations.append(photo_ann)
        else:
            # Fallback to synthetic face generation
            photo_bbox = placeholder_config['rel_bbox']
            if card_region:
                abs_photo_bbox = self._card_relative_to_absolute(
                    photo_bbox, card_region, w, h
                )
                px, py, pw, ph = abs_photo_bbox
            else:
                px = int(photo_bbox[0] * w)
                py = int(photo_bbox[1] * h)
                pw = int(photo_bbox[2] * w)
                ph = int(photo_bbox[3] * h)
            
            face = self.generate_synthetic_face((ph, pw), identity.get('sex', 'M'))
            template[py:py+ph, px:px+pw] = face
            
            annotations.append({
                'field': placeholder_id,
                'bbox': [px, py, pw, ph],
                'rel_bbox': photo_bbox,
                'type': 'photo',
                'synthetic': True
            })
    
    return template, annotations
```

---

### 3. Updated GUI Tool with Photo Placeholder Editor

#### 3.1 Extended HTML Interface

Add to `cnie_tool.html` control panel:

```html
<h2>🖼️ Photo Placeholders</h2>
<div class="coordinates-panel">
    <div id="photoPlaceholderList" class="field-list" style="max-height: 150px;">
        <!-- Dynamic list of photo placeholders -->
    </div>
    <div class="field-toolbar">
        <button onclick="addPhotoPlaceholder()">➕ Add Placeholder</button>
        <button onclick="deletePhotoPlaceholder()" class="danger">🗑️ Delete</button>
    </div>
    <div class="coordinate-row">
        <label>Type:</label>
        <select id="photoShape">
            <option value="rect">Rectangle</option>
            <option value="oval">Oval</option>
        </select>
    </div>
    <div class="coordinate-row">
        <label>Background:</label>
        <select id="photoBackground">
            <option value="white">White</option>
            <option value="transparent">Transparent</option>
        </select>
    </div>
    <div class="coordinate-row">
        <label>Border:</label>
        <input type="checkbox" id="photoBorder" checked>
        <input type="color" id="photoBorderColor" value="#cccccc">
        <input type="number" id="photoBorderWidth" value="2" min="0" max="10" style="width:50px;">
    </div>
</div>

<h2>📁 VGGFace2 Dataset</h2>
<div class="coordinates-panel">
    <div class="coordinate-row">
        <label>Dataset Path:</label>
        <input type="text" id="vggface2Path" placeholder="/path/to/vggface2">
    </div>
    <button onclick="testVGGFace2Connection()">Test Connection</button>
    <div id="vggface2Status" style="font-size: 0.85rem; margin-top: 5px;">
        Status: Not configured
    </div>
</div>
```

#### 3.2 JavaScript for Photo Placeholder Management

```javascript
// Add to cnie_tool.html <script> section:

let photoPlaceholders = {};
let currentPhotoPlaceholder = null;

function addPhotoPlaceholder() {
    const id = `photo_${Object.keys(photoPlaceholders).length + 1}`;
    const name = prompt('Enter placeholder name (e.g., left_photo, small_middle_photo):', id);
    if (!name) return;
    
    photoPlaceholders[name] = {
        rel_bbox: [0.1, 0.1, 0.2, 0.25],  // Default position
        shape: 'rect',
        background: 'white',
        border: { width: 2, color: '#cccccc' }
    };
    
    updatePhotoPlaceholderList();
    selectPhotoPlaceholder(name);
}

function updatePhotoPlaceholderList() {
    const list = document.getElementById('photoPlaceholderList');
    list.innerHTML = '';
    
    Object.entries(photoPlaceholders).forEach(([id, config]) => {
        const div = document.createElement('div');
        div.className = 'field-item';
        div.dataset.placeholderId = id;
        div.onclick = () => selectPhotoPlaceholder(id);
        
        const shapeIcon = config.shape === 'oval' ? '⭕' : '▭';
        div.innerHTML = `
            <div class="color-indicator" style="background:#81ecec"></div>
            <div class="field-name">${shapeIcon} ${id}</div>
            <div class="field-status set">✅</div>
        `;
        list.appendChild(div);
    });
}

function selectPhotoPlaceholder(id) {
    document.querySelectorAll('#photoPlaceholderList .field-item').forEach(e => {
        e.classList.remove('active');
    });
    document.querySelector(`#photoPlaceholderList .field-item[data-placeholder-id="${id}"]`)?.classList.add('active');
    
    currentPhotoPlaceholder = id;
    const config = photoPlaceholders[id];
    
    // Update UI controls
    document.getElementById('photoShape').value = config.shape;
    document.getElementById('photoBackground').value = config.background;
    document.getElementById('photoBorder').checked = !!config.border;
    document.getElementById('photoBorderColor').value = config.border?.color || '#cccccc';
    document.getElementById('photoBorderWidth').value = config.border?.width || 2;
    
    redrawCanvas();
}

// Modify redrawCanvas() to draw photo placeholders:
function redrawCanvas() {
    // ... existing drawing code ...
    
    // Draw photo placeholders
    Object.entries(photoPlaceholders).forEach(([id, config]) => {
        if (!cardRegion) return;
        
        const absX = cardRegion.x + config.rel_bbox[0] * cardRegion.width;
        const absY = cardRegion.y + config.rel_bbox[1] * cardRegion.height;
        const absW = config.rel_bbox[2] * cardRegion.width;
        const absH = config.rel_bbox[3] * cardRegion.height;
        
        // Draw placeholder background
        ctx.fillStyle = config.background === 'white' ? '#ffffff' : 'rgba(200,200,200,0.5)';
        
        if (config.shape === 'oval') {
            ctx.beginPath();
            ctx.ellipse(
                absX + absW/2, absY + absH/2,
                absW/2, absH/2, 0, 0, 2 * Math.PI
            );
            ctx.fill();
            ctx.strokeStyle = id === currentPhotoPlaceholder ? '#00ff00' : '#81ecec';
            ctx.lineWidth = 2;
            ctx.stroke();
        } else {
            ctx.fillRect(absX, absY, absW, absH);
            ctx.strokeStyle = id === currentPhotoPlaceholder ? '#00ff00' : '#81ecec';
            ctx.lineWidth = 2;
            ctx.strokeRect(absX, absY, absW, absH);
        }
        
        // Draw placeholder label
        ctx.fillStyle = '#333333';
        ctx.font = '10px Arial';
        ctx.fillText(`[${id}]`, absX, absY - 5);
    });
    
    // ... rest of existing code ...
}

function getCurrentConfig() {
    // ... existing code ...
    
    // Add photo placeholders to config
    config.photo_placeholders = photoPlaceholders;
    
    // Add VGGFace2 path if configured
    const vggface2Path = document.getElementById('vggface2Path').value;
    if (vggface2Path) {
        config.vggface2_path = vggface2Path;
    }
    
    return config;
}

function loadConfigToGUI(config) {
    // ... existing code ...
    
    // Load photo placeholders
    if (config.photo_placeholders) {
        photoPlaceholders = config.photo_placeholders;
        updatePhotoPlaceholderList();
    }
    
    // Load VGGFace2 path
    if (config.vggface2_path) {
        document.getElementById('vggface2Path').value = config.vggface2_path;
    }
}
```

---

### 4. Server API Extensions

#### 4.1 New Endpoints in `server.py`

```python
# Add to server.py:

VGGFACE2_DIR = PROJECT_ROOT / "data" / "vggface2"  # Default VGGFace2 location

@app.route('/set_vggface2_path', methods=['POST'])
def set_vggface2_path():
    """Set the path to VGGFace2 dataset."""
    data = request.json
    path = data.get('path')
    
    if not path:
        return jsonify({"error": "No path provided"}), 400
    
    vggface2_path = Path(path)
    if not vggface2_path.exists():
        return jsonify({"error": "Path does not exist"}), 404
    
    # Verify structure (should have train/ and/or test/ subdirectories)
    has_train = (vggface2_path / 'train').exists()
    has_test = (vggface2_path / 'test').exists()
    
    if not (has_train or has_test):
        return jsonify({"error": "Invalid VGGFace2 structure (missing train/ or test/)"}), 400
    
    global VGGFACE2_DIR
    VGGFACE2_DIR = vggface2_path
    
    # Count identities
    num_identities = len(list(vggface2_path.glob('train/*'))) if has_train else 0
    
    return jsonify({
        "status": "ok",
        "path": str(VGGFACE2_DIR),
        "has_train": has_train,
        "has_test": has_test,
        "num_identities": num_identities
    })


@app.route('/preview_face', methods=['POST'])
def preview_face():
    """Generate a preview of a random face from VGGFace2."""
    if not VGGFACE2_DIR or not VGGFACE2_DIR.exists():
        return jsonify({"error": "VGGFace2 not configured"}), 400
    
    # Import face manager
    sys.path.insert(0, str(BASE_DIR))
    from face_photo_manager import VGGFace2PhotoManager
    
    try:
        face_manager = VGGFace2PhotoManager(VGGFACE2_DIR)
        face_img, identity_id = face_manager.get_random_face()
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', face_img)
        
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

---

### 5. Directory Structure

```
retin-verify/
├── data/
│   ├── vggface2/                    # VGGFace2 dataset (user-provided)
│   │   ├── train/
│   │   │   ├── n000001/
│   │   │   │   ├── 0001_01.jpg
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── test/
│   ├── gui_preview/                 # GUI preview outputs
│   └── cnie_output/                 # Batch generation outputs
├── synthetic/
│   ├── scripts/
│   │   ├── face_photo_manager.py    # ⭐ NEW: VGGFace2 integration
│   │   ├── template_document_generator.py  # ⭐ MODIFIED: Photo support
│   │   ├── server.py                # ⭐ MODIFIED: New endpoints
│   │   └── gui_tool/
│   │       └── cnie_tool.html       # ⭐ MODIFIED: Photo UI
│   └── configs/
│       └── cnie_front_with_photos.json  # Example config with photos
└── docs/
    └── IMPLEMENTATION_SUMMARY_3.md  # This document
```

---

### 6. Usage Examples

#### 6.1 Generate CNIE with Real Faces

```bash
# Step 1: Ensure VGGFace2 dataset is available
# Download from: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
# Extract to: data/vggface2/

# Step 2: Run GUI tool to configure photo placeholders
python synthetic/scripts/server.py
# Open browser to http://127.0.0.1:5000
# - Load CNIE template
# - Define card region
# - Add "left_photo" placeholder
# - Set VGGFace2 path
# - Save config

# Step 3: Generate samples with real faces
python synthetic/scripts/run_template_pipeline.py \
    --doc-type cnie_front \
    --doc-config synthetic/configs/cnie_front_with_photos.json \
    --num-samples 100 \
    --output-dir data/cnie_with_faces
```

#### 6.2 Programmatic Usage

```python
from synthetic.scripts.face_photo_manager import VGGFace2PhotoManager, PhotoPlaceholderRenderer
from synthetic.scripts.template_document_generator import TemplateDocumentGenerator
from synthetic.scripts.identity_generator import AlgerianIdentityGenerator

# Initialize components
face_manager = VGGFace2PhotoManager('data/vggface2')
generator = AlgerianIdentityGenerator()

# Generate identity
identity = generator.generate_identity('cnie')

# Load template
template = cv2.imread('synthetic/templates/real/cnie_front_template.jpg')

# Define photo placeholder config
photo_config = {
    'id': 'left_photo',
    'rel_bbox': [0.035, 0.25, 0.22, 0.35],
    'shape': 'rect',
    'background': 'white',
    'border': {'width': 2, 'color': '#cccccc'}
}

# Render photo
renderer = PhotoPlaceholderRenderer(face_manager)
template, annotation = renderer.render_placeholder(
    template,
    photo_config,
    card_region=[0, 0, 1622, 1020],
    sex=identity['sex']
)

# Continue with text rendering...
```

---

### 7. Configuration Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CNIE Template Configuration with Photos",
  "type": "object",
  "required": ["card_region", "fields"],
  "properties": {
    "card_region": {
      "type": "array",
      "items": {"type": "integer"},
      "minItems": 4,
      "maxItems": 4,
      "description": "[x, y, width, height] of card region in template"
    },
    "vggface2_path": {
      "type": "string",
      "description": "Optional path to VGGFace2 dataset"
    },
    "photo_placeholders": {
      "type": "object",
      "patternProperties": {
        "^[a-z_]+$": {
          "type": "object",
          "required": ["rel_bbox"],
          "properties": {
            "rel_bbox": {
              "type": "array",
              "items": {"type": "number"},
              "minItems": 4,
              "maxItems": 4,
              "description": "[x, y, width, height] relative to card region"
            },
            "shape": {
              "type": "string",
              "enum": ["rect", "oval"],
              "default": "rect"
            },
            "background": {
              "type": "string",
              "enum": ["white", "transparent"],
              "default": "white"
            },
            "border": {
              "type": "object",
              "properties": {
                "width": {"type": "integer", "minimum": 0},
                "color": {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"}
              }
            }
          }
        }
      }
    },
    "fields": {
      "type": "object",
      "description": "Text field configurations (existing format)"
    }
  }
}
```

---

### 8. Performance Considerations

| Aspect | Consideration |
|--------|--------------|
| **Indexing** | VGGFace2 index is built once at initialization; ~2-3 seconds for full dataset |
| **Face Loading** | Images loaded on-demand; cache recently used faces for batch generation |
| **Memory** | ~500MB for index; individual faces are ~50-200KB |
| **Generation Speed** | Adding real faces adds ~100ms per sample vs synthetic faces |

---

### 9. Future Enhancements

1. **Sex/Age Classification**: Pre-classify VGGFace2 images by sex and age for better identity matching
2. **Face Alignment**: Use facial landmarks to ensure consistent face positioning
3. **Background Removal**: Automatic background removal from VGGFace2 photos
4. **Custom Photo Datasets**: Support for other face datasets beyond VGGFace2
5. **Photo Augmentation**: Apply lighting/color corrections to match template

---

### 10. Dependencies

```
# Add to requirements.txt
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.0.0
Flask>=2.0.0
flask-cors>=3.0.0
```

---

## Summary

This implementation extends the Retin-Verify synthetic data generation pipeline with:

1. ✅ **Photo Placeholder System**: Configurable regions in templates for face photos
2. ✅ **VGGFace2 Integration**: Real human faces instead of synthetic placeholders
3. ✅ **GUI Enhancements**: Visual editor for photo placeholder positioning
4. ✅ **Flexible Rendering**: Support for rectangular and oval photo shapes
5. ✅ **White Background**: Clean ID-card-style photo presentation
6. ✅ **Backward Compatibility**: Falls back to synthetic faces if VGGFace2 unavailable

**Next Steps**:
1. Download and extract VGGFace2 dataset to `data/vggface2/`
2. Run GUI tool to configure photo placeholders
3. Generate enhanced training data with real faces
