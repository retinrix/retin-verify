# Photo Placeholder System Guide

This guide explains how to use the new Photo Placeholder System with VGGFace2 integration in Retin-Verify.

## Overview

The Photo Placeholder System allows you to place real human face photos from the VGGFace2 dataset into ID document templates, creating more realistic synthetic training data for document verification models.

## Features

- ✅ **Real Face Integration**: Use faces from VGGFace2 dataset instead of synthetic placeholders
- ✅ **Multiple Photo Types**: Support for rectangular and oval photo shapes
- ✅ **Configurable Borders**: Add customizable borders around photos
- ✅ **GUI Editor**: Visual editor for positioning photo placeholders
- ✅ **Backward Compatibility**: Falls back to synthetic faces if VGGFace2 is unavailable

## Quick Start

### 1. Download VGGFace2 Dataset

Download the VGGFace2 dataset from: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

Extract to: `data/vggface2/`

The expected structure is:
```
data/vggface2/
├── train/
│   ├── n000001/
│   │   ├── 0001_01.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── n000002/
    └── ...
```

### 2. Run the GUI Tool

```bash
cd synthetic/scripts
python3 server.py
```

Open browser to http://127.0.0.1:5000

### 3. Configure Photo Placeholders

1. Load a CNIE template image
2. Define the card region (if not already set)
3. In the **Photo Placeholders** section:
   - Click "➕ Add Placeholder"
   - Enter a name (e.g., `left_photo`)
   - Select shape (Rectangle or Oval)
   - Configure border options
4. Position the placeholder on the template by clicking

### 4. Set VGGFace2 Path

1. In the **VGGFace2 Dataset** section:
   - Enter the path to your VGGFace2 dataset (e.g., `data/vggface2`)
   - Click "Test Connection"
   - You should see "Connected! X identities found"

### 5. Save and Generate

1. Save your configuration
2. Click "🖼️ Generate Preview" to see a sample
3. Click "📦 Generate Dataset" to create multiple samples

## Configuration Format

Photo placeholders are stored in the JSON configuration file:

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
      }
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
  "vggface2_path": "data/vggface2",
  "fields": { ... }
}
```

### Placeholder Configuration Options

| Field | Type | Description |
|-------|------|-------------|
| `rel_bbox` | Array | Relative bounding box `[x, y, width, height]` as fractions of card region |
| `shape` | String | `"rect"` or `"oval"` |
| `background` | String | `"white"` or `"transparent"` |
| `border` | Object | Optional border configuration |
| `border.width` | Integer | Border width in pixels |
| `border.color` | String | Border color as hex (e.g., `"#cccccc"`) |

## Programmatic Usage

### Generate Documents with Real Faces

```python
from synthetic.scripts.template_document_generator import TemplateDocumentGenerator
from pathlib import Path

generator = TemplateDocumentGenerator(
    template_dir=Path('synthetic/templates'),
    output_dir=Path('data/output'),
    config_file=Path('synthetic/configs/cnie_front_with_photos.json'),
    face_photos_dir=Path('data/vggface2'),  # Enable VGGFace2 integration
    seed=42
)

# Generate 100 samples
generator.generate_dataset('cnie_front', num_samples=100)
```

### Use Face Photo Manager Directly

```python
from synthetic.scripts.face_photo_manager import VGGFace2PhotoManager, PhotoPlaceholderRenderer
import cv2

# Initialize manager
face_manager = VGGFace2PhotoManager('data/vggface2')

# Get a random face
face_img, identity_id = face_manager.get_random_face()
print(f"Loaded face from identity: {identity_id}")

# Preprocess for a 200x250 rectangular photo
processed = face_manager.preprocess_face(face_img, (200, 250), shape='rect')

# Save result
cv2.imwrite('output_face.jpg', processed)
```

## API Endpoints

The server provides the following endpoints for VGGFace2 integration:

### Set VGGFace2 Path
```http
POST /set_vggface2_path
Content-Type: application/json

{
  "path": "/path/to/vggface2"
}
```

Response:
```json
{
  "status": "ok",
  "path": "/path/to/vggface2",
  "has_train": true,
  "has_test": true,
  "num_identities": 8631
}
```

### Get VGGFace2 Status
```http
GET /get_vggface2_status
```

Response:
```json
{
  "configured": true,
  "path": "/path/to/vggface2",
  "has_train": true,
  "has_test": true,
  "num_identities": 8631,
  "num_images": 3141890
}
```

### Preview Random Face
```http
POST /preview_face
```

Returns a JPEG image of a random face from the dataset.

## Performance Considerations

| Aspect | Details |
|--------|---------|
| **Indexing** | ~2-3 seconds for full VGGFace2 dataset (3M+ images) |
| **Memory** | ~500MB for index; individual faces are ~50-200KB |
| **Generation Speed** | ~100ms additional time per sample vs synthetic |
| **Disk Space** | VGGFace2 dataset is ~37GB for train+test |

## Troubleshooting

### VGGFace2 Not Found

If you see "VGGFace2 not configured":
1. Verify the path is correct
2. Check that the directory contains `train/` or `test/` subdirectories
3. Ensure the images are in `.jpg` format

### Faces Not Loading

If faces fail to load:
1. Check that images can be read by OpenCV: `cv2.imread('path/to/image.jpg')`
2. Verify file permissions
3. Try a smaller subset of the dataset for testing

### Out of Memory

For large-scale generation:
1. Process in smaller batches
2. Consider using a subset of VGGFace2 identities
3. Monitor memory usage with `top` or `htop`

## Future Enhancements

1. **Sex/Age Classification**: Pre-classify VGGFace2 images for better identity matching
2. **Face Alignment**: Use facial landmarks for consistent positioning
3. **Background Removal**: Automatic background removal from photos
4. **Custom Datasets**: Support for other face datasets beyond VGGFace2
5. **Photo Augmentation**: Apply lighting/color corrections to match templates

## References

- VGGFace2 Dataset: http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
- Paper: "VGGFace2: A dataset for recognising faces across pose and age"
