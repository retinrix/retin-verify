# Synthetic Data Generation Pipeline

This directory contains the complete synthetic data generation pipeline for Retin-Verify, creating realistic Algerian ID document images using Blender.

## Overview

The pipeline consists of:
1. **Identity Generator** (`scripts/identity_generator.py`) - Generates fake Algerian identity data
2. **Blender Generator** (`scripts/blender_document_generator.py`) - Renders document images with variations
3. **Annotation Utilities** (`scripts/annotation_utils.py`) - Handles annotation formats and validation
4. **Pipeline Runner** (`scripts/data_acquisition_pipeline.py`) - Orchestrates the entire process

## Directory Structure

```
synthetic/
├── scripts/              # Python scripts for data generation
│   ├── identity_generator.py
│   ├── blender_document_generator.py
│   ├── annotation_utils.py
│   └── data_acquisition_pipeline.py
├── scenes/               # Blender scene files (.blend)
├── templates/            # Clean document templates (PNG with transparency)
├── backgrounds/          # Background images for rendering
└── output/               # Generated dataset output
```

## Quick Start

### 1. Prepare Document Templates

You need clean templates of each document type with personal data removed:

**Option A: Create from your own documents (Recommended)**
1. Scan your document at 300-600 DPI
2. Open in GIMP/Photoshop
3. Remove all personal data (name, photo, numbers, MRZ)
4. Keep security patterns, layout, field labels
5. Save as transparent PNG: `synthetic/templates/passport_blank.png`

**Option B: Create synthetic templates**
Use graphic design software to recreate document layouts with:
- Accurate dimensions (passport: 125×88mm, CNIE: 85.6×54mm)
- Security pattern placeholders
- Field label positions

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Blender (Ubuntu)
sudo apt-get install blender

# Or download from blender.org
```

### 3. Generate Dataset

**Option A: Full pipeline with Blender rendering**
```bash
# Generate complete dataset with images
python synthetic/scripts/data_acquisition_pipeline.py --config configs/data_acquisition.yaml
```

**Option B: Generate identities only (no Blender)**
```bash
# Generate just the synthetic identity data
python synthetic/scripts/data_acquisition_pipeline.py --skip-blender
```

**Option C: Manual Blender generation**
```bash
# Run directly in Blender
blender --background --python synthetic/scripts/blender_document_generator.py -- \
    --template-dir synthetic/templates \
    --output-dir synthetic/output/images/passport \
    --doc-type passport \
    --template passport_blank \
    --num-samples 1000
```

## Generated Variations

The pipeline creates realistic variations in:

| Aspect | Range | Description |
|--------|-------|-------------|
| **Camera Angle** | -45° to +45° | Pitch, yaw, roll variations |
| **Distance** | 50-90% frame | Document scale variation |
| **Lighting** | Multiple types | Daylight, indoor warm/cool, low light |
| **Background** | 100+ options | Solid colors, desks, outdoor scenes |
| **Document State** | Flat to curved | Simulates physical bending |

## Output Format

Each generated sample includes:

```
synthetic/output/images/passport/000001/
├── image.png           # RGB render
├── annotations.json    # Bounding boxes, labels, metadata
└── depth.exr          # Depth map (optional)
```

### Annotation Schema

```json
{
  "sample_id": 1,
  "document_type": "passport",
  "identity": {
    "surname": "BENALI",
    "given_names": "MOHAMED",
    "date_of_birth": "15/03/1985",
    "passport_number": "DZ1234567",
    "mrz": {
      "line1": "P<DZABENALI<<MOHAMED<<<<<<<<<<<<<<",
      "line2": "12345678DZA8503157M2704159<<<<<<06"
    }
  },
  "image_path": "passport/000001/image.png",
  "image_size": [1920, 1080],
  "camera_pose": {...},
  "lighting": {...},
  "bounding_boxes": [
    {
      "field": "surname",
      "bbox": [450, 320, 200, 40],
      "text": "BENALI",
      "confidence": 1.0
    }
  ]
}
```

## Export Formats

The pipeline automatically exports to:

- **COCO** - Standard object detection format
- **YOLO** - Darknet format for training
- **Pascal VOC** - XML-based format
- **JSON** - Custom structured format

Exports are saved to: `synthetic/output/exports/`

## Dataset Statistics

| Document Type | Target Samples | Disk Usage (est.) |
|---------------|----------------|-------------------|
| Passport | 10,000 | ~5 GB |
| CNIE Front | 5,000 | ~2.5 GB |
| CNIE Back | 5,000 | ~2.5 GB |
| Carte Grise | 3,000 | ~1.5 GB |
| **Total** | **23,000** | **~12 GB** |

## Advanced Configuration

Edit `configs/data_acquisition.yaml` to customize:

```yaml
# Number of samples per document type
document_types:
  passport:
    num_samples: 5000  # Reduce for testing
    
# Render quality vs speed
quality:
  render_engine: "BLENDER_EEVEE"  # Faster than CYCLES
  samples: 64  # Lower for faster renders
  
# Output resolution
output:
  resolution: [1280, 720]  # Lower for smaller files
```

## Creating Templates from Your Documents

### Step-by-Step Guide

1. **Scan at high resolution**
   ```
   Settings: 300-600 DPI, Color, TIFF or PNG
   ```

2. **Remove personal data in GIMP**
   ```
   - Use Clone Stamp tool to copy security patterns over text
   - Content-aware fill for larger areas
   - Heal tool for seamless blending
   ```

3. **Preserve these elements:**
   - Document dimensions and proportions
   - Security pattern backgrounds
   - Field labels ("Nom/Name", "Date de naissance")
   - Government logos (if not identifying)
   - Border patterns

4. **Create layers for:**
   - Background (document base)
   - Text placeholders (where data will be overlaid)
   - Photo area (transparent placeholder)
   - Signature area

5. **Export settings:**
   ```
   Format: PNG
   Color mode: RGBA (with transparency)
   Compression: None or minimal
   ```

## Troubleshooting

### Blender not found
```bash
# Specify Blender path in config
paths:
  blender_executable: "/usr/bin/blender"  # Linux
  # or "/Applications/Blender.app/Contents/MacOS/Blender"  # macOS
```

### Out of memory
- Reduce `samples` in config (64 → 32)
- Lower `resolution` (1920×1080 → 1280×720)
- Generate in smaller batches

### Template not loading
- Ensure PNG format with transparency
- Check file is in `synthetic/templates/`
- Verify template name matches config

## Legal Notice

The synthetic data generated by this pipeline:
- Contains **no real personal data**
- Uses fake identities and document numbers
- Includes invalid (synthetic) MRZ codes
- Is safe for ML training without privacy concerns

**Important**: Real document scans used as templates should:
- Have all personal data removed
- Be stored securely
- Not be distributed
