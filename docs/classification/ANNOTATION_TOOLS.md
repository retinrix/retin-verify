# Image Annotation Tools

Two tools to make annotating your photos fast and reusable.

---

## Tool 1: `annotate_images.py` (GUI with Image Preview)

**Best for:** When you want to see each image and possibly rotate it

### Features
- Shows image with overlay UI
- Keyboard shortcuts (1-4 for labels, arrows to navigate)
- Rotate images in 90° increments
- Mark bad photos for deletion
- Saves reusable JSON annotations

### Usage

```bash
# Start annotating
python annotate_images.py --input-dir ./raw_photos

# Review and fix existing annotations
python annotate_images.py --input-dir ./raw_photos --review annotations.json

# Annotate and auto-organize into folders
python annotate_images.py --input-dir ./raw_photos --organize ./dataset
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **1** | Label as CNIE Front |
| **2** | Label as CNIE Back |
| **3** | Label as Passport |
| **4** | Label as Carte Grise |
| **→ or SPACE** | Next image |
| **←** | Previous image |
| **r** | Rotate 90° clockwise |
| **R** | Rotate 90° counter-clockwise |
| **d** | Mark for deletion |
| **q or ESC** | Save and quit |
| **?** | Show help |

---

## Tool 2: `batch_annotate.py` (Fast CLI)

**Best for:** When you already know what's in your photos and want speed

### Features
- No GUI - fast terminal input
- Pattern-based auto-annotation
- Keyword-based auto-annotation
- Batch operations

### Usage

#### Option A: Interactive Mode (Fast Keyboard)

```bash
python batch_annotate.py --input-dir ./raw_photos
```

You'll see:
```
🖼️  Found 50 images

Annotation keys:
  1 = CNIE Front
  2 = CNIE Back
  3 = Passport
  4 = Carte Grise
  s = Skip
  q = Save and quit
--------------------------------------------------

📝 IMG_2024_001.jpg
Label [1/2/3/4/s/q]: 1
   ✅ Labeled as: cnie_front

📝 IMG_2024_002.jpg
Label [1/2/3/4/s/q]: 2
   ✅ Labeled as: cnie_back
...
```

#### Option B: Pattern Mode (Smart Auto-Label)

If your filenames have patterns like:
- `cardA_front_01.jpg`
- `cardA_back_01.jpg`
- `cardB_front_01.jpg`
- `cardB_back_01.jpg`

```bash
python batch_annotate.py --input-dir ./raw_photos --pattern
```

The tool will:
1. Analyze common patterns in filenames
2. Ask you to assign labels to each pattern
3. Auto-apply labels to all matching files

Example:
```
Pattern 'front' matches 16 files:
  - cardA_front_01.jpg
  - cardA_front_02.jpg
  ...

Label for 'front' [1/2/3/4/s/Enter=skip]: 1
   ✅ 'front' → cnie_front
```

#### Option C: Keyword Auto-Annotation

```bash
python batch_annotate.py --input-dir ./raw_photos --auto \
    --front "flat,front,recto" \
    --back "back,dos,verso" \
    --passport "passport" \
    --grise "grise,carte"
```

This scans filenames for keywords and auto-labels:
- Files with "flat", "front", or "recto" → cnie_front
- Files with "back", "dos", or "verso" → cnie_back
- etc.

---

## Recommended Workflow

### For Your 16 CNIE Photos (Strategy 2)

**If you followed the naming convention:**

```bash
# Auto-annotate based on keywords in filenames
python batch_annotate.py \
    --input-dir ./my_4_cards_cnie_only/raw \
    --auto \
    --front "flat,angled,handheld,dim,front" \
    --back "back" \
    --organize ./my_4_cards_cnie_only
```

This will:
1. Auto-label images with "front" in name → cnie_front
2. Auto-label images with "back" in name → cnie_back
3. Copy organized images to:
   ```
   my_4_cards_cnie_only/
   ├── cnie_front/
   │   ├── cardA_flat.jpg
   │   ├── cardA_angled.jpg
   │   └── ...
   └── cnie_back/
       ├── cardA_flat.jpg
       └── ...
   ```

### For Many Photos (Strategy 1 or 3)

**Step 1:** Put all raw photos in one folder
```
raw_photos/
├── IMG_001.jpg
├── IMG_002.jpg
└── ...
```

**Step 2:** Annotate with GUI (to see and rotate)
```bash
python annotate_images.py --input-dir ./raw_photos
```

**Step 3:** Organize into dataset
```bash
python batch_annotate.py \
    --input-dir ./raw_photos \
    --organize ./dataset
```

---

## Reusing Annotations

The tools create `annotations.json` which you can:

1. **Review later:**
   ```bash
   python annotate_images.py --review annotations.json
   ```

2. **Edit manually:**
   ```json
   {
     "annotations": [
       {
         "filename": "cardA_flat.jpg",
         "filepath": "/path/to/cardA_flat.jpg",
         "label": "cnie_front",
         "rotation": 0,
         "marked_for_deletion": false,
         "notes": ""
       }
     ]
   }
   ```

3. **Re-organize with different structure:**
   ```python
   import json
   
   with open('annotations.json') as f:
       data = json.load(f)
   
   for ann in data['annotations']:
       if ann['label'] == 'cnie_front':
           # Copy to new location
           pass
   ```

---

## Tips for Speed

1. **Use consistent naming when taking photos:**
   - `cardA_front_01.jpg` not `IMG_4729.jpg`

2. **Use Pattern Mode when possible:**
   - One command labels 50+ photos

3. **Annotate in batches:**
   - Take 20 photos → Annotate → Take more

4. **Skip bad photos:**
   - Press `d` or `s` to skip blurry/dark photos

---

## Output Format

Both tools create `annotations.json`:

```json
{
  "annotations": [
    {
      "filename": "cardA_flat.jpg",
      "filepath": "/home/user/.../cardA_flat.jpg",
      "label": "cnie_front",
      "rotation": 0,
      "marked_for_deletion": false,
      "notes": ""
    }
  ],
  "metadata": {
    "total_images": 50,
    "annotated": 48,
    "by_class": {
      "cnie_front": 20,
      "cnie_back": 20,
      "passport": 8
    }
  }
}
```

This JSON is reusable - you can:
- Re-annotate later
- Import into other tools
- Use as ground truth for evaluation
