# Adding "No Card" Collection to UI

## Option 1: Quick Add to Existing UI

Add a "No Card" button to the feedback section:

```html
<!-- In index.html, feedback section -->
<div class="feedback-section" id="feedbackSection" style="display: none;">
    <h4>⚠️ Was this prediction wrong?</h4>
    
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0;">
        <button class="btn btn-warning" onclick="flagForRetraining('wrong')">
            🚩 Wrong (Front/Back)
        </button>
        <button class="btn btn-secondary" onclick="flagForRetraining('no_card')">
            📷 No Card / Other Doc
        </button>
    </div>
    
    <!-- ... rest of feedback section -->
</div>
```

## Option 2: Collection Mode Toggle

Add a mode selector at the top:

```html
<div class="capture-mode">
    <label>Collection Mode:</label>
    <select id="collectionMode" onchange="setCollectionMode(this.value)">
        <option value="normal">Normal Classification</option>
        <option value="front">Collect FRONT samples</option>
        <option value="back">Collect BACK samples</option>
        <option value="no_card">Collect NO CARD samples</option>
    </select>
</div>
```

When in "no_card" mode, all captures go to `feedback_data_3class/no_card/`.

## Backend Modification

Add new endpoint for 3-class feedback:

```python
@app.post("/feedback_3class")
async def submit_feedback_3class(request: Request):
    """Submit feedback for 3-class retraining."""
    data = await request.json()
    
    image_data = base64.b64decode(data['image_base64'])
    label_class = data['label_class']  # 'cnie_front', 'cnie_back', or 'no_card'
    
    # Save to 3-class feedback directory
    save_path = Path("feedback_data_3class") / label_class
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    img_path = save_path / f"{timestamp}_{hash}.jpg"
    
    with open(img_path, 'wb') as f:
        f.write(image_data)
    
    return {"success": True, "saved_to": str(img_path)}
```

## Simplest Approach: Manual Collection

For now, the easiest approach is:

1. **Use the existing UI** to capture images
2. **Download images** when no card is present
3. **Move them manually** to the no_card folder

### Collection Instructions

```bash
# Create no_card folder
mkdir -p ~/retin-verify/apps/classification/feedback_data_3class/no_card

# After capturing images with no card, 
# move them from feedback_data/misclassified/ to feedback_data_3class/no_card/
```

Or use this simple script:

```bash
#!/bin/bash
# save_no_card.sh
# Usage: ./save_no_card.sh image.jpg

dst="$HOME/retin-verify/apps/classification/feedback_data_3class/no_card/"
mkdir -p "$dst"
cp "$1" "$dst/$(date +%Y%m%d_%H%M%S)_$(basename $1)"
echo "Saved to $dst"
```

## Target Collection

| Class | Current | Target | Needed |
|-------|---------|--------|--------|
| cnie_front | 28 | 30 | +2 |
| cnie_back | 18 | 30 | +12 |
| no_card | 0 | 40 | +40 |
| **TOTAL** | **46** | **100** | **+54** |

## Collection Strategy

### Phase 1: No Card Samples (40 needed)

Capture:
- [ ] 10x Empty room/background
- [ ] 10x Credit cards
- [ ] 10x Driver's license or other ID
- [ ] 5x Random objects (phone, keys, etc.)
- [ ] 5x Hand without card / blurry

### Phase 2: Balance Front/Back

If needed, capture more:
- [ ] Additional front samples until 30 total
- [ ] Additional back samples until 30 total

## Quick Collection Guide

For fastest collection, use this workflow:

1. **Open terminal**
2. **Run collection script:**
   ```bash
   cd ~/retin-verify/apps/classification
   
   # Create alias for quick save
   alias save_nc='cp /tmp/last_capture.jpg feedback_data_3class/no_card/$(date +%Y%m%d_%H%M%S).jpg && echo "Saved"'
   ```

3. **In browser:**
   - Capture image
   - Click "📤 Upload" button to download
   - In terminal: `save_nc`

Or simply download and move files manually.
