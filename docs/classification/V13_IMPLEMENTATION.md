# V13 Implementation Guide: Preprocessing Enhancement

## Overview
V13 adds real-time preprocessing capabilities to enhance hidden security features (micro-text, guilloche patterns, ghost images) during annotation and inference. This is a **backward-compatible enhancement** - all existing functionality remains unchanged.

---

## Implementation Status

### ✅ Phase 1: Visual Prototype (COMPLETED)
**Location:** Manual Review page (`/manual`)

**Features Implemented (Phase 1.0):**
- Dual-engine preprocessing (Canvas + OpenCV.js)
- Side-by-side / overlay / split-slider comparison views
- Basic enhancement controls:
  - CLAHE Contrast (0-8)
  - Unsharp Mask Sharpening (1.0-5.0)
  - Gamma Correction (0.5-3.0)
  - Sobel Edge Detection (0-3)
  - High-Pass Filtering (0-2)
  - Local Contrast Enhancement (0-5)
  - Histogram Equalization
  - Color Inversion
- Quick presets for specific features:
  - 🔍 Micro-Text
  - 🎨 Guilloche Patterns
  - 👤 Ghost Image
  - 💡 UV/Security Features
- Save preprocessed images to dual dataset

**Advanced Features Planned (Phase 1.1):**
- 🌈 **Color Space Separation** - HSV/LAB hologram isolation
- 🔬 **Morphological Operations** - Dilation/erosion for micro-text
- 📊 **FFT Band-Pass Filtering** - Guilloché pattern enhancement
- 💡 **UV/IR Simulation** - Channel boosting for security features
- 🎥 **Multi-Exposure Fusion** - Dynamic range extension (video)

**Technical Implementation:**
- Canvas API with `willReadFrequently` optimization
- Tile-based CLAHE in JavaScript
- Laplacian/Sobel edge detection
- Local contrast normalization
- Histogram equalization
- **Next:** HSV color space conversion, FFT, morphological kernels

---

### 🔄 Phase 2: Dual Dataset Strategy (NEXT)

**Goal:** Automatically save both original and preprocessed versions during capture

**Implementation Plan:**

#### 2.1 Smart Capture Enhancement
```
Current:  v10_manual_capture/train/front/img.jpg
New:      v10_manual_capture_preprocessed/train/front/img.jpg
```

**Changes to `/capture` page:**
1. Add toggle: "Auto-preprocess captures" (default: ON after Phase 1 validation)
2. Apply preprocessing pipeline before saving
3. Save to `_preprocessed` folder with same filename
4. Maintain identical bounding boxes (geometry unchanged)

#### 2.2 Training Data Preparation
```python
# Training script enhancement
def load_dual_dataset():
    original_images = glob('v10_manual_capture/**/*/*.jpg')
    preprocessed_images = glob('v10_manual_capture_preprocessed/**/*/*.jpg')
    
    # Combine both datasets
    all_images = original_images + preprocessed_images
    # Same labels, same annotations
    return all_images
```

**Backward Compatibility:**
- Original `v10_manual_capture` unchanged
- New `_preprocessed` folder is additive
- Existing training scripts work without modification
- Optional dual-loader for enhanced training

---

### ⏳ Phase 3: Live Inference Enhancement (FUTURE)

**Goal:** Real-time preprocessing during evaluation with user guidance

**Implementation:**

#### 3.1 Preprocessing Toggle in Evaluate Page
```
┌─────────────────────────────────────────┐
│  [Camera Feed]                          │
│                                         │
│  🔘 Enhance Security Features (V13)    │
│     Contrast: [━━━●━━━━]               │
│     Sharpen:  [━━●━━━━━]               │
│                                         │
│  Detection: Front (0.94)               │
└─────────────────────────────────────────┘
```

#### 3.2 Performance Optimization
- Use lighter Canvas preprocessing (skip heavy algorithms)
- WebGL acceleration for real-time processing
- Frame skipping: process every Nth frame
- Cache preprocessed frames

#### 3.3 User Guidance Integration
```javascript
if (confidence < 0.7) {
    showTip("Try enabling 'Micro-Text Mode' to enhance card details");
}
```

---

## Backward Compatibility Matrix

| Component | Original Behavior | With V13 Enabled | Migration |
|-----------|-------------------|------------------|-----------|
| **Dataset Structure** | Flat folder hierarchy | Original + `_preprocessed` subfolder | Automatic, additive |
| **Training Scripts** | Load original images only | Optional dual-loader available | No changes required |
| **Saved Models** | Work with original images | Work with original images | None |
| **API Endpoints** | Existing endpoints | New `/api/preprocess/*` endpoints | Old endpoints unchanged |
| **Manual Review** | Original image only | Toggle for preprocess preview | Default: original |
| **Smart Capture** | Save original only | Option to save both | Toggle control |
| **Evaluation** | Direct inference | Optional preprocessing | Default: direct |

---

## Implementation Steps

### Step 1: Validate Phase 1 (Current)
**Verify hidden features are visible after preprocessing:**
- [ ] Micro-text on card edges
- [ ] Guilloche patterns
- [ ] Ghost image (secondary portrait)
- [ ] Holographic elements
- [ ] UV-reactive areas

**Success Criteria:** Features must be clearly eye-visible without squinting

### Step 2: Implement Dual Dataset (Week 1)
1. **Update Smart Capture** (`/capture` page)
   - Add preprocessing toggle
   - Implement auto-save to `_preprocessed` folder
   - Test with 50+ captures

2. **Update Training Script**
   - Add dual dataset loader option
   - Test training with combined data
   - Compare accuracy vs original-only

### Step 3: Live Inference (Week 2)
1. **Update Evaluate Page**
   - Add real-time preprocessing toggle
   - Optimize for 15+ FPS
   - Add feature-specific presets

2. **Performance Testing**
   - Measure FPS impact
   - Test on low-end devices
   - Implement fallback to lighter processing

---

## Advanced Preprocessing Techniques (V13.1)

To make hidden security features learnable by YOLO, specialized preprocessing goes beyond basic contrast enhancement. These advanced techniques target specific feature types:

### 1. Color Space Separation (Hologram Detection)
Holograms have rainbow iridescence that shifts with angle. Isolate them via HSV/LAB color spaces:

```javascript
// HSV Channel Extraction for Holograms
function extractHologramHSV(data, width, height, hueRange, satThreshold) {
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i] / 255, g = data[i+1] / 255, b = data[i+2] / 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        const delta = max - min;
        
        // Calculate HSV
        let h = 0, s = max > 0 ? delta / max : 0, v = max;
        if (delta > 0) {
            if (max === r) h = ((g - b) / delta + 6) % 6;
            else if (max === g) h = (b - r) / delta + 2;
            else h = (r - g) / delta + 4;
            h *= 60;
        }
        
        // Filter by hue range and saturation
        const inHueRange = h >= hueRange[0] && h <= hueRange[1];
        const highSaturation = s >= satThreshold;
        
        if (!inHueRange || !highSaturation) {
            data[i] = data[i+1] = data[i+2] = 0; // Mask out non-hologram pixels
        }
    }
}
```

**Use Case:** Isolating holographic strips, rainbow patterns on cards.

---

### 2. Edge Detection & Morphology (Micro-Text Enhancement)
Tiny text requires aggressive edge detection and morphological operations:

```javascript
// Canny Edge + Morphological Dilation
function enhanceMicroText(data, width, height, lowThreshold, highThreshold) {
    // Convert to grayscale
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
        gray[i/4] = Math.round(0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2]);
    }
    
    // Apply Canny edge detection
    const edges = cannyEdgeDetector(gray, width, height, lowThreshold, highThreshold);
    
    // Morphological dilation to connect broken text
    const dilated = dilate(edges, width, height, 2);
    
    // Blend with original
    for (let i = 0; i < data.length; i += 4) {
        const edgeVal = dilated[i/4];
        const blend = 0.3; // 30% edge overlay
        data[i] = data[i] * (1-blend) + edgeVal * blend;
        data[i+1] = data[i+1] * (1-blend) + edgeVal * blend;
        data[i+2] = data[i+2] * (1-blend) + edgeVal * blend;
    }
}

// Dilation with 2x2 kernel
function dilate(gray, width, height, iterations) {
    let result = new Uint8Array(gray);
    for (let iter = 0; iter < iterations; iter++) {
        const temp = new Uint8Array(result);
        for (let y = 1; y < height-1; y++) {
            for (let x = 1; x < width-1; x++) {
                const idx = y * width + x;
                // Max of 3x3 neighborhood
                let max = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        max = Math.max(max, temp[(y+dy)*width + (x+dx)]);
                    }
                }
                result[idx] = max;
            }
        }
    }
    return result;
}
```

**Use Case:** Making micro-text on card edges readable by YOLO.

---

### 3. Frequency Domain Filtering (Guilloché Patterns)
Periodic security patterns are enhanced via FFT band-pass filtering:

```javascript
// FFT-based band-pass filter (simplified DFT for small regions)
function fftBandPass(gray, width, height, lowFreq, highFreq) {
    // Apply window function to reduce edge artifacts
    const windowed = applyHannWindow(gray, width, height);
    
    // 2D FFT (using separate 1D FFTs for rows then columns)
    const fftRows = fft2DRows(windowed, width, height);
    const fftResult = fft2DCols(fftRows, width, height);
    
    // Create circular band-pass mask
    const centerX = width / 2, centerY = height / 2;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const dx = x - centerX, dy = y - centerY;
            const dist = Math.sqrt(dx*dx + dy*dy);
            const inBand = dist >= lowFreq && dist <= highFreq;
            
            if (!inBand) {
                const idx = (y * width + x) * 2; // Complex number
                fftResult[idx] = fftResult[idx+1] = 0;
            }
        }
    }
    
    // Inverse FFT
    const ifftResult = ifft2D(fftResult, width, height);
    return magnitude(ifftResult);
}
```

**Use Case:** Revealing hidden guilloché patterns, moiré effects.

---

### 4. Multi-Exposure Fusion (Dynamic Range Extension)
Simulate multi-exposure capture for cards with extreme lighting variations:

```javascript
// Not implemented in prototype - requires video capture
// Concept: Capture 3 frames with different exposures, merge
function multiExposureFusion(frames) {
    // Weight each pixel by its Laplacian variance (sharpness)
    const weights = frames.map(f => laplacianVariance(f));
    
    // Weighted average of all exposures
    const result = new Float32Array(frames[0].length);
    for (let i = 0; i < result.length; i++) {
        let sumWeight = 0, sumValue = 0;
        for (let f = 0; f < frames.length; f++) {
            sumWeight += weights[f][i];
            sumValue += frames[f][i] * weights[f][i];
        }
        result[i] = sumValue / sumWeight;
    }
    return result;
}
```

**Use Case:** Cards photographed in harsh lighting (overexposed hologram + underexposed text).

---

### 5. UV/IR Channel Simulation
Simulate UV fluorescence effects by boosting specific channels:

```javascript
// UV Simulation: Boost blue channel + add glow
function simulateUV(data, width, height, blueBoost, glowRadius) {
    // Boost blue channel
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.min(255, data[i] * blueBoost); // Blue is at i (BGR in Canvas)
    }
    
    // Add glow effect (simple Gaussian blur of blue, add back)
    const blueChannel = extractChannel(data, 0);
    const glow = gaussianBlur(blueChannel, width, height, glowRadius);
    
    // Blend glow back
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.min(255, data[i] + glow[i/4] * 0.5);
    }
}

// IR Simulation: Use only red channel (near-IR response)
function simulateIR(data) {
    for (let i = 0; i < data.length; i += 4) {
        const red = data[i+2]; // Red is at i+2 in RGBA
        data[i] = data[i+1] = data[i+2] = red; // Grayscale from red
    }
}
```

**Use Case:** Making UV-reactive security features visible without UV lamp.

---

### 6. Combined Advanced Pipeline

```javascript
function advancedPreprocessV13(imageData, width, height, featureType) {
    const data = new Uint8ClampedArray(imageData.data);
    
    switch(featureType) {
        case 'hologram':
            // Color space separation
            extractHologramHSV(data, width, height, [30, 90], 0.4); // Green-gold range
            applyCLAHE(data, width, height, 3.0);
            break;
            
        case 'microtext':
            // Edge detection + morphology
            enhanceMicroText(data, width, height, 30, 100);
            applySharpen(data, width, height, 2.5);
            break;
            
        case 'guilloche':
            // Frequency domain filtering
            const gray = rgbToGray(data);
            const filtered = fftBandPass(gray, width, height, 5, 50);
            grayToRgb(data, filtered);
            applyLocalContrast(data, width, height, 3.0);
            break;
            
        case 'uv_features':
            // UV simulation
            simulateUV(data, width, height, 2.5, 5);
            applyHistogramEqualization(data, width, height);
            break;
            
        case 'ghost':
            // Ghost image enhancement
            applyGamma(data, 1.8); // Brighten
            applyLocalContrast(data, width, height, 4.0);
            applyHistogramEqualization(data, width, height);
            break;
    }
    
    return new ImageData(data, width, height);
}
```

---

## Technical Specifications

### Preprocessing Pipeline (JavaScript)
```javascript
const PIPELINE = {
    // Phase 1: Color/tonal adjustments
    invert:      false,     // Negative view
    equalize:    false,     // Histogram equalization
    gamma:       1.2,       // Brightness adjustment
    
    // Phase 2: Contrast enhancement
    contrast:    2.0,       // CLAHE clip limit
    local:       0,         // Local contrast strength
    
    // Phase 3: Detail enhancement
    highpass:    0,         // High-pass filter for fine details
    edge:        0,         // Sobel edge detection
    sharpen:     1.5,       // Unsharp mask
    
    // Phase 4: Noise reduction
    denoise:     0,         // Box blur strength
    
    // Phase 5: Advanced (V13.1)
    hologram:    false,     // HSV color space separation
    morphology:  0,         // Dilation iterations
    fftBands:    null,      // [low, high] frequency band
    uvSim:       false,     // UV channel boost
    irSim:       false      // IR grayscale
};
```

### File Naming Convention
```
Original:      feedback_front_20240321_121200.jpg
Preprocessed:  feedback_front_20240321_121200.jpg (in _preprocessed folder)
Metadata:      feedback_front_20240321_121200_meta.json
```

### Metadata Format
```json
{
    "original_path": ".../v10_manual_capture/train/front/img.jpg",
    "preprocessed_path": ".../v10_manual_capture_preprocessed/train/front/img.jpg",
    "params": {
        "contrast": 6.0,
        "sharpen": 4.0,
        "edge": 2.0,
        "preset": "microtext"
    },
    "engine": "canvas",
    "timestamp": "2024-03-21T12:12:00Z"
}
```

---

## API Additions

### New Endpoints (Additive Only)

```http
POST /api/preprocess/save
{
    "original_path": string,
    "image_data": base64,
    "params": object,
    "engine": "canvas" | "opencv"
}

GET /api/preprocess/status
{
    "opencv_loaded": boolean,
    "supported_features": [...]
}
```

### Existing Endpoints (Unchanged)
All existing `/api/*` endpoints continue to work exactly as before.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Preprocessing artifacts | Keep original images; model sees both |
| Performance degradation | Canvas engine is <50ms per frame; optional toggle |
| Dataset bloat | Preprocessed images optional; can be regenerated |
| Training instability | Start with 10% preprocessed, gradually increase |
| User confusion | Clear UI labels; default to familiar behavior |

---

## Success Metrics

### Phase 1.0 (Basic Preprocessing) ✅
- [x] UI implemented in Manual Review
- [x] 5+ preprocessing algorithms working
- [x] Side-by-side comparison functional
- [x] Save/load preprocessed images

### Phase 1.1 (Advanced Features) 🔄
- [ ] HSV color space separation for holograms
- [ ] Morphological operations for micro-text
- [ ] FFT band-pass for guilloché patterns
- [ ] UV/IR channel simulation
- [ ] **Hidden features clearly eye-visible**

### Phase 2 (Dual Dataset)
- [ ] 500+ dual images captured
- [ ] Training accuracy ≥ original baseline
- [ ] Model inference speed maintained

### Phase 3 (Live Inference)
- [ ] Real-time preprocessing at ≥15 FPS
- [ ] False positive rate reduced by 20%
- [ ] User satisfaction score ≥4/5

---

## Implementation Priority

### High Priority (This Week)
1. **HSV Color Space Separation** - Critical for hologram detection
2. **Morphological Dilation** - Essential for micro-text readability

### Medium Priority (Next Week)
3. **FFT Band-Pass Filter** - For guilloché patterns
4. **UV Simulation** - For security feature visibility

### Low Priority (Future)
5. **Multi-Exposure Fusion** - Requires video capture infrastructure
6. **IR Simulation** - Lower impact than UV

---

## Next Actions

### Immediate (Today)
1. **Implement HSV Color Space Separation** 
   - Add HSV extraction for hologram detection
   - Add hue/saturation range sliders
   - Test on cards with holographic strips

2. **Implement Morphological Operations**
   - Add dilation kernel for micro-text
   - Add Canny edge + morphology preset
   - Test on card edge text

### This Week
3. **Implement FFT Band-Pass Filtering**
   - Add frequency domain filtering
   - Create guilloché-specific preset
   - Test on background patterns

4. **Validate Phase 1.1**
   - Verify hidden features are eye-visible
   - Document which techniques work best

### Next Week
5. **Begin Phase 2 (Dual Dataset)**
   - Add toggle to Smart Capture
   - Implement `_preprocessed` auto-save

6. **Phase 3 Planning**
   - Benchmark OpenCV.js vs Canvas performance
   - Design Evaluate page integration

---

## References

- **Strategy Document:** `V13_TRAIN_STRATEGY.md`
- **Implementation:** `tools/web_app.py` (Manual Review section)
- **Test Images:** `v10_manual_capture/train/front/`
- **Server:** Running at `http://localhost:5000`
