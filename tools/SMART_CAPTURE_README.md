# Smart Dataset Capture Tools

## Problem Identified

The V10 cleaning revealed **56 images were mislabeled** - "front" folder contained back images and vice versa. This label noise was causing poor training accuracy.

## Solution: Smart Capture System

### 1. Smart Capture (`smart_capture.py`)

An intelligent data collection tool that:
- **Auto-detects card orientation** using face and chip detection
- **Auto-captures** when the card is stable and clearly visible
- **Shows real-time stats** on which class needs more data
- **No clicking required** - just move the card!

#### Features

- **Real-time Detection**: Uses face detection (front indicator) and chip template matching (back indicator)
- **Auto-Capture**: When card is stable for 5 frames with high confidence, automatically saves
- **Visual Feedback**: Shows detection status, stability counter, and dataset balance
- **Cooldown**: 1-second delay between captures to avoid duplicates
- **Keyboard Controls**:
  - `Q` - Quit
  - `R` - Reset stability counter
  - `S` - Show detailed stats

#### How It Works

```
┌─────────────────────────────────────────┐
│  SMART CARD CAPTURE                     │
│  Detected: FRONT (0.85) [STABLE ✓]     │
│  📸 FRONT: 45/300 (need 255 more)      │
│  ✅ BACK: 320/300 (complete!)          │
│  📸 NOCARD: 12/150 (need 138 more)     │
└─────────────────────────────────────────┘

When you hold a card steady:
1. Tool detects orientation (face = front, chip = back)
2. Stability counter increases
3. At 5 stable frames → AUTO-CAPTURE!
4. Counter resets, ready for next card
```

#### Usage

```bash
cd ~/retin-verify/tools
python3 smart_capture.py
```

Images saved to: `~/retin-verify/training_data/v10_manual_capture/`

---

### 2. Dataset Dashboard (`dataset_dashboard.py`)

View detailed statistics and get advice on dataset balance.

#### Features

- **Visual ASCII Charts**: See class distribution at a glance
- **Balance Analysis**: Detects imbalance and recommends actions
- **Split Breakdown**: Shows train/val/test distribution
- **Watch Mode**: Auto-refresh every 5 seconds during collection
- **Multi-Dataset Compare**: Compare multiple datasets side-by-side

#### Usage

```bash
# Basic stats
python3 dataset_dashboard.py ~/retin-verify/training_data/v8_stage2_clean

# Watch mode (auto-refresh)
python3 dataset_dashboard.py ~/retin-verify/training_data/v10_manual_capture --watch

# Compare datasets
python3 dataset_dashboard.py \
    ~/retin-verify/training_data/v8_stage2_clean \
    ~/retin-verify/training_data/v10_manual_capture \
    --compare

# Generate JSON report
python3 dataset_dashboard.py ~/retin-verify/training_data/v10_manual_capture \
    --report dataset_stats.json
```

#### Sample Output

```
======================================================================
DATASET STATISTICS DASHBOARD
======================================================================

Total Images: 582

front      │███████████████████████████████████████             │  232 (39.9%)
back       │████████████████████████████████████████████████████│  290 (49.8%)
no_card    │██████                                               │   60 (10.3%)

----------------------------------------------------------------------
RECOMMENDATIONS:

  ⚠️  IMBALANCED: Need 58 MORE FRONT images (ratio 1.3:1)
  ✅ FRONT: 232 images (sufficient)
  ✅ BACK: 290 images (sufficient)
  📸 NOCARD: Only 60 images, need 140 more

======================================================================
```

---

## Workflow Recommendation

### Step 1: Check Current Stats
```bash
python3 dataset_dashboard.py ~/retin-verify/training_data/v8_stage2_clean
```

See which classes need more data.

### Step 2: Launch Smart Capture
```bash
python3 smart_capture.py
```

Follow the priority indicator:
- Tool shows "⚠️  FRONT needs more images!" → Show front of cards
- Tool shows "⚠️  BACK needs more images!" → Show back of cards
- Just move cards naturally, tool auto-captures!

### Step 3: Monitor Progress
Open another terminal:
```bash
python3 dataset_dashboard.py ~/retin-verify/training_data/v10_manual_capture --watch
```

Watch the bars fill up in real-time!

### Step 4: Balance Check
After collection, verify balance:
```bash
python3 dataset_dashboard.py \
    ~/retin-verify/training_data/v8_stage2_clean \
    ~/retin-verify/training_data/v10_manual_capture \
    --compare
```

---

## Target Dataset Sizes

| Class | Minimum | Recommended | Status |
|-------|---------|-------------|--------|
| Front | 200 | 300 | Monitor |
| Back | 200 | 300 | Monitor |
| No-Card | 100 | 150 | Monitor |

**Balance Goal**: Front:Back ratio should be between 0.8:1 and 1.2:1

---

## Why This Approach Works

1. **No Mislabeling**: Auto-detection ensures correct labels
2. **Efficient**: No clicking, just move cards naturally
3. **Guided**: Real-time advice on what to capture
4. **Quality Control**: Only captures when detection is stable
5. **Balanced**: Visual feedback encourages balanced collection

---

## File Locations

```
~/retin-verify/tools/
├── smart_capture.py       # Main capture tool
├── dataset_dashboard.py   # Stats and analysis
├── dataset_cleaner_*.py   # Cleaning tools (v1, v2, CLI)
└── SMART_CAPTURE_README.md # This file

~/retin-verify/training_data/
├── v8_stage2_clean/       # Original (may have mislabels)
├── v10_manual_capture/    # New clean captures
│   ├── front/
│   ├── back/
│   └── no_card/
└── v11_balanced/          # Final balanced dataset (future)
```

---

## Next Steps

1. **Collect data** using `smart_capture.py` until balanced
2. **Combine** with cleaned original dataset
3. **Retrain** model V11 with clean, balanced data
4. **Achieve** >95% accuracy!
