# Feedback Collection System for V6 CNIE Classifier

## Overview

The feedback collection system enables continuous improvement of the V6 CNIE classifier by collecting user feedback on predictions. This data can be used for retraining to improve model accuracy.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend UI   │────▶│  API Endpoints   │────▶│  Feedback Store │
│  (index.html)   │     │  (FastAPI)       │     │  (Local FS)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ EnhancedFeedback │
                        │   Collector      │
                        └──────────────────┘
```

## Components

### 1. Enhanced Feedback Collector (`feedback_integration.py`)

Core module that manages feedback storage and retrieval.

**Features:**
- Duplicate detection (via image hash)
- Image optimization (resize, compress)
- Category classification (misclassified, correct, low_confidence, no_card)
- Statistics generation
- Retraining dataset preparation

**Storage Structure:**
```
feedback_data/
├── misclassified/
│   ├── cnie_front/      # Images that are actually front but misclassified
│   └── cnie_back/       # Images that are actually back but misclassified
├── correct/
│   ├── cnie_front/      # Correctly classified front images
│   └── cnie_back/       # Correctly classified back images
├── low_confidence/
│   ├── cnie_front/
│   └── cnie_back/
├── no_card/             # Images without CNIE cards
├── retraining_dataset/  # Auto-generated retraining data
│   ├── train/
│   │   ├── cnie_front/
│   │   ├── cnie_back/
│   │   └── no_card/
│   └── val/
│       ├── cnie_front/
│       ├── cnie_back/
│       └── no_card/
└── feedback_annotations.json  # Metadata for all feedback
```

### 2. Enhanced API Server (`api_server_v6_enhanced.py`)

Extended API server with comprehensive feedback endpoints.

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with feedback system status |
| `/info` | GET | Model info with feedback statistics |
| `/predict` | POST | Classify an image |
| `/feedback` | POST | Generic feedback submission |
| `/feedback/correct` | POST | Confirm prediction was correct |
| `/feedback/misclassified` | POST | Report misclassification |
| `/feedback/no_card` | POST | Report no-card detection |
| `/feedback/stats` | GET | Get feedback statistics |
| `/feedback/recent` | GET | Get recent feedback entries |
| `/feedback/prepare-retraining` | POST | Prepare retraining dataset |
| `/feedback/export` | GET | Export feedback summary |
| `/feedback/clean` | DELETE | Clean old feedback |

### 3. Feedback Manager CLI (`scripts/feedback_manager.py`)

Command-line tool for managing feedback data.

**Commands:**

```bash
# Show statistics
python scripts/feedback_manager.py stats

# Prepare retraining dataset
python scripts/feedback_manager.py prepare

# Export feedback summary
python scripts/feedback_manager.py export

# Clean old feedback (default: 90 days)
python scripts/feedback_manager.py clean [days]

# Visualize patterns
python scripts/feedback_manager.py visualize

# Interactive review
python scripts/feedback_manager.py review
```

### 4. Frontend Feedback UI

Enhanced feedback interface with clear options:

- ✅ **Correct!** - Confirm the prediction was correct
- 🔀 **Actually Front** - Predicted Back, but it's Front
- 🔀 **Actually Back** - Predicted Front, but it's Back
- 🚫 **Not CNIE Card** - Credit card, other doc, or empty

## Usage

### Starting the Enhanced API Server

```bash
cd inference/apps/classification/backend
python api_server_v6_enhanced.py
```

The server will start on `http://localhost:8000` with all feedback endpoints active.

### Submitting Feedback via API

**Example: Report Misclassification**

```bash
curl -X POST http://localhost:8000/feedback/misclassified \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "...base64encodedimage...",
    "predicted_class": "cnie_front",
    "predicted_confidence": 0.75,
    "correct_class": "cnie_back",
    "notes": "User flagged as incorrect"
  }'
```

**Example: Confirm Correct Prediction**

```bash
curl -X POST http://localhost:8000/feedback/correct \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "...base64encodedimage...",
    "predicted_class": "cnie_front",
    "predicted_confidence": 0.92
  }'
```

### Preparing Retraining Dataset

When enough feedback is collected (10+ misclassified images):

```bash
# Via CLI
python scripts/feedback_manager.py prepare

# Via API
curl -X POST http://localhost:8000/feedback/prepare-retraining
```

This creates a ready-to-use dataset at:
`~/retin-verify/feedback_data/retraining_dataset/`

## Statistics

The system tracks:

- **Total feedback count**
- **By category**: misclassified, correct, low_confidence, no_card
- **By predicted class**: cnie_front, cnie_back
- **By correct class**: For misclassified images
- **Confidence distribution**: High (≥80%), Medium (50-80%), Low (<50%)
- **Misclassification patterns**: Most common error types

## Retraining Threshold

Retraining is recommended when:
- **10+ misclassified images** collected

This threshold balances:
- Having enough hard negatives to improve the model
- Not waiting too long to incorporate feedback
- Avoiding retraining with insufficient data

## Best Practices

1. **Encourage feedback**: Show the feedback UI after every prediction
2. **Be specific**: Use the specific feedback options ("Actually Front" vs "Actually Back")
3. **Monitor stats**: Regularly check `/feedback/stats` to track collection progress
4. **Export before retraining**: Always export feedback summary before starting retraining
5. **Clean periodically**: Remove feedback older than 90 days to manage storage

## Migration from Old System

The old feedback system stored images without metadata. To migrate:

1. Old images are preserved in `feedback_data/cnie_front/` and `feedback_data/cnie_back/`
2. New feedback uses the structured system with `feedback_annotations.json`
3. Old images can be manually reviewed and re-submitted through the new system if needed

## Storage Management

**Default Location**: `~/retin-verify/feedback_data/`

**Storage Requirements**:
- Average image size: ~100KB (optimized)
- 1000 feedback images: ~100MB

**Cleanup**:
```bash
# Remove feedback older than 90 days
python scripts/feedback_manager.py clean 90
```

## Integration with Retraining Pipeline

1. Collect feedback via UI/API
2. Monitor statistics until threshold reached
3. Run `prepare-retraining` to create dataset
4. Use generated dataset for Colab retraining
5. Download new model and deploy
6. Continue collecting feedback for next iteration

## Troubleshooting

**Feedback not saving:**
- Check API server logs
- Verify write permissions to `feedback_data/` directory
- Check disk space

**Duplicate detection not working:**
- Ensure images are being properly hashed
- Check that `feedback_annotations.json` is writable

**Statistics not updating:**
- Statistics are cached for 60 seconds
- Use `use_cache=false` to force refresh

## Future Enhancements

- [ ] Automatic retraining trigger when threshold reached
- [ ] Feedback quality scoring
- [ ] Active learning: prioritize uncertain predictions
- [ ] Integration with cloud storage for backup
- [ ] Web dashboard for feedback analytics
