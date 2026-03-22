# CNIE Dataset Tools - Web Application

A unified web interface with sidebar navigation for all dataset management tools.

## 🏠 Features

### 1. Dashboard (Home)
- Real-time statistics overview
- Quick action buttons
- Collection progress bars
- Recent captures gallery
- Imbalance alerts

### 2. Smart Capture
- Interface for auto-detection capture
- Real-time detection preview
- Stability counter display
- Capture counts by class

### 3. Manual Review
- Browse folder by folder
- View images one by one
- Move images between classes
- Delete bad images
- Keyboard shortcuts: Space (correct), B (move to back), F (move to front), Arrows (navigate)

### 4. Dataset Cleaner
- Automatic label verification
- Face and chip detection
- Flag suspicious images
- View mislabel suggestions

### 5. Statistics
- Detailed dataset analysis
- Balance visualization
- Split distribution (train/val/test)
- Recommendations

### 6. Train Model
- Training configuration interface
- Model selection
- Hyperparameter settings
- Training history

### 7. Evaluate
- Model evaluation interface
- Upload test images
- Camera-based testing

---

## 🚀 Quick Start

### Run the Web App

```bash
cd ~/retin-verify/tools
python3 web_app.py
```

Open browser to: **http://localhost:5000**

---

## 📁 Navigation Structure

```
┌─────────────────────────────────────────────────────────────┐
│  CNIE Tools              │  Dashboard                        │
│  Dataset Management Suite │                                   │
├──────────────────────────┤  ┌─────────────────────────────┐  │
│  Overview                │  │ Stats Overview              │  │
│  🏠 Dashboard            │  │ [cards]                     │  │
│                          │  └─────────────────────────────┘  │
│  Data Collection         │  ┌─────────────────────────────┐  │
│  📸 Smart Capture        │  │ Quick Actions               │  │
│  ✋ Manual Review        │  │ [buttons]                   │  │
│                          │  └─────────────────────────────┘  │
│  Quality Control         │  ┌─────────────────────────────┐  │
│  🧹 Dataset Cleaner      │  │ Progress Bars               │  │
│  📈 Statistics           │  │ [||||||||||] 45/300         │  │
│                          │  └─────────────────────────────┘  │
│  Training                │                                   │
│  🤖 Train Model          │                                   │
│  🧪 Evaluate             │                                   │
│                          │                                   │
│  v2.0.0                  │                                   │
└──────────────────────────┴───────────────────────────────────┘
```

---

## 🎯 Usage Workflow

### 1. Check Dataset Status
- Go to **Dashboard** or **Statistics**
- See current counts and balance
- Check recommendations

### 2. Collect More Data (if needed)
- Go to **Smart Capture**
- Start camera
- Show cards to camera
- Tool auto-captures when stable
- Watch progress bars fill

### 3. Review and Clean
- Go to **Manual Review**
- Select dataset, split, and class
- Review each image
- Move mislabeled images
- Delete bad images

### 4. Verify Balance
- Check **Statistics** again
- Ensure Front:Back ratio is ~1:1
- Capture more if needed

### 5. Train
- Go to **Train Model**
- Configure settings
- Start training (requires backend)

---

## 🛠️ Technical Details

### Built With
- **Flask** - Python web framework
- **HTML/CSS/JavaScript** - Frontend
- **Jinja2 Templates** - Server-side rendering

### File Structure
```
web_app.py          # Main Flask application
├── Dashboard       # Home page with stats
├── Smart Capture   # Auto-detection interface  
├── Manual Review   # Image-by-image review
├── Dataset Cleaner # Auto verification
├── Statistics      # Detailed analysis
├── Train Model     # Training interface
└── Evaluate        # Model evaluation
```

### API Endpoints
```
GET  /api/stats              # Quick stats
GET  /api/stats/detailed     # Detailed stats
GET  /api/recent             # Recent captures
GET  /api/manual/load        # Load folder for review
POST /api/manual/move        # Move image
POST /api/manual/delete      # Delete image
GET  /api/cleaner/scan       # Scan for mislabels
GET  /api/capture/frame      # Get capture status
```

---

## 📝 Keyboard Shortcuts (Manual Review)

| Key | Action |
|-----|--------|
| `Space` | Mark correct (next) |
| `→` | Next image |
| `←` | Previous image |
| `B` | Move to BACK |
| `F` | Move to FRONT |
| `Delete` | Delete image |

---

## 🔧 Customization

### Change Default Dataset
Edit `web_app.py`:
```python
DEFAULT_DATASET = DATASET_DIR / "your_dataset"
```

### Add New Tool/Page
1. Create page template (HTML)
2. Add route in Flask
3. Add nav item in sidebar

---

## 🐛 Troubleshooting

### Port already in use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or run on different port
python3 web_app.py --port 5001
```

### Cannot access from other machines
```python
# In web_app.py, change:
app.run(host='0.0.0.0', port=5000)  # Allows external access
```

### Flask not installed
```bash
pip3 install flask
```

---

## 🎨 Screenshot

The interface features:
- Dark sidebar with section grouping
- Clean card-based layout
- Progress bars for visualization
- Responsive grid for images
- Color-coded status indicators

---

## 📊 Next Steps

1. **Run the web app**: `python3 web_app.py`
2. **Navigate to Dashboard**: See current stats
3. **Use Smart Capture**: Collect balanced data
4. **Review with Manual tool**: Fix any mislabels
5. **Check Statistics**: Verify balance
6. **Train model**: Achieve >95% accuracy!
