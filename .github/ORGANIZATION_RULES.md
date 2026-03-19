# RetinVerify Project Organization Rules

> **Version:** 1.0  
> **Last Updated:** 2026-03-19  
> **Applies to:** All team members and automated tools

---

## 🎯 Philosophy

**A clean project is a maintainable project.**

Every file has a home. If it doesn't belong, it goes to archive or gets deleted.

---

## 📁 Golden Rules

### Rule 1: One Purpose Per Directory

| Directory | Purpose | What's Allowed |
|-----------|---------|----------------|
| `/apps/` | Production runtime | Only files needed to run the app |
| `/training/` | Model training | Training scripts, configs, notebooks |
| `/inference/` | Inference deployment | Standalone inference apps |
| `/data/` | Data storage | Datasets, annotations, feedback |
| `/models/` | Model artifacts | Trained models, checkpoints |
| `/docs/` | Documentation | Markdown files, diagrams |
| `/src/` | Reusable code | Modules, utilities |
| `/tests/` | Testing | Test scripts, fixtures |

### Rule 2: No Duplication

**One source of truth for everything.**

- Models only in `/models/`
- Data only in `/data/`
- Training scripts only in `/training/`

Use symlinks for backward compatibility, not copies.

### Rule 3: Archive, Don't Delete

When something is no longer needed:
1. Move to appropriate `archive/` folder
2. Add date prefix: `archive/20260319_old_file.py`
3. Document why in commit message

### Rule 4: Name Things Clearly

```python
# Good
train_cnie_3class.py
cnie_classifier_v3_20260319.pth
DATA_COLLECTION_GUIDE.md

# Bad
train.py
model.pth
README.md (too generic)
```

### Rule 5: Document Your Changes

Every significant change needs documentation:
- Code comments for complex logic
- Docstrings for functions
- Markdown for architecture decisions
- Changelog entries for releases

---

## 🗂️ Module Organization

### Classification Module

```
classification/
├── apps/                  # Runtime (frontend + backend)
├── inference/apps/        # Standalone inference
├── training/              # Training scripts & configs
├── models/                # Trained models
├── data/processed/        # Datasets
├── data/feedback/         # User feedback
└── docs/                  # Module documentation
```

### Detection Module

Same pattern as classification.

### Extraction Module

Same pattern as classification.

---

## 📝 File Type Guidelines

### Python Files (.py)

| Location | Purpose |
|----------|---------|
| `/src/` | Reusable modules |
| `/training/*/scripts/` | Training scripts |
| `/inference/apps/` | Inference applications |
| `/apps/` | Production runtime |
| `/tests/` | Test files |

**Naming:**
- `train_*.py` - Training scripts
- `inference_*.py` - Inference scripts
- `test_*.py` - Test files
- `utils_*.py` - Utilities

### Data Files

| Extension | Location |
|-----------|----------|
| `.pth`, `.pt`, `.onnx` | `/models/{task}/` |
| `.jpg`, `.png` | `/data/{type}/{task}/` |
| `.json` (annotations) | `/data/annotations/` |
| `.yaml`, `.json` (configs) | `/configs/{task}/` |

### Documentation

| Type | Location | Naming |
|------|----------|--------|
| Architecture | `/docs/{task}/` | `ARCHITECTURE_*.md` |
| Guides | `/docs/guides/` | `GUIDE_*.md` |
| API | `/docs/api/` | `API_*.md` |
| Workflows | `/docs/guides/` | `WORKFLOW_*.md` |

---

## 🔄 Workflow Rules

### Automatic Session Saving

**MANDATORY: Save session state at the end of EVERY session.**

```bash
# Quick save (minimal)
python3 .kimi/session_manager.py save "Task Name" "status" "progress" "next_steps"

# Examples:
python3 .kimi/session_manager.py save \
    "v3 Training Deployment" \
    "in_progress" \
    "Deployed training to Colab, monitoring epoch 15/50" \
    "Download model when training completes, test accuracy"

python3 .kimi/session_manager.py save \
    "Data Collection" \
    "complete" \
    "Collected 50 new front images, 45 back images" \
    "Augment dataset and retrain"
```

**What gets saved:**
- Task name and status
- Progress description
- Next steps
- Timestamp
- Git status snapshot

**Files created/updated:**
- `.kimi/session_state.json` - Machine-readable state
- `CURRENT_STATUS.txt` - Human-readable status
- `.kimi/session_history.jsonl` - Session history log

### Starting a New Task

1. Check if relevant directory exists
2. Create if needed following structure
3. Add README.md explaining purpose
4. Update this doc if new pattern emerges

### Adding a New Model

1. Train model in `/training/`
2. Evaluate thoroughly
3. Move to `/models/{task}/`
4. Name properly: `{task}_v{version}_{date}.pth`
5. Archive old version if needed
6. Update documentation

### Adding New Data

1. Raw data → `/data/raw/`
2. Processed data → `/data/processed/{task}/`
3. Annotations → `/data/annotations/`
4. Update dataset documentation

### Creating Documentation

1. Choose correct subdirectory in `/docs/`
2. Follow naming convention
3. Link from main README if important
4. Update table of contents

---

## 🧹 Maintenance Rules

### Weekly Cleanup

- [ ] Remove `__pycache__` directories
- [ ] Clear temporary files
- [ ] Check for uncommitted changes
- [ ] Archive old experimental files

### Monthly Review

- [ ] Review `/models/` - archive old versions
- [ ] Review `/data/` - remove duplicates
- [ ] Review `/docs/` - archive outdated docs
- [ ] Check symlink health
- [ ] Update this document if needed

### Before Commits

```bash
# Run cleanup
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name ".DS_Store" -delete 2>/dev/null

# Verify structure
tree -L 2  # or ls -R
```

### Ending a Session (MANDATORY CHECKLIST)

```bash
# 1. SAVE SESSION STATE (REQUIRED)
python3 .kimi/session_manager.py save \
    "Task Name" \
    "in_progress|complete|blocked" \
    "What was accomplished" \
    "What to do next"

# 2. Update status file
cat CURRENT_STATUS.txt

# 3. Clean temporary files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 4. Verify no misplaced files
find apps/ -name "train*.py" 2>/dev/null  # Should be empty
find apps/ -name "*.pth" 2>/dev/null      # Should be empty

# 5. Commit changes
git add [files]
git commit -m "[descriptive message]"
git push origin main
```

---

## 🚫 Anti-Patterns (Don't Do This)

### ❌ Wrong: Mixed Concerns
```
apps/classification/
├── train.py          # Training in apps/
├── dataset/          # Data in apps/
├── model.pth         # Model in apps/
└── api_server.py     # Runtime ✓
```

### ✅ Right: Separated Concerns
```
apps/classification/
├── backend/          # Runtime only
├── frontend/
└── start_server.sh

training/classification/
└── scripts/
    └── train.py      # Training here

data/processed/
└── classification/
    └── dataset/      # Data here

models/classification/
└── model.pth         # Model here
```

### ❌ Wrong: Vague Naming
```
README.md
train.py
model.pth
data/
```

### ✅ Right: Descriptive Naming
```
README_CLASSIFICATION.md
train_cnie_3class_v3.py
cnie_classifier_3class_v3_20260319.pth
dataset_3class_20260319/
```

---

## 🤖 Automation

### Pre-commit Hooks (Recommended)

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Check for large files
find . -type f -size +10M | grep -v ".git" | grep -v "data/" && echo "Warning: Large files detected"

# Verify structure
if [ -f "train.py" ] && [ ! -d "training" ]; then
    echo "Error: Training scripts should be in /training/"
    exit 1
fi
```

### CI/CD Checks

- Verify directory structure
- Check file naming conventions
- Ensure no secrets in code
- Validate symlinks

---

## 📚 Reference

### Full Structure

See `PROJECT_STRUCTURE.md` for complete directory layout.

### Quick Commands

```bash
# View structure
tree -L 3 --dirsfirst

# Find misplaced files
find apps/ -name "train*.py"
find apps/ -name "*.pth"

# Check symlinks
find . -type l -ls

# Archive old files
mv old_file.py archive/$(date +%Y%m%d)_old_file.py
```

---

## 🔄 Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-03-19 | 1.0 | Initial organization rules |

---

## ✅ Checklist for New Sessions

Before starting work:

- [ ] Review current structure with `tree -L 2`
- [ ] Identify correct directory for new work
- [ ] Check for existing similar files
- [ ] Follow naming conventions
- [ ] Update documentation as needed

Before ending session:

- [ ] Run cleanup commands
- [ ] Verify no files in wrong locations
- [ ] Commit with descriptive message
- [ ] Update this doc if patterns changed

---

**Questions?** Refer to `PROJECT_STRUCTURE.md` or ask the team lead.
