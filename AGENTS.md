# AGENTS.md - Kimi Session Context

> **AUTO-LOADED:** This file is automatically read at the start of every Kimi session.
> **Last Updated:** 2026-03-19

---

## 🚨 START HERE - Every Session

### 1. Read Organization Rules
**ALWAYS read this first:** `.github/ORGANIZATION_RULES.md`

### 2. Verify Project Structure
```bash
# Check current structure
tree -L 2 --dirsfirst 2>/dev/null || ls -la

# Verify key directories exist
ls -la {apps,training,inference,models,data,docs,src,tests}/ 2>/dev/null | head -20
```

### 3. Check Git Status
```bash
git status --short
git log --oneline -3
```

---

## 📁 PROJECT STRUCTURE (Quick Reference)

```
retin-verify/
├── apps/                    # Production runtime ONLY
│   └── classification/      # Frontend + Backend (clean, no training scripts)
│
├── training/                # Training scripts & configs
│   └── classification/
│       ├── scripts/         # Training Python scripts
│       ├── configs/         # YAML configs
│       └── new_training/    # v3 training (synthetic + real)
│           ├── train_with_synthetic.py
│           └── deploy_v3_with_synthetic.py
│
├── inference/               # Standalone inference apps
│   └── apps/
│       └── classification/  # Self-contained deployment
│
├── models/                  # Trained models
│   ├── classification/      # Current models
│   └── archive/             # Old models
│
├── data/                    # All data
│   ├── raw/                 # Raw input
│   ├── processed/           # Processed datasets
│   │   └── classification/
│   │       └── dataset_3class/
│   ├── synthetic/           # Synthetic data (16K images)
│   └── feedback/            # User feedback
│       └── classification/
│
├── docs/                    # Documentation by subject
│   ├── classification/      # Classification docs
│   ├── guides/              # General guides
│   ├── deployment/          # Deployment guides
│   └── synthetic/           # Synthetic data docs
│
├── src/                     # Reusable source code
├── tests/                   # Test suites
└── synthetic/               # Synthetic data generation tools
```

---

## 📋 ORGANIZATION RULES (Summary)

### Golden Rules
1. **One purpose per directory** - No mixing concerns
2. **No duplication** - Use symlinks, not copies
3. **Archive, don't delete** - Move to `archive/` with date prefix
4. **Clear naming** - `{task}_{description}_v{version}_{date}.{ext}`
5. **Document changes** - Update docs when changing structure

### Directory Purposes
| Directory | What's Allowed | What's NOT Allowed |
|-----------|----------------|-------------------|
| `/apps/` | Runtime files, API, UI | Training scripts, data, models |
| `/training/` | Training scripts, configs, notebooks | Runtime apps, large data |
| `/inference/` | Standalone inference apps | Training code |
| `/data/` | Datasets, annotations, feedback | Scripts, models |
| `/models/` | Trained model files (.pth, .onnx) | Training code |
| `/docs/` | Markdown documentation | Code (except examples) |
| `/src/` | Reusable Python modules | Scripts, apps |

### File Naming Conventions
- **Training:** `train_{task}_{version}.py`
- **Models:** `{task}_{description}_v{version}_{YYYYMMDD}.pth`
- **Docs:** `{TYPE}_{topic}.md` (e.g., `GUIDE_deployment.md`)
- **Data:** `{timestamp}_{uuid}.jpg` or `{split}_{class}_{index}.json`

---

## 💾 AUTOMATIC SESSION SAVING (MANDATORY)

**EVERY session MUST end with saving state:**

```bash
# Save session (run this before ending)
python3 .kimi/session_manager.py save \
    "Task Name" \
    "in_progress|complete|blocked" \
    "Progress made this session" \
    "Next steps to take"
```

**What this does:**
- Creates `.kimi/session_state.json` (machine-readable)
- Updates `CURRENT_STATUS.txt` (human-readable)
- Appends to `.kimi/session_history.jsonl` (full history)

**Why this matters:**
- Next session knows exactly where you left off
- No lost context between sessions
- Full history of project progress
- Kimi can auto-resume with full context

### Quick Save Examples

```bash
# After training deployment
python3 .kimi/session_manager.py save \
    "v3 Training Deployment" \
    "in_progress" \
    "Deployed to Colab, training at epoch 15/50, balance 85%" \
    "Download model at epoch 50, test on validation set"

# After data collection
python3 .kimi/session_manager.py save \
    "Data Collection - Back Images" \
    "complete" \
    "Collected 50 back images, total dataset: 150 per class" \
    "Run augmentation and start v4 training"

# When blocked
python3 .kimi/session_manager.py save \
    "Model Optimization" \
    "blocked" \
    "ONNX conversion failing with opset 13 error" \
    "Research ONNX opset compatibility, try opset 11"
```

## 🔄 SESSION WORKFLOW

### Starting a Session

1. **Check previous session state**
   ```bash
   cat CURRENT_STATUS.txt 2>/dev/null || echo "No status file"
   ```

2. **Verify you're in the right directory**
   ```bash
   pwd  # Should be: /home/retinrix/retin-verify
   ```

3. **Check for uncommitted changes**
   ```bash
   git status --short
   ```

4. **Review recent commits**
   ```bash
   git log --oneline -5
   ```

### During a Session

1. **Follow organization rules** - Put files in correct locations
2. **Name things clearly** - Use conventions above
3. **Document as you go** - Update relevant docs
4. **Commit regularly** - Small, descriptive commits

### Ending a Session

1. **SAVE SESSION (MANDATORY)**
   ```bash
   # This is REQUIRED - never skip this step
   python3 .kimi/session_manager.py save \
       "Task Name" \
       "in_progress" \
       "What was accomplished" \
       "Next steps"
   
   # Or let Kimi do it automatically:
   # Just say: "save session" and Kimi will run the command
   ```

2. **Clean up temporary files**
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
   find . -type f -name "*.pyc" -delete 2>/dev/null
   find . -type f -name ".DS_Store" -delete 2>/dev/null
   ```

3. **Verify no misplaced files**
   ```bash
   # Check for training scripts in apps/
   find apps/ -name "train*.py" 2>/dev/null
   
   # Check for data in wrong places
   find . -name "*.jpg" -path "*/apps/*" 2>/dev/null
   ```

4. **Commit changes**
   ```bash
   git add [files]
   git commit -m "[descriptive message]"
   git push origin main
   ```

---

## 🛠️ COMMON COMMANDS

### Development
```bash
# Start classification API
cd apps/classification && ./start_server.sh

# Check API health
curl http://localhost:8000/health

# Test classification
curl -X POST -F "file=@test.jpg" http://localhost:8000/predict
```

### Training
```bash
# Deploy v3 training to Colab
cd training/classification/new_training
python3 deploy_v3_with_synthetic.py

# Monitor training
ssh root@your-host "tail -f /content/retin_v3_synthetic/train_v3_synthetic.log"

# Download model
scp root@your-host:/content/cnie_classifier_3class_v3_synthetic.pth \
    ../../models/classification/
```

### Data
```bash
# Check dataset structure
ls -la data/processed/classification/dataset_3class/

# Check feedback
ls -la data/feedback/classification/

# Count images
find data/processed/classification/dataset_3class -name "*.jpg" | wc -l
```

### Git
```bash
# Quick status
git status --short

# View recent changes
git diff --stat

# Archive old file
mv old_file.py archive/$(date +%Y%m%d)_old_file.py
```

---

## 📚 ESSENTIAL DOCUMENTATION

Always refer to these docs:

| Document | When to Read |
|----------|--------------|
| `.github/ORGANIZATION_RULES.md` | **Every session start** |
| `PROJECT_STRUCTURE.md` | When creating new files/directories |
| `README.md` | For project overview |
| `docs/classification/V3_SYNTHETIC_INTEGRATION.md` | Before v3 training |
| `CURRENT_STATUS.txt` | To check previous session state |

---

## ⚠️ ANTI-PATTERNS (Don't Do This)

### ❌ Wrong
```
apps/classification/train.py          # Training in apps/
apps/classification/dataset/          # Data in apps/
model.pth                              # Model in root
train.py                               # Script in root
```

### ✅ Right
```
training/classification/scripts/train.py     # Training in training/
data/processed/classification/dataset/       # Data in data/
models/classification/model.pth              # Models in models/
```

---

## 🔗 IMPORTANT PATHS

| What | Where |
|------|-------|
| Current models | `models/classification/` |
| Training data | `data/processed/classification/dataset_3class/` |
| Synthetic data | `data/cnie_dataset_10k/cnie_pairs/` |
| Feedback data | `data/feedback/classification/` |
| v3 training scripts | `training/classification/new_training/` |
| API server | `apps/classification/backend/api_server.py` |
| Inference engines | `apps/classification/backend/inference_engine*.py` |

---

## 🎯 CURRENT PROJECT STATE

### Active Model
- **Name:** cnie_classifier_3class_v2.pth
- **Accuracy:** 88.5% validation
- **Status:** Production (with 35% front bias)
- **Location:** `models/classification/`

### Next Priority
- **Task:** v3 Training with synthetic + real data
- **Goal:** 90%+ balance without bias
- **Location:** `training/classification/new_training/`

---

## 📝 SESSION CHECKLIST

### At Start:
- [ ] Read `.github/ORGANIZATION_RULES.md`
- [ ] Check `CURRENT_STATUS.txt` for previous state
- [ ] Verify `git status` is clean
- [ ] Confirm you're in `/home/retinrix/retin-verify`

### At End:
- [ ] **SAVE SESSION** (REQUIRED): `python3 .kimi/session_manager.py save "Task" "status" "progress" "next"`
- [ ] Files in correct directories
- [ ] Named following conventions
- [ ] Temporary files cleaned
- [ ] `CURRENT_STATUS.txt` updated
- [ ] Changes committed
- [ ] Pushed to GitHub

---

## 💡 REMEMBER

1. **This file auto-loads** - You should see this at every session start
2. **Read the rules first** - Organization is critical
3. **When in doubt, archive** - Don't delete, move to `archive/`
4. **Document everything** - Future you will thank you
5. **Commit often** - Small commits are better than large ones

---

**Ready to start? Read `.github/ORGANIZATION_RULES.md` now!**
