# Manual vs Automated Operations Guide

## Overview

This document defines which operations in the VS Code + Colab Pro hybrid workflow are:
- **Manual** (you must do)
- **Kimi-Automated** (Kimi can handle entirely)
- **Hybrid** (Kimi assists, you review/confirm)

---

## 📊 Operations Matrix

| Operation | Category | One-time/Recurring | Time Required |
|-----------|----------|-------------------|---------------|
| **Account Setup** | Manual | One-time | 15 min |
| **Google Drive Upload** | Manual | One-time | 1-2 hours |
| **Colab Pro Subscription** | Manual | One-time | 5 min |
| **GitHub Repo Setup** | Hybrid | One-time | 10 min |
| **VS Code SSH Config** | Hybrid | One-time | 15 min |
| **Code Development** | Hybrid | Daily | Varies |
| **Training Execution** | Kimi-Automated | Recurring | 0 min |
| **Monitoring & Analysis** | Hybrid | Recurring | 15-30 min |
| **Model Download** | Manual | Recurring | 5 min |
| **Sync to Colab** | Kimi-Automated | Recurring | 0 min |

---

## 🔴 MANUAL OPERATIONS (You Must Do)

### 1. Account & Subscription Management

| Task | Why Manual | Instructions |
|------|-----------|--------------|
| **Sign up for Colab Pro** | Payment required | [colab.research.google.com](https://colab.research.google.com) → Upgrade |
| **Create ngrok account** | API token needed | [dashboard.ngrok.com](https://dashboard.ngrok.com) → Sign up |
| **Verify Google Account** | Security | Use your existing Google account |
| **GitHub account setup** | SSH keys, 2FA | [github.com](https://github.com) → Settings |

### 2. Data Upload to Google Drive

**Must be done manually** (large file transfers):

```bash
# Option 1: rclone (fastest, ~30 min for 3GB)
# You run this command
rclone copy data/cnie_dataset_10k gdrive:retin-verify/data/cnie_dataset_10k --progress

# Option 2: Google Drive for Desktop
# You drag and drop folder to Drive

# Option 3: Browser upload
# You use Google Drive web interface
```

**Why manual:**
- Large dataset (~3GB)
- Authentication required
- Bandwidth intensive
- One-time operation

### 3. Colab Runtime Management

| Action | Frequency | Your Task |
|--------|-----------|-----------|
| **Start Colab runtime** | Each session | Click "Connect" in Colab |
| **Select GPU type** | Each session | Runtime → Change runtime type → GPU |
| **Keep-alive** | During training | Keep browser/tab open or use keep-alive script |
| **Stop runtime** | When done | Runtime → Manage sessions → Terminate |

### 4. Model Download to Local

After training completes, you download models:

```bash
# Option 1: Google Drive web interface
# You: Right-click → Download

# Option 2: rclone (recommended)
# You run this:
rclone copy gdrive:retin-verify/models ./models --progress

# Option 3: Mount Drive locally
# You: Use Google Drive for Desktop
```

### 5. Decision Making

Kimi cannot make these decisions:

- **When to start training** (you decide based on readiness)
- **Which model to train first** (you prioritize)
- **Early stopping** (you check validation metrics)
- **Hyperparameter tuning** (you approve changes)
- **Production deployment** (you make the call)

---

## 🤖 KIMI-AUTOMATED OPERATIONS (Kimi Handles)

### 1. Code Development & Editing

Kimi can fully automate:

```
You: "Create a training script for EfficientNet-B0"
Kimi: ✍️ Writes complete train.py with:
  - Data loading
  - Model definition
  - Training loop
  - Checkpointing
  - Logging
```

| Task | Kimi Action | Output |
|------|-------------|--------|
| Write training scripts | Generate Python code | `training/*/train.py` |
| Create config files | Generate YAML configs | `training/*/configs/*.yaml` |
| Write data loaders | Implement Dataset classes | `training/utils/data_loaders.py` |
| Add error handling | Add try/except blocks | Robust code |
| Refactor code | Restructure for clarity | Cleaner code |
| Generate tests | Write unit tests | `tests/**/*.py` |

### 2. Git Operations

Kimi can automate (with your approval):

```bash
# Kimi generates commands, you approve:
git add .
git commit -m "feat: add training pipeline with checkpointing"
git push origin main

# Or Kimi creates sync script:
./scripts/sync_to_colab.sh  # Kimi wrote this
```

### 3. Colab Notebook Creation

Kimi generates complete notebooks:

```
You: "Create a training notebook for LayoutLMv3"
Kimi: ✍️ Generates notebook with:
  - Environment setup cells
  - Data loading
  - Model configuration
  - Training loop
  - Visualization
  - Export cells
```

### 4. Configuration Management

Kimi creates and updates configs:

```yaml
# Kimi generates:
# training/classification/configs/efficientnet_b0.yaml

model:
  name: "efficientnet_b0"
  num_classes: 4
  
training:
  batch_size: 32  # Kimi suggests based on GPU
  epochs: 50
  learning_rate: 1e-4
  
# Kimi adjusts based on your GPU specs
```

### 5. Troubleshooting Scripts

Kimi writes diagnostic scripts:

```python
# Kimi creates scripts/check_environment.py
# to verify setup before training

def check_colab_setup():
    check_gpu()
    check_drive_mount()
    check_dataset()
    check_dependencies()
```

### 6. Documentation

Kimi generates all documentation:

- `docs/VSCODE_COLAB_PRO_SETUP.md` (Kimi wrote this)
- `docs/TRAINING_SPECIFICATION.md` (Kimi wrote this)
- `README.md` updates
- Code comments
- Docstrings

---

## 🟡 HYBRID OPERATIONS (Kimi Assists, You Review)

### 1. SSH Setup for VS Code Remote

**Kimi does:**
- Generates setup script
- Creates SSH config template
- Writes instructions

**You do:**
- Run script in Colab
- Copy ngrok token (security)
- Add SSH config to `~/.ssh/config`
- Test connection

```
Kimi: "Here's the script and config..."
You: [runs script, copies token, tests connection]
Kimi: "Now try connecting from VS Code..."
```

### 2. Hyperparameter Adjustments

**Kimi suggests:**

```python
# Kimi: "Based on your GTX 950M specs, I suggest:"
batch_size = 4  # Instead of 32 (OOM risk)
gradient_accumulation_steps = 8  # Effective batch = 32
learning_rate = 5e-5  # Lower for stability
```

**You decide:**
- Accept suggestion
- Request alternatives
- Provide constraints

### 3. Debugging

**Kimi helps:**

```
You: "Training crashes with CUDA OOM"
Kimi: "Here are 5 solutions:
  1. Reduce batch size (current: 32 → suggested: 4)
  2. Enable gradient checkpointing
  3. Use mixed precision (fp16)
  4. Clear cache between epochs
  5. Use gradient accumulation"

You: "Try solution 1 and 3"
Kimi: [implements changes]
```

### 4. Training Monitoring

**Kimi creates:**
- Monitoring scripts
- Dashboard layouts
- Alert mechanisms

**You do:**
- Open TensorBoard
- Check metrics
- Decide when to stop

```python
# Kimi creates monitoring script
# You run it and interpret results
```

### 5. Model Evaluation

**Kimi generates:**
- Evaluation code
- Metric calculations
- Visualization plots

**You interpret:**
- Is 95% F1 good enough?
- Which model to deploy?
- Need more training?

### 6. Architecture Decisions

**Kimi presents options:**

```
Kimi: "For field extraction, we have 3 options:
  A) LayoutLMv3 (recommended, 95% accuracy)
  B) Rule-based (fast, 70% accuracy)
  C) Custom model (flexible, more work)
  
  Recommendation: A) for production"

You: "Go with A, but also implement B as fallback"
Kimi: [implements both]
```

---

## 📅 DAILY WORKFLOW BREAKDOWN

### Morning: Development (VS Code + Kimi)

```
┌──────────────────────────────────────────────────────────────┐
│ YOU                                                          │
│ ├── Open VS Code                                             │
│ ├── Start Kimi agent (kimi --continue)                       │
│ └── Plan today's work                                        │
├──────────────────────────────────────────────────────────────┤
│ KIMI                                                         │
│ ├── Write code based on your requirements                    │
│ ├── Create/modify training scripts                           │
│ ├── Generate configs                                         │
│ └── Write documentation                                      │
├──────────────────────────────────────────────────────────────┤
│ YOU                                                          │
│ ├── Review Kimi's code changes                               │
│ ├── Run local tests (pytest)                                 │
│ ├── Approve changes                                          │
│ └── Sync to GitHub                                           │
└──────────────────────────────────────────────────────────────┘
```

### Afternoon: Training (Colab)

```
┌──────────────────────────────────────────────────────────────┐
│ YOU                                                          │
│ ├── Open Google Colab                                        │
│ ├── Start GPU runtime                                        │
│ └── Open training notebook                                   │
├──────────────────────────────────────────────────────────────┤
│ KIMI (automated via notebook)                                │
│ ├── Mount Drive                                              │
│ ├── Pull latest code                                         │
│ ├── Prepare dataset                                          │
│ ├── Train model                                              │
│ ├── Save checkpoints to Drive                                │
│ └── Log metrics                                              │
├──────────────────────────────────────────────────────────────┤
│ YOU                                                          │
│ ├── Monitor TensorBoard (open in browser)                    │
│ ├── Check metrics every hour                                 │
│ └── Keep Colab tab open                                      │
└──────────────────────────────────────────────────────────────┘
```

### Evening: Analysis & Planning

```
┌──────────────────────────────────────────────────────────────┐
│ YOU                                                          │
│ ├── Download trained model from Drive                        │
│ ├── Run evaluation locally                                   │
│ └── Review results                                           │
├──────────────────────────────────────────────────────────────┤
│ KIMI                                                         │
│ ├── Generate evaluation report                               │
│ ├── Create comparison plots                                  │
│ ├── Suggest improvements                                     │
│ └── Update documentation                                     │
├──────────────────────────────────────────────────────────────┤
│ YOU + KIMI                                                   │
│ ├── Decide on next steps                                     │
│ ├── Adjust hyperparameters if needed                         │
│ └── Plan tomorrow's work                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 WHAT TO DELEGATE TO KIMI

### ✅ Delegate These (Safe & Efficient)

1. **Code Writing**
   - "Write a data loader for CNIE dataset"
   - "Create a training loop with early stopping"
   - "Implement F1 score calculation"

2. **Configuration**
   - "Generate YAML config for EfficientNet training"
   - "Create Colab notebook template"
   - "Write VS Code settings for remote development"

3. **Documentation**
   - "Document the training pipeline"
   - "Create README for setup instructions"
   - "Write API docs for inference module"

4. **Refactoring**
   - "Refactor train.py to use classes"
   - "Move common utilities to shared module"
   - "Add type hints to all functions"

5. **Testing**
   - "Write unit tests for data loaders"
   - "Create integration tests for pipeline"
   - "Add test fixtures for dataset"

6. **Debugging Assistance**
   - "Fix this CUDA OOM error"
   - "Why is training so slow?"
   - "Resolve this import error"

### ⚠️ Don't Delegate These (Need Your Input)

1. **Security**
   - Passwords, API keys, tokens
   - SSH private keys
   - Payment information

2. **Critical Decisions**
   - Which model to deploy
   - When to stop training
   - Budget allocation
   - Timeline commitments

3. **Data Verification**
   - Check dataset quality
   - Verify annotations
   - Validate outputs

4. **External Communications**
   - GitHub commits (you review)
   - Documentation publishing
   - Sharing results

---

## 🤖 AUTOMATION SCRIPTS PROVIDED

### Kimi-Created Automation

| Script | What It Automates | You Run It |
|--------|-------------------|------------|
| `scripts/sync_to_colab.sh` | Git add, commit, push | ✅ Manual |
| `scripts/init_colab.py` | Full Colab environment setup | ✅ In Colab |
| `scripts/setup_colab_ssh.py` | SSH tunnel for VS Code | ✅ In Colab |
| `scripts/prepare_dataset.py` | Dataset split preparation | ✅ Once |

### What Each Script Does

```bash
# sync_to_colab.sh - Kimi wrote this for you
#!/bin/bash
git add .
git commit -m "Sync to Colab - $(date)"
git push
# You just run: ./scripts/sync_to_colab.sh
```

```python
# init_colab.py - Kimi wrote this for you
# Mounts Drive, pulls code, installs dependencies, verifies data
# You just run: !python scripts/init_colab.py in Colab
```

---

## 💡 BEST PRACTICES

### For Manual Operations

1. **Use bookmarks:**
   - Colab: https://colab.research.google.com
   - Drive: https://drive.google.com
   - TensorBoard: Bookmark after first run

2. **Set reminders:**
   - Check Colab every 2-3 hours during training
   - Download best models immediately after training

3. **Keep notes:**
   - Document which hyperparameters worked
   - Note training times for planning

### For Kimi Operations

1. **Be specific:**
   ```
   ❌ "Fix this"
   ✅ "Fix the CUDA OOM in train.py by reducing batch size from 32 to 4"
   ```

2. **Provide context:**
   ```
   "I'm using Colab Pro with V100 GPU. 
   Create a training config optimized for 16GB VRAM."
   ```

3. **Review before applying:**
   - Check Kimi's code changes
   - Run tests before committing
   - Understand what changed

4. **Iterate:**
   ```
   You: "Create training script"
   Kimi: [generates script]
   You: "Add early stopping and checkpointing"
   Kimi: [updates script]
   You: "Also log to TensorBoard"
   Kimi: [final version]
   ```

---

## 📋 CHECKLIST: WHO DOES WHAT

### One-Time Setup

| Task | Who | Status |
|------|-----|--------|
| Sign up Colab Pro | You | ☐ |
| Upload dataset to Drive | You | ☐ |
| Create GitHub repo | You | ☐ |
| Push initial code | Kimi assists | ☐ |
| Create Colab notebook | Kimi | ☐ |
| Write setup scripts | Kimi | ☐ |
| Configure VS Code SSH | Hybrid | ☐ |
| Test full pipeline | Hybrid | ☐ |

### Daily Operations

| Task | Who | Status |
|------|-----|--------|
| Write/edit code | Kimi | ☐ |
| Review changes | You | ☐ |
| Run local tests | You | ☐ |
| Sync to GitHub | Kimi script | ☐ |
| Start Colab runtime | You | ☐ |
| Run training notebook | Kimi notebook | ☐ |
| Monitor training | You | ☐ |
| Download results | You | ☐ |
| Analyze results | Hybrid | ☐ |

---

## 🎓 LEARNING PATH

### Week 1: Setup (Mostly Manual)
- You: Set up accounts, upload data
- Kimi: Create initial scripts and configs
- Together: Test pipeline

### Week 2: Development (Hybrid)
- You: Define requirements, review code
- Kimi: Implement features, write tests
- Together: Debug issues

### Week 3: Training (Mostly Automated)
- You: Start training, monitor
- Kimi: Notebook runs automatically
- You: Download and evaluate

### Week 4: Optimization (Hybrid)
- Kimi: Analyze results, suggest improvements
- You: Decide on changes
- Kimi: Implement optimizations

---

## Summary

| Category | Percentage | Examples |
|----------|------------|----------|
| **Manual (You)** | 20% | Accounts, decisions, monitoring |
| **Hybrid (Both)** | 30% | Setup, debugging, optimization |
| **Kimi-Automated** | 50% | Coding, configs, docs, scripts |

**Key Insight:** 
- Kimi handles the 80% of tedious coding work
- You focus on the 20% of critical decisions and oversight
- Together you achieve 10x productivity

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-14
