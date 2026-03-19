# Session Rules & Guidelines

**Document:** Session Management Rules  
**Version:** 1.0  
**Updated:** 2026-03-18

---

## 1. File & Folder Structure Rules

### 1.1 Models Directory Structure

All models MUST follow this structure:

```
models/
├── classification/              # Current classification models
│   ├── *.pth                   # PyTorch models
│   ├── *.onnx                  # ONNX exports
│   └── *.json                  # Metadata/history
├── detection/                   # Detection models
├── extraction/                  # Extraction models
├── exported/                    # Final exports
├── classification_production/   # Production baselines (protected)
└── archive/                     # Historical versions
    └── YYYY-MM-DD/             # Date-stamped archives
```

### 1.2 Cleanup Rules

**After EVERY session:**

1. **Move new models** to appropriate folder (e.g., `classification/`)
2. **Archive old versions** to `archive/YYYY-MM-DD/`
3. **Remove temp files** (*.tar.gz, temp_*, etc.)
4. **Never leave files in models root** - only subdirectories

**Archive Process:**
```bash
# Create date-stamped archive
mkdir -p models/archive/YYYY-MM-DD

# Move old versions
mv models/classification/old_model.pth models/archive/YYYY-MM-DD/

# Remove empty directories
rmdir models/classification_colab 2>/dev/null || true
```

---

## 2. Session State Management

### 2.1 Session State File

Location: `~/.kimi/session_state.json`

**Required Fields:**
```json
{
  "session_id": "unique_identifier",
  "created_at": "ISO timestamp",
  "last_updated": "ISO timestamp",
  "status": "IN_PROGRESS|COMPLETE|ERROR",
  "phase": "current_phase",
  "project": "project_name",
  "current_task": "description",
  "steps": [
    {"step": 1, "name": "...", "status": "pending|in_progress|completed"}
  ],
  "next_action": "what to do next",
  "cleanup_rules_applied": true
}
```

### 2.2 Session Lifecycle

1. **Initialize:** Create session state with all planned steps
2. **Update:** Mark steps complete as progress happens
3. **Complete:** Final update with results, cleanup flag set
4. **Archive:** Move to `sessions/archive/` after completion

---

## 3. Documentation Requirements

### 3.1 Required Documents per Session

| Document | Location | Purpose |
|----------|----------|---------|
| Session Report | `docs/SESSION_REPORT_YYYY-MM-DD.md` | Complete session log |
| Session State | `~/.kimi/session_state.json` | Machine-readable status |
| Code Changes | Git commits | Track all modifications |

### 3.2 Session Report Template

Every session report MUST include:

1. **Executive Summary** - One-paragraph overview
2. **Phase-by-Phase Results** - What was done, metrics
3. **File Structure Changes** - Before/after cleanup
4. **Deliverables** - What was produced
5. **Usage Instructions** - How to use the outputs
6. **Next Steps** - Recommended follow-up actions

---

## 4. Model Management Rules

### 4.1 Model Naming Convention

```
{purpose}_{variant}_{version}.{ext}

Examples:
- cnie_front_back_real.pth          (production model)
- cnie_front_back_v2.pth            (version 2)
- classification_baseline_effnet.pth (baseline)
```

### 4.2 Model Metadata

Every model MUST include metadata:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': ['class1', 'class2'],
    'val_acc': 90.0,
    'epoch': 10,
    'created_at': '2026-03-18T12:00:00',
    'training_data': '7,600 images'
}, 'model.pth')
```

### 4.3 Export Requirements

For every PyTorch model:
1. ✅ Save .pth with metadata
2. ✅ Export to ONNX format
3. ✅ Validate ONNX with onnx.checker
4. ✅ Test inference on sample images

---

## 5. SSH/Colab Workflow Rules

### 5.1 SSH Key Management

- SSH key: `~/.ssh/id_colab` (private), `~/.ssh/id_colab.pub` (public)
- Config: `~/.ssh/config` with `colab-gpu` host entry
- Deploy key at start of each new Colab session

### 5.2 Colab Session Workflow

```
1. Start Colab with cloudflared tunnel
2. Update ~/.ssh/config with new hostname
3. Deploy SSH key: ./scripts/deploy_ssh_key.sh colab-gpu
4. Sync code: ./scripts/colab_workflow.sh colab-gpu sync
5. Upload data (rsync/scp)
6. Run training
7. Download results
8. Cleanup remote (optional)
```

---

## 6. Data Management Rules

### 6.1 Directory Structure

```
apps/classification/
├── my_cards/                   # Raw captured photos
│   ├── cnie_front/
│   └── cnie_back/
├── augmented/                  # Augmented dataset
│   ├── train/
│   ├── val/
│   └── annotations.json
└── cnie_only_augmented/        # Current session data
```

### 6.2 Data Retention

- **Raw photos:** Keep indefinitely (in `my_cards/`)
- **Augmented data:** Can be regenerated, archive if >10GB
- **Annotations:** Keep with model version

---

## 7. Session Checklist

### Before Starting
- [ ] Update session state
- [ ] Review previous session report
- [ ] Check file structure compliance
- [ ] Verify SSH connectivity (if using Colab)

### During Session
- [ ] Update session state after each phase
- [ ] Document decisions and rationale
- [ ] Test intermediate results
- [ ] Commit code changes regularly

### After Completion
- [ ] Finalize session state (status: COMPLETE)
- [ ] Write session report
- [ ] **Apply cleanup rules** (CRITICAL)
- [ ] Move new models to correct folders
- [ ] Archive old versions
- [ ] Remove temp files
- [ ] Verify file structure
- [ ] Update session state with `cleanup_rules_applied: true`

---

## 8. Quick Reference

### Cleanup Commands
```bash
cd ~/retin-verify/models

# Archive old classification models
mkdir -p archive/$(date +%Y-%m-%d)
mv classification/old_*.pth archive/$(date +%Y-%m-%d)/ 2>/dev/null || true

# Clean temp files
rm -f *.tar.gz model_temp*

# Verify structure
find . -maxdepth 2 -type f -name "*.pth" -o -name "*.onnx" | sort
```

### Session State Update
```bash
# Mark session complete
cat > ~/.kimi/session_state.json << 'EOF'
{
  "status": "COMPLETE",
  "phase": "done",
  "cleanup_rules_applied": true,
  ...
}
EOF
```

---

## 9. Violations & Corrections

### Common Violations

| Violation | Correction |
|-----------|------------|
| Files in models/ root | Move to appropriate subfolder |
| Multiple classification_* folders | Archive old, keep one |
| Missing session report | Create retroactively |
| No metadata in model | Reload and re-save with metadata |

### Emergency Cleanup

If structure is severely disorganized:

```bash
# 1. Create backup
cd ~/retin-verify
tar czf models_backup_$(date +%Y%m%d).tar.gz models/

# 2. Create clean structure
mkdir -p models/{classification,detection,extraction,exported,archive}

# 3. Move files to appropriate folders
# (manual review required)
```

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-18 | Initial rules established after Strategy 2 completion |

---

**Remember: Clean structure = Faster development, fewer errors, easier handoffs.**
