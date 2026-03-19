# Hybrid Retraining Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID RETRAINING LOOP                               │
│                     Local Machine ←────→ Google Colab                        │
└─────────────────────────────────────────────────────────────────────────────┘


PHASE 1: FEEDBACK COLLECTION (Local - Ongoing)
══════════════════════════════════════════════════════════════════════════════

    User Interface (http://127.0.0.1:8000)
    ┌─────────────────────────────────────────┐
    │  📷 Capture → 🎯 Classify → ✅/❓ Flag  │
    └─────────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────┐
    │     feedback_data/misclassified/        │
    │     ├── cnie_front/ (wrong preds)       │
    │     └── cnie_back/  (wrong preds)       │
    └─────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    [Count < 10]          [Count >= 10]
         │                     │
         │                     ▼
         │              ┌─────────────┐
         │              │  ✅ READY   │
         │              │   TO TRAIN  │
         │              └─────────────┘
         │                     │
    Continue              TRIGGER
    Collecting            RETRAINING


PHASE 2: DEPLOY TO COLAB (Local → Colab)
══════════════════════════════════════════════════════════════════════════════

    Local Machine                                    Google Colab
    ┌─────────────────────┐                         ┌─────────────────────┐
    │ 1. Package data     │                         │                     │
    │    retrain_data     │    ┌──────────────┐     │                     │
    │    .tar.gz          ├───▶│  SSH + SCP   │────▶│  /content/retin_    │
    │                     │    │              │     │  retrain/           │
    │ 2. Upload model     │    │  Cloudflare  │     │                     │
    │    base_model.pth   ├───▶│  Tunnel      │────▶│  + base_model.pth   │
    │                     │    └──────────────┘     │                     │
    │ 3. Upload script    │                         │  + train_on_colab.py│
    │    train_on_colab.py│────────────────────────▶│                     │
    └─────────────────────┘                         └─────────────────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────┐
                                                    │  Extract    │
                                                    │  Data       │
                                                    └─────────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────┐
                                                    │ Start Train │
                                                    │ (background)│
                                                    └─────────────┘


PHASE 3: TRAINING (Colab - GPU Accelerated)
══════════════════════════════════════════════════════════════════════════════

    Google Colab GPU (T4/V100/A100)
    ┌─────────────────────────────────────────┐
    │                                         │
    │  Epoch 1/10  ████░░░░░░  Train: 85%    │
    │  Epoch 2/10  ████████░░  Train: 89%    │
    │     ...                                 │
    │  Epoch 10/10 ██████████ Best: 94%      │
    │                                         │
    │  ┌─────────────────────────────────┐    │
    │  │  cnie_front_back_real_retrained │    │
    │  │         .pth                    │    │
    │  │  (Best validation accuracy)     │    │
    │  └─────────────────────────────────┘    │
    │                                         │
    │  [Write DONE file when complete]        │
    │                                         │
    └─────────────────────────────────────────┘
                    │
                    │ Monitor via:
                    │ ssh root@host \
                    │   'tail -f training.log'
                    ▼
              [TRAINING COMPLETE]


PHASE 4: DOWNLOAD & DEPLOY (Colab → Local)
══════════════════════════════════════════════════════════════════════════════

    Google Colab                                    Local Machine
    ┌─────────────────────┐                         ┌─────────────────────┐
    │                     │    ┌──────────────┐     │  1. Backup current  │
    │  cnie_front_back_   │    │              │     │     model           │
    │  real_retrained.pth │───▶│  SSH + SCP   │────▶│     → backup_*.pth  │
    │                     │    │              │     │                     │
    │  + history.json     │───▶│  Download    │────▶│  2. Replace with    │
    │                     │    └──────────────┘     │     new model       │
    │  + DONE             │                         │                     │
    │                     │                         │  3. Restart server  │
    └─────────────────────┘                         │     pkill + start   │
                                                    └─────────────────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────┐
                                                    │  Server     │
                                                    │  Running    │
                                                    │  New Model  │
                                                    └─────────────┘


PHASE 5: TEST & REPEAT
══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────┐
    │  Test at http://127.0.0.1:8000          │
    │                                         │
    │  • Capture new images                   │
    │  • Verify improved accuracy             │
    │  • Continue flagging wrong predictions  │
    │                                         │
    │  [Collect 10+ more samples]             │
    │         │                               │
    │         └───────────────────────────────┤
    │                                         │
    │    ↓ GO BACK TO PHASE 1                │
    └─────────────────────────────────────────┘


══════════════════════════════════════════════════════════════════════════════
                               COMMANDS SUMMARY
══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ FULL WORKFLOW (One command does it all)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   python retrain_manager.py --host abc123.trycloudflare.com --full          │
│                                                                             │
│   This runs:                                                                │
│   1. Check feedback status                                                  │
│   2. Deploy to Colab via SSH                                                │
│   3. Monitor training until complete                                        │
│   4. Download new model                                                     │
│   5. Restart local server                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP BY STEP (More control)                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   # Check if ready to retrain                                               │
│   python retrain_manager.py --status                                        │
│                                                                             │
│   # Deploy to Colab                                                         │
│   python retrain_manager.py --host abc123.trycloudflare.com --deploy        │
│                                                                             │
│   # Monitor training (optional, can Ctrl+C)                                 │
│   python retrain_manager.py --host abc123.trycloudflare.com --monitor       │
│                                                                             │
│   # Download and restart when ready                                         │
│   python retrain_manager.py --host abc123.trycloudflare.com --download \\   │
│       --restart                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


══════════════════════════════════════════════════════════════════════════════
                              TIME ESTIMATES
══════════════════════════════════════════════════════════════════════════════

┌────────────────────┬─────────────────┬─────────────────────────────────────┐
│ Phase              │ Time            │ Notes                               │
├────────────────────┼─────────────────┼─────────────────────────────────────┤
│ Collect feedback   │ Days/Weeks      │ Until 10+ samples accumulated       │
│ Deploy to Colab    │ 10-30 seconds   │ Depends on data size                │
│ Training on Colab  │ 5-15 minutes    │ T4 GPU, 10 epochs                   │
│ Download model     │ 10-20 seconds   │ Depends on connection               │
│ Restart server     │ 5-10 seconds    │ Local CPU inference                 │
├────────────────────┼─────────────────┼─────────────────────────────────────┤
│ TOTAL CYCLE        │ ~10-20 min      │ Excluding feedback collection       │
└────────────────────┴─────────────────┴─────────────────────────────────────┘


══════════════════════════════════════════════════════════════════════════════
                              FILE LOCATIONS
══════════════════════════════════════════════════════════════════════════════

Local Machine:
  ~/retin-verify/apps/classification/feedback_data/      # Collected feedback
  ~/retin-verify/models/classification/                  # Model storage
  /tmp/retrain_data_*.tar.gz                             # Deployment package

Google Colab:
  /content/retin_retrain/                                # Working directory
  /content/retin_retrain/retrain_data/                   # Training data
  /content/retin_retrain/base_model.pth                  # Original model
  /content/retin_retrain/training.log                    # Training logs
  /content/retin_retrain/DONE                            # Completion flag
  /content/retin_retrain/cnie_front_back_real_retrained.pth  # New model
