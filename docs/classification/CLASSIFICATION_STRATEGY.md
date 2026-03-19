# Document Classification Strategy Options

## Option A: Progressive Training (What You Described) ✅ Recommended

Train in stages as data becomes available:

### Stage 1: CNIE Only (Now)
- Classes: `cnie_front`, `cnie_back` (2 classes)
- Dataset: 5,942 samples (current)
- Train EfficientNet-B0 now

### Stage 2: Add Passport & Carte Grise (Later)
When you have passport/carte_grise images:

**Approach 1: Retrain from Scratch (Simplest)**
```yaml
model:
  num_classes: 4
  class_names: ["cnie_front", "cnie_back", "passport", "carte_grise"]
```
- Combine all data
- Train new model
- Best accuracy, no legacy issues

**Approach 2: Fine-tune Existing Model (Faster)**
- Load Stage 1 model (2-class)
- Extend classifier layer to 4 outputs
- Fine-tune on all 4 classes
- Risk: "Catastrophic forgetting" (may lose CNIE performance)

**Approach 3: Freeze Backbone (Safest)**
- Freeze all layers except final classifier
- Train only new classes
- Preserve CNIE performance
- May have lower accuracy on new classes

---

## Option B: True Hierarchical Classification (2-Stage Model)

### Model 1: Document Type Detector
```yaml
classes: ["cnie", "passport", "carte_grise"]  # 3 classes
```

### Model 2: CNIE Side Classifier  
```yaml
classes: ["cnie_front", "cnie_back"]  # 2 classes
```

**Inference Flow:**
```
Image → Model 1 (Document Type)
           ↓
      If CNIE → Model 2 (Front/Back)
      If Passport → Done
      If CarteGrise → Done
```

**Pros:**
- Can train Model 1 later when passport/carte_grise data available
- Model 2 can be trained NOW with current data
- Modular - update one without affecting other
- Better accuracy for each sub-problem

**Cons:**
- Two inference steps (slower)
- More complex deployment

---

## Recommendation: Start with Option A (Progressive)

### Current Action: Fix to 2 Classes

```yaml
# training/classification/configs/efficientnet_b0.yaml
model:
  name: "efficientnet_b0"
  num_classes: 2  # Changed from 4
  class_names: ["cnie_front", "cnie_back"]  # Only CNIE classes
  pretrained: true
  dropout: 0.3
```

### Later: Extend to 4 Classes

When passport/carte_grise data is ready:

```yaml
# Update config:
model:
  num_classes: 4
  class_names: ["cnie_front", "cnie_back", "passport", "carte_grise"]
```

Then either:
1. **Retrain from scratch** (recommended for best results)
2. **Fine-tune** the existing model (faster)

---

## Summary

| Approach | Now | Later | Complexity | Accuracy |
|----------|-----|-------|------------|----------|
| Progressive (Option A) | 2 classes | 4 classes | Low | High |
| Hierarchical (Option B) | Model 2 only | Add Model 1 | Medium | Very High |

**My recommendation:** 
1. **Stop current training**
2. **Fix config to 2 classes**
3. **Retrain** with current data
4. **Later**, collect passport/carte_grise and retrain with 4 classes

Would you like me to stop the current training and fix the config to 2 classes?
