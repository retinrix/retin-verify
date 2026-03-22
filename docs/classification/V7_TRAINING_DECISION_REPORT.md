# V7 Training Decision Report: Strategy Comparison & Recommendation

**Date:** March 20, 2026  
**Decision Required:** V7 Training Strategy Selection  
**Input Documents:**
- V7_TRAIN_STRATEGY.md (Existing training plan)
- FEEDBACK_TESTING_REPORT_V6.md (Feedback analysis)

---

## 1. Executive Summary

### Updated Front Misclassification Count
| Source | Count | Notes |
|--------|-------|-------|
| **Merged Feedback** | **135 images** | Historical misclassified fronts |
| **Recent Feedback** | **87 images** | Current session (updated count) |
| **feedback_data/misclassified** | **57 images** | Session-specific misclassified |
| **classification/misclassified** | **91 images** | Archive misclassified |
| **TOTAL UNIQUE** | **~220+ images** | Significant front failure data |

**Critical Update:** The recent front misclassification count is **87** (not 24 as initially reported), representing a much more severe problem requiring immediate action.

---

## 2. Strategy Comparison Matrix

### 2.1 Recommendation Areas Comparison

| Aspect | V7_TRAIN_STRATEGY.md | FEEDBACK_TESTING_REPORT_V6.md | Agreement Level |
|--------|---------------------|------------------------------|-----------------|
| **Primary Action** | Retrain Stage 2 with feedback | Retrain Stage 2 with feedback | ✅ Full |
| **Front Augmentation** | Glare, warping, texture, perspective | Glare, low-light, occlusion | ✅ Partial |
| **Loss Function** | Weighted (1.2 front, 0.9 back) | Weighted (penalize front errors) | ✅ Full |
| **Regularization** | MixUp/CutMix | MixUp/CutMix mentioned | ✅ Full |
| **Architecture** | Keep EfficientNet-B0 | Suggest attention mechanism | ⚠️ Different |
| **Threshold Tuning** | Not mentioned | Lower front to 0.45 | ❌ Gap |
| **V7 Definition** | Stage 2 Retraining | 3-Class single model | ❌ Different |

### 2.2 Detailed Differences

#### A. Scope Definition
| Document | "V7" Definition | Timeline |
|----------|----------------|----------|
| **V7_TRAIN_STRATEGY.md** | Retrained Stage 2 (still cascade) | Immediate (1-2 days) |
| **FEEDBACK_REPORT_V6** | New 3-class single model | Long-term (1-2 weeks) |

**Analysis:** The documents define "V7" differently. V7_TRAIN_STRATEGY treats it as incremental improvement; FEEDBACK_REPORT treats it as architectural change.

#### B. Augmentation Strategy
| Document | Front-Specific Augmentations |
|----------|------------------------------|
| **V7_TRAIN_STRATEGY.md** | Glare overlay, photo warping, background texture, extreme perspective, color temp |
| **FEEDBACK_REPORT_V6** | Photo glare, low-light, portrait occlusion, color distortion |

**Synthesis:** V7_TRAIN_STRATEGY has more detailed augmentations. Both agree on glare and lighting issues.

#### C. Architecture Changes
| Document | Proposal | Complexity |
|----------|----------|------------|
| **V7_TRAIN_STRATEGY.md** | Keep EfficientNet-B0, adjust training | Low |
| **FEEDBACK_REPORT_V6** | Add attention, larger resolution (299x299), ensemble | High |

**Analysis:** V7_TRAIN_STRATEGY favors quick wins; FEEDBACK_REPORT suggests deeper changes.

#### D. Quick Fixes vs Long-term
| Document | Quick Fix | Long-term |
|----------|-----------|-----------|
| **V7_TRAIN_STRATEGY.md** | None specified | Stage 2 retraining |
| **FEEDBACK_REPORT_V6** | Threshold 0.45, more feedback | 3-class model, active learning |

---

## 3. Synthesis: Unified V7 Strategy

### 3.1 Recommended Hybrid Approach

Based on comparing both documents, the optimal strategy combines:

**Phase 1: Immediate Hotfix (Today)**
```python
# Quick threshold adjustment from FEEDBACK_REPORT
if stage2_score > 0.45:  # Was 0.5
    return "front"
else:
    return "back"
```
- **Effort:** 5 minutes
- **Expected Impact:** +5-10% front recall
- **Risk:** May increase false positives

**Phase 2: V6.1 Stage 2 Retraining (This Week)**
Following V7_TRAIN_STRATEGY.md specifications:
- **Dataset:** Original + all 135 front + 120 back misclassified
- **Augmentation:** All 5 front-specific transforms
- **Loss:** Weighted [1.2, 0.9] for [front, back]
- **Regularization:** MixUp α=0.2, CutMix α=1.0
- **Training:** Differential LR (backbone 1e-5, classifier 1e-3)
- **Expected Result:** ≥95% front accuracy

**Phase 3: V7 Architecture (Next Sprint)**
From FEEDBACK_REPORT long-term recommendations:
- Evaluate if cascade meets >95% target
- If not, design 3-class single model
- Implement active learning pipeline

### 3.2 Augmentation Synthesis (Combined Best)

| Augmentation | Source | Priority |
|--------------|--------|----------|
| Random glare on photo area | Both | 🔴 Critical |
| Photo area warping | V7_STRATEGY | 🟡 High |
| Low-light simulation | FEEDBACK_RPT | 🔴 Critical |
| Background texture noise | V7_STRATEGY | 🟢 Medium |
| Extreme perspective | V7_STRATEGY | 🟡 High |
| Portrait occlusion | FEEDBACK_RPT | 🟡 High |
| Color temperature shift | V7_STRATEGY | 🟢 Medium |
| Color distortion | FEEDBACK_RPT | 🟢 Medium |

---

## 4. Data Inventory for Training

### 4.1 Available Training Data

| Class | Original | Misclassified | **Total Available** |
|-------|----------|---------------|---------------------|
| Front | 169 | 135 + 87 = **222** | **~391** |
| Back | 176 | 120 | **~296** |

**Key Insight:** We now have **222 front misclassifications** - more than the original training set! This is sufficient for aggressive retraining.

### 4.2 Proposed V6.1 Dataset Split

| Class | Train | Validation | Test | Augmentation Factor |
|-------|-------|------------|------|---------------------|
| Front | 300 | 50 | 41 | 2x (front-specific) |
| Back | 250 | 30 | 16 | 1.5x (standard) |
| **Total** | **550** | **80** | **57** | - |

**Notes:**
- Use 80% of misclassified in training, 20% in validation
- Keep 10% of original data as untouched test set
- Apply heavier augmentation to front class

---

## 5. Implementation Decision Matrix

### 5.1 Option A: Conservative (V7_TRAIN_STRATEGY.md only)
- Retrain Stage 2 with feedback
- Standard augmentations + front-specific
- Weighted loss
- **Risk:** May not address fundamental architecture limitations
- **Timeline:** 2-3 days

### 5.2 Option B: Aggressive (FEEDBACK_REPORT_V6 only)
- Immediate threshold hotfix
- Design new 3-class architecture
- Implement attention mechanism
- **Risk:** Longer development, may over-engineer
- **Timeline:** 2-3 weeks

### 5.3 Option C: Hybrid (RECOMMENDED)
| Phase | Action | Timeline | Expected Front Accuracy |
|-------|--------|----------|------------------------|
| 1 | Threshold hotfix (0.45) | Today | 77% → 82% |
| 2 | Stage 2 retraining (V6.1) | This week | 82% → 95% |
| 3 | Evaluate cascade performance | Next week | Validate >95% |
| 4 | Design V7 if needed | Following week | Target 98% |

---

## 6. Risk Assessment

### 6.1 If We Only Follow V7_TRAIN_STRATEGY.md
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Still miss front cases | Medium | High | Add threshold tuning |
| Overfit to misclassified | Low | Medium | Keep original test set |
| Chip bias persists | Medium | High | Add attention mechanism |

### 6.2 If We Only Follow FEEDBACK_REPORT_V6
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Delayed fix | High | High | Implement threshold hotfix |
| Over-engineering | Medium | Medium | Validate cascade first |
| Resource waste | Low | Medium | Phase approach |

### 6.3 Hybrid Approach Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Threshold causes FP | Medium | Medium | Monitor and adjust |
| Phase 2 doesn't reach 95% | Low | High | Proceed to Phase 3 |

---

## 7. Why Phased Approach vs Direct to Phase 4 (V7)?

You might ask: *"If Phase 4 gives the best accuracy (98%+), why not jump directly to it?"*

### 7.1 The Trade-off Matrix

| Factor | Direct to V7 (Phase 4) | Phased Approach (Option C) |
|--------|------------------------|---------------------------|
| **Time to 95%** | 2-3 weeks | 1 week |
| **Time to any improvement** | 2-3 weeks | 5 minutes (threshold) |
| **Risk** | HIGH (new architecture) | LOW (proven cascade) |
| **User pain continues** | 2-3 weeks | Stops in 24 hours |
| **Resource cost** | High (new infrastructure) | Low (retrain existing) |
| **Guarantee of success** | None | Phase 2 proven pattern |

### 7.2 Five Key Reasons for Phased Approach

#### 1. 🎯 Time-to-Value (Users Are Suffering NOW)
- **87 recent front misclassifications** = users actively struggling
- Threshold hotfix (Phase 1) takes 5 minutes, reduces errors immediately
- Waiting 2-3 weeks for V7 means 2-3 more weeks of user complaints
- **95% accuracy THIS WEEK** is better than **98% in 3 weeks**

#### 2. 📊 Prove the Cascade Can't Work FIRST
- Stage 1 is already **100% accurate** (perfect CNIE detection)
- Stage 2 has **220+ hard negatives** to learn from
- If retrained Stage 2 reaches 95%, you **SAVE weeks of work**
- Only abandon cascade IF Phase 2 fails
- Don't fix what isn't broken - the cascade architecture is sound

#### 3. 💰 Resource Efficiency
- Phase 2 uses **existing infrastructure** (just retrain Stage 2)
- Phase 4 requires:
  - New architecture design
  - New training pipeline
  - New inference engine
  - New API endpoints
  - Full regression testing
- If Phase 2 works, you **skip ALL of Phase 4's overhead**

#### 4. 🧪 Risk Mitigation
- V7 (3-class) is **UNPROVEN** - might not reach 98%
- Could introduce NEW failure modes (no-card classification regression)
- Stage 2 retraining is **PROVEN** - V5 went from 72% → 88% with feedback
- With 220+ samples, 95% is very achievable
- If Phase 2 fails, you still have V7 as backup

#### 5. 📈 Data Utilization
- Your **220+ misclassified images** are PERFECT for Stage 2 retraining
- They directly teach the model its failure modes
- V7 would need to relearn CNIE detection from scratch
- Stage 1 (100% accurate) is **WASTED** if you abandon cascade

### 7.3 When to Skip Directly to Phase 4 (V7)

Go directly to V7 **ONLY if**:

| Condition | Rationale |
|-----------|-----------|
| Phase 2 fails to reach 90%+ | After using all 220+ misclassified images, weighted loss, and augmentations |
| Cascade is fundamentally limited | Consistent front/back confusion regardless of training data |
| You have 3+ weeks before production | Can afford time to design, train, validate new architecture |
| You need >95% accuracy | Cascade ceiling appears to be ~95%, V7 target is 98%+ |

### 7.4 Decision Flowchart

```
START
  │
  ▼
┌────────────────────────┐     NO     ┌────────────────────────┐
│ Can you wait 3 weeks   │───────────►│ Do Phase 1 (today)     │
│ for 98% accuracy?      │            │ Threshold 0.45         │
└────────────────────────┘            └────────────────────────┘
         │ YES                                    │
         ▼                                        ▼
┌────────────────────────┐            ┌────────────────────────┐
│ Do you have 220+       │            │ Do Phase 2 (this week) │
│ misclassified images?  │            │ Retrain Stage 2        │
└────────────────────────┘            └────────────────────────┘
         │ NO                                       │
         ▼                                          ▼
┌────────────────────────┐     ┌────────────────────────────────┐
│ Go to Phase 4 (V7)     │     │ Did Phase 2 reach 95%?         │
│ 3-class model          │     └────────────────────────────────┘
└────────────────────────┘            │ YES            │ NO     │
                                      ▼                ▼        │
                               ┌──────────┐    ┌──────────┐     │
                               │ DEPLOY   │    │ Go to    │◄────┘
                               │ V6.1     │    │ Phase 4  │
                               └──────────┘    └──────────┘
```

### 7.5 The Win-Win Argument

**Phased approach is WIN-WIN:**
- **WIN:** If Phase 2 succeeds → You have 95% accuracy in 1 week (saved 2 weeks)
- **WIN:** If Phase 2 fails → You lost only 1 week, still have V7 as backup

**Direct to V7 is LOSE-WIN:**
- **LOSE:** If V7 has issues → You spent 3 weeks, users suffered, and need fallback
- **WIN:** If V7 works → You have 98% accuracy (but 2-3 weeks later)

## 8. Final Recommendation

### 8.1 Decision: **Implement Hybrid Option C**

**Rationale:**
1. **Immediate relief:** Threshold hotfix addresses user pain today
2. **Data-driven:** 222 front misclassifications provide rich training data
3. **Incremental validation:** Prove cascade can work before redesigning
4. **Risk mitigation:** Don't abandon working architecture prematurely
5. **Win-win:** Fast solution now, with V7 as backup plan

### 8.2 Action Items

#### TODAY (Phase 1)
- [ ] Update inference threshold to 0.45 for front
- [ ] Deploy hotfix to production
- [ ] Monitor front classification rate

#### THIS WEEK (Phase 2)
- [ ] Prepare V6.1 dataset (550 train / 80 val / 57 test)
- [ ] Implement all 8 augmentations (synthesized list)
- [ ] Configure weighted loss [1.2, 0.9]
- [ ] Train on Colab with MixUp/CutMix
- [ ] Validate on held-out test set
- [ ] Target: ≥95% front accuracy

#### NEXT WEEK (Phase 3)
- [ ] A/B test V6.1 vs V6
- [ ] Collect new feedback metrics
- [ ] Evaluate if ≥95% achieved
- [ ] **Decision point:** Continue with cascade or design V7 3-class?

#### IF NEEDED (Phase 4)
- [ ] Design 3-class architecture
- [ ] Plan active learning pipeline
- [ ] Estimate resource requirements

---

## 9. Success Metrics

| Metric | V6 Current | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|------------|----------------|----------------|----------------|
| Front Accuracy | 72% | 82% | ≥95% | ≥95% sustained |
| Front Recall | 72% | 82% | ≥95% | ≥95% sustained |
| Front Precision | 84% | 80% | ≥93% | ≥93% sustained |
| Misclass Ratio (F:B) | 2.7:1 | 2.0:1 | 1.2:1 | 1.0:1 |
| User Complaints | High | Medium | Low | Minimal |

---

## 10. Conclusion

**The V7_TRAIN_STRATEGY.md and FEEDBACK_TESTING_REPORT_V6.md are complementary, not contradictory.**

- **V7_TRAIN_STRATEGY.md** provides the detailed training recipe for immediate improvement
- **FEEDBACK_TESTING_REPORT_V6.md** provides the analysis context and long-term architectural vision

**With 222 front misclassifications now identified (not 24), the data strongly supports aggressive retraining.** The hybrid approach maximizes short-term gains while preserving optionality for architectural changes if needed.

**Decision:** Proceed with Hybrid Option C - Threshold hotfix today, Stage 2 retraining this week, V7 evaluation next week.

---

**Report Generated:** March 20, 2026  
**Updated:** With Section 7 rationale for phased approach  
**Decision Status:** Pending Approval  
**Next Review:** After Phase 2 completion
