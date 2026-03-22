# COURS: Cross-Entropy Loss - Short Explanation

---

## The Problem

You have probabilities from Softmax: `[0.7, 0.2, 0.1]`
True label is: `[1, 0, 0]` (Class 0 is correct)

**How "wrong" is your prediction?**

---

## The Formula

```
L = -Σ yᵢ × log(ŷᵢ)

Where:
- yᵢ = 1 if class i is correct, 0 otherwise
- ŷᵢ = predicted probability for class i (from softmax)
- log = natural logarithm
```

---

## Component Breakdown

| Component | Meaning | Purpose |
|-----------|---------|---------|
| `yᵢ` | True label (0 or 1) | Only correct class matters |
| `log(ŷᵢ)` | Log of predicted probability | Penalize wrong guesses heavily |
| `-` | Negative sign | Make loss positive |
| `Σ` | Sum | Aggregate across classes |

---

## Why log()?

**Property: log(x) approaches -∞ as x approaches 0**

| Prediction | log(pred) | Loss |
|------------|-----------|------|
| 0.99 (confident correct) | -0.01 | **0.01** ✓ Low |
| 0.50 (uncertain) | -0.69 | **0.69** Medium |
| 0.10 (wrong) | -2.30 | **2.30** High |
| 0.01 (very wrong) | -4.61 | **4.61** Very High! |

**Key insight:** Small mistakes in probability → Big penalty in loss

---

## Simple Example

**Scenario:** 3 classes, true answer is Class 0

### Case 1: Correct and Confident ✓
```
Prediction: [0.9, 0.05, 0.05]
True:       [1,   0,    0   ]

L = -(1 × log(0.9) + 0 × log(0.05) + 0 × log(0.05))
L = -log(0.9)
L = -(-0.105)
L = 0.105 ✓ (Low loss!)
```

### Case 2: Wrong and Confident ✗
```
Prediction: [0.05, 0.9, 0.05]  (Very sure it's Class 1, but wrong!)
True:       [1,    0,   0   ]

L = -(1 × log(0.05) + 0 + 0)
L = -log(0.05)
L = -(-2.996)
L = 2.996 ✗ (High loss! 28× worse than correct)
```

### Case 3: Uncertain
```
Prediction: [0.4, 0.35, 0.25]
True:       [1,   0,    0   ]

L = -log(0.4)
L = 0.916 (Medium loss - model should be more confident)
```

---

## Why This Formula?

### 1. Only Correct Class Matters

Because `yᵢ` is one-hot encoded `[1, 0, 0]`, only the correct class contributes:

```
L = -log(ŷ_correct)
```

All other terms are multiplied by 0 and disappear!

### 2. Heavily Penalizes Wrong Confidence

| Confidence in WRONG class | Loss |
|---------------------------|------|
| 50% | 0.69 |
| 90% | 2.30 |
| 99% | 4.61 |

**The model pays heavily for being confidently wrong!**

### 3. Information Theory Connection

Cross-entropy measures the "surprise" of seeing the true label given your prediction.

- Low loss = Not surprised (prediction matches reality)
- High loss = Very surprised (prediction was wrong)

---

## The Magic: Softmax + Cross-Entropy

When combined, the gradient becomes beautifully simple:

```
∂L/∂zᵢ = ŷᵢ - yᵢ

Meaning:
- Prediction - True label
```

| Scenario | ŷ | y | Gradient | Action |
|----------|---|---|----------|--------|
| Perfect | 1.0 | 1 | 0 | Do nothing ✓ |
| Under-confident correct | 0.6 | 1 | -0.4 | Increase this logit ↑ |
| Over-confident wrong | 0.9 | 0 | +0.9 | Decrease this logit ↓ |

**This is why Softmax + Cross-Entropy is the standard!**

---

## Summary

| Question | Answer |
|----------|--------|
| **What does it measure?** | How wrong the prediction is |
| **Why log()?** | Punish wrong confidence heavily |
| **Why negative?** | Make loss positive (log of prob < 0) |
| **Why sum?** | Handle multi-class |
| **Best case?** | L = 0 (100% confident and correct) |
| **Worst case?** | L = ∞ (100% confident and wrong) |

---

## One-Line Intuition

> **"Cross-entropy measures how surprised you should be by the true answer, given your prediction"**

- Not surprised (low loss) = Good prediction
- Very surprised (high loss) = Bad prediction
