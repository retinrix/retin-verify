# COURS: Gradient Derivation Demonstration
## ∂L/∂w = -2x(y - (wx + b))

---

## 1. Mathematical Derivation

### Starting Point: Mean Squared Error (MSE) Loss

For a single data point, the loss is:

```
L = (y - ŷ)²

where:
- y = true value (target)
- ŷ = predicted value = wx + b
- w = weight (what we want to update)
- b = bias
- x = input feature
```

### Step-by-Step Derivation

**Step 1: Write the full loss function**
```
L = (y - (wx + b))²
```

**Step 2: Apply the chain rule**

Let u = y - (wx + b), so L = u²

```
∂L/∂w = ∂L/∂u · ∂u/∂w        (Chain Rule)

∂L/∂u = 2u = 2(y - (wx + b))  (Power rule)

∂u/∂w = ∂/∂w[y - wx - b]
      = 0 - x - 0              (y and b are constant w.r.t. w)
      = -x
```

**Step 3: Multiply the partials**
```
∂L/∂w = 2(y - (wx + b)) · (-x)

∂L/∂w = -2x(y - (wx + b))
```

---

## 2. Numerical Example

### Setup
```python
# Given values
x = 2.0        # Input feature (e.g., pixel intensity)
y = 5.0        # True target (e.g., actual label)
w = 1.0        # Current weight
b = 0.5        # Current bias
```

### Forward Pass (Prediction)
```
ŷ = wx + b
ŷ = 1.0 × 2.0 + 0.5
ŷ = 2.5
```

### Calculate Loss
```
L = (y - ŷ)²
L = (5.0 - 2.5)²
L = (2.5)²
L = 6.25
```

### Calculate Gradient ∂L/∂w
```
∂L/∂w = -2x(y - (wx + b))
∂L/∂w = -2 × 2.0 × (5.0 - (1.0 × 2.0 + 0.5))
∂L/∂w = -4.0 × (5.0 - 2.5)
∂L/∂w = -4.0 × 2.5
∂L/∂w = -10.0
```

### Interpretation

| Value | Meaning |
|-------|---------|
| ∂L/∂w = **-10.0** | Loss decreases as w increases |
| Negative sign | We need to increase w to reduce loss |
| Magnitude (10) | Large gradient = steep slope = big update needed |

---

## 3. Python Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_gradient(x, y, w, b):
    """
    Compute ∂L/∂w manually
    L = (y - (wx + b))²
    ∂L/∂w = -2x(y - (wx + b))
    """
    # Prediction
    y_hat = w * x + b
    
    # Error (residual)
    error = y - y_hat
    
    # Gradient
    gradient = -2 * x * error
    
    return gradient, y_hat, error

def compute_gradient_numerical(x, y, w, b, epsilon=1e-5):
    """
    Verify using numerical differentiation
    ∂L/∂w ≈ (L(w + ε) - L(w - ε)) / (2ε)
    """
    # L(w + ε)
    w_plus = w + epsilon
    L_plus = (y - (w_plus * x + b))**2
    
    # L(w - ε)
    w_minus = w - epsilon
    L_minus = (y - (w_minus * x + b))**2
    
    # Numerical gradient
    grad_numerical = (L_plus - L_minus) / (2 * epsilon)
    
    return grad_numerical

# ============== DEMONSTRATION ==============

print("=" * 60)
print("GRADIENT CALCULATION DEMONSTRATION")
print("=" * 60)

# Example values
x = 2.0
y = 5.0
w = 1.0
b = 0.5
learning_rate = 0.1

print(f"\n📊 Input Values:")
print(f"  x (input)     = {x}")
print(f"  y (target)    = {y}")
print(f"  w (weight)    = {w}")
print(f"  b (bias)      = {b}")

# Compute gradient
gradient, y_hat, error = compute_gradient(x, y, w, b)

print(f"\n🔮 Forward Pass:")
print(f"  ŷ = wx + b = {w} × {x} + {b} = {y_hat}")

print(f"\n📉 Loss Calculation:")
print(f"  L = (y - ŷ)² = ({y} - {y_hat})² = {error}² = {(y - y_hat)**2:.4f}")

print(f"\n🔍 Gradient Calculation (∂L/∂w):")
print(f"  ∂L/∂w = -2x(y - (wx + b))")
print(f"  ∂L/∂w = -2 × {x} × ({y} - ({w} × {x} + {b}))")
print(f"  ∂L/∂w = -2 × {x} × {error}")
print(f"  ∂L/∂w = {gradient:.4f}")

# Verify with numerical gradient
grad_numerical = compute_gradient_numerical(x, y, w, b)
print(f"\n✅ Verification (Numerical):")
print(f"  Numerical gradient = {grad_numerical:.4f}")
print(f"  Analytical gradient = {gradient:.4f}")
print(f"  Difference = {abs(gradient - grad_numerical):.10f} (should be ~0)")

# Gradient Descent Update
w_new = w - learning_rate * gradient
print(f"\n⚡ Gradient Descent Update:")
print(f"  w_new = w - lr × ∂L/∂w")
print(f"  w_new = {w} - {learning_rate} × {gradient}")
print(f"  w_new = {w_new:.4f}")

# Verify loss decreased
loss_old = (y - (w * x + b))**2
loss_new = (y - (w_new * x + b))**2
print(f"\n📉 Loss Improvement:")
print(f"  Loss before = {loss_old:.4f}")
print(f"  Loss after  = {loss_new:.4f}")
print(f"  Improvement = {loss_old - loss_new:.4f} ✓")
```

---

## 4. Visualizing the Gradient

```python
# Create visualization
w_values = np.linspace(-2, 4, 100)
loss_values = []
gradient_values = []

for w_test in w_values:
    # Compute loss
    y_hat = w_test * x + b
    loss = (y - y_hat)**2
    loss_values.append(loss)
    
    # Compute gradient
    grad, _, _ = compute_gradient(x, y, w_test, b)
    gradient_values.append(grad)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(w_values, loss_values, 'b-', linewidth=2, label='Loss L(w)')
ax1.scatter([w], [(y - (w * x + b))**2], color='red', s=100, zorder=5, label=f'Current w={w}')
ax1.set_xlabel('Weight (w)', fontsize=12)
ax1.set_ylabel('Loss L(w)', fontsize=12)
ax1.set_title('Loss vs Weight', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mark minimum
w_optimal = (y - b) / x  # Where gradient = 0
ax1.axvline(w_optimal, color='green', linestyle='--', alpha=0.5, label=f'Optimal w={w_optimal:.2f}')
ax1.legend()

# Gradient curve
ax2.plot(w_values, gradient_values, 'r-', linewidth=2, label='∂L/∂w')
ax2.scatter([w], [gradient], color='blue', s=100, zorder=5, label=f'Current ∇={gradient:.2f}')
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.axvline(w_optimal, color='green', linestyle='--', alpha=0.5, label=f'Optimal w={w_optimal:.2f}')
ax2.set_xlabel('Weight (w)', fontsize=12)
ax2.set_ylabel('Gradient ∂L/∂w', fontsize=12)
ax2.set_title('Gradient vs Weight', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_visualization.png', dpi=150)
plt.show()

print(f"\n📊 Visualization shows:")
print(f"  - Loss is minimized at w = {w_optimal:.2f} (where gradient = 0)")
print(f"  - Current w = {w}, gradient = {gradient:.2f}")
print(f"  - Negative gradient → need to INCREASE w")
```

---

## 5. Interactive Understanding

### Effect of Each Variable

```python
print("\n" + "=" * 60)
print("HOW EACH VARIABLE AFFECTS THE GRADIENT")
print("=" * 60)

base_x, base_y, base_w, base_b = 2.0, 5.0, 1.0, 0.5

# 1. Effect of x (input)
print("\n1️⃣ Effect of x (input magnitude):")
x_values = [0.5, 1.0, 2.0, 4.0]
for x_test in x_values:
    grad, _, _ = compute_gradient(x_test, base_y, base_w, base_b)
    print(f"  x = {x_test:4.1f} → ∂L/∂w = {grad:7.2f}")
print("  ↑ Larger input → Larger gradient → Bigger update")

# 2. Effect of error
print("\n2️⃣ Effect of error (y - ŷ):")
y_values = [2.5, 3.5, 5.0, 7.0]  # Different targets
for y_test in y_values:
    grad, y_hat, error = compute_gradient(base_x, y_test, base_w, base_b)
    print(f"  y = {y_test:4.1f} (error = {error:5.2f}) → ∂L/∂w = {grad:7.2f}")
print("  ↑ Larger error → Larger gradient → Bigger correction needed")

# 3. Effect of sign
print("\n3️⃣ Effect of gradient sign:")
test_cases = [
    (2.0, 5.0),   # y > ŷ (underpredicting)
    (2.0, 2.5),   # y < ŷ (overpredicting)
    (2.0, 2.5),   # y = ŷ (perfect prediction)
]
for x_test, y_test in test_cases:
    grad, y_hat, error = compute_gradient(x_test, y_test, base_w, base_b)
    direction = "INCREASE w" if grad < 0 else "DECREASE w"
    print(f"  y = {y_test}, ŷ = {y_hat:.1f}, error = {error:+.1f} → ∂L/∂w = {grad:+6.2f} → {direction}")
```

---

## 6. Why This Formula Makes Sense

### Intuitive Explanation

```
∂L/∂w = -2x(y - ŷ)
        └──┘└─────┘
         │     │
         │     └── Error: How far off is our prediction?
         │         Large error → Large update needed
         │
         └── Input: How much did this feature contribute?
             Large x → Feature is important → Weight update matters more
```

### Physical Analogy: Learning to Throw a Ball

Imagine learning to throw a ball distance `y` by adjusting arm strength `w`:

- **x** = Weight of the ball (heavier ball = more sensitive to arm strength)
- **y** = Target distance (how far you want to throw)
- **ŷ** = Actual distance (how far you actually threw)
- **∂L/∂w** = How much to adjust arm strength

**Cases:**
1. **Heavy ball (large x)** + **Far miss (large error)** = **BIG adjustment**
2. **Light ball (small x)** + **Close miss (small error)** = **Small adjustment**
3. **Hit target (error = 0)** = **No adjustment needed**

---

## 7. Extension to Multiple Samples (Batch)

For a batch of N samples:

```python
def compute_batch_gradient(X, Y, w, b):
    """
    X: array of shape (N,) - N input samples
    Y: array of shape (N,) - N targets
    Returns average gradient over batch
    """
    N = len(X)
    
    # Predictions
    Y_hat = w * X + b
    
    # Errors
    errors = Y - Y_hat
    
    # Gradients for each sample
    gradients = -2 * X * errors
    
    # Average gradient (for SGD/batch GD)
    avg_gradient = np.mean(gradients)
    
    return avg_gradient

# Example batch
X_batch = np.array([1.0, 2.0, 3.0, 4.0])
Y_batch = np.array([2.0, 4.0, 6.0, 8.0])
w, b = 0.5, 0.0

grad_batch = compute_batch_gradient(X_batch, Y_batch, w, b)
print(f"Batch gradient: {grad_batch:.4f}")

# This is what happens in PyTorch:
# loss = criterion(predictions, targets)
# loss.backward()  # Computes ∂L/∂w for all parameters
```

---

## 8. Connection to PyTorch Autograd

```python
import torch

# PyTorch automatically computes this for us!
x = torch.tensor(2.0)
y = torch.tensor(5.0)
w = torch.tensor(1.0, requires_grad=True)  # Track gradients
b = torch.tensor(0.5, requires_grad=True)

# Forward pass
y_hat = w * x + b
loss = (y - y_hat)**2

print(f"PyTorch computation:")
print(f"  y_hat = {y_hat.item()}")
print(f"  loss = {loss.item()}")

# Backward pass - computes all gradients automatically
loss.backward()

print(f"\nGradients computed by PyTorch:")
print(f"  ∂L/∂w = {w.grad.item():.4f}")
print(f"  ∂L/∂b = {b.grad.item():.4f}")

# Verify with our manual formula
manual_grad_w = -2 * x.item() * (y.item() - y_hat.item())
manual_grad_b = -2 * (y.item() - y_hat.item())

print(f"\nManual verification:")
print(f"  ∂L/∂w = {manual_grad_w:.4f}")
print(f"  ∂L/∂b = {manual_grad_b:.4f}")

print(f"\n✅ Match: {torch.isclose(w.grad, torch.tensor(manual_grad_w))}")
```

---

## Summary

### Key Takeaways

1. **Formula**: ∂L/∂w = -2x(y - ŷ)
   - Negative sign: Increase w to decrease loss (when y > ŷ)
   - x multiplier: Important features get larger updates
   - Error multiplier: Bigger mistakes get bigger corrections

2. **Verification Methods**:
   - Analytical (formula)
   - Numerical (finite differences)
   - PyTorch autograd (automatic)

3. **Physical Meaning**:
   - Gradient points in direction of steepest ascent
   - Negative gradient points toward minimum
   - Magnitude indicates update size

4. **In Practice**:
   - PyTorch handles this automatically via `backward()`
   - Understanding helps with debugging and optimization
   - Extension to matrices is conceptually similar

---

**Exercise**: Try different values of x, y, w, b and observe how the gradient changes. What happens when:
- The prediction is perfect (y = ŷ)?
- The input x is zero?
- The weight w is already optimal?
