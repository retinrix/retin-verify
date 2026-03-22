# COURS: EfficientNet Masterclass - From Zero to Hero

**Course Overview:** Complete deep dive into EfficientNet architecture, from mathematical foundations to practical implementation in the Retin-Verify project.

**Target Audience:** Developers with basic Python knowledge wanting to understand CNNs and EfficientNet

**Prerequisites:** Basic Python, high school math (algebra, basic calculus)

---

# TABLE OF CONTENTS

1. [Chapter 1: Mathematical Foundations](#chapter-1-mathematical-foundations)
2. [Chapter 2: Python & Deep Learning Libraries](#chapter-2-python--deep-learning-libraries)
3. [Chapter 3: Neural Network Fundamentals](#chapter-3-neural-network-fundamentals)
4. [Chapter 4: Convolutional Neural Networks (CNNs)](#chapter-4-convolutional-neural-networks-cnns)
5. [Chapter 5: EfficientNet Architecture Deep Dive](#chapter-5-efficientnet-architecture-deep-dive)
6. [Chapter 6: Training Process & Optimization](#chapter-6-training-process--optimization)
7. [Chapter 7: Implementation in Retin-Verify](#chapter-7-implementation-in-retin-verify)
8. [Chapter 8: Performance Analysis & Expectations](#chapter-8-performance-analysis--expectations)
9. [Chapter 9: Advanced Topics & Fine-tuning](#chapter-9-advanced-topics--fine-tuning)
10. [Chapter 10: Practical Exercises](#chapter-10-practical-exercises)

---

# Chapter 1: Mathematical Foundations

## 1.1 Linear Algebra - The Language of Deep Learning

### Vectors and Matrices

**Vectors** are 1D arrays of numbers. In Python:
```python
import numpy as np

# A vector representing pixel intensities
pixel_vector = np.array([255, 128, 64, 32, 16])
print(f"Vector shape: {pixel_vector.shape}")  # (5,)
print(f"Vector: {pixel_vector}")
```

**Matrices** are 2D arrays. An image is a matrix:
```python
# A 3x3 grayscale image (matrix)
image_matrix = np.array([
    [255, 128, 64],
    [128, 64, 32],
    [64, 32, 16]
])
print(f"Matrix shape: {image_matrix.shape}")  # (3, 3)
```

### Matrix Multiplication - The Core Operation

Matrix multiplication is the fundamental operation in neural networks:

```
Given: A (m×n) and B (n×p)
Result: C (m×p) where C[i,j] = Σ(A[i,k] × B[k,j])
```

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2
B = np.array([[7, 8, 9], [10, 11, 12]])  # 2×3

C = np.dot(A, B)  # or A @ B
print(f"Result shape: {C.shape}")  # (3, 3)
```

**In Neural Networks:**
- Input: Vector of features (e.g., 224×224×3 = 150,528 pixels)
- Weights: Matrix learned during training
- Output: Transformed features

### Dot Product

The dot product measures similarity between vectors:

```python
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)  # 1×4 + 2×5 + 3×6 = 32
```

## 1.2 Calculus for Deep Learning

### Derivatives - Understanding Gradients

**Derivative** = Rate of change (slope)

```python
import matplotlib.pyplot as plt

# Function: f(x) = x²
# Derivative: f'(x) = 2x

def f(x):
    return x ** 2

def derivative_f(x):
    return 2 * x

x = np.linspace(-5, 5, 100)
y = f(x)
dy_dx = derivative_f(x)

# At x=3, slope = 6 (steep upward)
# At x=0, slope = 0 (minimum point)
```

### Partial Derivatives

When a function has multiple inputs, we take partial derivatives:

```python
# Loss function: L(w, b) = (y - (wx + b))²
# ∂L/∂w = -2x(y - (wx + b))  <- How loss changes with weight
# ∂L/∂b = -2(y - (wx + b))   <- How loss changes with bias
```

### Chain Rule - Backpropagation Foundation

If `z = f(g(x))`, then `dz/dx = dz/dg × dg/dx`

```python
# Example: z = (2x + 1)²
# Let u = 2x + 1, then z = u²
# dz/dx = dz/du × du/dx = 2u × 2 = 4(2x + 1)

def z(x):
    u = 2 * x + 1
    return u ** 2

# Derivative: 4(2x + 1)
# At x=1: 4(3) = 12
```

## 1.3 Statistics & Probability

### Mean and Standard Deviation

**Normalization** is crucial for neural networks:

```python
data = np.array([10, 20, 30, 40, 50])

mean = np.mean(data)      # 30
std = np.std(data)        # ~14.14

# Normalize: (x - mean) / std
normalized = (data - mean) / std
# Result: [-1.41, -0.71, 0, 0.71, 1.41]
```

**Why normalize?**
- Helps gradients flow better
- Prevents exploding/vanishing gradients
- Makes training faster and more stable

### Probability & Softmax

Softmax converts raw scores to probabilities:

```python
def softmax(x):
    """Convert logits to probabilities"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

# Example: Raw model outputs (logits)
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)
# Result: [0.66, 0.24, 0.10] (sum = 1.0)
```

### Cross-Entropy Loss

Measures difference between predicted and true distribution:

```python
def cross_entropy_loss(y_true, y_pred):
    """
    y_true: One-hot encoded [0, 1, 0] (class 1)
    y_pred: Softmax probabilities [0.2, 0.7, 0.1]
    """
    # Only the correct class contributes
    return -np.sum(y_true * np.log(y_pred + 1e-8))

# Example
y_true = np.array([0, 1, 0])  # Correct class is index 1
y_pred = np.array([0.2, 0.7, 0.1])

loss = cross_entropy_loss(y_true, y_pred)
print(f"Loss: {loss:.4f}")  # ~0.3567

# Perfect prediction
y_pred_perfect = np.array([0.0, 1.0, 0.0])
loss_perfect = cross_entropy_loss(y_true, y_pred_perfect)
print(f"Perfect loss: {loss_perfect:.4f}")  # ~0.0
```

---

# Chapter 2: Python & Deep Learning Libraries

## 2.1 NumPy - The Foundation

### Array Operations

```python
import numpy as np

# Creating arrays
zeros = np.zeros((3, 3))          # 3×3 matrix of zeros
ones = np.ones((2, 4))            # 2×4 matrix of ones
random = np.random.randn(3, 3)    # Random values from normal distribution
arange = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]

# Reshaping (crucial for images)
image_flat = np.random.randn(224 * 224 * 3)  # 150,528 elements
image_3d = image_flat.reshape(224, 224, 3)   # Back to image shape

# Broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])

# Add vector to each row
result = matrix + vector
# [[11, 22, 33],
#  [14, 25, 36]]
```

### Array Indexing and Slicing

```python
# Image shape: (Height, Width, Channels)
image = np.random.randn(224, 224, 3)

# Get red channel only
red_channel = image[:, :, 0]  # Shape: (224, 224)

# Crop center 112×112
center = image[56:168, 56:168, :]  # Shape: (112, 112, 3)

# Batch of images: (Batch, H, W, C)
batch = np.random.randn(32, 224, 224, 3)  # 32 images
first_image = batch[0]  # Shape: (224, 224, 3)
```

## 2.2 PyTorch - Deep Learning Framework

### Tensors

Tensors are multi-dimensional arrays with GPU support:

```python
import torch

# Creating tensors
scalar = torch.tensor(3.14)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])

# GPU tensor (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_tensor = torch.randn(3, 3).to(device)

# Converting between NumPy and PyTorch
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
numpy_from_tensor = tensor_from_numpy.numpy()
```

### Autograd - Automatic Differentiation

PyTorch automatically computes gradients:

```python
# Create a tensor that requires gradient
x = torch.tensor(2.0, requires_grad=True)

# Define computation
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

# Compute gradient
y.backward()

# dy/dx at x=2
print(f"Gradient: {x.grad}")  # 2*2 + 3 = 7
```

**Real Example - Linear Regression:**

```python
# Generate data: y = 2x + 1 + noise
x = torch.randn(100, 1)
y_true = 2 * x + 1 + 0.1 * torch.randn(100, 1)

# Initialize parameters
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    y_pred = w * x + b
    
    # Compute loss (MSE)
    loss = torch.mean((y_pred - y_true) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Update parameters (gradient descent)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")
```

### Neural Network Modules

```python
import torch.nn as nn
import torch.nn.functional as F

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(784, 256)   # Input: 28×28 image
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model
model = SimpleNet(num_classes=2)
print(model)

# Forward pass
input_batch = torch.randn(32, 784)  # Batch of 32 images
output = model(input_batch)
print(f"Output shape: {output.shape}")  # (32, 2)
```

## 2.3 torchvision - Computer Vision

### Data Transforms

```python
from torchvision import transforms

# Common transforms for EfficientNet
transform = transforms.Compose([
    # Resize to 224×224 (EfficientNet input size)
    transforms.Resize(256),
    transforms.CenterCrop(224),
    
    # Convert PIL Image to Tensor (0-255 -> 0-1)
    transforms.ToTensor(),
    
    # Normalize with ImageNet statistics
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

# Apply to image
from PIL import Image
image = Image.open("document.jpg")
tensor_image = transform(image)  # Shape: (3, 224, 224)
```

### Pre-trained Models

```python
from torchvision import models

# Load pre-trained EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)

# View architecture
print(model)

# Modify for your task (2 classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Move to GPU
model = model.to('cuda')
```

## 2.4 Other Essential Libraries

```python
# PIL/Pillow - Image processing
from PIL import Image, ImageEnhance

img = Image.open("image.jpg")
img_resized = img.resize((224, 224))
img_gray = img.convert('L')  # Grayscale

# Matplotlib - Visualization
import matplotlib.pyplot as plt

plt.imshow(image_array)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# tqdm - Progress bars
from tqdm import tqdm

for epoch in tqdm(range(100), desc="Training"):
    # Training code
    pass

# TensorBoard - Logging
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', loss, epoch)
writer.add_image('Input', image, epoch)
writer.close()
```

---

# Chapter 3: Neural Network Fundamentals

## 3.1 The Neuron - Building Block

A neuron performs: `output = activation(w·x + b)`

```python
class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = 0.0
    
    def forward(self, x):
        # Weighted sum + bias
        z = np.dot(self.weights, x) + self.bias
        # Apply activation (ReLU)
        return max(0, z)  # ReLU

# Create and use neuron
neuron = Neuron(num_inputs=3)
input_vector = np.array([1.0, 2.0, 3.0])
output = neuron.forward(input_vector)
print(f"Neuron output: {output}")
```

## 3.2 Activation Functions

### ReLU (Rectified Linear Unit)

```python
def relu(x):
    return np.maximum(0, x)

# Plot
x = np.linspace(-5, 5, 100)
y = relu(x)
# Simple, fast, solves vanishing gradient
# Problem: "Dead ReLU" (neurons can get stuck at 0)
```

### Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Maps any value to (0, 1)
# Used for binary classification output
# Problem: Vanishing gradients for extreme inputs
```

### Softmax (for Multi-class)

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Converts logits to probabilities
# Sum of outputs = 1
# Used for multi-class classification
```

## 3.3 Layers - Connecting Neurons

### Fully Connected (Dense) Layer

```python
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
    
    def forward(self, x):
        # x: (batch_size, input_size)
        # weights: (input_size, output_size)
        return np.dot(x, self.weights) + self.bias

# Example
layer = DenseLayer(784, 256)  # MNIST image to 256 neurons
batch_input = np.random.randn(32, 784)  # 32 images
output = layer.forward(batch_input)
print(f"Output shape: {output.shape}")  # (32, 256)
```

## 3.4 Forward and Backward Propagation

### Forward Pass

```python
class SimpleNN:
    def __init__(self):
        self.layer1 = DenseLayer(2, 3)
        self.layer2 = DenseLayer(3, 1)
    
    def forward(self, x):
        # Layer 1: Linear -> ReLU
        z1 = self.layer1.forward(x)
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2: Linear -> Sigmoid
        z2 = self.layer2.forward(a1)
        output = sigmoid(z2)
        
        return output
```

### Backward Pass (Backpropagation)

```python
def backward(self, x, y_true, y_pred):
    """
    Compute gradients using chain rule
    """
    # Output layer gradient
    d_loss = y_pred - y_true  # Derivative of MSE
    
    # Layer 2 gradients
    d_layer2 = d_loss * sigmoid_derivative(z2)
    grad_w2 = np.dot(a1.T, d_layer2)
    grad_b2 = np.sum(d_layer2, axis=0)
    
    # Layer 1 gradients (chain rule)
    d_layer1 = np.dot(d_layer2, self.layer2.weights.T) * relu_derivative(z1)
    grad_w1 = np.dot(x.T, d_layer1)
    grad_b1 = np.sum(d_layer1, axis=0)
    
    return grad_w1, grad_b1, grad_w2, grad_b2
```

## 3.5 Loss Functions

### Mean Squared Error (Regression)

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
```

### Cross-Entropy Loss (Classification)

```python
def cross_entropy(y_true, y_pred):
    # Clip to prevent log(0)
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
```

## 3.6 Gradient Descent Optimization

```python
def gradient_descent(params, grads, learning_rate):
    """
    params: list of parameters (weights, biases)
    grads: list of gradients
    learning_rate: step size
    """
    for param, grad in zip(params, grads):
        param -= learning_rate * grad
    return params

# Variants:
# - SGD: Basic gradient descent
# - Momentum: Adds velocity term (smoother convergence)
# - Adam: Adaptive learning rates per parameter
```

---

# Chapter 4: Convolutional Neural Networks (CNNs)

## 4.1 Why CNNs for Images?

**Problem with Dense Networks:**
- 224×224 image = 50,176 pixels
- First layer with 1000 neurons = 50M parameters!
- No spatial awareness

**CNN Solution:**
- Local connectivity (receptive fields)
- Weight sharing (same filter across image)
- Translation invariance

## 4.2 Convolution Operation

```python
def conv2d(input_tensor, kernel, stride=1, padding=0):
    """
    Manual 2D convolution implementation
    
    input_tensor: (H, W) - single channel
    kernel: (kH, kW) - filter
    """
    if padding > 0:
        input_tensor = np.pad(input_tensor, padding, mode='constant')
    
    H, W = input_tensor.shape
    kH, kW = kernel.shape
    
    outH = (H - kH) // stride + 1
    outW = (W - kW) // stride + 1
    
    output = np.zeros((outH, outW))
    
    for i in range(outH):
        for j in range(outW):
            # Extract patch
            patch = input_tensor[i*stride:i*stride+kH, j*stride:j*stride+kW]
            # Element-wise multiply and sum
            output[i, j] = np.sum(patch * kernel)
    
    return output

# Example: Edge detection kernel
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Apply to image
from PIL import Image
image = np.array(Image.open("document.jpg").convert('L'))
edges = conv2d(image, edge_kernel)
```

## 4.3 Convolution in PyTorch

```python
import torch.nn as nn

# Define a convolutional layer
conv_layer = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=32,    # 32 filters
    kernel_size=3,      # 3×3 filter
    stride=1,           # Step size
    padding=1           # Zero padding (keeps size)
)

# Apply to batch of images
batch = torch.randn(8, 3, 224, 224)  # (B, C, H, W)
output = conv_layer(batch)
print(f"Output shape: {output.shape}")  # (8, 32, 224, 224)

# Number of parameters
params = sum(p.numel() for p in conv_layer.parameters())
print(f"Parameters: {params}")  # 3×3×3×32 + 32 = 896
```

## 4.4 Pooling Layers

```python
# Max Pooling - Reduces spatial dimensions
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Input: (8, 32, 224, 224)
output = max_pool(output)
# Output: (8, 32, 112, 112) - Halved spatially

# Average Pooling
avg_pool = nn.AdaptiveAvgPool2d((1, 1))
# Output: (8, 32, 1, 1) - Global average
```

## 4.5 Batch Normalization

**Problem:** Internal Covariate Shift (layer inputs change during training)

**Solution:** Normalize layer inputs

```python
bn_layer = nn.BatchNorm2d(num_features=32)

# During training: normalize using batch statistics
# During inference: normalize using running statistics

output = bn_layer(conv_output)
# Benefits:
# - Faster training
# - Higher learning rates
# - Regularization effect
```

## 4.6 Complete CNN Example

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=2)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

# Chapter 5: EfficientNet Architecture Deep Dive

## 5.1 The Problem with Previous Approaches

**ResNet, VGG, etc.:**
- Scale only one dimension: depth (layers) or width (channels)
- Inefficient resource usage
- Not optimal accuracy vs. computation trade-off

**EfficientNet Insight:**
Scale **all three dimensions** together: depth, width, and resolution

## 5.2 Compound Scaling

```python
# Compound scaling formula
def compound_scaling(phi, alpha=1.2, beta=1.1, gamma=1.15):
    """
    phi: compound coefficient (0 for B0, 1 for B1, etc.)
    alpha: depth scaling
    beta: width scaling  
    gamma: resolution scaling
    """
    depth_scale = alpha ** phi
    width_scale = beta ** phi
    resolution_scale = gamma ** phi
    
    return depth_scale, width_scale, resolution_scale

# B0 (baseline): phi=0 -> all scales = 1.0
# B1: phi=1
# B2: phi=2
# etc.

for phi in [0, 1, 2]:
    d, w, r = compound_scaling(phi)
    print(f"B{phi}: depth={d:.2f}, width={w:.2f}, resolution={r:.2f}")
```

## 5.3 EfficientNet-B0 Architecture

**Input:** 224×224×3 RGB image
**Output:** 1000 classes (ImageNet)

### Building Block: MBConv (Mobile Inverted Bottleneck Convolution)

```python
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block
    Core building block of EfficientNet
    """
    def __init__(self, in_channels, out_channels, expand_ratio, 
                 kernel_size, stride, se_ratio=0.25):
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        # Expansion phase (1×1 conv)
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)  # Swish activation
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze-and-Excitation (attention mechanism)
        if se_ratio:
            se_channels = max(1, int(hidden_dim * se_ratio))
            layers.append(
                SqueezeExcitation(hidden_dim, se_channels)
            )
        
        # Projection phase (1×1 conv)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Skip connection
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)  # Residual connection
        else:
            return self.shortcut(x) + self.conv(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block
    Learns channel-wise attention
    """
    def __init__(self, channels, reduction_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, reduction_channels, 1)
        self.fc2 = nn.Conv2d(reduction_channels, channels, 1)
    
    def forward(self, x):
        # Squeeze: Global average pooling
        scale = F.adaptive_avg_pool2d(x, 1)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        
        # Scale
        return x * scale
```

## 5.4 Complete EfficientNet-B0 Structure

```
EfficientNet-B0 Architecture:

Input: 224×224×3

Stem:
  Conv 3×3, stride 2, 32 filters → 112×112×32
  BatchNorm + SiLU

Block 1 (×1):
  MBConv1, k3×3, 16 filters, stride 1 → 112×112×16

Block 2 (×2):
  MBConv6, k3×3, 24 filters, stride 2 → 56×56×24
  MBConv6, k3×3, 24 filters, stride 1 → 56×56×24

Block 3 (×2):
  MBConv6, k5×5, 40 filters, stride 2 → 28×28×40
  MBConv6, k5×5, 40 filters, stride 1 → 28×28×40

Block 4 (×3):
  MBConv6, k3×3, 80 filters, stride 2 → 14×14×80
  MBConv6, k3×3, 80 filters, stride 1 → 14×14×80 (×2)

Block 5 (×3):
  MBConv6, k5×5, 112 filters, stride 1 → 14×14×112 (×3)

Block 6 (×4):
  MBConv6, k5×5, 192 filters, stride 2 → 7×7×192 (×4)

Block 7 (×1):
  MBConv6, k3×3, 320 filters, stride 1 → 7×7×320

Head:
  Conv 1×1, 1280 filters → 7×7×1280
  Global Average Pooling → 1×1×1280
  Dropout (0.2)
  FC: 1280 → num_classes

Total Parameters: ~5.3M (B0)
```

## 5.5 Key Design Decisions

### Depthwise Separable Convolutions

```python
# Standard convolution: expensive
# Input: (H, W, C_in), Kernel: (K, K, C_in, C_out)
# Params: K² × C_in × C_out

# Depthwise separable: cheaper
# Step 1: Depthwise (per-channel)
#   Params: K² × C_in
# Step 2: Pointwise (1×1 conv)
#   Params: C_in × C_out
# Total: K² × C_in + C_in × C_out (much smaller!)

# Example
C_in, C_out, K = 32, 64, 3
standard = K*K * C_in * C_out      # 18,432 params
separable = K*K*C_in + C_in*C_out  # 4,928 params (73% reduction!)
```

### Swish Activation (SiLU)

```python
def swish(x):
    return x * torch.sigmoid(x)
    # Smooth, non-monotonic
    # Better than ReLU for deep networks
    # PyTorch: F.silu(x) or nn.SiLU()

# Visualization
x = torch.linspace(-5, 5, 100)
y = F.silu(x)
# At x=0: y=0
# For x>>0: y≈x (like ReLU)
# For x<<0: y≈0 (smooth, not abrupt)
```

### Stochastic Depth (Drop Connect)

```python
class StochasticDepth(nn.Module):
    """
    Randomly drop residual connections during training
    Acts as regularization
    """
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
    
    def forward(self, x):
        if not self.training:
            return x
        
        # Random binary mask
        mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < self.keep_prob
        x = x / self.keep_prob * mask  # Scale to maintain expectation
        return x
```

---

# Chapter 6: Training Process & Optimization

## 6.1 Data Pipeline

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class DocumentDataset(Dataset):
    """
    Custom dataset for document classification
    """
    def __init__(self, annotations_file, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform
        
        # Map class names to indices
        self.class_to_idx = {
            'cnie_front': 0,
            'cnie_back': 1
        }
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Load image
        image_path = f"/content/data/{item['image_path']}"
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.class_to_idx[item['document_type']]
        
        return image, label

# Transforms for training
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transforms for validation (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = DocumentDataset(
    '/content/data/processed/classification/train.json',
    transform=train_transform
)
val_dataset = DocumentDataset(
    '/content/data/processed/classification/val.json',
    transform=val_transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4
)
```

## 6.2 Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc
```

## 6.3 Optimizers

### AdamW (Used in EfficientNet)

```python
# AdamW: Adam with proper weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,           # Learning rate
    weight_decay=0.01, # L2 regularization
    betas=(0.9, 0.999) # Momentum coefficients
)

# How it works:
# m_t = β1*m_{t-1} + (1-β1)*g_t     (first moment - momentum)
# v_t = β2*v_{t-1} + (1-β2)*g_t²    (second moment - adaptive LR)
# m̂_t = m_t / (1-β1^t)              (bias correction)
# v̂_t = v_t / (1-β2^t)
# θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε) - wd*θ_{t-1}
```

### Learning Rate Scheduling

```python
# Cosine Annealing with Warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

# Warmup: gradually increase LR for first few epochs
# Cosine: decay LR following cosine curve

def get_lr_scheduler(optimizer, epochs, warmup_epochs=5):
    """Cosine annealing with linear warmup"""
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer, epochs=100, warmup_epochs=5)

# Usage in training loop
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    
    scheduler.step()  # Update learning rate
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: LR={current_lr:.6f}, Val Acc={val_acc:.2f}%")
```

## 6.4 Mixed Precision Training (FP16)

```python
from torch.cuda.amp import autocast, GradScaler

# Enable mixed precision training
scaler = GradScaler()

def train_epoch_fp16(model, dataloader, criterion, optimizer, device):
    model.train()
    scaler = GradScaler()
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
# Benefits:
# - ~2x faster training on modern GPUs (Tensor Cores)
# - 50% less memory usage
# - Minimal accuracy loss
```

## 6.5 Regularization Techniques

### Dropout

```python
# Randomly zero out neurons during training
self.dropout = nn.Dropout(p=0.5)  # 50% dropout

# In forward pass
x = self.dropout(x)

# At inference: automatically disabled
model.eval()  # Disables dropout
```

### Label Smoothing

```python
# Instead of [0, 1], use [0.1, 0.9]
# Prevents overconfidence

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# True label [0, 1] becomes [0.05, 0.95] for 2 classes
```

### Data Augmentation

```python
# Already covered in transforms:
# - Random crop
# - Random flip
# - Color jitter
# - Rotation

# Mixup/CutMix (advanced)
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Loss: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)
```

---

# Chapter 7: Implementation in Retin-Verify

## 7.1 Project Structure

```
retin-verify/
├── training/
│   └── classification/
│       ├── train_cli.py          # Main training script
│       └── configs/
│           └── efficientnet_b0.yaml  # Model config
├── configs/
│   └── model_configs.yaml        # Global configs
└── scripts/
    └── run_training.py           # Training launcher
```

## 7.2 Configuration File Deep Dive

```yaml
# training/classification/configs/efficientnet_b0.yaml

model:
  name: "efficientnet_b0"
  num_classes: 2                    # CNIE front/back only
  class_names: ["cnie_front", "cnie_back"]
  pretrained: true                  # Use ImageNet weights
  dropout: 0.3                      # 30% dropout for regularization

training:
  batch_size: 32                    # Images per batch
  epochs: 50                        # Training iterations
  learning_rate: 0.0001            # Step size (small for fine-tuning)
  weight_decay: 0.01               # L2 regularization
  optimizer: "adamw"               # Optimizer choice
  scheduler: "cosine"              # LR schedule
  warmup_epochs: 5                 # Gradual LR increase
  early_stopping_patience: 10      # Stop if no improvement
  
data:
  train_split: 0.8                 # 80% train
  val_split: 0.1                   # 10% validation
  test_split: 0.1                  # 10% test
  input_size: 224                  # EfficientNet-B0 input
  num_workers: 4                   # Data loading threads
  augment: true                    # Enable augmentation
  
augmentation:
  random_crop: true
  random_flip: true
  random_rotation: 15              # ±15 degrees
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.1
    hue: 0.05

checkpointing:
  save_every: 10                   # Save every 10 epochs
  keep_best: true                  # Save best model
  keep_last: true                  # Save last model
  output_dir: "training/classification/checkpoints"

logging:
  use_tensorboard: true
  log_every: 10                    # Log every 10 batches
  log_dir: "training/classification/logs"

performance_targets:
  accuracy: 0.99                   # Target 99% accuracy
  max_latency_ms: 50               # Max 50ms inference time
```

## 7.3 Model Loading and Modification

```python
# From training/classification/train_cli.py (simplified)

from torchvision import models
import torch.nn as nn

def create_model(model_name='efficientnet_b0', num_classes=2, pretrained=True):
    """
    Create and modify EfficientNet for our task
    """
    # Load pre-trained model
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Get input features of classifier
    in_features = model.classifier[1].in_features  # 1280 for B0
    
    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    return model

# Transfer learning strategy
def setup_transfer_learning(model, fine_tune=True):
    """
    Configure model for transfer learning
    """
    if not fine_tune:
        # Freeze all layers except classifier
        for param in model.parameters():
            param.requires_grad = False
        
        # Only train classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Fine-tune all layers (with lower LR)
        pass
    
    return model

# Usage
model = create_model(num_classes=2, pretrained=True)
model = setup_transfer_learning(model, fine_tune=True)
model = model.to('cuda')
```

## 7.4 Training Execution Flow

```python
# scripts/run_training.py (simplified flow)

1. Load configuration from YAML
2. Setup data loaders (train/val/test)
3. Create model (EfficientNet-B0 with 2 classes)
4. Setup optimizer (AdamW) and scheduler (Cosine)
5. Setup loss function (CrossEntropyLoss)
6. Enable mixed precision (FP16)
7. For each epoch:
   a. Train on training set
   b. Validate on validation set
   c. Save checkpoints
   d. Log metrics to TensorBoard
8. Export final model
9. Push results to Google Drive
```

## 7.5 Inference (Production)

```python
def predict_document_type(model, image_path, device='cuda'):
    """
    Predict document type for a single image
    """
    from torchvision import transforms
    from PIL import Image
    
    # Same transforms as validation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['cnie_front', 'cnie_back']
    return {
        'class': class_names[predicted.item()],
        'confidence': confidence.item()
    }

# Example
result = predict_document_type(model, "document.jpg")
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")
```

---

# Chapter 8: Performance Analysis & Expectations

## 8.1 Expected Training Metrics

### For CNIE Classification (2 classes)

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| **Training Accuracy** | 95-99% | Should reach high quickly |
| **Validation Accuracy** | 95-98% | Slightly lower than train |
| **Test Accuracy** | 94-97% | Final performance |
| **Training Loss** | 0.01-0.1 | Cross-entropy loss |
| **Validation Loss** | 0.05-0.2 | Watch for increase |
| **Convergence Epoch** | 20-40 | Usually enough for this task |
| **Training Time** | 30-60 min | On H100 for 100 epochs |
| **Inference Time** | 5-15 ms | Per image on GPU |

## 8.2 Learning Curves

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Expected curve shapes:
# - Loss: Rapid decrease first 10 epochs, then plateau
# - Accuracy: Rapid increase first 10 epochs, then plateau
# - Train/Val gap should be small (<5%) to avoid overfitting
```

## 8.3 Overfitting Detection

### Signs of Overfitting:
- Training accuracy >> Validation accuracy (>5% gap)
- Validation loss increases while training loss decreases
- Validation accuracy plateaus or decreases

### Solutions:
```python
# 1. Increase dropout
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),  # Increase from 0.3
    nn.Linear(in_features, num_classes)
)

# 2. Add more augmentation
transform = transforms.Compose([
    # ... existing transforms
    transforms.RandomErasing(p=0.2),  # Randomly erase regions
])

# 3. Early stopping (already in config)
early_stopping_patience: 10

# 4. Reduce model complexity (if severe)
# Use smaller EfficientNet (B0 is already small)
# Or reduce learning rate
```

## 8.4 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['cnie_front', 'cnie_back'],
                yticklabels=['cnie_front', 'cnie_back'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Classification report
    print(classification_report(all_labels, all_preds,
                               target_names=['cnie_front', 'cnie_back']))
    
    return cm

# Expected for 2-class problem:
# - High diagonal values (correct predictions)
# - Low off-diagonal values (misclassifications)
# - Front/back confusion might occur for low-quality images
```

## 8.5 Model Size & Speed

```python
# Model statistics
def model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print(f"Model size: {total_params * 2 / 1024 / 1024:.2f} MB (FP16)")

# EfficientNet-B0:
# Total parameters: ~5,300,000
# Model size: ~20 MB (FP32), ~10 MB (FP16)

# Inference speed test
def benchmark_model(model, device, batch_size=1, num_iterations=100):
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / num_iterations
    
    print(f"Average inference time: {elapsed_ms:.2f} ms")
    print(f"Throughput: {1000/elapsed_ms:.1f} images/sec")
    
    return elapsed_ms

# Expected on H100:
# Batch=1: ~5-10 ms
# Batch=32: ~50-100 ms total (~1.5-3 ms per image)
```

---

# Chapter 9: Advanced Topics & Fine-tuning

## 9.1 Progressive Resizing

```python
# Start with smaller images, gradually increase
# Speeds up early training, improves final accuracy

class ProgressiveResize:
    def __init__(self, sizes=[128, 160, 192, 224], epochs_per_size=10):
        self.sizes = sizes
        self.epochs_per_size = epochs_per_size
    
    def get_transform(self, epoch):
        size_idx = min(epoch // self.epochs_per_size, len(self.sizes) - 1)
        size = self.sizes[size_idx]
        
        return transforms.Compose([
            transforms.Resize(size + 32),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])

# Usage
progressive = ProgressiveResize()
for epoch in range(total_epochs):
    transform = progressive.get_transform(epoch)
    # Update dataloader with new transform
    # Train epoch...
```

## 9.2 Test Time Augmentation (TTA)

```python
def predict_with_tta(model, image_path, device, num_augmentations=5):
    """
    Average predictions over multiple augmentations
    Improves accuracy by 1-2%
    """
    from PIL import Image
    
    image = Image.open(image_path).convert('RGB')
    
    # Base transform
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    # Augmented transforms
    aug_transforms = [
        base_transform,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ]),
        # Add more augmentations...
    ]
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for transform in aug_transforms[:num_augmentations]:
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs)
    
    # Average predictions
    avg_probs = torch.stack(all_probs).mean(dim=0)
    confidence, predicted = torch.max(avg_probs, 1)
    
    return predicted.item(), confidence.item()
```

## 9.3 Model Ensemble

```python
class ModelEnsemble:
    """
    Combine predictions from multiple models
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.weights = [w / sum(self.weights) for w in self.weights]
    
    def predict(self, image):
        all_probs = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                output = model(image)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs * weight)
        
        # Weighted average
        ensemble_probs = torch.stack(all_probs).sum(dim=0)
        _, predicted = torch.max(ensemble_probs, 1)
        
        return predicted

# Usage: Train 3 models with different seeds
# ensemble = ModelEnsemble([model1, model2, model3])
```

## 9.4 Knowledge Distillation

```python
# Train a small student model to mimic a large teacher
# Useful for deploying on edge devices

def distillation_loss(student_logits, teacher_logits, true_labels, 
                     temperature=3.0, alpha=0.5):
    """
    Combine soft targets from teacher with hard targets
    """
    # Soft targets (teacher)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
    
    distillation = F.kl_div(soft_predictions, soft_targets, 
                           reduction='batchmean') * (temperature ** 2)
    
    # Hard targets (ground truth)
    ce_loss = F.cross_entropy(student_logits, true_labels)
    
    # Combined loss
    return alpha * ce_loss + (1 - alpha) * distillation

# Student model: Smaller EfficientNet or MobileNet
# Teacher model: Larger EfficientNet (B3/B4) or ensemble
```

## 9.5 ONNX Export & Optimization

```python
# Export to ONNX for deployment
def export_to_onnx(model, output_path='model.onnx'):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to('cuda')
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {output_path}")

# Optimize with TensorRT for NVIDIA GPUs
import tensorrt as trt

def build_tensorrt_engine(onnx_path, engine_path):
    """
    Build optimized TensorRT engine
    2-3x faster inference
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    engine = builder.build_engine(network, config)
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to {engine_path}")
```

---

# Chapter 10: Practical Exercises

## Exercise 1: Visualize Convolutions

```python
# Visualize what different filters detect
import cv2

def visualize_conv_layers(model, image_path):
    """
    Visualize activations of convolutional layers
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to('cuda')
    
    # Hook to capture activations
    activations = {}
    def hook_fn(module, input, output, name):
        activations[name] = output.detach()
    
    # Register hooks on conv layers
    hooks = []
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)
            )
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize first layer filters
    first_conv = list(activations.values())[0]
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < first_conv.shape[1]:
            ax.imshow(first_conv[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
    plt.suptitle('First Convolution Layer Activations')
    plt.savefig('conv_activations.png')
    plt.show()

# Run
visualize_conv_layers(model, 'document.jpg')
```

## Exercise 2: Feature Extraction

```python
def extract_features(model, image_path):
    """
    Extract 1280-dimensional features before classifier
    Useful for similarity search, clustering
    """
    # Remove classifier head
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to('cuda')
    
    # Extract features
    with torch.no_grad():
        features = feature_extractor(input_tensor)
        features = features.squeeze()  # (1280,)
    
    return features.cpu().numpy()

# Compare two images
feat1 = extract_features(model, 'doc1.jpg')
feat2 = extract_features(model, 'doc2.jpg')

similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
print(f"Cosine similarity: {similarity:.4f}")
```

## Exercise 3: Grad-CAM Visualization

```python
# Visualize which parts of image the model focuses on

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, target_class):
        # Forward pass
        output = self.model(input_image)
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= np.max(heatmap)  # Normalize
        
        return heatmap

# Usage
target_layer = model.features[-1]  # Last conv layer
grad_cam = GradCAM(model, target_layer)
heatmap = grad_cam.generate_cam(input_tensor, target_class=0)

# Overlay on original image
import cv2
heatmap = cv2.resize(heatmap, (image.width, image.height))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
plt.imshow(overlay)
```

## Exercise 4: Hyperparameter Tuning

```python
# Grid search for optimal hyperparameters

import itertools

def grid_search(param_grid):
    """
    Try different combinations of hyperparameters
    """
    best_acc = 0
    best_params = None
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        print(f"\nTrying: {params}")
        
        # Create model with these params
        model = create_model(num_classes=2)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Quick training (few epochs)
        val_acc = quick_train(model, optimizer, epochs=10)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            print(f"New best! Acc: {best_acc:.2f}%")
    
    return best_params, best_acc

# Define search space
param_grid = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'weight_decay': [0.001, 0.01, 0.1],
    'dropout': [0.2, 0.3, 0.5],
    'batch_size': [16, 32, 64]
}

# Run search (takes time!)
# best_params, best_acc = grid_search(param_grid)
```

---

# APPENDICES

## A. Math Cheat Sheet

### Derivatives
| Function | Derivative |
|----------|------------|
| f(x) = c | f'(x) = 0 |
| f(x) = x | f'(x) = 1 |
| f(x) = x² | f'(x) = 2x |
| f(x) = eˣ | f'(x) = eˣ |
| f(x) = ln(x) | f'(x) = 1/x |
| f(x) = sigmoid(x) | f'(x) = f(x)(1-f(x)) |
| f(x) = tanh(x) | f'(x) = 1 - f(x)² |
| f(x) = ReLU(x) | f'(x) = 0 if x<0, 1 if x>0 |

### Matrix Operations
```
(A + B)ᵢⱼ = Aᵢⱼ + Bᵢⱼ
(A · B)ᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ
(Aᵀ)ᵢⱼ = Aⱼᵢ
tr(A) = Σᵢ Aᵢᵢ
det(AB) = det(A) · det(B)
```

## B. PyTorch Quick Reference

### Tensor Operations
```python
# Creation
x = torch.zeros(2, 3)
x = torch.ones(2, 3)
x = torch.randn(2, 3)  # Normal distribution
x = torch.arange(0, 10, 2)

# Operations
y = x + 2
y = x * 2
y = x @ y.T  # Matrix multiplication
y = x.view(-1)  # Reshape
y = x.permute(0, 2, 1)  # Transpose dimensions

# GPU
x = x.cuda()
x = x.cpu()
x.device  # Check device
```

### nn.Module
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = MyModel()
model.train()   # Training mode
model.eval()    # Evaluation mode
model.to('cuda')
```

### Optimizers
```python
# SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

## C. EfficientNet Variants

| Variant | Input Size | Parameters | FLOPs | Top-1 Acc |
|---------|-----------|------------|-------|-----------|
| B0 | 224×224 | 5.3M | 0.39B | 77.1% |
| B1 | 240×240 | 7.8M | 0.70B | 79.1% |
| B2 | 260×260 | 9.2M | 1.0B | 80.1% |
| B3 | 300×300 | 12M | 1.8B | 81.6% |
| B4 | 380×380 | 19M | 4.2B | 82.9% |
| B5 | 456×456 | 30M | 9.9B | 83.6% |
| B6 | 528×528 | 43M | 19B | 84.0% |
| B7 | 600×600 | 66M | 37B | 84.3% |

**Recommendation for Document Classification:**
- B0: Fast, mobile-friendly
- B3: Good balance of speed/accuracy
- B4+: Only if accuracy is critical

## D. Common Issues & Solutions

### Issue: CUDA Out of Memory
```python
# Solutions:
# 1. Reduce batch size
batch_size = 16  # Instead of 32

# 2. Use gradient accumulation
accumulation_steps = 2
for i, (images, labels) in enumerate(dataloader):
    loss = model(images, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Issue: NaN Loss
```python
# Causes: Learning rate too high, gradient explosion

# Solutions:
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Lower learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 3. Check for inf/nan
if torch.isnan(loss):
    print("NaN loss detected!")
    break
```

### Issue: Slow Training
```python
# Solutions:
# 1. More workers
dataloader = DataLoader(dataset, num_workers=8, pin_memory=True)

# 2. Mixed precision
with autocast():
    output = model(input)

# 3. Compile model (PyTorch 2.0+)
model = torch.compile(model)

# 4. Prefetch data
from torch.utils.data import prefetch_generator
```

---

**END OF COURSE**

**Next Steps:**
1. Run through the exercises
2. Experiment with the Retin-Verify training
3. Try different architectures (ResNet, ConvNeXt)
4. Build your own CNN from scratch
5. Read the EfficientNet paper: https://arxiv.org/abs/1905.11946
