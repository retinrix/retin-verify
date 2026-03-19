# Domain Adaptation Techniques Guide

## The Problem: Synthetic-to-Real Domain Gap

**Source Domain (Synthetic):**
- Perfectly flat cards
- Uniform lighting
- No noise or artifacts
- Consistent resolution
- Clean backgrounds

**Target Domain (Real):**
- Perspective distortion
- Variable lighting/shadows
- Sensor noise
- Different resolutions
- Complex backgrounds

**The Gap:**
```
Model trained on:  [Synthetic Images]  →  [100% accuracy]
Model tested on:   [Real Images]       →  [~40% accuracy] ❌

Goal:               [Real Images]       →  [~95% accuracy] ✓
```

## Technique 1: Adversarial Domain Adaptation (GAN-based)

### Overview
Train a discriminator to distinguish between synthetic and real images while training the classifier to "fool" the discriminator.

### How It Works

```
┌─────────────────┐     ┌─────────────────┐
│  Synthetic Img  │────▶│   Feature       │
└─────────────────┘     │   Extractor     │◄────┐
                        └────────┬────────┘     │
                                 │              │
                        ┌────────▼────────┐     │
                        │  Classifier     │     │
                        │  (4 classes)    │     │
                        └────────┬────────┘     │
                                 │              │
                        ┌────────▼────────┐     │
                        │   Prediction    │     │
                        └─────────────────┘     │
                                                │
┌─────────────────┐     ┌─────────────────┐     │
│   Real Image    │────▶│   Feature       │─────┤
└─────────────────┘     │   Extractor     │     │
                        └────────┬────────┘     │
                                 │              │
                        ┌────────▼────────┐     │
                        │  Domain         │◄────┘
                        │  Discriminator  │ (tries to tell if features
                        └─────────────────┘  come from synthetic or real)
```

### Implementation: Domain-Adversarial Neural Network (DANN)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL)
    Forward: identity
    Backward: multiply gradient by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class DomainAdaptationModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Feature Extractor (shared)
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0')
        self.feature_dim = self.feature_extractor._fc.in_features
        
        # Remove original classifier
        self.feature_extractor._fc = nn.Identity()
        
        # Task Classifier (for document type)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Domain Discriminator (tells synthetic vs real)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # synthetic vs real
        )
        
    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Classification (always)
        class_output = self.classifier(features)
        
        # Domain prediction (with gradient reversal)
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_discriminator(reversed_features)
        
        return class_output, domain_output

# Training Loop
def train_dann(model, synthetic_loader, real_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        # Progressively increase domain adaptation weight
        # Start with alpha=0 (no adaptation), increase to alpha=1
        p = epoch / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        for (syn_images, syn_labels), (real_images, _) in zip(synthetic_loader, real_loader):
            # Combine batches
            images = torch.cat([syn_images, real_images])
            
            # Domain labels: 0 = synthetic, 1 = real
            domain_labels = torch.cat([
                torch.zeros(len(syn_images)),
                torch.ones(len(real_images))
            ]).long()
            
            # Forward pass
            class_output, domain_output = model(images, alpha)
            
            # Classification loss (only on synthetic - we have labels)
            class_loss = F.cross_entropy(class_output[:len(syn_images)], syn_labels)
            
            # Domain loss (on all images)
            domain_loss = F.cross_entropy(domain_output, domain_labels)
            
            # Total loss
            loss = class_loss + 0.5 * domain_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### When to Use GAN-based Adaptation

✅ **Best for:**
- Large amount of unlabeled real images (1000+)
- Significant visual differences between domains
- When you can't collect many labeled real images

❌ **Avoid when:**
- Very limited unlabeled real data (<500 images)
- Domains are already quite similar
- Training stability is critical

---

## Technique 2: Self-Supervised Contrastive Learning

### Overview
Learn representations that are invariant to domain by training on pretext tasks using both synthetic and real images.

### Key Concept: "Similar images should have similar features"

```
┌─────────────────────────────────────────────────────────────┐
│                    Contrastive Learning                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Anchor (CNIE Front - Real)                                  │
│       ↓                                                      │
│  [Feature Extractor] ───────────────────┐                   │
│       ↓                                  │                   │
│  Embedding Vector                        │ Contrastive Loss │
│       ↘                                  ↓                   │
│        ↘  (should be similar)    ┌──────────────┐           │
│         ↘───────────────────────▶│   Positive   │           │
│                                  │ (CNIE Front  │           │
│                                  │  - Synthetic)│           │
│  ┌──────────────┐                └──────────────┘           │
│  │   Negative   │◄─────────────── (should be different)     │
│  │ (CNIE Back   │                                           │
│  │  - Real)     │                                           │
│  └──────────────┘                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation: SimCLR-style Domain Adaptation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveDomainAdaptation(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
        # Encoder ( EfficientNet-B0 )
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.encoder._fc = nn.Identity()
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # Embedding space
        )
        
    def forward(self, x):
        features = self.encoder(x)
        embedding = self.projection_head(features)
        return F.normalize(embedding, dim=1)

# NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    z_i: embeddings from augmented view 1
    z_j: embeddings from augmented view 2
    """
    batch_size = z_i.shape[0]
    
    # Concatenate embeddings
    z = torch.cat([z_i, z_j], dim=0)  # Shape: (2*batch_size, dim)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels + batch_size, labels])
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    
    # Compute loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# Domain-Invariant Contrastive Loss
def domain_invariant_contrastive_loss(
    syn_embeddings,      # Synthetic embeddings
    real_embeddings,     # Real embeddings  
    labels,              # Document class labels
    temperature=0.5
):
    """
    Pull together same-class embeddings from different domains
    Push apart different-class embeddings
    """
    batch_size = syn_embeddings.shape[0]
    
    # Normalize embeddings
    syn_embeddings = F.normalize(syn_embeddings, dim=1)
    real_embeddings = F.normalize(real_embeddings, dim=1)
    
    # Compute cross-domain similarities
    sim_matrix = torch.mm(syn_embeddings, real_embeddings.t()) / temperature
    
    # Positive pairs: same document class
    # Create label matrix where True = same class
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # Compute contrastive loss
    positives = sim_matrix[label_matrix].view(batch_size, -1)
    negatives = sim_matrix[~label_matrix].view(batch_size, -1)
    
    # Log-sum-exp trick for numerical stability
    logits = torch.cat([positives, negatives], dim=1)
    labels_pos = torch.zeros(batch_size, dtype=torch.long).to(syn_embeddings.device)
    
    return F.cross_entropy(logits, labels_pos)

# Training Pipeline
def train_contrastive_adaptation(model, synthetic_loader, real_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        for (syn_images, syn_labels), (real_images, _) in zip(synthetic_loader, real_loader):
            # Get embeddings from both domains
            syn_emb = model(syn_images)
            real_emb = model(real_images)
            
            # Domain-invariant contrastive loss
            loss = domain_invariant_contrastive_loss(syn_emb, real_emb, syn_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Advanced: Domain-Consistent Augmentation

```python
class DomainConsistentAugmentation:
    """
    Apply same augmentations to both synthetic and real images
    to learn domain-invariant features
    """
    
    def __init__(self):
        self.augmentations = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        # Generate two augmented views
        view1 = self.augmentations(image)
        view2 = self.augmentations(image)
        return view1, view2
```

### When to Use Contrastive Learning

✅ **Best for:**
- Large unlabeled datasets from both domains
- When domain shift is moderate (not extreme)
- As pre-training before supervised fine-tuning

❌ **Avoid when:**
- Very small datasets (<500 images per domain)
- Domains are extremely different (e.g., sketches vs photos)

---

## Technique 3: Style Transfer (CycleGAN)

### Overview
Transform synthetic images to look like real images (or vice versa) while preserving content.

```
Synthetic ──[Generator G]──▶ "Real-looking" Synthetic ──[Classifier]──▶ Prediction
                ↑__________________________↓
                           Cycle Consistency
```

### Quick Implementation

```python
# Use pre-trained CycleGAN or train your own
# Then apply to synthetic training data:

# Step 1: Train CycleGAN (or use pretrained)
# Synthetic ↔ Real translation

# Step 2: Augment synthetic data
augmented_synthetic = cyclegan(synthetic_images, direction='synthetic2real')

# Step 3: Train classifier on augmented data
classifier.train(augmented_synthetic, labels)
```

### Pros/Cons

✅ **Pros:**
- Can use existing synthetic labels
- Visual verification of adaptation

❌ **Cons:**
- Requires training GAN (compute intensive)
- May introduce artifacts
- Need paired or unpaired examples from both domains

---

## Technique 4: Mixup & CutMix (Simple & Effective)

### Overview
Blend synthetic and real images during training to encourage the model to learn domain-invariant features.

```python
def mixup(synthetic_img, real_img, alpha=0.4):
    """Mix two images with random ratio"""
    lam = np.random.beta(alpha, alpha)
    mixed = lam * synthetic_img + (1 - lam) * real_img
    return mixed, lam

# Training
for syn_img, syn_label in synthetic_loader:
    real_img = sample_random_real_image()
    
    mixed_img, lam = mixup(syn_img, real_img)
    
    # Soft label
    target = lam * one_hot(syn_label) + (1 - lam) * uniform_distribution
    
    output = model(mixed_img)
    loss = cross_entropy(output, target)
```

---

## Recommended Approach: Hybrid Strategy

### Phase 1: Contrastive Pre-training (1-2 days)
```python
# Use all unlabeled real + synthetic images
# Learn domain-invariant representations
pretrain_contrastive(
    labeled_synthetic=synthetic_data,
    unlabeled_real=real_data,
    epochs=100
)
```

### Phase 2: Supervised Fine-tuning (3-5 hours)
```python
# Fine-tune on labeled real images
# Start with frozen backbone
finetune_supervised(
    model=pretrained_model,
    labeled_real=real_labeled_data,
    epochs=20
)
```

### Phase 3: Adversarial Refinement (Optional, 1-2 days)
```python
# If accuracy still not satisfactory
# Add domain adversarial training
refine_adversarial(
    model=finetuned_model,
    synthetic=synthetic_data,
    real=real_unlabeled_data,
    epochs=30
)
```

---

## Decision Flowchart

```
How much labeled real data do you have?
│
├─► Less than 100 images
│   └─► Use: Contrastive Learning + Heavy Augmentation
│
├─► 100-500 images
│   └─► Use: Fine-tuning with Mixup/CutMix
│
├─► 500-2000 images
│   └─► Use: Contrastive Pre-training → Fine-tuning
│
└─► More than 2000 images
    └─► Use: Full Adversarial Training (DANN)
```

---

## Implementation Priority

| Technique | Difficulty | Effectiveness | Data Needed | Priority |
|-----------|-----------|---------------|-------------|----------|
| Simple Fine-tuning | ⭐ Easy | ⭐⭐ Moderate | 300+ labeled | 1st |
| Mixup/CutMix | ⭐ Easy | ⭐⭐ Good | 300+ labeled | 2nd |
| Contrastive Learning | ⭐⭐ Medium | ⭐⭐⭐ Very Good | 1000+ unlabeled | 3rd |
| Adversarial (DANN) | ⭐⭐⭐ Hard | ⭐⭐⭐⭐ Excellent | 1000+ unlabeled | 4th |
| CycleGAN | ⭐⭐⭐⭐ Very Hard | ⭐⭐⭐ Good | 2000+ unpaired | 5th |

**Start with:** Fine-tuning + Mixup (easiest, good results)
**Then try:** Contrastive learning (if you have unlabeled real data)
