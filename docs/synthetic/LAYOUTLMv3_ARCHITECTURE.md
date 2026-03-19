# LayoutLMv3 Architecture

## Overview

**LayoutLMv3** is a pre-trained multimodal Transformer model for Document AI (Document Understanding). It was developed by Microsoft Research and published in 2022. It represents the third generation of the LayoutLM family, designed to jointly learn text, layout, and image representations from document images.

## Why LayoutLMv3 for ID Document Processing?

### Key Advantages for Retin-Verify:
1. **Multimodal Understanding**: Combines text content + spatial layout + visual appearance
2. **Pre-trained on Documents**: Already understands document structures (forms, invoices, IDs)
3. **Token-Level Classification**: Perfect for field extraction (surname, DOB, etc.)
4. **Multilingual Support**: Handles Arabic and French text
5. **Fast Inference**: Efficient for production deployment

---

## Architecture Components

### 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Text Input │  │ Layout Input│  │  Image Input│            │
│  │  (Tokens)   │  │  (Boxes)    │  │  (Pixels)   │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EMBEDDING LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Text Embeds  │  │Layout Embeds│  │Image Embeds │            │
│  │  + Word     │  │  + x, y,    │  │  + CNN      │            │
│  │  + Position │  │  + width,   │  │  + Patch    │            │
│  │             │  │  + height   │  │  + Linear   │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MULTIMODAL FUSION (Concatenation)                  │
│                    + Linear Projection                          │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           TRANSFORMER ENCODER (Multi-Layer)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Self-Attention + Layer Norm + FFN + Layer Norm        │   │
│  │  (Repeated 12 times for LayoutLMv3-base)                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ Token Class.  │  │ Seq. Class.   │  │ Bounding Box  │      │
│  │ (BIO Labels)  │  │ (Doc Type)    │  │ Regression    │      │
│  │ B-SURNAME     │  │ CNIE_FRONT    │  │ [x, y, w, h]  │      │
│  │ I-SURNAME     │  │ CNIE_BACK     │  │               │      │
│  │ B-DATE        │  │ PASSPORT      │  │               │      │
│  │ O             │  │               │  │               │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Component Analysis

### 2.1 Text Embedding Layer

**Purpose**: Convert text tokens into vector representations

```
Text Input: "Surname: Hamidi"
     │
     ▼
┌─────────────────────────────────────┐
│  WordPiece Tokenizer               │
│  "Surname" → [101, 2345, 102]     │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Word Embeddings (768-dim)         │
│  Lookup table: token_id → vector   │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Position Embeddings               │
│  Position 0, 1, 2, ... → vector    │
└─────────────────────────────────────┘
     │
     ▼
Text Embedding = Word Embedding + Position Embedding
```

**Parameters**:
- Vocabulary size: 30,522 (WordPiece)
- Embedding dimension: 768 (base model)
- Max sequence length: 512 tokens

### 2.2 Layout Embedding Layer

**Purpose**: Encode spatial position of text on the page

```
Bounding Box Input: [x1, y1, x2, y2] (normalized 0-1000)
     │
     ▼
┌─────────────────────────────────────┐
│  Spatial Embeddings                │
│                                    │
│  x1_embedding = Embedding(x1)      │
│  y1_embedding = Embedding(y1)      │
│  x2_embedding = Embedding(x2)      │
│  y2_embedding = Embedding(y2)      │
│  w_embedding  = Embedding(x2-x1)   │
│  h_embedding  = Embedding(y2-y1)   │
└─────────────────────────────────────┘
     │
     ▼
Layout Embedding = x1 + y1 + x2 + y2 + w + h
```

**Key Innovation**: LayoutLMv3 uses **continuous relative position embeddings** instead of discrete bins used in v1/v2.

**Why continuous?**
- Smoother gradients during training
- Better generalization to unseen layouts
- More precise spatial relationships

### 2.3 Image Embedding Layer

**Purpose**: Encode visual appearance of the document

```
Image Input: (3, H, W) RGB pixels
     │
┌─────────────────────────────────────┐
│  CNN Backbone (ResNet-50)          │
│  - Extracts visual features        │
│  - Output: (C, H/32, W/32)         │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Patch Embedding                     │
│  - Flatten spatial features          │
│  - Project to Transformer dimension  │
│  - Output: (N, 768) patches          │
└─────────────────────────────────────┘
     │
     ▼
Image Embedding = Patch features
```

**LayoutLMv3 Improvement**: Uses CNN (ResNet) + patch embedding instead of ViT in v2.
- Better for document images with irregular layouts
- More efficient than pure ViT approach

### 2.4 Multimodal Fusion

**Fusion Strategy**: Early fusion via concatenation + linear projection

```
Inputs:
  Text Embeddings:    (N, 768)
  Layout Embeddings:  (N, 768)
  Image Embeddings:   (M, 768)  [separate patches]

Fusion:
  1. Concat Text + Layout: (N, 1536)
  2. Linear projection:    (N, 768)
  3. Concat with Image:    (N+M, 768)
  4. Add segment embeddings to distinguish modalities
```

**Segment Embeddings**:
- Type A: Text + Layout tokens
- Type B: Image patches
- Helps model distinguish between text regions and visual elements

---

## 3. Transformer Encoder

### 3.1 Architecture Details

```
LayoutLMv3-base (standard configuration):
- Layers: 12
- Hidden size: 768
- Attention heads: 12
- FFN intermediate size: 3072
- Parameters: ~120M

LayoutLMv3-large:
- Layers: 24
- Hidden size: 1024
- Attention heads: 16
- Parameters: ~340M
```

### 3.2 Self-Attention Mechanism

```
Input: X (sequence_length, hidden_size)

Multi-Head Self-Attention:
  Q = X @ W_q    (Queries)
  K = X @ W_k    (Keys)
  V = X @ W_v    (Values)

  Attention(Q, K, V) = softmax(QK^T / √d_k) @ V

  Output = Concat(head_1, ..., head_h) @ W_o

Each head learns different attention patterns:
  - Head 1: Focus on text content
  - Head 2: Focus on spatial proximity
  - Head 3: Focus on visual similarity
  - ...
```

**Attention Visualization**:
```
Text Token: "Hamidi"
     │
     ├───→ "Surname" (text relationship)
     ├───→ [bbox nearby] (spatial relationship)
     └───→ [photo region] (visual context)
```

### 3.3 Pre-Training Objectives

LayoutLMv3 uses **three pre-training tasks**:

#### 1. Masked Language Modeling (MLM)
```
Input:  "Surname: [MASK]"
Target: "Surname: Hamidi"

Randomly mask 15% of text tokens
Model learns to predict masked tokens using:
  - Context words
  - Layout information (where is the word?)
  - Visual appearance (what does it look like?)
```

#### 2. Masked Image Modeling (MIM)
```
Input:   Document with random image patches masked
Target:  Reconstruct pixel values or discrete tokens

Uses discrete VAE (dVAE) to tokenize image patches
Model learns visual representations of document elements
```

#### 3. Word-Patch Alignment (WPA)
```
Input:  Text tokens + Image patches
Task:   Predict which image patch corresponds to each text token

Aligns text and image modalities
Critical for field extraction: "This text is in this region"
```

---

## 4. Fine-Tuning for Field Extraction

### 4.1 Token Classification Task

```
Input: Document image with OCR text
      "Surname: Hamidi"
       
Output: BIO labels for each token
  "Surname" → B-LABEL      (Begin label field)
  ":"       → O            (Outside any field)
  "Hamidi"  → B-SURNAME   (Begin surname value)
            → I-SURNAME   (Inside surname value)

BIO Scheme:
  B-XXX: Beginning of field XXX
  I-XXX: Inside field XXX
  O:     Outside any field
```

### 4.2 Label Set for CNIE Documents

```python
LABELS = [
    # Fields on Front
    "B-SURNAME", "I-SURNAME",
    "B-GIVEN_NAMES", "I-GIVEN_NAMES",
    "B-DATE_OF_BIRTH", "I-DATE_OF_BIRTH",
    "B-PLACE_OF_BIRTH", "I-PLACE_OF_BIRTH",
    "B-NATIONAL_ID", "I-NATIONAL_ID",
    "B-DATE_OF_ISSUE", "I-DATE_OF_ISSUE",
    "B-DATE_OF_EXPIRY", "I-DATE_OF_EXPIRY",
    "B-SEX", "I-SEX",
    "B-BLOOD_GROUP", "I-BLOOD_GROUP",
    "B-PHOTO", "I-PHOTO",
    
    # Fields on Back
    "B-MRZ_LINE1", "I-MRZ_LINE1",
    "B-MRZ_LINE2", "I-MRZ_LINE2",
    "B-MRZ_LINE3", "I-MRZ_LINE3",
    "B-BIRTH_YEAR", "I-BIRTH_YEAR",
    
    # Other
    "O"  # Outside any field
]
```

### 4.3 Training Process

```python
# 1. Load pre-trained LayoutLMv3
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(LABELS)
)

# 2. Prepare input
inputs = processor(
    image=document_image,           # (H, W, 3)
    text=ocr_text,                  # ["Surname", ":", "Hamidi"]
    boxes=bounding_boxes,           # [[x1,y1,x2,y2], ...]
    return_tensors="pt"
)

# 3. Forward pass
outputs = model(**inputs)
predictions = outputs.logits.argmax(-1)

# 4. Loss computation
loss = CrossEntropyLoss(predictions, labels)
```

### 4.4 Input Representation

```python
# Example input for a CNIE field
{
    "text": "Hamidi",
    "box": [1197, 660, 264, 46],  # x, y, w, h
    "label": "B-SURNAME"
}

# Encoded as:
text_embedding = embedding("Hamidi")        # Semantic meaning
layout_embedding = embedding([1197, 660,    # Spatial position
                               264,  46])
image_embedding = cnn_patch(image_region)   # Visual appearance

# Combined:
input_vector = text + layout + image        # Multimodal fusion
```

---

## 5. Comparison with Previous Versions

| Feature | LayoutLM | LayoutLMv2 | LayoutLMv3 |
|---------|----------|------------|------------|
| **Text Encoder** | BERT | UniLMv2 | RoBERTa |
| **Layout Encoding** | Discrete bins | Continuous | Continuous |
| **Visual Encoder** | Faster R-CNN | ViT | CNN (ResNet) |
| **Pre-training** | MLM + KD | MLM + MIM + WPA | MLM + MIM + WPA |
| **Multilingual** | No | Limited | Yes |
| **Speed** | Medium | Slow | Fast |
| **Accuracy** | Good | Better | Best |

**Key Improvements in v3**:
1. **Segment-level layout embeddings** instead of word-level
2. **CNN + patch** instead of ViT for visual features
3. **Better multilingual support** (important for Arabic)
4. **More efficient training** with improved objectives

---

## 6. Inference Pipeline

```
Input: CNIE Document Image
       ┌─────────────────┐
       │  Front Image    │
       └────────┬────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌─────────┐           ┌─────────┐
│   OCR   │           │  Layout │
│ Engine  │           │ Analysis│
└────┬────┘           └────┬────┘
     │                     │
     ▼                     ▼
┌─────────────┐     ┌─────────────┐
│ Text Tokens │     │ Bounding    │
│ ["Hamidi"]  │     │ Boxes       │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 ▼
       ┌───────────────────┐
       │  LayoutLMv3       │
       │  Model Inference  │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │  Token Labels     │
       │  B-SURNAME        │
       │  I-SURNAME        │
       └─────────┬─────────┘
                 │
                 ▼
       ┌───────────────────┐
       │  Field Extraction │
       │  surname: Hamidi  │
       └───────────────────┘
```

---

## 7. Implementation in Retin-Verify

### 7.1 Model Configuration

```python
# src/extraction/field_extractor.py
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

class FieldExtractor:
    def __init__(self, model_path="microsoft/layoutlmv3-base"):
        self.processor = LayoutLMv3Processor.from_pretrained(model_path)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_path,
            num_labels=42  # Number of BIO labels
        )
    
    def extract_fields(self, image, ocr_results):
        """
        Args:
            image: PIL Image or numpy array
            ocr_results: List of dicts with 'text' and 'bbox'
        Returns:
            Extracted fields dict
        """
        # Prepare inputs
        words = [r["text"] for r in ocr_results]
        boxes = [r["bbox"] for r in ocr_results]
        
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Inference
        outputs = self.model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Convert predictions to fields
        fields = self._decode_predictions(words, predictions)
        return fields
```

### 7.2 Training Script

```python
# src/extraction/train_extractor.py
def train_layoutlmv3(dataset_path, output_dir):
    """Fine-tune LayoutLMv3 on synthetic CNIE data"""
    
    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path)
    
    # Initialize model
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(ID2LABEL)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
```

### 7.3 Expected Performance

| Metric | Baseline (Rule-based) | LayoutLMv3 |
|--------|----------------------|------------|
| **Field Extraction F1** | 65% | 95%+ |
| **Surname Accuracy** | 70% | 98% |
| **Date Parsing** | 60% | 95% |
| **MRZ Alignment** | 80% | 99% |
| **Inference Time** | 50ms | 100ms |

---

## 8. Advantages for Algerian ID Documents

### 8.1 Arabic Text Support

```
LayoutLMv3 Pre-training includes Arabic:
  - Fine-tuned on multilingual documents
  - Understands Arabic script and diacritics
  - Handles right-to-left (RTL) text direction

Example:
  Input: "الاسم: حميدي"
  Output: B-LABEL, B-SURNAME_AR, I-SURNAME_AR
```

### 8.2 Handling Variable Layouts

```
Different CNIE versions have slight layout variations:
  - Old vs new CNIE designs
  - Photo on left vs right
  - Different font sizes

LayoutLMv3 handles this via:
  - Continuous position embeddings
  - Visual feature learning
  - Layout-invariant text understanding
```

### 8.3 Small Data Efficiency

```
Synthetic dataset: 10,000 samples
Real-world performance transfer:
  - LayoutLMv3: 90%+ with fine-tuning on 100 real samples
  - From-scratch model: Requires 10,000+ real samples
  
Pre-training acts as regularization:
  - Better generalization
  - Less overfitting to synthetic artifacts
```

---

## 9. Model Sizes & Deployment

### 9.1 Size Comparison

| Model | Parameters | Disk Size | Inference Time |
|-------|-----------|-----------|----------------|
| LayoutLMv3-base | 120M | ~500MB | 100ms/image |
| LayoutLMv3-large | 340M | ~1.3GB | 200ms/image |
| Distilled (custom) | 60M | ~250MB | 50ms/image |

### 9.2 Deployment Options

```
Option 1: Full Model (GPU)
  - Hardware: NVIDIA T4 / GTX 1660+
  - Throughput: ~10 images/second
  - Use case: Cloud API

Option 2: ONNX Runtime (CPU)
  - Export to ONNX format
  - Hardware: 4-core CPU
  - Throughput: ~2 images/second
  - Use case: On-premise deployment

Option 3: Quantized Model (Edge)
  - INT8 quantization
  - Hardware: Raspberry Pi 4 / Jetson Nano
  - Throughput: ~0.5 images/second
  - Use case: Edge devices
```

---

## 10. References

1. **LayoutLMv3 Paper**: "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking" (Huang et al., 2022)

2. **GitHub**: https://github.com/microsoft/unilm/tree/master/layoutlmv3

3. **Hugging Face**: https://huggingface.co/microsoft/layoutlmv3-base

4. **Documentation**: https://huggingface.co/docs/transformers/model_doc/layoutlmv3

---

## Summary

**LayoutLMv3 is the ideal choice for Retin-Verify because:**

1. ✅ **Multimodal**: Understands text + layout + image
2. ✅ **Pre-trained**: Document-specific knowledge from millions of docs
3. ✅ **Accurate**: State-of-the-art for form understanding
4. ✅ **Fast**: Efficient for production use
5. ✅ **Multilingual**: Supports Arabic and French
6. ✅ **Proven**: Used in production by Microsoft and others

**Integration path:**
1. Use synthetic data to fine-tune LayoutLMv3-base
2. Validate on small set of real CNIE scans
3. Deploy with ONNX for CPU inference
4. Continuously improve with real user data
