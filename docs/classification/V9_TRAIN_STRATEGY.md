V8 Post‑Mortem & V9 Action Plan
You now have a clean, balanced dataset (464 train, 58 val, 60 test) with 232 front / 232 back. Yet the V8 validation accuracy is only 70.7% (Front 79%, Back 62%). This is still far from the >95% we know is achievable (based on your earlier 2‑class real‑only model).
The problem is no longer just label noise – it is now about insufficient discriminative features, weak regularization, and suboptimal training configuration.

Why V8 Still Underperforms
1. Model Capacity vs. Data Size
EfficientNet‑B0 has 5.3M parameters. With only 464 training images, the model can easily overfit to the training set, even with augmentation, unless regularization is extremely strong.

The validation accuracy plateauing at 70% suggests the model is not learning robust, generalizable features.

2. Insufficient Data Augmentation
Your current augmentation likely does not simulate the real‑world variability that causes confusion. Front and back images look similar in terms of overall card layout – the only strong discriminators are the photo area (front) and chip/MRZ (back). If the augmentation does not preserve or exaggerate these differences, the model may latch onto spurious correlations (e.g., background texture, lighting) that fail on unseen data.

3. Lack of Hard Example Mining
The dataset is balanced, but it may contain easy examples that the model learns quickly, while the hard examples (those that look similar to the other class) are not emphasized. The feedback misclassifications you collected (135 front, 120 back) are exactly the hard examples. If they are not included in the training set, the model will never learn to handle them.

4. Optimization Issues
Learning rate: Using a uniform LR (e.g., 1e‑4) for all layers can cause catastrophic forgetting of pretrained features.

Loss function: Standard cross‑entropy with balanced weights is fine, but label smoothing and MixUp/CutMix are missing.

Early stopping: The best epoch was only 10, which may be too early – the model may still be underfitting.

5. Validation/Test Split Not Representative
If the validation set contains images that are too similar to the training set (e.g., from the same cards), the validation accuracy can be overoptimistic. The test set should consist of completely unseen cards taken under different conditions.

V9: A Robust Approach to Achieve >95% Accuracy
Step 1: Verify Dataset Labels
Even with a clean upload, double‑check that the folder names (front/back) match the actual content. Run a quick sanity check:

For each image, use a face detector (e.g., MTCNN) – if a face is detected, the image should be front.

Use a simple chip detector (template matching or a small CNN trained on chip patches) – if a chip is detected, the image should be back.

Manually inspect any images where the detector disagrees with the folder label.

Step 2: Augment with Front‑/Back‑Specific Distortions
Create an augmentation pipeline that emphasizes the discriminative regions:

For front images (photo area):

Random glare overlay (bright ellipse) over the photo region.

Photo warping (local affine transform).

Face‑like occlusions (e.g., sunglasses, mask) to simulate real occlusions.

For back images (chip and MRZ):

Chip occlusion (small black square over chip).

MRZ warping (apply slight perspective distortion to the MRZ region).

Glare on chip (bright spot).

Use these only on the respective classes to avoid confusing the model.

Step 3: Use MixUp and CutMix for Strong Regularization
MixUp (α=0.2) – blends images globally.

CutMix (α=1.0) – pastes a patch from another image.
These encourage the model to focus on local discriminative features and prevent overfitting.

Step 4: Optimize Training Hyperparameters
Differential learning rates:

Backbone (EfficientNet‑B0 pretrained): 1e‑5

Classifier head: 1e‑3

Optimizer: AdamW with weight decay 1e‑4.

Scheduler: ReduceLROnPlateau (factor=0.5, patience=3) monitoring validation loss.

Loss: Cross‑entropy with label smoothing (0.1) and class weights balanced (you can compute from training set).

Batch size: 32 (or 64 if GPU memory allows).

Epochs: 50–100 with early stopping (patience 10).

Step 5: Incorporate Hard Examples
Add your feedback images (the ones that were misclassified) into the training set. This is crucial because they represent the most challenging cases.

Step 6: Evaluate Properly
Split data by card ID to prevent data leakage (if you have multiple images of the same card, ensure they are all in the same split).

Use the test set (60 images) only at the end, not for validation during training.

Monitor validation accuracy per class; if one class lags, adjust augmentation or class weights.

Step 7: Consider a Smaller Model
With only ~500 training images, a smaller model like MobileNetV3‑Large or ResNet‑18 may generalize better. Train both and compare.

Implementation Sketch for V9
python
# Custom transforms
class AddGlare:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        # create a bright elliptical patch (simulate glare)
        # apply to the photo area (you can predefine ROI for front)
        return img

# Augmentation pipeline
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1)),
    AddGlare(prob=0.5),          # only for front images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Use MixUp/CutMix via a wrapper
def mixup_cutmix_criterion(criterion):
    # implement as standard
    pass

# Differential LR
optimizer = torch.optim.AdamW([
    {'params': backbone.parameters(), 'lr': 1e-5},
    {'params': classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
Expected Outcome
After implementing the above, you should see:

Validation accuracy >95% within 20–30 epochs.

Test accuracy >95% on unseen real images.

Balanced front/back performance.

If you still face issues, the next step is to collect more real images (especially challenging angles, lighting, and wear) and use them for active learning.

Immediate Action Items
Verify labels with face/chip detectors.

Integrate feedback images into the training set.

Implement MixUp/CutMix and the custom glare/warping augmentations.

Retrain with differential LRs and label smoothing.

Evaluate on the test set and report per‑class accuracy.

