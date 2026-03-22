V11 Strategy: Clean Data, Smaller Model, Stronger Regularization
2.1 Use MobileNetV3‑Small (2.5M params)
Why: Lower capacity reduces overfitting risk while still having enough parameters to learn the subtle differences between front/back. The task does not require a 5M+ model.

2.2 Clean the Dataset Automatically
Use the face and chip detection results to flag potential mislabels. For each image:

If the folder says front but no face detected, flag it as uncertain.

If the folder says back but no chip detected, flag it as uncertain.

Manually review flagged images (or at least a sample) and correct labels. This is critical to remove label noise.

After cleaning, you can exclude images that remain ambiguous, or keep them but with lower weight.

2.3 Balance Hard Examples
Collect more back hard examples (e.g., from your feedback system or by capturing challenging back images). If not possible, use oversampling (duplicate back images) or class weights to give back class higher importance during training.

2.4 Enhance Augmentation with Back‑Specific Distortions
For back images, add chip glare (bright spot), chip occlusion (random black patch), MRZ warping (local affine transform), and text blur to simulate real scanning/lighting.

For front images, keep the existing glare and warping, but also add photo‑area specific distortions.

2.5 Use MixUp/CutMix Only During Training, But Monitor
These are powerful regularizers, but they can also confuse the model if the dataset is small. We'll keep them but reduce the probability of application (e.g., apply to 50% of batches).

2.6 Implement Stratified K‑Fold Cross‑Validation
Instead of a single train/val/test split, use 5‑fold cross‑validation to get a more reliable estimate of performance and to detect overfitting earlier.

2.7 Add Weighted Loss for Back Class
Compute class weights inversely proportional to class frequency in the training set. Give back class a slightly higher weight (e.g., 1.2) to compensate for fewer hard examples.

3. Implementation Plan for V11
We'll modify the V10 script to:

Load and clean data using face/chip detection to filter or relabel uncertain images.

Use MobileNetV3‑Small as the backbone.

Apply class‑specific augmentations (via separate transform pipelines for front/back).

Use stratified K‑fold for training and early stopping based on average validation accuracy across folds.

Train with lower learning rates and stronger weight decay (1e‑3 for backbone? Actually we'll keep 1e‑5 backbone, 1e‑3 classifier).

Evaluate on a final held‑out test set after cross‑validation to get unbiased metrics.

Here's a simplified pseudocode for the dataset cleaning step:

python
def clean_dataset(data_root, output_root):
    """
    Use face and chip detection to relabel images.
    For each image, if label matches detection -> keep.
    If mismatch, move to a 'mismatch' folder for manual review.
    """
    # Implement detection functions (as in V10)
    for class_name in ['front', 'back']:
        class_dir = data_root / class_name
        for img_path in class_dir.glob('*.jpg'):
            if class_name == 'front':
                has_face = detect_face(img_path)
                if not has_face:
                    # Flag for manual review
                    dest = mismatch_dir / f"{class_name}_{img_path.name}"
                    shutil.move(img_path, dest)
            else: # back
                has_chip = detect_chip(img_path)
                if not has_chip:
                    dest = mismatch_dir / f"{class_name}_{img_path.name}"
                    shutil.move(img_path, dest)
For augmentation, we'll create separate pipelines:

python
front_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3,0.3,0.2),
    transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)),
    AddGlare(prob=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

back_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3,0.3,0.2),
    transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)),
    AddGlare(prob=0.3, max_intensity=0.5),  # chip glare
    AddChipOcclusion(prob=0.3),  # custom transform
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
4. Expected Outcomes
After implementing V11:

Test accuracy should exceed 95% for both front and back.

Validation accuracy will be closer to test accuracy (no overfitting).

The model will generalize well to new, unseen images.

5. If Accuracy Remains Low
If after these steps you still see low accuracy, the next step is to collect more real data (especially challenging back images) and consider synthetic data generation with domain randomization to artificially expand the dataset.



