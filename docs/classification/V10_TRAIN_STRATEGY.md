V10 Strategy: Solving the Low Accuracy with Hard Examples & Model Selection
Your V9 results show that even with a clean dataset (464 train images) and strong regularization, accuracy only reached 63.79% – far from the 95% target. This confirms that the dataset size and/or quality are still insufficient. The next step is to incorporate hard examples (feedback images), verify labels, and choose a more suitable model (MobileNetV3-Large) that can generalize better from small data.

1. Dataset Augmentation with Hard Examples
Feedback images (135 front misclassified, 120 back misclassified) represent the most challenging cases. They should be added to the training set (after verifying their correct labels).

Goal: Increase training size from 464 to ~720 images (including feedback). This alone will help, but we also need to ensure the images are correctly labeled.

2. Automatic Label Verification
Before adding feedback, run a script to verify each image's label using:

Face detector (MTCNN or OpenCV Haar cascade) – if a face is detected, it should be front.

Chip detector – train a simple CNN on cropped chip regions, or use template matching.

MRZ detector – use easyOCR to detect MRZ text; if MRZ present, it should be back.

Flag any image where the detector's verdict disagrees with the folder label for manual review.

3. Model Selection: MobileNetV3‑Large
EfficientNet‑B0 (5.3M parameters) is too large for ~700 images. MobileNetV3‑Large (5.4M) is similar, but MobileNetV3‑Small (2.5M) or ResNet‑18 (11.7M) may be better. We'll use MobileNetV3‑Large as it balances size and accuracy, and we'll apply stronger regularization.

4. Enhanced Augmentation & Training
Augmentations: Add random erasing, random grayscale, Gaussian blur to simulate real conditions.

MixUp & CutMix: Keep both with α=0.2 and α=1.0.

Label smoothing: 0.1.

Differential learning rates: backbone 1e-5, classifier 1e-3.

Learning rate scheduling: CosineAnnealingWarmRestarts.

Weight decay: 1e-4.

Dropout: 0.5 in classifier.

5. Cross‑Validation & Test Set
Use stratified k‑fold (5 folds) to ensure robust evaluation.

Reserve a final test set (e.g., 20% of all real + feedback) that is never used for validation during training.

Implementation Plan
We'll create a new script train_v10.py that:

Loads the original clean dataset (train/val/test) and the feedback images.

Verifies labels using detectors.

Splits all data into train/val/test while preserving class balance and hard examples.

Builds the model (MobileNetV3‑Large) with a custom classifier.

Trains with the described settings and logs detailed metrics.

Outputs the best model and final test accuracy.

====================================================================================

1. Root Causes of Poor Performance
1.1 Model Still Too Large
MobileNetV3‑Large (5.4M params) is similar to EfficientNet‑B0 (5.3M). With only 663 images, the model can memorize the training set, even with regularization.

The gap between validation (69%) and test (52%) confirms overfitting.

1.2 Label Noise Remains
Only 66% of front images had a detected face, and 63% of back images had a detected chip.

This suggests many images may be mislabeled or are extremely challenging (e.g., angles where face/chip not visible). The model learns from these noisy labels, leading to poor generalization.

1.3 Imbalance in Hard Examples
Feedback images: 57 front, only 24 back. The back class lacks challenging examples, so the model fails to learn robust back features.

1.4 Weak Augmentation for Back Class
The current augmentation (rotation, color jitter, blur) does not simulate chip-specific challenges (e.g., glare on chip, partial occlusion). Back images may require different augmentation than front.

1.5 Validation/Test Split Not Representative
The 15% test split might contain images that are too similar to training (e.g., same cards). Overfitting shows the model is memorizing the training set; a more diverse test set would expose this.

