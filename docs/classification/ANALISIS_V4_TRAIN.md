Analysis: Why the 3‑Class Model Misclassifies Real Front Images as Back
Based on the provided files and your additional information, the core issue is that the model trained with train_v4.py (which mixes synthetic data) has learned features that do not generalize to real CNIE images, causing symmetric confusion between front and back. Below is a breakdown of the contributing factors, followed by actionable recommendations.

🔍 Root Causes
1. Synthetic Data Does Not Match Real‑World Distribution
Construction method: Synthetic images are created by extracting backgrounds, adding text fields, and inserting a photo from the ICAO dataset. This process likely produces images that are too clean and lack the variability of real photos (lighting, shadows, angles, glare, background clutter).

Result: The model learns to associate the synthetic “style” with each class. When faced with a real front image, which may have different lighting, texture, or perspective, the model falls back on the synthetic features it memorized, often leading to misclassification.

2. Synthetic Front and Back Are Not Sufficiently Distinct
The confusion is symmetric (135 front→back, 120 back→front). This suggests that the learned representations for front and back overlap heavily.

If synthetic front and back images share similar backgrounds, text fonts, or synthetic artifacts, the model cannot build a robust boundary. Real images, with their natural variations, then fall on the wrong side of that boundary.

3. Overwhelming Proportion of Synthetic Data
train_v4.py uses up to 10,000 synthetic images (5,000 pairs). The number of real images is unknown but likely far smaller (maybe a few hundred). Even with a 3× weight on real samples, the model still sees many synthetic batches and can overfit to synthetic patterns.

The 2‑class model from your session report achieved 99–100% accuracy on real images because it was trained exclusively on real data (with augmentation). This proves that real data alone is sufficient for the task.

4. Third Class (“no_card”) Adds Complexity Without Enough Real Data
The no‑card class includes credit cards, passports, etc. These may share visual elements with CNIE (e.g., photos, text blocks). If the model hasn’t seen enough real no‑card examples, it might incorrectly map some real front images to the no‑card region, but your feedback shows confusion between front and back, not with no‑card. So this is a secondary issue.

5. Training Configuration May Hinder Generalization
Layer freezing: Freezing the first 100 layers of EfficientNet‑B0 might prevent the model from adapting low‑level features (edges, textures) to the real domain.

Uniform learning rate (1e‑4 for all parameters) can cause catastrophic forgetting of useful pretrained features, especially when fine‑tuning on synthetic data. The incident report recommended differential learning rates (backbone 1e‑5, classifier 1e‑3) – this was not implemented.

Augmentation in train_v4.py is strong, but it may not simulate the specific challenges of real CNIE photos (e.g., glare on glossy surfaces, extreme angles, partial occlusion).

6. Potential Label Noise in Synthetic Data
If any synthetic front images were incorrectly labeled as back (or vice versa) during generation, the model receives conflicting signals, further blurring the front/back boundary.

📊 Supporting Evidence from Your Feedback
Misclassification Type	Count	Observation
Front → Back	135	Model consistently mistakes real fronts for backs
Back → Front	120	Symmetric confusion confirms poor feature separation
No‑Card	107	Third class is not the main source of front/back confusion
The symmetric confusion points directly to a weak discriminative feature space for front vs. back, likely caused by the synthetic training data.

🛠️ Recommended Solutions
Immediate Steps (Low Effort, High Impact)
Retrain Without Synthetic Data

Use only real images + augmentation, as in the successful 2‑class experiment.

If you need a 3‑class model (including no‑card), collect more real no‑card examples or use the existing ones (107 feedback images) and augment them heavily.

This will likely eliminate the front/back confusion immediately.

Improve Synthetic Data Realism

Apply domain randomization: add realistic lighting, shadows, camera noise, and perspective distortions to synthetic images.

Mix synthetic and real images in a controlled ratio (e.g., 1:1) and monitor validation on a real test set.

Use Differential Learning Rates

Set backbone LR to 1e‑5 (or even freeze it entirely) and classifier LR to 1e‑3. This preserves pretrained features while adapting the classifier to your task.

Enhance Augmentation to Mimic Real‑World Challenges

Add random glare simulation, motion blur, extreme rotations (up to 45°), and perspective warping to cover real photo conditions.

Evaluate with a Clean Test Set

Ensure you have a held‑out set of real front/back images (different from training/feedback) to measure true performance.

Medium‑Term Improvements
Collect More Real Data

Even 100–200 additional real images per class can significantly improve robustness. Use your feedback pipeline to gather challenging cases.

Use MixUp or CutMix

These regularization techniques can help the model generalize better and reduce overfitting to synthetic artifacts.

Monitor Confidence and Feature Embeddings

Track average confidence and logit distributions during training. If confidence on real validation images is low or highly variable, it signals a domain gap.

Long‑Term Strategy
Domain Adaptation

If synthetic data is unavoidable, use adversarial domain adaptation or feature‑level alignment to force the model to learn domain‑invariant features.

Ensemble with 2‑Class Model

Combine predictions from the 2‑class real‑only model and the 3‑class model, using a rule‑based fallback when confidence is low.

🧪 Suggested Experiment
Train a 3‑class model only on real images (your existing real data + the 107 no‑card feedback images) with heavy augmentation, differential LRs, and early stopping.

Evaluate on your real test set.

If accuracy on front/back is high (>95%), the problem was indeed synthetic data.

Then gradually reintroduce synthetic data with improved realism and monitor the impact on real test accuracy.

📌 Summary
The poor performance on real front images is a classic case of domain shift: the model learned from synthetic data that does not represent real‑world variability. The symmetric front/back confusion confirms that the synthetic features are not robust. The solution is to prioritize real data and either eliminate or drastically improve the synthetic pipeline.

Let me know if you need help implementing any of these steps!

