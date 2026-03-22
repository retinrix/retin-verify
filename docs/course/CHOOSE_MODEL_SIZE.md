When you have a small dataset (like ~500–700 images), the ratio of training samples to model parameters is critical. If the model has too many parameters relative to the data, it will overfit – memorizing the training set instead of learning generalizable features. This explains why even with a clean, balanced dataset, your V9 model (EfficientNet‑B0, 5.3M params) still performed poorly (63.8% test accuracy). The model simply had too much capacity for the amount of data.

1. Model Size & Parameters
Model	Parameters	Type
EfficientNet‑B0	5.3M	High capacity
MobileNetV3‑Large	5.4M	High capacity (similar to B0)
MobileNetV3‑Small	2.5M	Medium capacity
ResNet‑18	11.7M	Very high capacity
Custom CNN (small)	< 1M	Low capacity
Rule of thumb – For classification tasks, a safe sample‑to‑parameter ratio is at least 100:1 (i.e., 100 training samples per million parameters). For your 700 images, that would allow up to ~7M parameters if the data is highly diverse and you use strong regularization. But with only 500–700 images, even 5M parameters is risky.

2. Why Model Size Matters
Too many parameters → The model can easily memorize the training data, achieving near‑perfect training accuracy but failing on validation/test because it hasn't learned the underlying patterns.

Too few parameters → Underfitting – the model cannot capture the complexity of the task.

Optimal size → The model has enough capacity to learn the discriminative features but not enough to memorize noise.

In your case, the task (front vs back) is not visually trivial, but the distinguishing features (photo vs chip/MRZ) are localized and can be learned with a moderate‑sized model. EfficientNet‑B0 is overkill.

3. Reasoning for Model Selection
✅ MobileNetV3‑Small (2.5M params) – Recommended
Parameter count: 2.5M → safe with ~700 images (ratio ~280:1)

Architecture: Designed for mobile and embedded devices; efficient and fast.

Generalization: With strong regularization (dropout, label smoothing, MixUp), it can learn robust features without overfitting.

Expected performance: Should easily achieve >95% test accuracy on a well‑prepared dataset (with hard examples).

⚠️ MobileNetV3‑Large (5.4M params)
Parameter count: 5.4M → borderline; requires more data or extreme regularization.

Why not: You already tried EfficientNet‑B0 (5.3M) and got poor results. Likely still too large for your current dataset size, even with augmentation.

⚠️ ResNet‑18 (11.7M params)
Parameter count: 11.7M → too large for <1000 images. Will severely overfit unless you have massive augmentation and a huge dataset.

✅ Custom Tiny CNN (< 500k params)
If you want ultimate safety, you could design a small CNN with a few convolutional layers and a couple of fully‑connected layers. This would be extremely fast to train and resistant to overfitting. However, it may not achieve >95% accuracy if the distinguishing features are subtle.

4. The Sample‑to‑Parameter Ratio in Practice
General rule: For a small dataset (few hundred images), aim for 1M parameters or less for safe generalization.

With heavy regularization (dropout, weight decay, MixUp/CutMix, label smoothing) you can push a bit higher.

With transfer learning (pretrained on ImageNet) you can use a larger model because the features are already powerful and you only need to fine‑tune the top layers. But the fine‑tuning itself still needs enough examples to adapt to your domain.

Your V9 used EfficientNet‑B0 with strong regularization but still failed – suggesting the dataset size was the bottleneck, not the regularization strength.

5. Practical Advice for Your Next Experiment
Use MobileNetV3‑Small (2.5M params) as the base model.

Keep all the strong regularizations:

Label smoothing (0.1)

MixUp & CutMix

Dropout (0.5 in classifier)

Weight decay (1e-4)

Differential learning rates (backbone 1e-5, classifier 1e-3) to preserve pretrained features.

Increase dataset size by including all feedback images (hard examples).

Use stratified splitting to ensure the test set is representative.

With these changes, you should see test accuracy >95%.

6. If You Still Have Issues
If MobileNetV3‑Small still underperforms, you could:

Add more real images (even from different angles, lighting conditions, and card types).

Use a slightly larger input resolution (e.g., 256×256 instead of 224×224) to give the model more spatial detail.

Try a custom CNN (like a 3‑layer convnet) to see if the task is simple enough for a tiny model – if that also fails, the problem may be in the data (label noise, insufficient variability).

Summary
Model	Params	Suitable for ~700 images?	Reasoning
EfficientNet‑B0	5.3M	❌ Overfits	Too large; you saw poor performance
MobileNetV3‑Large	5.4M	❌ Overfits	Similar capacity; likely still too big
MobileNetV3‑Small	2.5M	✅ Good fit	Balances capacity and generalization
ResNet‑18	11.7M	❌ Severe overfitting	Way too large
Tiny Custom	< 0.5M	✅ Safe but may underfit	Could work if features are simple
Recommendation: Switch to MobileNetV3‑Small for your next training run (V11). Combine it with the full dataset (clean + feedback) and the strong augmentation/regularization pipeline. This should finally push accuracy above 95%.

