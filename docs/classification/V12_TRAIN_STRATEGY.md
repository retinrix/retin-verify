Unified Detector + Classifier with YOLOv8: End‑to‑End Strategy
Training a single YOLOv8 model to detect and classify cards (CNIE front, CNIE back, other_card) simplifies your pipeline, eliminates the need for a separate stage‑1 classifier, and provides bounding boxes for visual feedback. Here’s a complete strategy.

1. Data Preparation
You already have 1279 labeled images (front, back, no‑card). For YOLO, you need bounding box annotations for each image. The boxes should tightly enclose the card (or the card region) and be assigned one of three classes:

cnie_front

cnie_back

other_card (formerly no‑card)

1.1 Annotation Tools
LabelImg (free, cross‑platform): draw rectangles and assign class labels.

CVAT (online/self‑hosted): more advanced, supports multiple users.

Roboflow (cloud‑based) – can also help with dataset management.

Process:

Open each image in the tool.

Draw a bounding box around the card (the entire card, not just the photo).

Choose the appropriate class (front, back, other).

Export annotations in YOLO format (one .txt file per image, with lines: class_id x_center y_center width height normalized to [0,1]).

1.2 Dataset Splits
Create train/ and val/ folders, each containing images/ and labels/ subdirectories.

80% of images go to train/, 20% to val/ (stratified by class to maintain balance).

If you have multiple images of the same card, ensure they are not split across train/val (group by card ID if known). If not, a random stratified split is acceptable.

2. Dataset Configuration File
Create a dataset.yaml file:

yaml
# dataset.yaml
path: /path/to/your/dataset   # absolute path
train: train/images
val: val/images

nc: 3                         # number of classes
names: ['cnie_front', 'cnie_back', 'other_card']
3. Model Selection
YOLOv8 offers several sizes:

YOLOv8n (nano): fastest, smallest (≈3M parameters). Suitable for real‑time on CPU/GPU.

YOLOv8s (small): better accuracy, still fast.

YOLOv8m (medium): heavier but more accurate.

Recommendation: Start with yolov8n.pt (pretrained on COCO). It will fine‑tune quickly on your card dataset and run at high frame rates.

4. Training
Install ultralytics:

bash
pip install ultralytics
Run training:

bash
yolo train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16
Key parameters:

epochs=100 – enough to converge, can be reduced if early stopping triggers.

imgsz=640 – good balance between speed and detail (cards are relatively large).

batch – adjust based on GPU memory.

You can add patience=10 for early stopping.

The model will be saved in runs/detect/train/weights/best.pt.

5. Evaluation
After training, compute metrics:

bash
yolo val model=runs/detect/train/weights/best.pt data=dataset.yaml
This outputs mAP@0.5, mAP@0.5:0.95, precision, recall.
You should aim for mAP@0.5 > 0.95 for a well‑tuned detector.

Test on a few unseen images or a short video to verify visual quality.

6. Integration with Live Stream
Use OpenCV to capture frames and YOLOv8 to detect and classify in real time:

python
import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, conf=0.5, iou=0.45)

    # Draw boxes and labels
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            # Draw rectangle and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow('CNIE Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Tuning:

Adjust conf (confidence threshold) to balance false positives vs. false negatives.

If detection is slow, reduce imgsz in training or use a smaller model (YOLOv8n).

7. Handling Misclassifications (e.g., French ID mistaken for CNIE)
Ensure your "other_card" class includes many examples of the offending cards (French ID, passport, etc.). The more varied, the better the model will discriminate.

Use hard negative mining: After training, run the model on a set of non‑CNIE images; collect those that are incorrectly predicted as CNIE and add them to the training set (with correct bounding boxes and class label “other_card”). Retrain.

Consider adding OCR as a post‑processing step: if the model predicts CNIE, run OCR on the detected region; if the text does not contain “بطاقة التعريف الوطنية” (Algerian ID), override the prediction to “other_card”.

8. Iterative Improvement with Feedback
Deploy the model and collect user‑flagged misclassifications (via your capture tool).

Add those images (with correct annotations) to the dataset.

Periodically retrain the model to incorporate new hard examples.

9. Why This Approach is Superior
Single model: No cascade, no separate stage‑1 classifier – simpler deployment and maintenance.

Bounding boxes: Provides visual feedback and allows cropping for further processing (e.g., OCR).

Real‑time performance: YOLOv8n runs at >30 FPS on a GPU (or >10 FPS on CPU with OpenVINO).

End‑to‑end learning: The model learns to associate the card’s layout with its class, not just the presence of a face or chip.

10. Next Steps
Annotate bounding boxes for your 1279 images (or at least 500–800 well‑selected images).

Create the dataset.yaml and split the data.

Train YOLOv8n and evaluate.

Integrate into your live‑stream application.

Iterate by collecting more hard examples.

This unified approach will give you a robust, real‑time card detector and classifier with high accuracy, solving both the bounding box requirement and the misclassification of non‑CNIE cards in one go.