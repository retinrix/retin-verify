• Based on your GTX 950M (2 GB VRAM), here are smaller model alternatives that will fit and still achieve good accuracy:

  Model Downsizing Strategy

   Component                 Original Model                  Smaller Alternative          Memory     Trade-off
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Document Classification   EfficientNet-B3 (~48M params)   EfficientNet-B0 (~5.3M)      ~100 MB    2% accuracy drop, 9x faster
                                                             MobileNetV3-Small (~2.5M)    ~50 MB     Faster, slight accuracy drop
   Text Detection            DB-ResNet50                     DB-MobileNetV3               ~50 MB     Good for documents
                                                             PaddleOCR PP-OCRv4 (det)     ~20 MB     Optimized for edge
   OCR Engine                PaddleOCR full                  PP-OCRv4 lightweight         ~100 MB    Still <1% CER on MRZ
                                                             Tesseract 5 + LSTM           CPU only   No VRAM usage
   Field Extraction          LayoutLMv3-base (~500 MB)       LayoutLMv3-small (~100 MB)   ~200 MB    May need batch_size=1
                                                             Rule-based + Regex           0 MB       For MRZ (100% accurate)
   Face Detection            MTCNN                           YuNet (OpenCV DNN)           ~10 MB     Real-time on CPU
                                                             RetinaFace-Mobile            ~20 MB     Good accuracy
   Face Comparison           ArcFace-ResNet100               MobileFaceNet                ~20 MB     99%+ on LFW

  Recommended Pipeline for 2 GB VRAM

  ┌─────────────────────────────────────────────────────────────┐
  │  Pipeline Component              │  Model                   │
  ├─────────────────────────────────────────────────────────────┤
  │  1. Document Classification      │  EfficientNet-B0         │
  │  2. Text Detection               │  PP-OCRv4_det (light)    │
  │  3. OCR (General Text)           │  PP-OCRv4_rec (light)    │
  │  4. OCR (MRZ only)               │  Tesseract OCR-B         │
  │  5. Field Extraction             │  LayoutLMv3-small        │
  │  6. Face Detection               │  YuNet (OpenCV)          │
  │  7. Face Comparison              │  MobileFaceNet           │
  └─────────────────────────────────────────────────────────────┘
  Total VRAM: ~500-800 MB (leaves headroom for batch processing)

  Key Adjustments for Your GPU

  1. Training Considerations

  • Batch size = 1 or 2 for LayoutLMv3-small
  • Use gradient accumulation to simulate larger batches
  • Freeze backbone layers, train only classification heads

  2. Inference Optimizations

  # Use half precision (FP16)
  model = model.half().cuda()

  # Enable torch.compile (PyTorch 2.0+)
  model = torch.compile(model)

  # Batch documents when possible

  3. Hybrid Approach

   Task                      Device          Reason
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Document Classification   GPU             Fast, small model
   Text Detection            GPU             Parallelizable
   OCR                       GPU or CPU      PP-OCRv4 is efficient
   LayoutLMv3                GPU (batch=1)   Borderline, but works
   MRZ (Tesseract)           CPU             No GPU benefit
   Face Detection            CPU             YuNet is fast enough

  4. Model Sizes Comparison

   Model                   Parameters   Disk Size   VRAM (inference)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   EfficientNet-B3         48M          ~180 MB     ~600 MB
   EfficientNet-B0         5.3M         ~20 MB      ~100 MB
   LayoutLMv3-base         133M         ~500 MB     ~1.5 GB
   LayoutLMv3-small        31M          ~120 MB     ~400 MB
   PP-OCRv4_det (server)   -            ~100 MB     ~300 MB
   PP-OCRv4_det (mobile)   -            ~20 MB      ~80 MB

  Recommended Code Snippets

  Document Classification (EfficientNet-B0):

  from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

  model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
  model.classifier[1] = nn.Linear(1280, num_classes)  # 5 doc types

  Text Detection (PaddleOCR lightweight):

  from paddleocr import PaddleOCR

  ocr = PaddleOCR(
      det_model_dir='models/ch_PP-OCRv4_det_infer',  # Mobile version
      rec_model_dir='models/ch_PP-OCRv4_rec_infer',
      use_gpu=True,
      gpu_mem=1000  # Limit to 1GB
  )

  LayoutLMv3-small (if VRAM tight):

  from transformers import LayoutLMv3ForTokenClassification

  model = LayoutLMv3ForTokenClassification.from_pretrained(
      "microsoft/layoutlmv3-base",  # Use small if available
      num_labels=len(label_list)
  )
  # Run with batch_size=1, gradient_accumulation_steps=8