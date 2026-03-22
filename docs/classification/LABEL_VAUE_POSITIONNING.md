Here are the next steps to complete your synthetic CNIE dataset:

Step 1: Measure Real CNIE Positions
cd /home/retinrix/retin-verify
python3 synthetic/scripts/start_measure_server.py
Then open http://localhost:8765 and measure all 10 fields + 2 card corners.

Step 2: Update Config
Paste the generated config into synthetic/scripts/template_document_generator.py lines 70-88.

Step 3: Generate Test Samples
cd /home/retinrix/retin-verify
python3 synthetic/scripts/run_template_pipeline.py --sample-only
Step 4: Verify & Adjust
Check data/synthetic_review/cnie_front/:

All text within card boundaries?
Values to the right of labels?
Blood group visible on photo?
Nothing cut off?
If issues, tweak coordinates in template_document_generator.py (nudge by ±0.01-0.02).

Step 5: Generate Full Dataset
python3 synthetic/scripts/run_template_pipeline.py
Step 6: Export Training Formats
The pipeline auto-exports to:

COCO format (data/synthetic_coco/)
YOLO format (data/synthetic_yolo/)