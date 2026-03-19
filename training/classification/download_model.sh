#!/bin/bash
# Download the trained model from Colab

echo "Note: Make sure the Cloudflare tunnel is active first!"
echo ""
echo "Run these commands to download the model:"
echo ""
echo "mkdir -p models logs"
echo "scp root@<your-colab-host>:/content/retin_retrain_3class/cnie_classifier_3class_v2.pth models/"
echo "scp root@<your-colab-host>:/content/retin_retrain_3class/train_simple.log logs/ 2>/dev/null || true"
echo ""
echo "Then update the inference engine to use v2 model:"
echo "  Edit inference_engine_3class.py line 14:"
echo "    MODEL_FILENAME = 'cnie_classifier_3class_v2.pth'"
