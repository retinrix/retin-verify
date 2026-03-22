📥 Download & Deploy (Manual Steps)
Step 1: Download Model (On your local terminal)
# Download the trained model from Colab
scp root@accompanied-informed-junction-dispatched.trycloudflare.com:/content/v6_stage2_best.pth ~/retin-verify/models/classification/v6_stage2_front_back_new.pth
Step 2: Backup Old Model
# Backup current model
cd ~/retin-verify/models/classification
mkdir -p ../archive/2026-03-21
cp v6_stage2_front_back.pth ../archive/2026-03-21/v6_stage2_front_back_before_fix.pth

# Replace with new model
cp v6_stage2_front_back_new.pth v6_stage2_front_back.pth
Step 3: Update Inference Engine
The inference engine at ~/retin-verify/inference/apps/classification/backend/inference_engine_v6_cascade.py should already be using Class 0 = Front, Class 1 = Back (as the new model was trained).

Step 4: Restart Server
pkill -f api_server
cd ~/retin-verify/inference/apps/classification/backend
python3 api_server_v6_enhanced.py