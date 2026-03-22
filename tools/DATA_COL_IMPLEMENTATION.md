We'll design a production-ready web-based capture tool using Flask (Python) for the backend and vanilla JavaScript for the frontend. The tool will:

Access the user's camera via getUserMedia.

Provide three big buttons: Front, Back, No‑Card.

Start capturing frames automatically when a button is pressed.

Use motion detection to save only when the card moves/changes significantly.

Show live statistics (counts per class) and advice (e.g., "Need more back images").

Upload captured images to the server for storage and metadata logging.

Optionally allow pausing/resuming.

We'll include a simple dashboard with a progress bar for each class and a target (e.g., 1000 per class). The tool will be modular and can be extended with face/chip verification.

Architecture Overview
Frontend: HTML/CSS/JS, uses MediaDevices API to get camera stream, draws frames on a canvas, and periodically (e.g., every 0.5 sec) sends the frame to the backend via fetch if a significant change is detected.

Backend: Flask server that receives images and saves them to a structured directory (captured/front/, captured/back/, captured/no_card/). Also maintains counts in a JSON file to display real-time stats.

Motion detection: On frontend, compare current frame with previous using simple pixel difference; if change exceeds threshold, send the frame.

Stats API: Endpoint to get current counts; frontend polls periodically to update the dashboard.

Detailed Implementation
Backend (Flask)
We'll create a minimal Flask app with routes:

/ – serves the main HTML page.

/upload – POST endpoint to receive image (as base64 or form data) with class label and timestamp.

/stats – GET endpoint returns JSON with current counts per class.

We'll store images in a directory structure and maintain a JSON file for counts.

python
# app.py
import os
import json
import base64
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

DATA_DIR = 'captured'
STATS_FILE = 'stats.json'

# Ensure directories exist
for label in ['front', 'back', 'no_card']:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {'front': 0, 'back': 0, 'no_card': 0}

def save_stats(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats', methods=['GET'])
def stats():
    stats = load_stats()
    return jsonify(stats)

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    label = data.get('label')
    image_data = data.get('image')  # base64 string without header
    if not label or not image_data:
        return jsonify({'error': 'Missing label or image'}), 400

    # Decode base64
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({'error': 'Invalid image data'}), 400

    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(DATA_DIR, label, filename)

    # Save image
    with open(filepath, 'wb') as f:
        f.write(image_bytes)

    # Update stats
    stats = load_stats()
    stats[label] += 1
    save_stats(stats)

    return jsonify({'status': 'ok', 'count': stats[label]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
Frontend (HTML/JS)
We'll create a simple but effective interface:

A video element displaying the camera feed.

A canvas for frame extraction and motion detection.

Buttons to start capture for each class.

A status area with live stats and advice.

Motion detection to avoid saving similar frames.

html
<!-- templates/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>CNIE Data Collector</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        video, canvas { width: 640px; height: 480px; border: 1px solid #ccc; background: #000; }
        canvas { display: none; }
        .buttons { margin: 20px 0; }
        button { font-size: 24px; padding: 10px 20px; margin: 0 10px; cursor: pointer; border: none; border-radius: 5px; color: white; }
        button.front { background-color: #4CAF50; }
        button.back { background-color: #2196F3; }
        button.no-card { background-color: #FF9800; }
        button.stop { background-color: #f44336; }
        .stats { margin-top: 20px; font-size: 18px; }
        .stat-item { margin: 5px 0; }
        .advice { font-size: 20px; font-weight: bold; color: #d32f2f; margin-top: 10px; }
        .progress { background-color: #ddd; border-radius: 5px; height: 20px; width: 100%; margin-top: 5px; }
        .progress-bar { background-color: #4CAF50; height: 100%; border-radius: 5px; width: 0%; }
    </style>
</head>
<body>
<div class="container">
    <h1>CNIE Data Collection Tool</h1>
    <p>Select a class, then move the card in front of the camera. Frames will be captured automatically.</p>
    <video id="video" autoplay playsinline></video>
    <canvas id="canvas"></canvas>

    <div class="buttons">
        <button id="btn-front" class="front">Front</button>
        <button id="btn-back" class="back">Back</button>
        <button id="btn-no-card" class="no-card">No Card</button>
        <button id="btn-stop" class="stop">Stop</button>
    </div>

    <div class="stats">
        <h3>Capture Statistics</h3>
        <div id="stats-front" class="stat-item">Front: 0</div>
        <div id="stats-back" class="stat-item">Back: 0</div>
        <div id="stats-no-card" class="stat-item">No Card: 0</div>
        <div id="advice" class="advice"></div>
        <div id="progress-front" class="progress"><div class="progress-bar" id="progress-front-bar"></div></div>
        <div id="progress-back" class="progress"><div class="progress-bar" id="progress-back-bar"></div></div>
        <div id="progress-no-card" class="progress"><div class="progress-bar" id="progress-no-card-bar"></div></div>
    </div>
</div>

<script>
    // DOM elements
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    // Buttons
    const btnFront = document.getElementById('btn-front');
    const btnBack = document.getElementById('btn-back');
    const btnNoCard = document.getElementById('btn-no-card');
    const btnStop = document.getElementById('btn-stop');

    // Stats elements
    const statsFront = document.getElementById('stats-front');
    const statsBack = document.getElementById('stats-back');
    const statsNoCard = document.getElementById('stats-no-card');
    const adviceDiv = document.getElementById('advice');

    // State
    let capturing = false;
    let currentLabel = null;
    let lastFrame = null;
    let frameCounter = 0;
    let captureInterval = null;

    // Target counts (adjust as needed)
    const TARGET = { front: 1000, back: 1000, no_card: 500 };
    let currentStats = { front: 0, back: 0, no_card: 0 };

    // Helper: update stats display and advice
    function updateStatsDisplay() {
        statsFront.innerText = `Front: ${currentStats.front}`;
        statsBack.innerText = `Back: ${currentStats.back}`;
        statsNoCard.innerText = `No Card: ${currentStats.no_card}`;

        // Update progress bars
        document.getElementById('progress-front-bar').style.width = Math.min(100, (currentStats.front / TARGET.front) * 100) + '%';
        document.getElementById('progress-back-bar').style.width = Math.min(100, (currentStats.back / TARGET.back) * 100) + '%';
        document.getElementById('progress-no-card-bar').style.width = Math.min(100, (currentStats.no_card / TARGET.no_card) * 100) + '%';

        // Advice: which class needs more
        const missing = [];
        if (currentStats.front < TARGET.front) missing.push(`front (${TARGET.front - currentStats.front} more)`);
        if (currentStats.back < TARGET.back) missing.push(`back (${TARGET.back - currentStats.back} more)`);
        if (currentStats.no_card < TARGET.no_card) missing.push(`no-card (${TARGET.no_card - currentStats.no_card} more)`);
        if (missing.length === 0) adviceDiv.innerText = "Great! All targets met. You can stop.";
        else adviceDiv.innerText = `Need more: ${missing.join(', ')}`;
    }

    // Fetch current stats from server
    async function refreshStats() {
        try {
            const response = await fetch('/stats');
            const data = await response.json();
            currentStats = data;
            updateStatsDisplay();
        } catch (e) {
            console.error('Error fetching stats', e);
        }
    }

    // Upload a frame (base64) with label
    async function uploadFrame(base64Image) {
        if (!currentLabel) return;
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: currentLabel, image: base64Image })
            });
            const result = await response.json();
            if (response.ok) {
                // Update stats locally from server response
                currentStats[currentLabel] = result.count;
                updateStatsDisplay();
            }
        } catch (e) {
            console.error('Upload failed', e);
        }
    }

    // Motion detection: compare two canvas frames
    function detectMotion(frame1, frame2, threshold = 30) {
        if (!frame1 || !frame2) return false;
        // Compute absolute difference
        let diff = 0;
        for (let i = 0; i < frame1.data.length; i += 4) {
            const r = Math.abs(frame1.data[i] - frame2.data[i]);
            const g = Math.abs(frame1.data[i+1] - frame2.data[i+1]);
            const b = Math.abs(frame1.data[i+2] - frame2.data[i+2]);
            diff += (r + g + b) / 3;
        }
        const avgDiff = diff / (frame1.data.length / 4);
        return avgDiff > threshold;
    }

    // Capture a frame from video, optionally compare with last frame
    async function captureFrame() {
        if (!capturing) return;

        // Draw current video frame to canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Get image data for motion detection
        const currentImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        let shouldSave = false;
        if (lastFrame === null) {
            shouldSave = true; // first frame
        } else {
            shouldSave = detectMotion(lastFrame, currentImageData, 25);
        }

        if (shouldSave) {
            // Convert canvas to base64 JPEG (quality 0.9)
            const base64 = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
            await uploadFrame(base64);
            lastFrame = currentImageData;
        }
    }

    // Start capturing for a given label
    function startCapture(label) {
        if (capturing) {
            // If already capturing, stop current first
            stopCapture();
        }
        currentLabel = label;
        capturing = true;
        lastFrame = null;
        // Reset frame counter? Not needed.
        // Start capturing at ~2 fps (500ms)
        captureInterval = setInterval(captureFrame, 500);
        // Update UI to show active capture
        document.body.style.backgroundColor = '#e8f5e9'; // light green
    }

    function stopCapture() {
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }
        capturing = false;
        currentLabel = null;
        document.body.style.backgroundColor = '';
    }

    // Initialize camera
    async function initCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            video.srcObject = stream;
            await video.play();
        } catch (err) {
            console.error('Camera access denied', err);
            alert('Camera access is required. Please grant permission.');
        }
    }

    // Event listeners
    btnFront.onclick = () => startCapture('front');
    btnBack.onclick = () => startCapture('back');
    btnNoCard.onclick = () => startCapture('no_card');
    btnStop.onclick = stopCapture;

    // Refresh stats periodically
    setInterval(refreshStats, 2000);
    initCamera();
    refreshStats();
</script>
</body>
</html>
