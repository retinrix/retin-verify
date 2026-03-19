/**
 * CNIE Classification App
 * 4-Panel UI with camera, results, info, and history
 */

// Global state
let currentStream = null;
let currentCapture = null;
let currentPrediction = null;
let history = [];
let isUploading = false;

// DOM Elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const cameraPreview = document.getElementById('cameraPreview');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const loadingOverlay = document.getElementById('loadingOverlay');

// Buttons
const startBtn = document.getElementById('startBtn');
const captureBtn = document.getElementById('captureBtn');
const retakeBtn = document.getElementById('retakeBtn');
const uploadBtn = document.getElementById('uploadBtn');

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    loadModelInfo();
    updateHistoryUI();
});

// ==================== CAMERA FUNCTIONS ====================

async function startCamera() {
    try {
        showToast('Starting camera...', 'warning');
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        
        currentStream = stream;
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            video.style.display = 'block';
            cameraPlaceholder.style.display = 'none';
            cameraPreview.style.display = 'none';
            
            startBtn.style.display = 'none';
            captureBtn.disabled = false;
            
            document.getElementById('cameraStatus').style.color = '#28a745';
            showToast('Camera started', 'success');
            
            // Show guidance overlay
            showGuidance();
        };
    } catch (err) {
        console.error('Camera error:', err);
        showToast('Failed to start camera: ' + err.message, 'error');
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    video.style.display = 'none';
    document.getElementById('cameraStatus').style.color = '#dc3545';
    hideGuidance();
}

// ==================== GUIDANCE SYSTEM ====================

let guidanceDemoInterval = null;

function showGuidance() {
    const overlay = document.getElementById('guidanceOverlay');
    if (overlay) {
        overlay.classList.add('active');
        // Start cycling demo to show all states
        startGuidanceDemo();
    }
}

function hideGuidance() {
    const overlay = document.getElementById('guidanceOverlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
    stopGuidanceDemo();
}

function startGuidanceDemo() {
    // Cycle through guidance states to educate user
    const states = ['neutral', 'too-far', 'perfect', 'too-close'];
    let current = 0;
    
    // Show first state immediately
    updateGuidance(states[current]);
    
    // Cycle every 2 seconds
    guidanceDemoInterval = setInterval(() => {
        current = (current + 1) % states.length;
        updateGuidance(states[current]);
    }, 2000);
}

function stopGuidanceDemo() {
    if (guidanceDemoInterval) {
        clearInterval(guidanceDemoInterval);
        guidanceDemoInterval = null;
    }
}

function updateGuidance(state) {
    // state: 'too-close', 'too-far', 'perfect', 'neutral', 'angle'
    const targetBox = document.getElementById('targetBox');
    const guideTooClose = document.getElementById('guideTooClose');
    const guideTooFar = document.getElementById('guideTooFar');
    const guidePerfect = document.getElementById('guidePerfect');
    const guideAngle = document.getElementById('guideAngle');
    const distanceLabel = document.getElementById('distanceLabel');
    const distanceFill = document.getElementById('distanceFill');
    
    if (!targetBox) return; // Elements not found
    
    // Reset all
    targetBox.classList.remove('too-close', 'too-far', 'perfect');
    if (guideTooClose) guideTooClose.classList.remove('visible');
    if (guideTooFar) guideTooFar.classList.remove('visible');
    if (guidePerfect) guidePerfect.classList.remove('visible');
    if (guideAngle) guideAngle.classList.remove('visible');
    
    switch(state) {
        case 'too-close':
            targetBox.classList.add('too-close');
            if (guideTooClose) guideTooClose.classList.add('visible');
            if (distanceLabel) distanceLabel.textContent = '🔴 Too close! Move back';
            if (distanceFill) {
                distanceFill.style.width = '90%';
                distanceFill.style.background = '#f44336';
            }
            break;
            
        case 'too-far':
            targetBox.classList.add('too-far');
            if (guideTooFar) guideTooFar.classList.add('visible');
            if (distanceLabel) distanceLabel.textContent = '🟠 Too far! Move closer';
            if (distanceFill) {
                distanceFill.style.width = '20%';
                distanceFill.style.background = '#ff9800';
            }
            break;
            
        case 'perfect':
            targetBox.classList.add('perfect');
            if (guidePerfect) guidePerfect.classList.add('visible');
            if (distanceLabel) distanceLabel.textContent = '✅ Perfect! Hold steady';
            if (distanceFill) {
                distanceFill.style.width = '50%';
                distanceFill.style.background = '#4caf50';
            }
            break;
            
        case 'angle':
            if (guideAngle) guideAngle.classList.add('visible');
            if (distanceLabel) distanceLabel.textContent = '🔵 Keep card flat';
            break;
            
        default: // neutral
            if (distanceLabel) distanceLabel.textContent = '📍 Position card in box';
            if (distanceFill) {
                distanceFill.style.width = '50%';
                distanceFill.style.background = 'linear-gradient(90deg, #f44336 0%, #4caf50 100%)';
            }
    }
}

// Manual guidance controls
function setGuidanceTooClose() { updateGuidance('too-close'); }
function setGuidanceTooFar() { updateGuidance('too-far'); }
function setGuidancePerfect() { updateGuidance('perfect'); }
function setGuidanceAngle() { updateGuidance('angle'); }

function toggleBackTips() {
    const backTips = document.getElementById('backTips');
    const btn = document.getElementById('backTipsBtn');
    if (backTips) {
        if (backTips.style.display === 'none') {
            backTips.style.display = 'block';
            btn.textContent = '🔙 Hide Tips';
            btn.classList.add('active');
        } else {
            backTips.style.display = 'none';
            btn.textContent = '🔙 Back Tips';
            btn.classList.remove('active');
        }
    }
}

function captureImage() {
    if (!currentStream) return;
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get image data
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    currentCapture = imageData;
    
    // Show preview
    cameraPreview.src = imageData;
    cameraPreview.style.display = 'block';
    video.style.display = 'none';
    
    // Update buttons
    captureBtn.style.display = 'none';
    retakeBtn.style.display = 'inline-block';
    uploadBtn.style.display = 'inline-block';
    
    // Stop camera to save resources
    stopCamera();
    
    // Automatically classify
    classifyImage(imageData);
}

function retakeImage() {
    // Reset UI
    cameraPreview.style.display = 'none';
    cameraPreview.src = '';
    cameraPlaceholder.style.display = 'flex';
    
    retakeBtn.style.display = 'none';
    uploadBtn.style.display = 'none';
    startBtn.style.display = 'none';  // Hide start button during restart
    captureBtn.style.display = 'inline-block';
    captureBtn.disabled = true;  // Will be enabled when camera ready
    
    // Reset results
    document.getElementById('resultPlaceholder').style.display = 'block';
    document.getElementById('resultContent').classList.remove('active');
    document.getElementById('feedbackSection').style.display = 'none';
    const noCardWarning = document.getElementById('noCardWarning');
    if (noCardWarning) noCardWarning.style.display = 'none';
    
    currentCapture = null;
    currentPrediction = null;
    
    // Restart camera
    startCamera();
}

function uploadCapture() {
    if (!currentCapture) return;
    
    // Create download link
    const link = document.createElement('a');
    link.download = `capture_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.jpg`;
    link.href = currentCapture;
    link.click();
    
    showToast('Image downloaded', 'success');
}

// ==================== CLASSIFICATION ====================

async function classifyImage(imageData) {
    showLoading(true);
    
    try {
        // Convert base64 to blob
        const response = await fetch(imageData);
        const blob = await response.blob();
        
        // Create form data
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        
        // Call API
        const apiResponse = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!apiResponse.ok) {
            throw new Error('Classification failed');
        }
        
        const result = await apiResponse.json();
        currentPrediction = result;
        
        // Display results
        displayResults(result);
        
        // Add to history
        addToHistory(result, imageData);
        
        // ALWAYS show feedback section - user can flag any image
        const maxScore = Math.max(...Object.values(result.all_scores));
        const isNoCard = result.confidence < 0.6 || maxScore < 0.6;
        
        document.getElementById('feedbackSection').style.display = 'block';
        
        if (isNoCard) {
            showToast('⚠️ No CNIE card detected - please try again', 'warning');
        } else {
            showToast(`Classified as: ${result.predicted_class}`, 'success');
        }
        
    } catch (err) {
        console.error('Classification error:', err);
        showToast('Classification failed: ' + err.message, 'error');
    } finally {
        showLoading(false);
    }
}

function displayResults(result) {
    document.getElementById('resultPlaceholder').style.display = 'none';
    document.getElementById('resultContent').classList.add('active');
    
    // Check for "no card" - if confidence is low, likely no CNIE card
    const confidence = result.confidence;
    const maxScore = Math.max(...Object.values(result.all_scores));
    const isNoCard = confidence < 0.6 || maxScore < 0.6;  // 60% threshold
    
    // Update prediction
    const predClassEl = document.getElementById('predictedClass');
    const noCardWarning = document.getElementById('noCardWarning');
    if (isNoCard) {
        predClassEl.textContent = '⚠️ NO CNIE CARD';
        predClassEl.classList.add('no-card');
        if (noCardWarning) noCardWarning.style.display = 'block';
    } else {
        predClassEl.textContent = result.predicted_class;
        predClassEl.classList.remove('no-card');
        if (noCardWarning) noCardWarning.style.display = 'none';
    }
    
    // Update confidence display
    const confidencePercent = result.confidence * 100;
    document.getElementById('confidenceValue').textContent = confidencePercent.toFixed(1) + '%';
    document.getElementById('confidenceBar').style.width = confidencePercent + '%';
    document.getElementById('confidenceText').textContent = confidencePercent.toFixed(0) + '%';
    
    // Color code confidence
    const confBar = document.getElementById('confidenceBar');
    confBar.classList.remove('low', 'medium');
    if (confidencePercent < 50) {
        confBar.classList.add('low');
    } else if (confidencePercent < 70) {
        confBar.classList.add('medium');
    }
    
    // Update scores table
    const scoresBody = document.getElementById('scoresBody');
    scoresBody.innerHTML = '';
    
    Object.entries(result.all_scores)
        .sort((a, b) => b[1] - a[1])
        .forEach(([cls, score]) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${cls}</td>
                <td>${(score * 100).toFixed(1)}%</td>
            `;
            scoresBody.appendChild(row);
        });
    
    // Update inference time
    document.getElementById('inferenceTime').textContent = result.inference_time_ms.toFixed(1);
}

// ==================== FEEDBACK ====================

async function flagForRetraining() {
    if (!currentCapture || !currentPrediction || isUploading) return;
    
    isUploading = true;
    const flagBtn = document.getElementById('flagBtn');
    const progressDiv = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusText = document.getElementById('uploadStatus');
    
    flagBtn.disabled = true;
    progressDiv.classList.add('active');
    
    try {
        // Simulate progress
        progressFill.style.width = '30%';
        progressText.textContent = 'Preparing...';
        
        // Determine correct class (the other one)
        const correctClass = currentPrediction.predicted_class === 'cnie_front' 
            ? 'cnie_back' : 'cnie_front';
        
        progressFill.style.width = '60%';
        progressText.textContent = 'Uploading...';
        
        // Send feedback
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_base64: currentCapture.split(',')[1],
                predicted_class: currentPrediction.predicted_class,
                predicted_confidence: currentPrediction.confidence,
                is_correct: false,
                correct_class: correctClass,
                notes: 'Flagged via UI'
            })
        });
        
        progressFill.style.width = '100%';
        
        if (response.ok) {
            const result = await response.json();
            
            // Update history item status
            updateHistoryStatus(currentCapture, 'uploaded');
            
            statusText.innerHTML = '<span style="color: #28a745;">✓ Uploaded successfully!</span>';
            showToast('Image flagged for retraining', 'success');
            
            // Update model info
            loadModelInfo();
        } else {
            throw new Error('Upload failed');
        }
        
    } catch (err) {
        console.error('Upload error:', err);
        statusText.innerHTML = '<span style="color: #dc3545;">✗ Failed. <a href="#" onclick="flagForRetraining()">Retry</a></span>';
        updateHistoryStatus(currentCapture, 'failed');
        showToast('Upload failed: ' + err.message, 'error');
        flagBtn.disabled = false;
    } finally {
        isUploading = false;
        setTimeout(() => {
            progressDiv.classList.remove('active');
        }, 2000);
    }
}

async function saveAsNoCard() {
    // Save current capture as 'no_card' sample for 3-class training.
    if (!currentCapture || isUploading) return;
    
    isUploading = true;
    const progressDiv = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusText = document.getElementById('uploadStatus');
    
    progressDiv.classList.add('active');
    progressFill.style.width = '30%';
    progressText.textContent = 'Saving as no_card...';
    
    try {
        // Send to special no_card feedback endpoint
        const response = await fetch('/feedback_no_card', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_base64: currentCapture.split(',')[1],
                predicted_class: currentPrediction ? currentPrediction.predicted_class : 'unknown',
                predicted_confidence: currentPrediction ? currentPrediction.confidence : 0,
                notes: 'Manually marked as no_card'
            })
        });
        
        progressFill.style.width = '100%';
        
        if (response.ok) {
            updateHistoryStatus(currentCapture, 'uploaded');
            statusText.innerHTML = '<span style="color: #28a745;">✓ Saved as no_card!</span>';
            showToast('Saved as "no card" sample', 'success');
            loadModelInfo();
        } else {
            throw new Error('Save failed');
        }
    } catch (err) {
        // Fallback: just download the image
        progressFill.style.width = '100%';
        const link = document.createElement('a');
        link.download = `no_card_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.jpg`;
        link.href = currentCapture;
        link.click();
        
        statusText.innerHTML = '<span style="color: #ffc107;">⚠ Downloaded - manually move to feedback_data_3class/no_card/</span>';
        showToast('Downloaded - save to feedback_data_3class/no_card/', 'warning');
    } finally {
        isUploading = false;
        setTimeout(() => {
            progressDiv.classList.remove('active');
        }, 3000);
    }
}

// ==================== HISTORY ====================

function addToHistory(result, imageData) {
    const item = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        thumbnail: imageData,
        predicted_class: result.predicted_class,
        confidence: result.confidence,
        status: 'pending'
    };
    
    history.unshift(item);
    
    // Keep only last 20 (to avoid localStorage quota)
    if (history.length > 20) {
        history = history.slice(0, 20);
    }
    
    saveHistory();
    updateHistoryUI();
    updateModelInfo();
}

function updateHistoryStatus(imageData, status) {
    const item = history.find(h => h.thumbnail === imageData);
    if (item) {
        item.status = status;
        saveHistory();
        updateHistoryUI();
        updateModelInfo();
    }
}

function updateHistoryUI() {
    const container = document.getElementById('historyList');
    
    if (history.length === 0) {
        container.innerHTML = `
            <div class="history-empty">
                <p>No captures yet</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = history.map(item => {
        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleTimeString();
        const confidence = (item.confidence * 100).toFixed(1);
        
        let statusIcon = '⏳';
        let statusClass = '';
        let actions = '';
        
        switch (item.status) {
            case 'uploaded':
                statusIcon = '✓';
                statusClass = 'uploaded';
                break;
            case 'failed':
                statusIcon = '✗';
                statusClass = 'failed';
                actions = `<button class="btn btn-small btn-warning" onclick="retryUpload('${item.id}')">Retry</button>`;
                break;
            case 'flagged':
                statusIcon = '🚩';
                statusClass = 'flagged';
                break;
            default:
                actions = `<button class="btn btn-small btn-warning" onclick="flagHistoryItem('${item.id}')">Flag</button>`;
        }
        
        return `
            <div class="history-item ${statusClass}">
                <img class="history-thumb" src="${item.thumbnail}" alt="Capture">
                <div class="history-info">
                    <div class="history-class">${item.predicted_class}</div>
                    <div class="history-confidence">${confidence}% confidence</div>
                    <div class="history-time">${timeStr}</div>
                    <div class="history-actions">
                        ${actions}
                    </div>
                </div>
                <div class="history-status">
                    <span>${statusIcon}</span>
                </div>
            </div>
        `;
    }).join('');
}

async function flagHistoryItem(id) {
    const item = history.find(h => h.id === id);
    if (!item) return;
    
    item.status = 'flagged';
    updateHistoryUI();
    
    // Auto-upload
    try {
        const correctClass = item.predicted_class === 'cnie_front' 
            ? 'cnie_back' : 'cnie_front';
        
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_base64: item.thumbnail.split(',')[1],
                predicted_class: item.predicted_class,
                predicted_confidence: item.confidence,
                is_correct: false,
                correct_class: correctClass,
                notes: 'Flagged from history'
            })
        });
        
        if (response.ok) {
            item.status = 'uploaded';
            showToast('Image uploaded for retraining', 'success');
            loadModelInfo();
        } else {
            throw new Error('Upload failed');
        }
    } catch (err) {
        item.status = 'failed';
        showToast('Upload failed', 'error');
    }
    
    updateHistoryUI();
    saveHistory();
}

function retryUpload(id) {
    flagHistoryItem(id);
}

function clearHistory() {
    if (confirm('Clear all history?')) {
        history = [];
        saveHistory();
        updateHistoryUI();
        showToast('History cleared', 'success');
    }
}

function saveHistory() {
    try {
        localStorage.setItem('cnie_history', JSON.stringify(history));
    } catch (e) {
        if (e.name === 'QuotaExceededError') {
            console.warn('LocalStorage full, clearing old history');
            // Keep only last 10 items
            history = history.slice(0, 10);
            try {
                localStorage.setItem('cnie_history', JSON.stringify(history));
            } catch (e2) {
                // If still failing, clear completely
                localStorage.removeItem('cnie_history');
            }
        }
    }
}

function loadHistory() {
    const saved = localStorage.getItem('cnie_history');
    if (saved) {
        history = JSON.parse(saved);
    }
}

// ==================== MODEL INFO ====================

async function loadModelInfo() {
    try {
        const response = await fetch('/info');
        if (!response.ok) throw new Error('Failed to load info');
        
        const info = await response.json();
        
        document.getElementById('modelStatus').textContent = 'Ready';
        document.getElementById('modelName').textContent = info.model_path.split('/').pop();
        document.getElementById('modelClasses').textContent = info.classes.join(', ');
        document.getElementById('modelDevice').textContent = info.device;
        document.getElementById('modelInputSize').textContent = info.input_size + 'x' + info.input_size;
        
    } catch (err) {
        document.getElementById('modelStatus').textContent = 'Error';
        document.getElementById('modelStatus').className = 'status-badge status-error';
    }
    
    // Load feedback stats
    try {
        const statsResponse = await fetch('/feedback/stats');
        const stats = await statsResponse.json();
        
        document.getElementById('totalFlagged').textContent = stats.misclassified;
    } catch (err) {
        document.getElementById('totalFlagged').textContent = '?';
    }
    
    // Update total captures
    document.getElementById('totalCaptures').textContent = history.length;
}

function updateModelInfo() {
    document.getElementById('totalCaptures').textContent = history.length;
    const flagged = history.filter(h => h.status === 'uploaded' || h.status === 'flagged').length;
    document.getElementById('totalFlagged').textContent = flagged;
}

// ==================== UTILS ====================

function showLoading(show) {
    if (show) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}
