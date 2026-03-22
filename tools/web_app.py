#!/usr/bin/env python3
"""
CNIE Dataset Tools - Web Application with Browser Camera
Uses getUserMedia API for reliable camera access.
"""

import os
import sys
import json
import shutil
import base64
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file
from PIL import Image

app = Flask(__name__)

# Configuration
DATASET_DIR = Path.home() / "retin-verify" / "training_data"
DEFAULT_DATASET = DATASET_DIR / "v8_stage2_clean"
CAPTURE_DIR = DATASET_DIR / "v10_manual_capture"
STATS_FILE = DATASET_DIR / "capture_stats.json"

# Ensure directories exist
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
for cls in ['front', 'back', 'no_card']:
    (CAPTURE_DIR / cls).mkdir(exist_ok=True)

# Target counts
TARGETS = {'front': 300, 'back': 300, 'no_card': 150}


def load_stats():
    """Load capture statistics."""
    if STATS_FILE.exists():
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    # Count existing files
    stats = {}
    for cls in ['front', 'back', 'no_card']:
        folder = CAPTURE_DIR / cls
        stats[cls] = len(list(folder.glob('*.jpg'))) if folder.exists() else 0
    return stats


def save_stats(stats):
    """Save capture statistics."""
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)


def get_dataset_stats():
    """Get combined stats from all datasets."""
    stats = {'front': 0, 'back': 0, 'no_card': 0}
    
    # Check capture directory
    for cls in ['front', 'back', 'no_card']:
        folder = CAPTURE_DIR / cls
        if folder.exists():
            stats[cls] = len(list(folder.glob('*.jpg')))
    
    # Check main dataset
    for split in ['train', 'val', 'test']:
        for cls in ['cnie_front', 'cnie_back']:
            folder = DEFAULT_DATASET / split / cls
            if folder.exists():
                count = len(list(folder.glob('*.jpg')))
                if 'front' in cls:
                    stats['front'] += count
                else:
                    stats['back'] += count
    
    return stats


def render_page(title, page_id, content_html, extra_scripts=''):
    """Render page with navigation."""
    
    nav_items = {
        'home': ('🏠', 'Dashboard', '/'),
        'capture': ('📸', 'Smart Capture', '/capture'),
        'manual': ('✋', 'Manual Review', '/manual'),
        'cleaner': ('🧹', 'Dataset Cleaner', '/cleaner'),
        'stats': ('📈', 'Statistics', '/stats'),
        'train': ('🤖', 'Train Model', '/train'),
        'evaluate': ('🧪', 'Evaluate', '/evaluate'),
    }
    
    nav_html = ''
    sections = [
        ('Overview', ['home']),
        ('Data Collection', ['capture', 'manual']),
        ('Quality Control', ['cleaner', 'stats']),
        ('Training', ['train', 'evaluate']),
    ]
    
    for section_title, items in sections:
        nav_html += f'<div class="nav-section"><div class="nav-section-title">{section_title}</div>'
        for item_id in items:
            icon, label, url = nav_items[item_id]
            active_class = 'active' if item_id == page_id else ''
            nav_html += f'<a href="{url}" class="nav-item {active_class}"><span>{icon}</span> {label}</a>'
        nav_html += '</div>'
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>CNIE Dataset Tools - {title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            height: 100vh;
            overflow: hidden;
        }}
        .container {{
            display: flex;
            height: 100vh;
        }}
        .sidebar {{
            width: 260px;
            background: #2c3e50;
            color: white;
            display: flex;
            flex-direction: column;
        }}
        .sidebar-header {{
            padding: 20px;
            background: #1a252f;
            border-bottom: 1px solid #34495e;
        }}
        .sidebar-header h1 {{ font-size: 18px; font-weight: 600; }}
        .sidebar-header p {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
        .nav-menu {{ flex: 1; overflow-y: auto; padding: 10px 0; }}
        .nav-section {{ margin-bottom: 10px; }}
        .nav-section-title {{
            padding: 10px 20px;
            font-size: 11px;
            text-transform: uppercase;
            color: #7f8c8d;
            letter-spacing: 1px;
        }}
        .nav-item {{
            display: flex;
            align-items: center;
            padding: 12px 20px;
            color: #bdc3c7;
            text-decoration: none;
            transition: all 0.2s;
            border-left: 3px solid transparent;
        }}
        .nav-item:hover {{ background: #34495e; color: white; }}
        .nav-item.active {{ background: #34495e; color: white; border-left-color: #3498db; }}
        .nav-item span {{ width: 24px; margin-right: 10px; font-size: 18px; }}
        .sidebar-footer {{
            padding: 15px 20px;
            background: #1a252f;
            border-top: 1px solid #34495e;
            font-size: 12px;
            color: #7f8c8d;
        }}
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .top-bar {{
            background: white;
            padding: 15px 25px;
            border-bottom: 1px solid #e1e4e8;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .top-bar h2 {{ font-size: 20px; color: #2c3e50; }}
        .content-area {{ flex: 1; overflow: auto; padding: 25px; }}
        
        /* Custom Scrollbar for Panels */
        .custom-scroll::-webkit-scrollbar {{
            width: 6px;
        }}
        .custom-scroll::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 3px;
        }}
        .custom-scroll::-webkit-scrollbar-thumb {{
            background: #888;
            border-radius: 3px;
        }}
        .custom-scroll::-webkit-scrollbar-thumb:hover {{
            background: #555;
        }}
        
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .card-title {{ font-size: 16px; font-weight: 600; color: #2c3e50; }}
        
        /* Capture Page Styles */
        .capture-container {{
            text-align: center;
            max-width: 900px;
            margin: 0 auto;
        }}
        .video-wrapper {{
            position: relative;
            display: inline-block;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        #video {{
            width: 640px;
            height: 480px;
            display: block;
        }}
        #canvas {{ display: none; }}
        .capture-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: left;
            font-size: 14px;
            min-width: 200px;
        }}
        .capture-overlay .status {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .capture-overlay .status.capturing {{ color: #4CAF50; }}
        .capture-overlay .status.waiting {{ color: #FFC107; }}
        .big-buttons {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        .big-btn {{
            font-size: 24px;
            padding: 20px 40px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
            color: white;
            font-weight: 600;
            min-width: 180px;
        }}
        .big-btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
        .big-btn:active {{ transform: translateY(0); }}
        .big-btn.front {{ background: linear-gradient(135deg, #4CAF50, #45a049); }}
        .big-btn.back {{ background: linear-gradient(135deg, #2196F3, #1976D2); }}
        .big-btn.no-card {{ background: linear-gradient(135deg, #FF9800, #F57C00); }}
        .big-btn.stop {{ background: linear-gradient(135deg, #f44336, #d32f2f); }}
        .big-btn.active {{ box-shadow: 0 0 0 4px rgba(255,255,255,0.5), 0 4px 12px rgba(0,0,0,0.3); transform: scale(1.05); }}
        
        .stats-panel {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e1e4e8;
        }}
        .stat-row:last-child {{ border-bottom: none; }}
        .progress-container {{
            flex: 1;
            margin: 0 20px;
        }}
        .progress-bar-bg {{
            background: #e1e4e8;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .progress-bar-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }}
        .progress-bar-fill.front {{ background: #4CAF50; }}
        .progress-bar-fill.back {{ background: #2196F3; }}
        .progress-bar-fill.no-card {{ background: #FF9800; }}
        
        .advice {{
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
        }}
        .advice.need-more {{ background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }}
        .advice.good {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: 700;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-block;
        }}
        .btn-primary {{ background: #3498db; color: white; }}
        .btn-primary:hover {{ background: #2980b9; }}
        .btn-success {{ background: #27ae60; color: white; }}
        .btn-danger {{ background: #e74c3c; color: white; }}
        .btn-warning {{ background: #f39c12; color: white; }}
        
        /* Pulse animation for target completion */
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(255,255,255,0.7); }}
            50% {{ transform: scale(1.02); box-shadow: 0 0 30px 10px rgba(255,255,255,0.3); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>📊 CNIE Tools</h1>
                <p>Dataset Management Suite</p>
            </div>
            <nav class="nav-menu">
                {nav_html}
            </nav>
            <div class="sidebar-footer">
                <div>v2.0.0</div>
                <div style="margin-top: 5px;">CNIE Classifier Tools</div>
            </div>
        </div>
        <div class="main-content">
            <div class="top-bar">
                <h2>{title}</h2>
            </div>
            <div class="content-area">
                {content_html}
            </div>
        </div>
    </div>
    {extra_scripts}
</body>
</html>'''
    return html


# ============ PAGES ============

@app.route('/')
def home():
    stats = get_dataset_stats()
    total = sum(stats.values())
    ratio = stats['back'] / stats['front'] if stats['front'] > 0 else 0
    
    content = f'''
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">{total}</div>
        <div class="stat-label">Total Images</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{stats['front']}</div>
        <div class="stat-label">Front Images</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{stats['back']}</div>
        <div class="stat-label">Back Images</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{ratio:.2f}:1</div>
        <div class="stat-label">Back:Front Ratio</div>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <div class="card-title">Quick Actions</div>
    </div>
    <div style="display: flex; gap: 15px; flex-wrap: wrap;">
        <a href="/capture" class="btn btn-success">📸 Start Capture</a>
        <a href="/manual" class="btn btn-primary">✋ Manual Review</a>
        <a href="/stats" class="btn btn-warning">📊 View Statistics</a>
        <a href="/cleaner" class="btn btn-danger">🧹 Clean Dataset</a>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <div class="card-title">Collection Progress</div>
    </div>
    <div class="stat-row">
        <span>Front</span>
        <div class="progress-container">
            <div class="progress-bar-bg">
                <div class="progress-bar-fill front" style="width: {min(stats['front']/TARGETS['front']*100, 100)}%"></div>
            </div>
        </div>
        <span>{stats['front']} / {TARGETS['front']}</span>
    </div>
    <div class="stat-row">
        <span>Back</span>
        <div class="progress-container">
            <div class="progress-bar-bg">
                <div class="progress-bar-fill back" style="width: {min(stats['back']/TARGETS['back']*100, 100)}%"></div>
            </div>
        </div>
        <span>{stats['back']} / {TARGETS['back']}</span>
    </div>
    <div class="stat-row">
        <span>No-Card</span>
        <div class="progress-container">
            <div class="progress-bar-bg">
                <div class="progress-bar-fill no-card" style="width: {min(stats['no_card']/TARGETS['no_card']*100, 100)}%"></div>
            </div>
        </div>
        <span>{stats['no_card']} / {TARGETS['no_card']}</span>
    </div>
</div>
'''
    return render_template_string(render_page('Dashboard', 'home', content))


@app.route('/capture')
def capture():
    # Get target parameters from URL (from advisor click)
    target_split = request.args.get('split', '')
    target_class = request.args.get('class', '')
    target_count = request.args.get('count', '')
    
    content = f'''
<div class="capture-container">
    <!-- Target Compensation Panel -->
    <div id="target-panel" style="display: {'block' if target_split else 'none'}; margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; text-align: center;">
        <div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">🎯 Compensation Target</div>
        <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
            <div><span style="opacity: 0.8;">Split:</span> <strong id="target-split-display">{target_split or 'Not set'}</strong></div>
            <div><span style="opacity: 0.8;">Class:</span> <strong id="target-class-display">{target_class or 'Not set'}</strong></div>
            <div><span style="opacity: 0.8;">Need:</span> <strong id="target-count-display">{target_count or '?'}</strong> images</div>
            <div><span style="opacity: 0.8;">Captured:</span> <strong id="target-captured">0</strong></div>
            <div><span style="opacity: 0.8;">Remaining:</span> <strong id="target-remaining">{target_count or '?'}</strong></div>
        </div>
        <div id="target-completion" style="display: none; margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 8px; font-size: 20px;">
            ✅ Target Complete! <a href="/stats" style="color: white; text-decoration: underline;">Check Stats →</a>
        </div>
    </div>

    <div class="video-wrapper">
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
        <div class="capture-overlay">
            <div class="status waiting" id="status-text">Ready</div>
            <div>Mode: <span id="current-mode">-</span></div>
            <div>Motion: <span id="motion-level">0</span>%</div>
            <div style="margin-top: 10px; border-top: 1px solid #555; padding-top: 10px;">
                <div>Front: <span id="live-front">0</span></div>
                <div>Back: <span id="live-back">0</span></div>
                <div>No-Card: <span id="live-nocard">0</span></div>
            </div>
        </div>
    </div>
    
    <!-- Split Selection -->
    <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border: 2px solid #e9ecef;">
        <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
            <label style="font-weight: 600; color: #495057;">📁 Save to Split:</label>
            <select id="split-select" class="form-select" style="width: 200px;" onchange="updateSplit()">
                <option value="train" {'selected' if target_split == 'train' else ''}>train (Training)</option>
                <option value="val" {'selected' if target_split == 'val' else ''}>val (Validation)</option>
                <option value="test" {'selected' if target_split == 'test' else ''}>test (Testing)</option>
            </select>
            <span style="color: #6c757d; font-size: 14px;">
                Images will be saved to: <code id="save-path">v10_manual_capture/train/</code>
            </span>
        </div>
    </div>
    
    <div class="big-buttons">
        <button class="big-btn front" id="btn-front" onclick="startCapture('front')">
            📷 FRONT
        </button>
        <button class="big-btn back" id="btn-back" onclick="startCapture('back')">
            📷 BACK
        </button>
        <button class="big-btn no-card" id="btn-no-card" onclick="startCapture('no_card')">
            📷 NO CARD
        </button>
        <button class="big-btn" style="background: #9c27b0;" onclick="manualCapture()">
            📸 CAPTURE NOW
        </button>
        <button class="big-btn stop" id="btn-stop" onclick="stopCapture()">
            ⏹ STOP
        </button>
    </div>
    
    <div style="margin: 15px 0; padding: 10px; background: #f0f0f0; border-radius: 8px;">
        <label>Motion Sensitivity: </label>
        <input type="range" id="sensitivity" min="5" max="100" value="25" onchange="updateSensitivity(this.value)">
        <span id="sensitivity-val">25</span>
        <span style="margin-left: 20px; color: #666; font-size: 14px;">
            (Lower = more sensitive, Higher = less sensitive)
        </span>
    </div>
    
    <div class="stats-panel">
        <div class="card-header">
            <div class="card-title">Capture Statistics</div>
        </div>
        <div class="stat-row">
            <span>🎯 Front</span>
            <div class="progress-container">
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill front" id="progress-front" style="width: 0%"></div>
                </div>
            </div>
            <span id="stat-front">0 / 300</span>
        </div>
        <div class="stat-row">
            <span>🎯 Back</span>
            <div class="progress-container">
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill back" id="progress-back" style="width: 0%"></div>
                </div>
            </div>
            <span id="stat-back">0 / 300</span>
        </div>
        <div class="stat-row">
            <span>🎯 No-Card</span>
            <div class="progress-container">
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill no-card" id="progress-nocard" style="width: 0%"></div>
                </div>
            </div>
            <span id="stat-nocard">0 / 150</span>
        </div>
        
        <div class="advice need-more" id="advice-box">
            Click a button above to start capturing. Move the card to trigger capture.
        </div>
    </div>
</div>
'''
    
    scripts = '''<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let capturing = false;
let currentLabel = null;
let lastFrame = null;
let captureInterval = null;
let motionThreshold = 15; // Lower = more sensitive
let currentStats = {front: 0, back: 0, no_card: 0};
const TARGETS = {front: 300, back: 300, no_card: 150};

// Target compensation tracking
let targetSplit = '';
let targetClass = '';
let targetCount = 0;
let targetCaptured = 0;
let currentSplit = 'train'; // Default split

// Initialize target from URL parameters
function initTargetFromURL() {
    const params = new URLSearchParams(window.location.search);
    targetSplit = params.get('split') || '';
    targetClass = params.get('class') || '';
    targetCount = parseInt(params.get('count')) || 0;
    
    if (targetSplit) {
        currentSplit = targetSplit;
        document.getElementById('split-select').value = targetSplit;
    }
    
    updateSavePath();
    
    // Auto-start capture if target class is specified
    if (targetClass && ['front', 'back', 'no_card'].includes(targetClass)) {
        setTimeout(() => startCapture(targetClass), 500);
    }
}

// Update split selection
function updateSplit() {
    currentSplit = document.getElementById('split-select').value;
    updateSavePath();
}

// Update displayed save path
function updateSavePath() {
    const path = `v10_manual_capture/${currentSplit}/`;
    document.getElementById('save-path').textContent = path;
}

// Update target tracker display
function updateTargetTracker() {
    if (!targetClass || !targetSplit) return;
    
    // Only count if current capture matches target
    if (currentLabel === targetClass) {
        targetCaptured++;
    }
    
    const remaining = Math.max(0, targetCount - targetCaptured);
    
    document.getElementById('target-captured').textContent = targetCaptured;
    document.getElementById('target-remaining').textContent = remaining;
    
    if (remaining === 0 && targetCount > 0) {
        document.getElementById('target-completion').style.display = 'block';
        // Flash success
        flashCompletion();
    }
}

// Flash completion effect
function flashCompletion() {
    const panel = document.getElementById('target-panel');
    panel.style.animation = 'pulse 1s ease-in-out 3';
    setTimeout(() => {
        panel.style.animation = '';
    }, 3000);
}

// Initialize camera
async function initCamera() {
    try {
        console.log('Requesting camera...');
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            console.log('Video ready:', video.videoWidth, 'x', video.videoHeight);
            video.play();
        };
        
        console.log('Camera stream acquired');
    } catch (err) {
        console.error('Camera error:', err);
        alert('Camera error: ' + err.message);
    }
}

// Motion detection
function detectMotion(frame1, frame2) {
    if (!frame1 || !frame2) return false;
    let diff = 0;
    for (let i = 0; i < frame1.data.length; i += 4) {
        diff += Math.abs(frame1.data[i] - frame2.data[i]);
        diff += Math.abs(frame1.data[i+1] - frame2.data[i+1]);
        diff += Math.abs(frame1.data[i+2] - frame2.data[i+2]);
    }
    const avgDiff = diff / (frame1.data.length / 4);
    return avgDiff;
}

// Capture and upload frame
let lastCaptureTime = 0;
const CAPTURE_COOLDOWN = 1000; // 1 second between captures

async function captureFrame() {
    if (!capturing || !currentLabel) return;
    
    // Check if video is ready
    if (!video.videoWidth) {
        console.log('Video not ready yet');
        return;
    }
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const currentImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const motionLevel = lastFrame ? detectMotion(lastFrame, currentImageData) : 0;
    
    // Update motion display (scale for display)
    const displayMotion = Math.min(100, Math.round(motionLevel / 2));
    document.getElementById('motion-level').textContent = displayMotion;
    
    // Debug info
    console.log('Motion level:', motionLevel, 'Threshold:', motionThreshold, 'Display:', displayMotion);
    
    // Capture if motion detected AND cooldown passed
    const now = Date.now();
    if (motionLevel > motionThreshold && (now - lastCaptureTime) > CAPTURE_COOLDOWN) {
        console.log('CAPTURING! Motion:', motionLevel);
        lastCaptureTime = now;
        
        const base64 = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({label: currentLabel, image: base64, split: currentSplit})
            });
            const result = await response.json();
            
            console.log('Upload result:', result);
            
            if (result.success) {
                currentStats[currentLabel] = result.count;
                updateStatsDisplay();
                updateTargetTracker();
                flashCapture();
                document.getElementById('status-text').textContent = 'SAVED!';
                setTimeout(() => {
                    if (capturing) document.getElementById('status-text').textContent = 'CAPTURING!';
                }, 500);
            } else {
                console.error('Upload failed:', result.error);
            }
        } catch (e) {
            console.error('Upload error:', e);
        }
        
        lastFrame = currentImageData;
    } else {
        // Update status with motion level
        if (capturing) {
            document.getElementById('status-text').textContent = 
                motionLevel > motionThreshold ? 'COOLDOWN...' : 'CAPTURING!';
        }
    }
}

// Visual feedback for capture
function flashCapture() {
    const overlay = document.querySelector('.video-wrapper');
    overlay.style.boxShadow = '0 0 50px #4CAF50';
    setTimeout(() => {
        overlay.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
    }, 200);
}

// Start capturing
function startCapture(label) {
    if (capturing) stopCapture();
    
    currentLabel = label;
    capturing = true;
    lastFrame = null;
    
    // Update UI
    document.querySelectorAll('.big-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('btn-' + label.replace('_', '-')).classList.add('active');
    document.getElementById('status-text').textContent = 'CAPTURING!';
    document.getElementById('status-text').className = 'status capturing';
    document.getElementById('current-mode').textContent = label.toUpperCase();
    
    // Start capture loop (5 fps)
    captureInterval = setInterval(captureFrame, 200);
    
    updateAdvice();
}

// Stop capturing
function stopCapture() {
    capturing = false;
    currentLabel = null;
    clearInterval(captureInterval);
    
    document.querySelectorAll('.big-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('status-text').textContent = 'Ready';
    document.getElementById('status-text').className = 'status waiting';
    document.getElementById('current-mode').textContent = '-';
    document.getElementById('motion-level').textContent = '0';
}

// Update stats display
function updateStatsDisplay() {
    // Update stat numbers
    document.getElementById('stat-front').textContent = currentStats.front + ' / ' + TARGETS.front;
    document.getElementById('stat-back').textContent = currentStats.back + ' / ' + TARGETS.back;
    document.getElementById('stat-nocard').textContent = currentStats.no_card + ' / ' + TARGETS.no_card;
    
    document.getElementById('live-front').textContent = currentStats.front;
    document.getElementById('live-back').textContent = currentStats.back;
    document.getElementById('live-nocard').textContent = currentStats.no_card;
    
    // Update progress bars
    document.getElementById('progress-front').style.width = Math.min(100, (currentStats.front / TARGETS.front) * 100) + '%';
    document.getElementById('progress-back').style.width = Math.min(100, (currentStats.back / TARGETS.back) * 100) + '%';
    document.getElementById('progress-nocard').style.width = Math.min(100, (currentStats.no_card / TARGETS.no_card) * 100) + '%';
    
    updateAdvice();
}

// Update advice text
function updateAdvice() {
    const missing = [];
    if (currentStats.front < TARGETS.front) missing.push(`front (${TARGETS.front - currentStats.front})`);
    if (currentStats.back < TARGETS.back) missing.push(`back (${TARGETS.back - currentStats.back})`);
    if (currentStats.no_card < TARGETS.no_card) missing.push(`no-card (${TARGETS.no_card - currentStats.no_card})`);
    
    const adviceBox = document.getElementById('advice-box');
    if (missing.length === 0) {
        adviceBox.textContent = '✅ All targets met! Great job!';
        adviceBox.className = 'advice good';
    } else {
        adviceBox.textContent = '📸 Need more: ' + missing.join(', ');
        adviceBox.className = 'advice need-more';
    }
}

// Load stats from server
async function loadStats() {
    try {
        const response = await fetch('/api/stats/capture');
        currentStats = await response.json();
        updateStatsDisplay();
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

// Manual capture function
async function manualCapture() {
    if (!currentLabel) {
        alert('Please click FRONT, BACK, or NO-CARD first!');
        return;
    }
    
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const base64 = canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
    
    try {
        document.getElementById('status-text').textContent = 'UPLOADING...';
        const response = await fetch('/api/upload', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({label: currentLabel, image: base64, split: currentSplit})
        });
        const result = await response.json();
        
        if (result.success) {
            currentStats[currentLabel] = result.count;
            updateStatsDisplay();
            updateTargetTracker();
            flashCapture();
            document.getElementById('status-text').textContent = 'SAVED! ✅';
        } else {
            alert('Failed: ' + result.error);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

function updateSensitivity(val) {
    motionThreshold = parseInt(val);
    document.getElementById('sensitivity-val').textContent = val;
}

// Initialize
initCamera();
initTargetFromURL();
loadStats();
setInterval(loadStats, 3000);
</script>'''
    
    return render_template_string(render_page('Smart Capture', 'capture', content, scripts))


# ============ API ROUTES ============

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Receive image upload from browser."""
    data = request.get_json()
    label = data.get('label')
    image_data = data.get('image')  # base64 without header
    split = data.get('split', '')  # train/val/test (optional)
    
    if not label or not image_data:
        return jsonify({'success': False, 'error': 'Missing label or image'}), 400
    
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{label}_{timestamp}.jpg"
        
        # Determine save path - support both flat and split structures
        if split and split in ['train', 'val', 'test']:
            # Split structure: v10_manual_capture/{split}/{label}/
            save_dir = CAPTURE_DIR / split / label
        else:
            # Flat structure: v10_manual_capture/{label}/
            save_dir = CAPTURE_DIR / label
        
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename
        
        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Update stats (for backward compatibility, also update flat stats)
        stats = load_stats()
        stats[label] = stats.get(label, 0) + 1
        save_stats(stats)
        
        return jsonify({'success': True, 'count': stats[label], 'split': split})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """Get combined dataset stats."""
    return jsonify(get_dataset_stats())


@app.route('/api/stats/capture')
def api_capture_stats():
    """Get capture stats only."""
    return jsonify(load_stats())


@app.route('/api/stats/detailed')
def api_stats_detailed():
    """Get detailed stats with mislabel detection."""
    import cv2
    import numpy as np
    
    dataset = request.args.get('dataset', 'combined')  # Default to combined
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    stats = {
        'total': 0, 
        'front': 0, 
        'back': 0, 
        'no_card': 0,
        'by_split': {'train': {}, 'val': {}, 'test': {}},
        'mislabel_count': 0,
        'mislabels': []
    }
    
    # Determine which datasets to include
    if dataset == 'combined':
        datasets_to_scan = ['v8_stage2_clean', 'v10_manual_capture']
    else:
        datasets_to_scan = [dataset]
    
    for ds_name in datasets_to_scan:
        dataset_path = DATASET_DIR / ds_name
        if not dataset_path.exists():
            continue
        
        # Check folder structure
        has_split_structure = (dataset_path / 'train').exists() or (dataset_path / 'val').exists()
        
        if has_split_structure:
            # Original structure: {dataset}/{split}/{class}/
            for split in ['train', 'val', 'test']:
                for cls in ['cnie_front', 'cnie_back']:
                    folder = dataset_path / split / cls
                    if not folder.exists():
                        continue
                    
                    images = list(folder.glob('*.jpg'))
                    if not images:
                        continue
                    
                    # Initialize split dict if not exists
                    if cls not in stats['by_split'][split]:
                        stats['by_split'][split][cls] = {'total': 0, 'front': 0, 'back': 0}
                    
                    folder_stats = process_folder_for_stats(folder, images, face_cascade, cls, stats)
                    
                    # Aggregate counts
                    stats['by_split'][split][cls]['total'] += folder_stats['total']
                    stats['by_split'][split][cls]['front'] += folder_stats['front']
                    stats['by_split'][split][cls]['back'] += folder_stats['back']
                    
                    if 'front' in cls:
                        stats['front'] += len(images)
                    else:
                        stats['back'] += len(images)
        
        # Also check for split structure in v10_manual_capture
        for split in ['train', 'val', 'test']:
            for cls_short, cls_full in [('front', 'cnie_front'), ('back', 'cnie_back'), ('no_card', 'no_card')]:
                folder = dataset_path / split / cls_short
                if not folder.exists():
                    continue
                
                images = list(folder.glob('*.jpg'))
                if not images:
                    continue
                
                if cls_short == 'no_card':
                    stats['no_card'] += len(images)
                    if 'no_card' not in stats['by_split'][split]:
                        stats['by_split'][split]['no_card'] = {'total': 0}
                    stats['by_split'][split]['no_card']['total'] += len(images)
                else:
                    if cls_full not in stats['by_split'][split]:
                        stats['by_split'][split][cls_full] = {'total': 0, 'front': 0, 'back': 0}
                    
                    folder_stats = process_folder_for_stats(folder, images, face_cascade, cls_full, stats)
                    
                    stats['by_split'][split][cls_full]['total'] += folder_stats['total']
                    stats['by_split'][split][cls_full]['front'] += folder_stats['front']
                    stats['by_split'][split][cls_full]['back'] += folder_stats['back']
                    
                    if 'front' in cls_full:
                        stats['front'] += len(images)
                    else:
                        stats['back'] += len(images)
        
        # Flat structure: {dataset}/{class}/
        if not has_split_structure or ds_name == 'v10_manual_capture':
            for folder_name, cls_key in [('front', 'cnie_front'), ('back', 'cnie_back'), ('no_card', 'no_card')]:
                folder = dataset_path / folder_name
                if not folder.exists():
                    continue
                
                images = list(folder.glob('*.jpg'))
                if not images:
                    continue
                
                if folder_name == 'no_card':
                    stats['no_card'] += len(images)
                else:
                    folder_stats = process_folder_for_stats(folder, images, face_cascade, cls_key, stats)
                    
                    if 'front' in cls_key:
                        stats['front'] += len(images)
                    else:
                        stats['back'] += len(images)
    
    stats['total'] = stats['front'] + stats['back'] + stats['no_card']
    
    return jsonify(stats)


def process_folder_for_stats(folder, images, face_cascade, cls, stats):
    """Process a folder for stats - detect mislabels via sampling."""
    folder_stats = {
        'total': len(images),
        'front': 0,
        'back': 0
    }
    
    # Sample up to 20 images for face detection
    sample_size = min(20, len(images))
    
    for img_path in images[:sample_size]:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            has_face = len(faces) > 0
            
            if 'front' in cls and not has_face:
                folder_stats['back'] += 1
                stats['mislabel_count'] += 1
            elif 'back' in cls and has_face:
                folder_stats['front'] += 1
                stats['mislabel_count'] += 1
            else:
                if 'front' in cls:
                    folder_stats['front'] += 1
                else:
                    folder_stats['back'] += 1
        except:
            continue
    
    # Assume remaining are correct
    remaining = len(images) - sample_size
    if remaining > 0:
        if 'front' in cls:
            folder_stats['front'] += remaining
        else:
            folder_stats['back'] += remaining
    
    return folder_stats


# ============ MANUAL REVIEW API ============

@app.route('/api/review/load')
def api_review_load():
    """Load images for manual review - supports both split and flat structures."""
    dataset = request.args.get('dataset', 'v8_stage2_clean')
    split = request.args.get('split', 'train')
    cls = request.args.get('class', 'cnie_front')
    
    images = []
    dataset_path = DATASET_DIR / dataset
    
    # Try multiple folder structures
    possible_paths = [
        # Split structure: dataset/split/class/
        dataset_path / split / cls,
        dataset_path / split / cls.replace('cnie_', ''),  # front instead of cnie_front
        # Flat structure: dataset/class/
        dataset_path / cls,
        dataset_path / cls.replace('cnie_', ''),
    ]
    
    for folder in possible_paths:
        if folder.exists():
            for img_path in sorted(folder.glob('*.jpg')):
                images.append({
                    'name': img_path.name,
                    'path': str(img_path),
                    'url': f'/api/image?path={img_path}'
                })
    
    return jsonify({'images': images, 'count': len(images)})


@app.route('/api/review/move', methods=['POST'])
def api_review_move():
    """Move image to different class folder."""
    data = request.get_json()
    img_path = data.get('path')
    from_dataset = data.get('from_dataset')
    from_split = data.get('from_split')
    to_class = data.get('to_class')
    
    if not all([img_path, from_dataset, from_split, to_class]):
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400
    
    try:
        src = Path(img_path)
        dest_dir = DATASET_DIR / from_dataset / from_split / to_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        
        # Remove if exists
        if dest.exists():
            dest.unlink()
        
        shutil.move(str(src), str(dest))
        
        return jsonify({'success': True, 'new_path': str(dest)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/review/delete', methods=['POST'])
def api_review_delete():
    """Delete image."""
    data = request.get_json()
    img_path = data.get('path')
    
    if not img_path:
        return jsonify({'success': False, 'error': 'Missing path'}), 400
    
    try:
        Path(img_path).unlink()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/move_between_splits', methods=['POST'])
def api_move_between_splits():
    """Move images between splits for rebalancing."""
    data = request.get_json()
    source_split = data.get('source_split')
    dest_split = data.get('dest_split')
    cls = data.get('class')  # 'front' or 'back'
    count = data.get('count', 1)
    dataset = data.get('dataset', 'combined')
    
    if not all([source_split, dest_split, cls]):
        return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
    
    if source_split == dest_split:
        return jsonify({'success': False, 'error': 'Source and destination must be different'}), 400
    
    try:
        count = int(count)
        if count < 1:
            return jsonify({'success': False, 'error': 'Count must be at least 1'}), 400
    except:
        return jsonify({'success': False, 'error': 'Invalid count'}), 400
    
    # Map class names
    cls_map = {'front': 'cnie_front', 'back': 'cnie_back', 'no_card': 'no_card'}
    cls_full = cls_map.get(cls, cls)
    cls_short = cls if cls in ['front', 'back', 'no_card'] else cls.replace('cnie_', '')
    
    moved = 0
    errors = []
    
    # Determine which datasets to scan for source images
    if dataset == 'combined':
        source_datasets = ['v8_stage2_clean', 'v10_manual_capture']
    else:
        source_datasets = [dataset]
    
    for ds_name in source_datasets:
        if moved >= count:
            break
            
        ds_path = DATASET_DIR / ds_name
        if not ds_path.exists():
            continue
        
        # Check different folder structures
        source_folders = [
            ds_path / source_split / cls_full,  # v8 style: train/cnie_front/
            ds_path / source_split / cls_short,  # v10 style: train/front/
        ]
        
        for source_folder in source_folders:
            if not source_folder.exists():
                continue
            
            # Find images in source folder
            images = list(source_folder.glob('*.jpg'))
            
            for img_path in images:
                if moved >= count:
                    break
                
                try:
                    # Determine destination folder (prefer v10 structure for new captures)
                    dest_folder = CAPTURE_DIR / dest_split / cls_short
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    
                    dest_path = dest_folder / img_path.name
                    
                    # Handle name collision
                    if dest_path.exists():
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                        name, ext = img_path.stem, img_path.suffix
                        dest_path = dest_folder / f"{name}_{timestamp}{ext}"
                    
                    shutil.move(str(img_path), str(dest_path))
                    moved += 1
                    
                except Exception as e:
                    errors.append(f"{img_path.name}: {str(e)}")
    
    if moved > 0:
        return jsonify({
            'success': True, 
            'moved': moved, 
            'source_split': source_split,
            'dest_split': dest_split,
            'class': cls,
            'errors': errors if errors else None
        })
    else:
        return jsonify({
            'success': False, 
            'error': 'No images could be moved. ' + '; '.join(errors[:3])
        }), 400


@app.route('/api/image')
def api_image():
    """Serve image file."""
    path = request.args.get('path')
    if path and os.path.exists(path):
        return send_file(path)
    return '', 404


# ============ ANNOTATION APIs (YOLO) ============

ANNOTATIONS_DIR = DATASET_DIR / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)

@app.route('/api/annotation/save', methods=['POST'])
def api_annotation_save():
    """Save bounding box annotation for YOLO training."""
    data = request.get_json()
    image_path = data.get('image_path')
    annotation = data.get('annotation')  # {x, y, width, height, label}
    img_width = data.get('image_width')
    img_height = data.get('image_height')
    
    if not all([image_path, annotation, img_width, img_height]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    try:
        # Convert to YOLO format (normalized x_center, y_center, width, height)
        # The annotation coords are relative to displayed image size
        # Need to convert to actual image coordinates
        img = Image.open(image_path)
        actual_width, actual_height = img.size
        
        # Scale factors
        scale_x = actual_width / img_width
        scale_y = actual_height / img_height
        
        # Convert annotation to actual image coordinates
        actual_x = annotation['x'] * scale_x
        actual_y = annotation['y'] * scale_y
        actual_w = annotation['width'] * scale_x
        actual_h = annotation['height'] * scale_y
        
        # Convert to YOLO format (center x, center y, width, height - all normalized)
        x_center = (actual_x + actual_w / 2) / actual_width
        y_center = (actual_y + actual_h / 2) / actual_height
        width_norm = actual_w / actual_width
        height_norm = actual_h / actual_height
        
        # Class mapping
        class_map = {'cnie_front': 0, 'cnie_back': 1, 'other_card': 2}
        class_id = class_map.get(annotation['label'], 0)
        
        # Save to annotations file
        annotation_file = ANNOTATIONS_DIR / f"{Path(image_path).stem}.txt"
        with open(annotation_file, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        return jsonify({
            'success': True,
            'annotation_file': str(annotation_file),
            'yolo_format': [class_id, x_center, y_center, width_norm, height_norm]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/preprocess/save', methods=['POST'])
def api_preprocess_save():
    """Save preprocessed image for V13 dual dataset strategy."""
    data = request.get_json()
    original_path = data.get('original_path')
    image_data = data.get('image_data')
    params = data.get('params', {})
    engine = data.get('engine', 'canvas')
    
    if not all([original_path, image_data]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Parse original path to determine save location
        orig_path = Path(original_path)
        
        # Create preprocessed folder alongside original
        # e.g., v10_manual_capture/train/front/ -> v10_manual_capture_preprocessed/train/front/
        dataset_dir = orig_path.parent.parent.parent  # training_data/v10_manual_capture
        relative_path = orig_path.parent.relative_to(dataset_dir)  # train/front
        
        preprocessed_dir = Path(str(dataset_dir) + '_preprocessed') / relative_path
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with same filename
        save_path = preprocessed_dir / orig_path.name
        
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
        
        # Save metadata about preprocessing
        meta_path = preprocessed_dir / (orig_path.stem + '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'original_path': str(original_path),
                'preprocessed_path': str(save_path),
                'params': params,
                'engine': engine,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return jsonify({
            'success': True,
            'save_path': str(save_path),
            'relative_path': str(relative_path),
            'filename': orig_path.name
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/annotation/export', methods=['POST'])
def api_annotation_export():
    """Export all annotations as YOLO dataset."""
    data = request.get_json()
    dataset_name = data.get('dataset_name', 'v12_yolo_annotated')
    
    try:
        export_dir = DATASET_DIR / dataset_name
        images_dir = export_dir / 'images'
        labels_dir = export_dir / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all annotated images and their labels
        image_count = 0
        label_count = 0
        
        for ann_file in ANNOTATIONS_DIR.glob('*.txt'):
            # Find corresponding image
            img_stem = ann_file.stem
            
            # Search for image in training_data directories
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                for dataset in ['v8_stage2_clean', 'v10_manual_capture']:
                    for split in ['train', 'val', 'test']:
                        for cls in ['cnie_front', 'cnie_back', 'front', 'back']:
                            potential_path = DATASET_DIR / dataset / split / cls / f"{img_stem}{ext}"
                            if potential_path.exists():
                                img_path = potential_path
                                break
                        if img_path:
                            break
                    if img_path:
                        break
                if img_path:
                    break
            
            if img_path:
                # Copy image
                shutil.copy2(img_path, images_dir / f"{img_stem}.jpg")
                image_count += 1
                
                # Copy label
                shutil.copy2(ann_file, labels_dir / ann_file.name)
                label_count += 1
        
        # Create dataset.yaml
        yaml_content = f"""# YOLO Dataset Configuration
path: {export_dir.absolute()}
train: images
val: images

nc: 3
names: ['cnie_front', 'cnie_back', 'other_card']
"""
        with open(export_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)
        
        return jsonify({
            'success': True,
            'export_path': str(export_dir),
            'image_count': image_count,
            'label_count': label_count
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============ OTHER PAGES ============

@app.route('/manual')
def manual():
    content = '''
<div class="review-container" id="review-setup">
    <div class="card">
        <div class="card-header">
            <div class="card-title">📁 Select Folder to Review</div>
        </div>
        <p>Choose which dataset folder you want to review and clean.</p>
        
        <div style="display: flex; gap: 15px; margin-top: 20px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Dataset</label>
                <select class="form-select" id="review-dataset">
                    <option value="v8_stage2_clean">v8_stage2_clean (Original)</option>
                    <option value="v10_manual_capture">v10_manual_capture (New)</option>
                </select>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Split</label>
                <select class="form-select" id="review-split">
                    <option value="train">train</option>
                    <option value="val">val</option>
                    <option value="test">test</option>
                </select>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Class</label>
                <select class="form-select" id="review-class">
                    <option value="cnie_front">cnie_front</option>
                    <option value="cnie_back">cnie_back</option>
                </select>
            </div>
        </div>
        
        <div style="margin-top: 20px;">
            <button class="btn btn-primary" onclick="loadReviewFolder()" style="font-size: 16px; padding: 12px 30px;">
                📂 Load Folder
            </button>
        </div>
        
        <div id="folder-stats" style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; display: none;">
        </div>
    </div>
</div>

<div class="review-container" id="review-interface" style="display: none;">
    <!-- Progress Header -->
    <div class="card" style="padding: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>📸 <span id="review-location">train/cnie_front</span></strong>
                <span style="margin-left: 20px; color: #666;">
                    Image <span id="current-index" style="font-weight: bold; color: #2196F3;">1</span> of <span id="total-images">0</span>
                </span>
            </div>
            <div>
                <span style="margin-right: 20px;">✓ Kept: <span id="count-kept" style="color: #4CAF50; font-weight: bold;">0</span></span>
                <span style="margin-right: 20px;">→ Moved: <span id="count-moved" style="color: #FF9800; font-weight: bold;">0</span></span>
                <span>✗ Deleted: <span id="count-deleted" style="color: #f44336; font-weight: bold;">0</span></span>
            </div>
        </div>
        
        <!-- Progress Bar -->
        <div class="progress-bar-bg" style="margin-top: 10px; height: 8px;">
            <div id="review-progress" class="progress-bar-fill" style="width: 0%; background: #2196F3;"></div>
        </div>
    </div>
    
    <!-- Main Review Area -->
    <div style="display: flex; gap: 20px; margin-top: 20px; align-items: flex-start;">
        <!-- Image Viewer (Sticky/Floating) -->
        <div style="flex: 1; position: sticky; top: 20px; z-index: 50;">
            <div class="card" style="padding: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
                <div id="image-container" style="position: relative; background: #000; border-radius: 8px; overflow: hidden; min-height: 500px; display: flex; align-items: center; justify-content: center; padding: 20px;">
                    <!-- Original Image -->
                    <img id="review-image" src="" style="max-width: 100%; max-height: 560px; display: block;">
                    
                    <!-- Preprocessed Image (hidden by default) -->
                    <img id="preprocess-image" src="" style="max-width: 100%; max-height: 560px; display: none; margin-left: 10px; border: 2px solid #38ef7d;">
                    
                    <!-- Preprocess Canvas (for processing) -->
                    <canvas id="preprocess-canvas" style="display: none;"></canvas>
                    
                    <!-- Split View Slider Container -->
                    <div id="split-view-container" style="display: none; position: relative; width: 100%; max-width: 800px; height: 500px;">
                        <div id="split-original" style="position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden; border-right: 2px solid #38ef7d;">
                            <img id="split-original-img" src="" style="height: 100%; max-width: none;">
                        </div>
                        <div id="split-preprocessed" style="position: absolute; top: 0; right: 0; width: 50%; height: 100%; overflow: hidden;">
                            <img id="split-preprocessed-img" src="" style="height: 100%; max-width: none; position: absolute; right: 0;">
                        </div>
                        <input type="range" id="split-slider" min="0" max="100" value="50" style="position: absolute; bottom: -30px; left: 0; width: 100%;" oninput="updateSplitPosition(this.value)">
                        <div style="position: absolute; bottom: -55px; left: 0; width: 100%; display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                            <span>Original</span>
                            <span>Preprocessed</span>
                        </div>
                    </div>
                    
                    <div id="image-overlay" style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: white; padding: 8px 12px; border-radius: 4px; font-size: 12px; z-index: 101;">
                    </div>
                    
                    <!-- Preprocess indicator -->
                    <div id="preprocess-indicator" style="position: absolute; top: 10px; right: 10px; background: linear-gradient(135deg, #11998e, #38ef7d); color: white; padding: 5px 10px; border-radius: 4px; font-size: 11px; display: none; z-index: 102;">
                        🔍 PREPROCESSED
                    </div>
                </div>
                
                <!-- Image Info -->
                <div id="image-info" style="margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 4px; font-size: 13px; font-family: monospace;">
                </div>
            </div>
            
            <!-- Navigation -->
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 15px;">
                <button class="btn" onclick="firstImage()">⏮ First</button>
                <button class="btn" onclick="prevImage()">◀ Prev</button>
                <input type="number" id="jump-input" style="width: 70px; text-align: center;" placeholder="#">
                <button class="btn" onclick="jumpToImage()">Go</button>
                <button class="btn" onclick="nextImage()">Next ▶</button>
                <button class="btn" onclick="lastImage()">Last ⏭</button>
            </div>
        </div>
        
        <!-- Action Panel (Scrollable) -->
        <div class="custom-scroll" style="width: 280px; max-height: calc(100vh - 150px); overflow-y: auto; padding-right: 5px;">
            <div class="card">
                <h3 style="margin-bottom: 15px;">Actions</h3>
                
                <button class="big-btn front" onclick="markCorrect()" style="width: 100%; margin-bottom: 10px; font-size: 16px; padding: 12px;">
                    ✓ CORRECT (Space)
                </button>
                <p style="font-size: 12px; color: #666; margin: -5px 0 15px 0;">Image is correctly labeled, go to next</p>
                
                <hr style="margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;">
                
                <p style="font-weight: 600; margin-bottom: 10px;">Move to:</p>
                <button class="big-btn back" onclick="moveImage('cnie_back')" style="width: 100%; margin-bottom: 10px; font-size: 16px; padding: 12px;">
                    → BACK (B)
                </button>
                <button class="big-btn front" onclick="moveImage('cnie_front')" style="width: 100%; margin-bottom: 10px; font-size: 16px; padding: 12px; background: linear-gradient(135deg, #4CAF50, #45a049);">
                    → FRONT (F)
                </button>
                <p style="font-size: 12px; color: #666; margin: -5px 0 15px 0;">Move to different class folder</p>
                
                <hr style="margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;">
                
                <button class="big-btn stop" onclick="deleteImage()" style="width: 100%; margin-bottom: 10px; font-size: 16px; padding: 12px;">
                    ✗ DELETE (Del)
                </button>
                <p style="font-size: 12px; color: #666; margin: -5px 0 15px 0;">Permanently remove image</p>
                
                <hr style="margin: 15px 0; border: none; border-top: 1px solid #e0e0e0;">
                
                <button class="btn" onclick="backToSelection()" style="width: 100%;">
                    ← Change Folder
                </button>
            </div>
            
            <!-- Annotation Mode -->
            <div class="card" style="margin-top: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <h4 style="margin-bottom: 10px;">🏷️ YOLO Annotation</h4>
                <p style="font-size: 12px; margin-bottom: 10px; opacity: 0.9;">
                    Draw bounding boxes for YOLO training data
                </p>
                
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <label class="switch" style="position: relative; display: inline-block; width: 50px; height: 24px;">
                        <input type="checkbox" id="annotation-mode" onchange="toggleAnnotationMode()" style="opacity: 0; width: 0; height: 0;">
                        <span style="position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(255,255,255,0.3); transition: .4s; border-radius: 24px;"></span>
                    </label>
                    <span style="font-size: 12px;">Enable Annotation Mode</span>
                </div>
                
                <div id="annotation-tools" style="display: none;">
                    <div style="margin-bottom: 10px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 5px;">Tool:</label>
                        <div style="display: flex; gap: 5px;">
                            <button class="btn btn-sm" id="tool-draw" onclick="setAnnotationTool('draw')" style="flex: 1; background: #4CAF50;">✏️ Draw</button>
                            <button class="btn btn-sm" id="tool-move" onclick="setAnnotationTool('move')" style="flex: 1; background: rgba(255,255,255,0.2);">↔️ Move</button>
                            <button class="btn btn-sm" id="tool-delete" onclick="setAnnotationTool('delete')" style="flex: 1; background: rgba(255,255,255,0.2);">🗑️ Del</button>
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 10px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 5px;">Label:</label>
                        <select id="annotation-label" class="form-select" style="width: 100%; font-size: 12px;">
                            <option value="cnie_front">cnie_front</option>
                            <option value="cnie_back">cnie_back</option>
                            <option value="other_card">other_card</option>
                        </select>
                    </div>
                    
                    <div id="annotation-info" style="font-size: 11px; margin-bottom: 10px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                        No annotation
                    </div>
                    
                    <button class="btn" onclick="saveAnnotation()" style="width: 100%; background: #4CAF50; color: white; margin-bottom: 5px;">
                        💾 Save Annotation
                    </button>
                    
                    <button class="btn btn-sm" onclick="clearAnnotation()" style="width: 100%; background: #f44336; color: white; margin-bottom: 5px; font-size: 11px;">
                        ❌ Clear/Cancel Box
                    </button>
                    
                    <button class="btn btn-sm" onclick="exportAnnotations()" style="width: 100%; background: rgba(255,255,255,0.2); color: white; font-size: 11px;">
                        📤 Export YOLO Dataset
                    </button>
                </div>
                
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);">
                    <div style="font-size: 11px; opacity: 0.9;">
                        Progress: <span id="annotation-progress">0</span> annotated
                    </div>
                </div>
            </div>
            
            <!-- V13: Preprocessing Preview Panel -->
            <div class="card" style="margin-top: 15px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white;">
                <h4 style="margin-bottom: 10px;">🔍 V13 Preprocess Preview</h4>
                <p style="font-size: 12px; margin-bottom: 10px; opacity: 0.9;">
                    Enhance hidden security features
                </p>
                
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <label class="switch" style="position: relative; display: inline-block; width: 50px; height: 24px;">
                        <input type="checkbox" id="preprocess-mode" onchange="togglePreprocessMode()" style="opacity: 0; width: 0; height: 0;">
                        <span style="position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(255,255,255,0.3); transition: .4s; border-radius: 24px;"></span>
                    </label>
                    <span style="font-size: 12px;">Enable Preview</span>
                </div>
                
                <div id="preprocess-tools" style="display: none;">
                    <!-- Engine Selection -->
                    <div style="margin-bottom: 10px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 5px;">Engine:</label>
                        <div style="display: flex; gap: 5px;">
                            <button class="btn btn-sm" id="engine-canvas" onclick="setPreprocessEngine('canvas')" style="flex: 1; background: #4CAF50; font-size: 11px;">⚡ Canvas (Fast)</button>
                            <button class="btn btn-sm" id="engine-opencv" onclick="setPreprocessEngine('opencv')" style="flex: 1; background: rgba(255,255,255,0.2); font-size: 11px;">🔬 OpenCV.js</button>
                        </div>
                        <div id="opencv-status" style="font-size: 10px; margin-top: 5px; color: #ffeb3b; display: none;">
                            ⏳ Loading OpenCV.js...
                        </div>
                    </div>
                    
                    <!-- View Mode -->
                    <div style="margin-bottom: 10px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 5px;">View:</label>
                        <select id="preprocess-view" onchange="updatePreprocessView()" class="form-select" style="width: 100%; font-size: 12px; color: #333;">
                            <option value="side">Side-by-Side</option>
                            <option value="overlay">Overlay Toggle</option>
                            <option value="split">Split Slider</option>
                        </select>
                    </div>
                    
                    <!-- Enhancement Controls -->
                    <div style="margin-bottom: 10px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 8px; font-weight: bold;">Basic Enhancements:</label>
                        
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>Contrast (CLAHE)</span>
                                <span id="val-contrast">2.0</span>
                            </div>
                            <input type="range" id="slider-contrast" min="0" max="8" step="0.5" value="2.0" onchange="updatePreprocessParams()" style="width: 100%;">
                        </div>
                        
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>Sharpen</span>
                                <span id="val-sharpen">1.5</span>
                            </div>
                            <input type="range" id="slider-sharpen" min="1.0" max="5.0" step="0.1" value="1.5" onchange="updatePreprocessParams()" style="width: 100%;">
                        </div>
                        
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>Gamma</span>
                                <span id="val-gamma">1.2</span>
                            </div>
                            <input type="range" id="slider-gamma" min="0.5" max="3.0" step="0.1" value="1.2" onchange="updatePreprocessParams()" style="width: 100%;">
                        </div>
                        
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>Denoise</span>
                                <span id="val-denoise">0</span>
                            </div>
                            <input type="range" id="slider-denoise" min="0" max="10" step="1" value="0" onchange="updatePreprocessParams()" style="width: 100%;">
                        </div>
                    </div>
                    
                    <!-- Phase 1.1: Advanced Hidden Feature Detection -->
                    <div style="margin-bottom: 10px; padding: 10px; background: rgba(255,0,0,0.15); border-radius: 4px; border: 1px solid rgba(255,0,0,0.3);">
                        <label style="font-size: 11px; display: block; margin-bottom: 8px; font-weight: bold; color: #ffeb3b;">🔬 Phase 1.1 - Hidden Feature Boosters:</label>
                        
                        <!-- HSV Color Space for Holograms -->
                        <div style="margin-bottom: 10px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                            <label style="font-size: 10px; display: block; margin-bottom: 5px; color: #4CAF50;">🌈 HSV Color Space (Holograms):</label>
                            
                            <div style="margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between; font-size: 9px; margin-bottom: 2px;">
                                    <span>Hue Range</span>
                                    <span id="val-hsv-hue">All</span>
                                </div>
                                <select id="select-hsv-hue" onchange="updatePreprocessParams()" style="width: 100%; font-size: 10px; padding: 3px;">
                                    <option value="all">All Colors</option>
                                    <option value="red">Red (0-30)</option>
                                    <option value="yellow">Yellow/Gold (30-60)</option>
                                    <option value="green">Green (60-90)</option>
                                    <option value="blue">Blue (90-120)</option>
                                    <option value="purple">Purple (120-150)</option>
                                    <option value="high-sat">High Saturation Only</option>
                                </select>
                            </div>
                            
                            <div>
                                <div style="display: flex; justify-content: space-between; font-size: 9px; margin-bottom: 2px;">
                                    <span>Saturation Threshold</span>
                                    <span id="val-hsv-sat">0.3</span>
                                </div>
                                <input type="range" id="slider-hsv-sat" min="0" max="1" step="0.1" value="0.3" onchange="updatePreprocessParams()" style="width: 100%;">
                            </div>
                        </div>
                        
                        <!-- Morphological Operations for Micro-Text -->
                        <div style="margin-bottom: 10px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                            <label style="font-size: 10px; display: block; margin-bottom: 5px; color: #2196F3;">🔬 Morphology (Micro-Text):</label>
                            
                            <div style="margin-bottom: 6px;">
                                <div style="display: flex; justify-content: space-between; font-size: 9px; margin-bottom: 2px;">
                                    <span>Dilation Iterations</span>
                                    <span id="val-dilate">0</span>
                                </div>
                                <input type="range" id="slider-dilate" min="0" max="5" step="1" value="0" onchange="updatePreprocessParams()" style="width: 100%;">
                            </div>
                            
                            <div>
                                <div style="display: flex; justify-content: space-between; font-size: 9px; margin-bottom: 2px;">
                                    <span>Canny Edge + Morph</span>
                                    <span id="val-canny-morph">OFF</span>
                                </div>
                                <input type="checkbox" id="check-canny-morph" onchange="updatePreprocessParams()" style="width: auto;">
                            </div>
                        </div>
                        
                        <!-- FFT Band-Pass for Guilloché -->
                        <div style="margin-bottom: 10px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                            <label style="font-size: 10px; display: block; margin-bottom: 5px; color: #9C27B0;">📊 FFT Band-Pass (Patterns):</label>
                            
                            <div style="display: flex; gap: 5px;">
                                <div style="flex: 1;">
                                    <div style="font-size: 9px; margin-bottom: 2px;">Low Freq</div>
                                    <input type="number" id="input-fft-low" value="5" min="0" max="100" style="width: 100%; font-size: 10px;" onchange="updatePreprocessParams()">
                                </div>
                                <div style="flex: 1;">
                                    <div style="font-size: 9px; margin-bottom: 2px;">High Freq</div>
                                    <input type="number" id="input-fft-high" value="50" min="0" max="100" style="width: 100%; font-size: 10px;" onchange="updatePreprocessParams()">
                                </div>
                                <div style="flex: 1;">
                                    <div style="font-size: 9px; margin-bottom: 2px;">Enable</div>
                                    <input type="checkbox" id="check-fft" onchange="updatePreprocessParams()" style="width: auto;">
                                </div>
                            </div>
                        </div>
                        
                        <!-- Previous controls -->
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>Edge Enhancement (Sobel)</span>
                                <span id="val-edge">0</span>
                            </div>
                            <input type="range" id="slider-edge" min="0" max="3" step="0.5" value="0" onchange="updatePreprocessParams()" style="width: 100%;">
                        </div>
                        
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>High-Pass Filter</span>
                                <span id="val-highpass">0</span>
                            </div>
                            <input type="range" id="slider-highpass" min="0" max="2" step="0.2" value="0" onchange="updatePreprocessParams()" style="width: 100%;">
                        </div>
                        
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>Local Contrast (Detail)</span>
                                <span id="val-local">0</span>
                            </div>
                            <input type="range" id="slider-local" min="0" max="5" step="0.5" value="0" onchange="updatePreprocessParams()" style="width: 100%;">
                        </div>
                        
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>Invert Colors</span>
                                <span id="val-invert">OFF</span>
                            </div>
                            <input type="checkbox" id="check-invert" onchange="updatePreprocessParams()" style="width: auto;">
                        </div>
                        
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 3px;">
                                <span>Equalize Histogram</span>
                                <span id="val-equalize">OFF</span>
                            </div>
                            <input type="checkbox" id="check-equalize" onchange="updatePreprocessParams()" style="width: auto;">
                        </div>
                    </div>
                    
                    <!-- Preset Buttons -->
                    <div style="margin-bottom: 10px;">
                        <label style="font-size: 11px; display: block; margin-bottom: 5px;">Quick Presets:</label>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
                            <button class="btn btn-sm" onclick="applyPreset('microtext')" style="background: #ff9800; color: white; font-size: 10px; padding: 5px;">🔍 Micro-Text</button>
                            <button class="btn btn-sm" onclick="applyPreset('guilloche')" style="background: #9c27b0; color: white; font-size: 10px; padding: 5px;">🎨 Guilloche</button>
                            <button class="btn btn-sm" onclick="applyPreset('hologram')" style="background: #4CAF50; color: white; font-size: 10px; padding: 5px;">🌈 Hologram</button>
                            <button class="btn btn-sm" onclick="applyPreset('ghost')" style="background: #00bcd4; color: white; font-size: 10px; padding: 5px;">👤 Ghost Image</button>
                            <button class="btn btn-sm" onclick="applyPreset('uv')" style="background: #e91e63; color: white; font-size: 10px; padding: 5px;">💡 UV/Security</button>
                        </div>
                    </div>
                    
                    <!-- Action Buttons -->
                    <button class="btn" onclick="applyPreprocessing()" style="width: 100%; background: #4CAF50; color: white; margin-bottom: 5px; font-size: 12px;">
                        🔄 Apply & Preview
                    </button>
                    
                    <button class="btn btn-sm" onclick="cancelPreprocessing()" style="width: 100%; background: #ff9800; color: white; margin-bottom: 5px; font-size: 11px;">
                        ↩️ Cancel / Show Original
                    </button>
                    
                    <button class="btn btn-sm" onclick="savePreprocessedImage()" style="width: 100%; background: #2196F3; color: white; margin-bottom: 5px; font-size: 11px;">
                        💾 Save Preprocessed Copy
                    </button>
                    
                    <!-- Session State Management -->
                    <div style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                        <div style="font-size: 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                            <span>💾 Session State:</span>
                            <span id="session-status" style="color: #4CAF50;">Auto-saved</span>
                        </div>
                        <div style="display: flex; gap: 5px;">
                            <button class="btn btn-sm" onclick="saveSessionStateManual()" style="flex: 1; background: #4CAF50; color: white; font-size: 9px; padding: 5px;">
                                💾 Save Now
                            </button>
                            <button class="btn btn-sm" onclick="resetSessionState()" style="flex: 1; background: #f44336; color: white; font-size: 9px; padding: 5px;">
                                🔄 Reset
                            </button>
                        </div>
                        <div id="session-timestamp" style="font-size: 8px; color: #aaa; margin-top: 5px; text-align: center;">
                            Last saved: Never
                        </div>
                    </div>
                    
                    <div style="font-size: 10px; opacity: 0.9; margin-top: 8px; padding: 8px; background: rgba(0,0,0,0.15); border-radius: 4px;">
                        <strong>Hidden Features to Check:</strong><br>
                        • Micro-text on edges<br>
                        • Guilloche patterns<br>
                        • Ghost image<br>
                        • UV-reactive areas
                    </div>
                </div>
            </div>
            
            <!-- Keyboard Shortcuts -->
            <div class="card" style="margin-top: 15px;">
                <h4 style="margin-bottom: 10px;">⌨️ Shortcuts</h4>
                <table style="font-size: 12px; width: 100%;">
                    <tr><td><kbd>Space</kbd></td><td>Correct/Next</td></tr>
                    <tr><td><kbd>B</kbd></td><td>Move to Back</td></tr>
                    <tr><td><kbd>F</kbd></td><td>Move to Front</td></tr>
                    <tr><td><kbd>Del</kbd></td><td>Delete</td></tr>
                    <tr><td><kbd>←</kbd></td><td>Previous</td></tr>
                    <tr><td><kbd>→</kbd></td><td>Next</td></tr>
                </table>
            </div>
        </div>
    </div>
</div>

<div class="review-container" id="review-complete" style="display: none;">
    <div class="card" style="text-align: center; padding: 40px;">
        <h2 style="color: #4CAF50; margin-bottom: 20px;">🎉 Folder Complete!</h2>
        <p style="font-size: 18px; margin-bottom: 20px;">
            All images have been reviewed.
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; margin: 30px 0;">
            <div style="text-align: center;">
                <div style="font-size: 36px; font-weight: bold; color: #4CAF50;" id="final-kept">0</div>
                <div>✓ Kept</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 36px; font-weight: bold; color: #FF9800;" id="final-moved">0</div>
                <div>→ Moved</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 36px; font-weight: bold; color: #f44336;" id="final-deleted">0</div>
                <div>✗ Deleted</div>
            </div>
        </div>
        <button class="btn btn-primary" onclick="backToSelection()" style="font-size: 16px; padding: 12px 30px;">
            Review Another Folder
        </button>
    </div>
</div>
'''
    
    scripts = '''<script>
// Review state
let reviewImages = [];
let reviewIndex = 0;
let reviewStats = { kept: 0, moved: 0, deleted: 0 };
let currentDataset = '';
let currentSplit = '';
let currentClass = '';

// Load folder for review
async function loadReviewFolder() {
    currentDataset = document.getElementById('review-dataset').value;
    currentSplit = document.getElementById('review-split').value;
    currentClass = document.getElementById('review-class').value;
    
    try {
        const response = await fetch(`/api/review/load?dataset=${currentDataset}&split=${currentSplit}&class=${currentClass}`);
        const data = await response.json();
        
        reviewImages = data.images;
        reviewIndex = 0;
        reviewStats = { kept: 0, moved: 0, deleted: 0 };
        
        if (reviewImages.length === 0) {
            alert('No images found in this folder!');
            return;
        }
        
        // Update UI
        document.getElementById('review-setup').style.display = 'none';
        document.getElementById('review-interface').style.display = 'block';
        document.getElementById('review-complete').style.display = 'none';
        
        document.getElementById('review-location').textContent = `${currentSplit}/${currentClass}`;
        document.getElementById('total-images').textContent = reviewImages.length;
        
        updateReviewDisplay();
        
        // Update session UI if preprocessing was previously enabled
        updateSessionUI();
        if (preprocessMode) {
            updateUIFromParams();
        }
    } catch (e) {
        console.error('Failed to load folder:', e);
        alert('Error loading folder: ' + e.message);
    }
}

// Update review display
let lastDisplayedIndex = -1;

function updateReviewDisplay() {
    if (reviewIndex >= reviewImages.length) {
        showComplete();
        return;
    }
    
    const img = reviewImages[reviewIndex];
    
    // Clear annotation when navigating to a DIFFERENT image
    if (reviewIndex !== lastDisplayedIndex) {
        console.log('Navigating to new image:', reviewIndex, 'clearing annotation');
        currentAnnotation = null;
        lastDisplayedIndex = reviewIndex;
        
        // Reset preprocessing view for new image - show original first
        if (preprocessMode) {
            // Clear preprocessed image and show original until processing completes
            document.getElementById('preprocess-image').src = '';
            resetPreprocessView();
        }
    }
    
    document.getElementById('review-image').src = img.url;
    document.getElementById('current-index').textContent = reviewIndex + 1;
    document.getElementById('image-overlay').textContent = img.name;
    document.getElementById('image-info').textContent = `File: ${img.path}`;
    
    // Update counts
    document.getElementById('count-kept').textContent = reviewStats.kept;
    document.getElementById('count-moved').textContent = reviewStats.moved;
    document.getElementById('count-deleted').textContent = reviewStats.deleted;
    
    // Update progress
    const progress = ((reviewIndex) / reviewImages.length) * 100;
    document.getElementById('review-progress').style.width = progress + '%';
}

// Navigation
function firstImage() {
    reviewIndex = 0;
    updateReviewDisplay();
}

function lastImage() {
    reviewIndex = reviewImages.length - 1;
    updateReviewDisplay();
}

function prevImage() {
    if (reviewIndex > 0) {
        reviewIndex--;
        updateReviewDisplay();
    }
}

function nextImage() {
    if (reviewIndex < reviewImages.length - 1) {
        reviewIndex++;
        updateReviewDisplay();
    }
}

function jumpToImage() {
    const input = document.getElementById('jump-input');
    const idx = parseInt(input.value) - 1;
    if (idx >= 0 && idx < reviewImages.length) {
        reviewIndex = idx;
        updateReviewDisplay();
    }
    input.value = '';
}

// Actions
function markCorrect() {
    reviewStats.kept++;
    reviewIndex++;
    updateReviewDisplay();
}

async function moveImage(targetClass) {
    if (reviewIndex >= reviewImages.length) return;
    
    const img = reviewImages[reviewIndex];
    
    try {
        const response = await fetch('/api/review/move', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                path: img.path,
                from_dataset: currentDataset,
                from_split: currentSplit,
                from_class: currentClass,
                to_class: targetClass
            })
        });
        
        const result = await response.json();
        if (result.success) {
            reviewStats.moved++;
            reviewImages.splice(reviewIndex, 1);
            
            if (reviewIndex >= reviewImages.length) {
                reviewIndex = Math.max(0, reviewImages.length - 1);
            }
            
            updateReviewDisplay();
        } else {
            alert('Failed to move: ' + result.error);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function deleteImage() {
    if (reviewIndex >= reviewImages.length) return;
    if (!confirm('Are you sure you want to DELETE this image?')) return;
    
    const img = reviewImages[reviewIndex];
    
    try {
        const response = await fetch('/api/review/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ path: img.path })
        });
        
        const result = await response.json();
        if (result.success) {
            reviewStats.deleted++;
            reviewImages.splice(reviewIndex, 1);
            
            if (reviewIndex >= reviewImages.length) {
                reviewIndex = Math.max(0, reviewImages.length - 1);
            }
            
            updateReviewDisplay();
        } else {
            alert('Failed to delete: ' + result.error);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

function showComplete() {
    document.getElementById('review-interface').style.display = 'none';
    document.getElementById('review-complete').style.display = 'block';
    
    document.getElementById('final-kept').textContent = reviewStats.kept;
    document.getElementById('final-moved').textContent = reviewStats.moved;
    document.getElementById('final-deleted').textContent = reviewStats.deleted;
}

function backToSelection() {
    document.getElementById('review-setup').style.display = 'block';
    document.getElementById('review-interface').style.display = 'none';
    document.getElementById('review-complete').style.display = 'none';
    
    reviewImages = [];
    reviewIndex = 0;
}

// ========== YOLO ANNOTATION FEATURES ==========

let annotationMode = false;
let annotationTool = 'draw';  // 'draw', 'move', 'delete'
let currentAnnotation = null;  // {x, y, width, height, label}
let annotations = {};  // Map of image path to annotation
let isDrawing = false;
let drawStart = null;

function toggleAnnotationMode() {
    annotationMode = document.getElementById('annotation-mode').checked;
    const toolsDiv = document.getElementById('annotation-tools');
    const imageContainer = document.querySelector('#review-interface .card');
    
    if (annotationMode) {
        toolsDiv.style.display = 'block';
        enableAnnotationCanvas();
    } else {
        toolsDiv.style.display = 'none';
        disableAnnotationCanvas();
    }
}

function setAnnotationTool(tool) {
    annotationTool = tool;
    
    // Update button styles
    ['draw', 'move', 'delete'].forEach(t => {
        const btn = document.getElementById('tool-' + t);
        if (t === tool) {
            btn.style.background = '#4CAF50';
        } else {
            btn.style.background = 'rgba(255,255,255,0.2)';
        }
    });
}

let canvasInitialized = false;

function enableAnnotationCanvas() {
    const img = document.getElementById('review-image');
    const container = document.getElementById('image-container');
    
    // Create canvas overlay only once
    let canvas = document.getElementById('annotation-canvas');
    if (!canvas) {
        console.log('Creating annotation canvas...');
        canvas = document.createElement('canvas');
        canvas.id = 'annotation-canvas';
        canvas.style.position = 'absolute';
        canvas.style.top = '20px';  // Match container padding
        canvas.style.left = '20px';
        canvas.style.cursor = 'crosshair';
        canvas.style.zIndex = '200';
        canvas.style.pointerEvents = 'auto';
        container.appendChild(canvas);
        
        // Add event listeners only once
        canvas.addEventListener('mousedown', onAnnotationMouseDown);
        canvas.addEventListener('mousemove', onAnnotationMouseMove);
        canvas.addEventListener('mouseup', onAnnotationMouseUp);
        canvas.addEventListener('mouseleave', onAnnotationMouseLeave);
        canvas.addEventListener('mouseenter', onAnnotationMouseEnter);
        canvas.addEventListener('contextmenu', onAnnotationRightClick);
        
        canvasInitialized = true;
    }
    
    // Size canvas to match image display size
    updateAnnotationCanvasSize();
    canvas.style.display = 'block';
    
    // Load existing annotation if any
    loadCurrentAnnotation();
}

function disableAnnotationCanvas() {
    const canvas = document.getElementById('annotation-canvas');
    if (canvas) {
        canvas.style.display = 'none';
    }
}

function updateAnnotationCanvasSize() {
    const img = document.getElementById('review-image');
    const canvas = document.getElementById('annotation-canvas');
    if (canvas && img.offsetWidth > 0) {
        // Position canvas over the actual image (accounting for container padding)
        canvas.style.top = img.offsetTop + 'px';
        canvas.style.left = img.offsetLeft + 'px';
        canvas.width = img.offsetWidth;
        canvas.height = img.offsetHeight;
        
        console.log('Canvas sized to:', canvas.width, 'x', canvas.height, 'at', canvas.style.left, canvas.style.top);
        console.log('currentAnnotation:', currentAnnotation);
        
        // Redraw after resize
        redrawAnnotation();
    }
}

function loadCurrentAnnotation() {
    console.log('loadCurrentAnnotation called, currentAnnotation:', currentAnnotation);
    if (!reviewImages[reviewIndex]) return;
    
    const imgPath = reviewImages[reviewIndex].path;
    
    // Only load from storage if we don't already have a current annotation
    // This prevents losing an annotation that was just drawn but not saved yet
    if (!currentAnnotation) {
        currentAnnotation = annotations[imgPath] || null;
        console.log('Loaded from storage:', currentAnnotation);
    } else {
        console.log('Preserving existing annotation:', currentAnnotation);
    }
    
    updateAnnotationInfo();
    redrawAnnotation();
    
    // Also sync the label dropdown to match the annotation's label
    if (currentAnnotation && currentAnnotation.label) {
        document.getElementById('annotation-label').value = currentAnnotation.label;
    }
}

function updateAnnotationInfo() {
    const infoDiv = document.getElementById('annotation-info');
    if (currentAnnotation) {
        infoDiv.innerHTML = `
            <strong style="color: #4CAF50;">✓ Box Drawn</strong><br>
            Label: <b>${currentAnnotation.label}</b><br>
            Size: ${Math.round(currentAnnotation.width)}×${Math.round(currentAnnotation.height)}px<br>
            <small>💾 Click Save | 🖱️ Right-click to Delete</small>
        `;
        infoDiv.style.background = 'rgba(76, 175, 80, 0.2)';
    } else {
        const savedAnnotation = reviewImages[reviewIndex] ? annotations[reviewImages[reviewIndex].path] : null;
        if (savedAnnotation) {
            infoDiv.innerHTML = `
                <strong style="color: #2196F3;">💾 Saved</strong><br>
                Label: <b>${savedAnnotation.label}</b><br>
                Size: ${Math.round(savedAnnotation.width)}×${Math.round(savedAnnotation.height)}px
            `;
            infoDiv.style.background = 'rgba(33, 150, 243, 0.2)';
        } else {
            infoDiv.innerHTML = '<strong>No annotation</strong><br><small>🖱️ Click & drag to draw box | Right-click to cancel</small>';
            infoDiv.style.background = 'rgba(0,0,0,0.2)';
        }
    }
}

function redrawAnnotation() {
    const canvas = document.getElementById('annotation-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (currentAnnotation) {
        const label = currentAnnotation.label;
        const colors = {
            'cnie_front': '#4CAF50',
            'cnie_back': '#2196F3',
            'other_card': '#FF9800'
        };
        const color = colors[label] || '#4CAF50';
        
        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(
            currentAnnotation.x,
            currentAnnotation.y,
            currentAnnotation.width,
            currentAnnotation.height
        );
        
        // Draw label background
        ctx.fillStyle = color;
        ctx.fillRect(
            currentAnnotation.x,
            currentAnnotation.y - 20,
            ctx.measureText(label).width + 10,
            20
        );
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.font = '12px sans-serif';
        ctx.fillText(label, currentAnnotation.x + 5, currentAnnotation.y - 5);
    }
}

let wasDrawingBeforeLeave = false;
let mouseLeftAt = null;

function onAnnotationMouseDown(e) {
    // Handle right-click for cancel
    if (e.button === 2) {
        e.preventDefault();
        if (currentAnnotation) {
            if (confirm('Delete current bounding box?')) {
                clearAnnotation();
            }
        }
        return;
    }
    
    console.log('Mouse DOWN, annotationMode:', annotationMode, 'tool:', annotationTool);
    if (!annotationMode || annotationTool !== 'draw') return;
    if (e.button !== 0) return; // Only left click
    
    const canvas = document.getElementById('annotation-canvas');
    const rect = canvas.getBoundingClientRect();
    
    isDrawing = true;
    wasDrawingBeforeLeave = false;
    drawStart = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
    console.log('Drawing started at:', drawStart);
}

function onAnnotationMouseMove(e) {
    if (!isDrawing || !annotationMode) return;
    
    const canvas = document.getElementById('annotation-canvas');
    const rect = canvas.getBoundingClientRect();
    
    // Clamp coordinates to canvas bounds
    let currentX = e.clientX - rect.left;
    let currentY = e.clientY - rect.top;
    currentX = Math.max(0, Math.min(currentX, canvas.width));
    currentY = Math.max(0, Math.min(currentY, canvas.height));
    
    // Redraw with temporary box
    redrawAnnotation();
    
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = '#4CAF50';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 2;
    ctx.strokeRect(
        drawStart.x,
        drawStart.y,
        currentX - drawStart.x,
        currentY - drawStart.y
    );
    ctx.setLineDash([]);
}

function onAnnotationMouseUp(e) {
    console.log('=== MOUSE UP ===');
    console.log('isDrawing:', isDrawing, 'wasDrawingBeforeLeave:', wasDrawingBeforeLeave);
    
    if (!isDrawing || !annotationMode) {
        console.log('Ignoring - not drawing or not in annotation mode');
        return;
    }
    
    isDrawing = false;
    
    const canvas = document.getElementById('annotation-canvas');
    const rect = canvas.getBoundingClientRect();
    
    // Clamp coordinates to canvas bounds
    let endX = e.clientX - rect.left;
    let endY = e.clientY - rect.top;
    endX = Math.max(0, Math.min(endX, canvas.width));
    endY = Math.max(0, Math.min(endY, canvas.height));
    
    const width = Math.abs(endX - drawStart.x);
    const height = Math.abs(endY - drawStart.y);
    
    console.log('Box size:', width, 'x', height, '(min: 20x20)');
    
    // Only save if box is large enough
    if (width > 20 && height > 20) {
        currentAnnotation = {
            x: Math.min(drawStart.x, endX),
            y: Math.min(drawStart.y, endY),
            width: width,
            height: height,
            label: document.getElementById('annotation-label').value
        };
        console.log('✅ Annotation CREATED successfully!');
        console.log('currentAnnotation is now:', JSON.stringify(currentAnnotation));
        updateAnnotationInfo();
        redrawAnnotation();
    } else {
        console.log('❌ Box too small, not saving');
        redrawAnnotation();
    }
    
    wasDrawingBeforeLeave = false;
}

function onAnnotationMouseLeave(e) {
    if (!isDrawing || !annotationMode) return;
    
    console.log('Mouse LEFT canvas while drawing');
    wasDrawingBeforeLeave = true;
    mouseLeftAt = { x: e.clientX, y: e.clientY };
    // Don't set isDrawing = false here - we want to continue if mouse comes back
}

function onAnnotationMouseEnter(e) {
    if (!annotationMode) return;
    
    console.log('Mouse ENTERED canvas, wasDrawingBeforeLeave:', wasDrawingBeforeLeave);
    
    // If we were drawing and mouse comes back, continue drawing
    if (wasDrawingBeforeLeave) {
        wasDrawingBeforeLeave = false;
        // Don't automatically resume - user needs to click again
        // Just clear the preview
        redrawAnnotation();
    }
}

function onAnnotationRightClick(e) {
    e.preventDefault(); // Hide context menu
    
    if (currentAnnotation) {
        if (confirm('Delete current bounding box?')) {
            clearAnnotation();
        }
    }
    return false;
}

async function saveAnnotation() {
    console.log('=== SAVE ANNOTATION ===');
    console.log('currentAnnotation:', currentAnnotation ? JSON.stringify(currentAnnotation) : 'NULL');
    console.log('reviewIndex:', reviewIndex);
    console.log('annotationMode:', annotationMode);
    
    if (!reviewImages[reviewIndex]) {
        alert('No image loaded!');
        return;
    }
    
    if (!currentAnnotation) {
        console.error('ERROR: currentAnnotation is null!');
        console.log('Did you draw a box? Check if mouse events are firing.');
        alert('No annotation to save! Please draw a bounding box first. Tip: Make sure Annotation Mode is ON and draw a box larger than 20x20 pixels.');
        return;
    }
    
    pendingAnnotationSave = true;
    const imgPath = reviewImages[reviewIndex].path;
    annotations[imgPath] = {...currentAnnotation}; // Copy to storage
    
    try {
        const img = document.getElementById('review-image');
        const response = await fetch('/api/annotation/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                image_path: imgPath,
                annotation: currentAnnotation,
                image_width: img.naturalWidth,
                image_height: img.naturalHeight
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log('✅ Annotation saved to server:', result);
            currentAnnotation = null; // Clear after successful save
            pendingAnnotationSave = false;
            updateAnnotationProgress();
            updateAnnotationInfo();
            redrawAnnotation();
            // Auto-advance to next image after short delay
            setTimeout(() => nextImage(), 500);
        } else {
            alert('Failed to save: ' + result.error);
            pendingAnnotationSave = false;
        }
    } catch (e) {
        console.error('Save error:', e);
        alert('Save error: ' + e.message);
        pendingAnnotationSave = false;
    }
}

function updateAnnotationProgress() {
    const count = Object.keys(annotations).length;
    document.getElementById('annotation-progress').textContent = count;
}

function clearAnnotation() {
    console.log('Clearing annotation, was:', currentAnnotation);
    currentAnnotation = null;
    redrawAnnotation();
    updateAnnotationInfo();
    console.log('Annotation cleared');
}

async function exportAnnotations() {
    const count = Object.keys(annotations).length;
    if (count === 0) {
        alert('No annotations to export!');
        return;
    }
    
    if (!confirm(`Export ${count} annotations as YOLO dataset?`)) return;
    
    try {
        const response = await fetch('/api/annotation/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                dataset_name: 'v12_yolo_annotated',
                annotations: annotations
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert(`✅ Export complete!\n\nImages: ${result.image_count}\nLabels: ${result.label_count}\n\nLocation: ${result.export_path}`);
        } else {
            alert('Export failed: ' + result.error);
        }
    } catch (e) {
        alert('Export error: ' + e.message);
    }
}

// ========== V13 PREPROCESSING FUNCTIONS ==========

let preprocessMode = false;
let preprocessEngine = 'canvas'; // 'canvas' or 'opencv'
let opencvLoaded = false;
// Session state management
const SESSION_STORAGE_KEY = 'v13_preprocess_session';

// Default preprocessing parameters
const DEFAULT_PARAMS = {
    contrast: 2.0,
    sharpen: 1.5,
    gamma: 1.2,
    denoise: 0,
    edge: 0,
    highpass: 0,
    local: 0,
    invert: false,
    equalize: false,
    // Phase 1.1 Advanced params
    hsvHue: 'all',
    hsvSat: 0.3,
    dilate: 0,
    cannyMorph: false,
    fftLow: 5,
    fftHigh: 50,
    fftEnable: false
};

// Load session from localStorage
function loadSessionState() {
    try {
        const saved = localStorage.getItem(SESSION_STORAGE_KEY);
        if (saved) {
            const session = JSON.parse(saved);
            console.log('Loaded session state:', session);
            return { ...DEFAULT_PARAMS, ...session.params };
        }
    } catch (e) {
        console.warn('Failed to load session state:', e);
    }
    return { ...DEFAULT_PARAMS };
}

// Save session to localStorage
function saveSessionState() {
    try {
        const session = {
            params: preprocessParams,
            timestamp: new Date().toISOString(),
            lastImage: reviewImages[reviewIndex] ? reviewImages[reviewIndex].path : null,
            lastIndex: reviewIndex
        };
        localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
        updateSessionUI();
        console.log('Session state saved');
    } catch (e) {
        console.warn('Failed to save session state:', e);
    }
}

// Manual save with visual feedback
function saveSessionStateManual() {
    saveSessionState();
    const statusEl = document.getElementById('session-status');
    const originalText = statusEl.textContent;
    statusEl.textContent = '✓ Saved!';
    statusEl.style.color = '#4CAF50';
    setTimeout(() => {
        statusEl.textContent = originalText;
        statusEl.style.color = '#4CAF50';
    }, 1500);
}

// Update session UI elements
function updateSessionUI() {
    try {
        const saved = localStorage.getItem(SESSION_STORAGE_KEY);
        if (saved) {
            const session = JSON.parse(saved);
            const timestampEl = document.getElementById('session-timestamp');
            if (timestampEl && session.timestamp) {
                const date = new Date(session.timestamp);
                timestampEl.textContent = 'Last saved: ' + date.toLocaleTimeString();
            }
        }
    } catch (e) {
        console.warn('Failed to update session UI:', e);
    }
}

// Reset session to defaults
function resetSessionState() {
    if (!confirm('Reset all preprocessing parameters to defaults?')) {
        return;
    }
    localStorage.removeItem(SESSION_STORAGE_KEY);
    preprocessParams = { ...DEFAULT_PARAMS };
    updateUIFromParams();
    if (preprocessMode) {
        applyPreprocessing();
    }
    // Update UI
    const statusEl = document.getElementById('session-status');
    const timestampEl = document.getElementById('session-timestamp');
    if (statusEl) statusEl.textContent = 'Reset to defaults';
    if (timestampEl) timestampEl.textContent = 'Last saved: Never';
    console.log('Session state reset to defaults');
}

// Update UI controls from current params
function updateUIFromParams() {
    document.getElementById('slider-contrast').value = preprocessParams.contrast;
    document.getElementById('slider-sharpen').value = preprocessParams.sharpen;
    document.getElementById('slider-gamma').value = preprocessParams.gamma;
    document.getElementById('slider-denoise').value = preprocessParams.denoise;
    document.getElementById('slider-edge').value = preprocessParams.edge;
    document.getElementById('slider-highpass').value = preprocessParams.highpass;
    document.getElementById('slider-local').value = preprocessParams.local;
    document.getElementById('check-invert').checked = preprocessParams.invert;
    document.getElementById('check-equalize').checked = preprocessParams.equalize;
    
    // Phase 1.1 params
    document.getElementById('select-hsv-hue').value = preprocessParams.hsvHue;
    document.getElementById('slider-hsv-sat').value = preprocessParams.hsvSat;
    document.getElementById('slider-dilate').value = preprocessParams.dilate;
    document.getElementById('check-canny-morph').checked = preprocessParams.cannyMorph;
    document.getElementById('input-fft-low').value = preprocessParams.fftLow;
    document.getElementById('input-fft-high').value = preprocessParams.fftHigh;
    document.getElementById('check-fft').checked = preprocessParams.fftEnable;
    
    // Update all value displays
    updatePreprocessParams();
}

// Initialize params from session
let preprocessParams = loadSessionState();

function togglePreprocessMode() {
    preprocessMode = document.getElementById('preprocess-mode').checked;
    const toolsDiv = document.getElementById('preprocess-tools');
    const indicator = document.getElementById('preprocess-indicator');
    
    if (preprocessMode) {
        toolsDiv.style.display = 'block';
        indicator.style.display = 'block';
        
        // Restore session state (params are already loaded, just update UI)
        updateUIFromParams();
        updateSessionUI();
        
        // Auto-apply preprocessing when enabled
        setTimeout(() => applyPreprocessing(), 100);
    } else {
        toolsDiv.style.display = 'none';
        indicator.style.display = 'none';
        // Reset to original view
        resetPreprocessView();
    }
}

function setPreprocessEngine(engine) {
    preprocessEngine = engine;
    
    // Update button styles
    document.getElementById('engine-canvas').style.background = engine === 'canvas' ? '#4CAF50' : 'rgba(255,255,255,0.2)';
    document.getElementById('engine-opencv').style.background = engine === 'opencv' ? '#4CAF50' : 'rgba(255,255,255,0.2)';
    
    if (engine === 'opencv' && !opencvLoaded) {
        loadOpenCV();
    } else if (preprocessMode) {
        applyPreprocessing();
    }
}

function loadOpenCV() {
    const statusDiv = document.getElementById('opencv-status');
    statusDiv.style.display = 'block';
    statusDiv.textContent = '⏳ Loading OpenCV.js...';
    
    // Load OpenCV.js dynamically
    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/4.8.0/opencv.js';
    script.onload = () => {
        opencvLoaded = true;
        statusDiv.textContent = '✅ OpenCV.js ready!';
        statusDiv.style.color = '#4CAF50';
        setTimeout(() => statusDiv.style.display = 'none', 2000);
        if (preprocessMode) applyPreprocessing();
    };
    script.onerror = () => {
        statusDiv.textContent = '❌ Failed to load OpenCV.js';
        statusDiv.style.color = '#f44336';
    };
    document.head.appendChild(script);
}

function updatePreprocessParams() {
    preprocessParams.contrast = parseFloat(document.getElementById('slider-contrast').value);
    preprocessParams.sharpen = parseFloat(document.getElementById('slider-sharpen').value);
    preprocessParams.gamma = parseFloat(document.getElementById('slider-gamma').value);
    preprocessParams.denoise = parseInt(document.getElementById('slider-denoise').value);
    preprocessParams.edge = parseFloat(document.getElementById('slider-edge').value);
    preprocessParams.highpass = parseFloat(document.getElementById('slider-highpass').value);
    preprocessParams.local = parseFloat(document.getElementById('slider-local').value);
    preprocessParams.invert = document.getElementById('check-invert').checked;
    preprocessParams.equalize = document.getElementById('check-equalize').checked;
    
    // Phase 1.1 Advanced params
    preprocessParams.hsvHue = document.getElementById('select-hsv-hue').value;
    preprocessParams.hsvSat = parseFloat(document.getElementById('slider-hsv-sat').value);
    preprocessParams.dilate = parseInt(document.getElementById('slider-dilate').value);
    preprocessParams.cannyMorph = document.getElementById('check-canny-morph').checked;
    preprocessParams.fftLow = parseInt(document.getElementById('input-fft-low').value);
    preprocessParams.fftHigh = parseInt(document.getElementById('input-fft-high').value);
    preprocessParams.fftEnable = document.getElementById('check-fft').checked;
    
    // Update value displays
    document.getElementById('val-contrast').textContent = preprocessParams.contrast.toFixed(1);
    document.getElementById('val-sharpen').textContent = preprocessParams.sharpen.toFixed(1);
    document.getElementById('val-gamma').textContent = preprocessParams.gamma.toFixed(1);
    document.getElementById('val-denoise').textContent = preprocessParams.denoise;
    document.getElementById('val-edge').textContent = preprocessParams.edge.toFixed(1);
    document.getElementById('val-highpass').textContent = preprocessParams.highpass.toFixed(1);
    document.getElementById('val-local').textContent = preprocessParams.local.toFixed(1);
    document.getElementById('val-invert').textContent = preprocessParams.invert ? 'ON' : 'OFF';
    document.getElementById('val-equalize').textContent = preprocessParams.equalize ? 'ON' : 'OFF';
    
    // Phase 1.1 displays
    document.getElementById('val-hsv-hue').textContent = preprocessParams.hsvHue.charAt(0).toUpperCase() + preprocessParams.hsvHue.slice(1);
    document.getElementById('val-hsv-sat').textContent = preprocessParams.hsvSat.toFixed(1);
    document.getElementById('val-dilate').textContent = preprocessParams.dilate;
    document.getElementById('val-canny-morph').textContent = preprocessParams.cannyMorph ? 'ON' : 'OFF';
    
    // Auto-save session state when params change
    saveSessionState();
}

function applyPreset(preset) {
    console.log('Applying preset:', preset);
    
    // Reset all advanced params first
    document.getElementById('slider-dilate').value = 0;
    document.getElementById('check-canny-morph').checked = false;
    document.getElementById('select-hsv-hue').value = 'all';
    document.getElementById('slider-hsv-sat').value = 0.3;
    document.getElementById('check-fft').checked = false;
    
    switch(preset) {
        case 'microtext':
            // Phase 1.1: Canny + Morphology for micro-text
            document.getElementById('check-canny-morph').checked = true;
            document.getElementById('slider-dilate').value = 2;
            document.getElementById('slider-contrast').value = 6;
            document.getElementById('slider-sharpen').value = 4;
            document.getElementById('slider-edge').value = 0; // Using Canny instead
            document.getElementById('slider-highpass').value = 1.5;
            document.getElementById('slider-gamma').value = 0.8;
            document.getElementById('check-invert').checked = false;
            break;
            
        case 'guilloche':
            // Phase 1.1: FFT Band-Pass for patterns
            document.getElementById('check-fft').checked = true;
            document.getElementById('input-fft-low').value = 5;
            document.getElementById('input-fft-high').value = 50;
            document.getElementById('slider-contrast').value = 3;
            document.getElementById('slider-sharpen').value = 2;
            document.getElementById('slider-edge').value = 0.5;
            document.getElementById('slider-highpass').value = 2;
            document.getElementById('slider-local').value = 4;
            document.getElementById('slider-gamma').value = 1.0;
            document.getElementById('check-invert').checked = false;
            break;
            
        case 'ghost':
            // Ghost image: brighten, equalize, enhance local contrast
            document.getElementById('slider-contrast').value = 2;
            document.getElementById('slider-sharpen').value = 1.5;
            document.getElementById('slider-edge').value = 0;
            document.getElementById('slider-highpass').value = 0;
            document.getElementById('slider-local').value = 3;
            document.getElementById('slider-gamma').value = 1.8;
            document.getElementById('check-equalize').checked = true;
            document.getElementById('check-invert').checked = false;
            break;
            
        case 'uv':
            // UV/security: invert, high contrast
            document.getElementById('slider-contrast').value = 8;
            document.getElementById('slider-sharpen').value = 3;
            document.getElementById('slider-edge').value = 2.5;
            document.getElementById('slider-highpass').value = 1.5;
            document.getElementById('slider-gamma').value = 0.6;
            document.getElementById('check-invert').checked = true;
            document.getElementById('check-equalize').checked = true;
            break;
            
        case 'hologram':
            // Phase 1.1: HSV extraction for holograms
            document.getElementById('select-hsv-hue').value = 'green';
            document.getElementById('slider-hsv-sat').value = 0.4;
            document.getElementById('slider-contrast').value = 4;
            document.getElementById('slider-sharpen').value = 2;
            document.getElementById('slider-gamma').value = 1.0;
            break;
    }
    
    updatePreprocessParams();
    applyPreprocessing();
}

function updatePreprocessView() {
    if (!preprocessMode) return;
    applyPreprocessing();
}

function resetPreprocessView() {
    const originalImg = document.getElementById('review-image');
    const preprocessedImg = document.getElementById('preprocess-image');
    const splitContainer = document.getElementById('split-view-container');
    
    // Reset to original single image view
    originalImg.style.display = 'block';
    originalImg.style.maxWidth = '100%';
    originalImg.style.marginLeft = '0';
    
    preprocessedImg.style.display = 'none';
    preprocessedImg.style.maxWidth = '100%';
    preprocessedImg.style.marginLeft = '0';
    
    splitContainer.style.display = 'none';
    
    // Clear preprocessed image source
    preprocessedImg.src = '';
}

// Cancel preprocessing and show original
function cancelPreprocessing() {
    console.log('Cancelling preprocessing, showing original');
    
    // Reset view to original only
    resetPreprocessView();
    
    // Clear the preprocessed image
    const preprocessedImg = document.getElementById('preprocess-image');
    preprocessedImg.src = '';
    
    // Hide preprocessed indicator
    document.getElementById('preprocess-indicator').style.display = 'none';
    
    // Optional: reset params to defaults (but keep session state)
    // Uncomment below if you want to also reset params when cancelling:
    // preprocessParams = { ...DEFAULT_PARAMS };
    // updateUIFromParams();
    
    console.log('Preprocessing cancelled, original image restored');
}

async function applyPreprocessing() {
    const originalImg = document.getElementById('review-image');
    const preprocessedImg = document.getElementById('preprocess-image');
    const viewMode = document.getElementById('preprocess-view').value;
    
    if (!originalImg.src) return;
    
    try {
        if (preprocessEngine === 'canvas' || !opencvLoaded) {
            // Use Canvas-based preprocessing
            const processedDataUrl = await preprocessWithCanvas(originalImg);
            preprocessedImg.src = processedDataUrl;
        } else {
            // Use OpenCV.js preprocessing
            const processedDataUrl = await preprocessWithOpenCV(originalImg);
            preprocessedImg.src = processedDataUrl;
        }
        
        // Update view based on selected mode
        updateViewMode(viewMode);
        
    } catch (e) {
        console.error('Preprocessing error:', e);
        alert('Preprocessing failed: ' + e.message);
    }
}

function updateViewMode(mode) {
    const originalImg = document.getElementById('review-image');
    const preprocessedImg = document.getElementById('preprocess-image');
    const splitContainer = document.getElementById('split-view-container');
    
    if (mode === 'side') {
        // Side-by-side view
        originalImg.style.display = 'block';
        originalImg.style.maxWidth = '48%';
        preprocessedImg.style.display = 'block';
        preprocessedImg.style.maxWidth = '48%';
        splitContainer.style.display = 'none';
    } else if (mode === 'overlay') {
        // Toggle overlay - show only preprocessed
        originalImg.style.display = 'none';
        preprocessedImg.style.display = 'block';
        preprocessedImg.style.maxWidth = '100%';
        splitContainer.style.display = 'none';
    } else if (mode === 'split') {
        // Split slider view
        originalImg.style.display = 'none';
        preprocessedImg.style.display = 'none';
        splitContainer.style.display = 'block';
        
        // Set images for split view
        document.getElementById('split-original-img').src = originalImg.src;
        document.getElementById('split-preprocessed-img').src = preprocessedImg.src;
    }
}

function updateSplitPosition(value) {
    const splitOriginal = document.getElementById('split-original');
    splitOriginal.style.width = value + '%';
}

// ========== PHASE 1.1: ADVANCED PREPROCESSING ==========

// Convert RGB data to grayscale array
function rgbToGray(data) {
    const gray = new Uint8Array(data.length / 4);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    return gray;
}

// Convert grayscale array back to RGB data
function grayToRgb(data, gray) {
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        const val = Math.max(0, Math.min(255, gray[j]));
        data[i] = data[i + 1] = data[i + 2] = val;
    }
}

// HSV Color Space Extraction for Holograms
function applyHSVExtraction(data, width, height, hueRange, satThreshold) {
    // Define hue ranges for different colors
    const hueRanges = {
        'red': [0, 30],
        'yellow': [30, 60],
        'green': [60, 90],
        'blue': [90, 120],
        'purple': [120, 150],
        'all': [0, 360]
    };
    
    const [hMin, hMax] = hueRanges[hueRange] || [0, 360];
    
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i] / 255;
        const g = data[i + 1] / 255;
        const b = data[i + 2] / 255;
        
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const delta = max - min;
        
        // Calculate HSV
        let h = 0, s = max > 0 ? delta / max : 0, v = max;
        
        if (delta > 0) {
            if (max === r) h = ((g - b) / delta + 6) % 6;
            else if (max === g) h = (b - r) / delta + 2;
            else h = (r - g) / delta + 4;
            h *= 60; // Convert to degrees
        }
        
        // Check if pixel matches criteria
        const inHueRange = hueRange === 'high-sat' ? true : (h >= hMin && h <= hMax);
        const highSaturation = s >= satThreshold;
        
        if (!inHueRange || !highSaturation) {
            // Dim non-matching pixels instead of black
            data[i] *= 0.2;
            data[i + 1] *= 0.2;
            data[i + 2] *= 0.2;
        } else {
            // Boost matching pixels
            data[i] = Math.min(255, data[i] * 1.5);
            data[i + 1] = Math.min(255, data[i + 1] * 1.5);
            data[i + 2] = Math.min(255, data[i + 2] * 1.5);
        }
    }
}

// Canny Edge Detection + Morphological Operations
function applyCannyMorphology(data, width, height, lowThreshold, highThreshold, dilateIterations) {
    // Convert to grayscale
    const gray = rgbToGray(data);
    
    // Apply Gaussian blur first
    const blurred = applyGaussianBlur(gray, width, height, 1.5);
    
    // Sobel operators for gradient
    const sobelX = new Float32Array(width * height);
    const sobelY = new Float32Array(width * height);
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            
            // Sobel X: [-1 0 1; -2 0 2; -1 0 1]
            sobelX[idx] = -blurred[(y-1)*width + (x-1)] + blurred[(y-1)*width + (x+1)]
                        -2*blurred[y*width + (x-1)] + 2*blurred[y*width + (x+1)]
                        -blurred[(y+1)*width + (x-1)] + blurred[(y+1)*width + (x+1)];
            
            // Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1]
            sobelY[idx] = -blurred[(y-1)*width + (x-1)] - 2*blurred[(y-1)*width + x] - blurred[(y-1)*width + (x+1)]
                        + blurred[(y+1)*width + (x-1)] + 2*blurred[(y+1)*width + x] + blurred[(y+1)*width + (x+1)];
        }
    }
    
    // Calculate gradient magnitude
    const magnitude = new Float32Array(width * height);
    let maxMag = 0;
    for (let i = 0; i < width * height; i++) {
        magnitude[i] = Math.sqrt(sobelX[i] * sobelX[i] + sobelY[i] * sobelY[i]);
        maxMag = Math.max(maxMag, magnitude[i]);
    }
    
    // Normalize and threshold (simple hysteresis)
    const edges = new Uint8Array(width * height);
    const low = lowThreshold;
    const high = highThreshold;
    
    for (let i = 0; i < width * height; i++) {
        const mag = (magnitude[i] / maxMag) * 255;
        if (mag >= high) edges[i] = 255;
        else if (mag >= low) edges[i] = 128;
        else edges[i] = 0;
    }
    
    // Dilate edges if requested
    if (dilateIterations > 0) {
        const dilated = applyDilation(edges, width, height, dilateIterations);
        grayToRgb(data, dilated);
    } else {
        grayToRgb(data, edges);
    }
}

// Gaussian blur for grayscale
function applyGaussianBlur(gray, width, height, sigma) {
    // Simple 3x3 Gaussian kernel
    const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
    const kernelSum = 16;
    
    const result = new Uint8Array(gray);
    const temp = new Uint8Array(gray);
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sum = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = (y + ky) * width + (x + kx);
                    const kidx = (ky + 1) * 3 + (kx + 1);
                    sum += temp[idx] * kernel[kidx];
                }
            }
            result[y * width + x] = sum / kernelSum;
        }
    }
    
    return result;
}

// Morphological Dilation
function applyDilation(gray, width, height, iterations) {
    let result = new Uint8Array(gray);
    let temp = new Uint8Array(gray);
    
    for (let iter = 0; iter < iterations; iter++) {
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                
                // Find maximum in 3x3 neighborhood
                let maxVal = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nidx = (y + dy) * width + (x + dx);
                        maxVal = Math.max(maxVal, temp[nidx]);
                    }
                }
                result[idx] = maxVal;
            }
        }
        // Swap for next iteration
        [temp, result] = [result, temp];
    }
    
    return temp;
}

// FFT Band-Pass Filter (Simplified 2D DFT for small images)
function applyFFTBandPass(gray, width, height, lowFreq, highFreq) {
    // For performance, process in smaller tiles or use simplified approach
    // Here we use a spatial domain approximation (Laplacian of Gaussian)
    
    // Step 1: Gaussian blur (low-pass)
    const blurred = applyGaussianBlur(gray, width, height, 2);
    
    // Step 2: Subtract from original (high-pass component)
    const highPass = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
        highPass[i] = gray[i] - blurred[i];
    }
    
    // Step 3: Additional band-pass filtering in spatial domain
    // Using difference of Gaussians (DoG) approximation
    const blurred2 = applyGaussianBlur(gray, width, height, 4);
    const dog = new Float32Array(width * height);
    
    for (let i = 0; i < width * height; i++) {
        // Difference of Gaussians
        dog[i] = blurred[i] - blurred2[i];
        
        // Scale by frequency range parameters
        const scale = (highFreq - lowFreq) / 50;
        dog[i] *= scale;
        
        // Add back to original with emphasis
        dog[i] = gray[i] + dog[i] * 2;
    }
    
    // Normalize and convert to uint8
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < width * height; i++) {
        min = Math.min(min, dog[i]);
        max = Math.max(max, dog[i]);
    }
    
    const result = new Uint8Array(width * height);
    const range = max - min || 1;
    for (let i = 0; i < width * height; i++) {
        result[i] = Math.round(((dog[i] - min) / range) * 255);
    }
    
    return result;
}

// ========== CANVAS-BASED PREPROCESSING ==========

async function preprocessWithCanvas(img) {
    const canvas = document.getElementById('preprocess-canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Set canvas size to match image
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    
    // Draw original image
    ctx.drawImage(img, 0, 0);
    
    // Get image data
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let data = imageData.data;
    const width = canvas.width;
    const height = canvas.height;
    
    console.log('Preprocessing with params:', preprocessParams);
    
    // ===== Phase 1.1: Advanced Preprocessing (First) =====
    
    // 1. HSV Color Space Separation for Holograms
    if (preprocessParams.hsvHue !== 'all' || preprocessParams.hsvSat > 0) {
        console.log('Applying HSV extraction...');
        applyHSVExtraction(data, width, height, preprocessParams.hsvHue, preprocessParams.hsvSat);
    }
    
    // 2. Canny Edge + Morphological Operations for Micro-Text
    if (preprocessParams.cannyMorph) {
        console.log('Applying Canny + Morphology...');
        applyCannyMorphology(data, width, height, 50, 150, preprocessParams.dilate);
    } else if (preprocessParams.dilate > 0) {
        // Just dilation without Canny
        console.log('Applying dilation...');
        const gray = rgbToGray(data);
        const dilated = applyDilation(gray, width, height, preprocessParams.dilate);
        grayToRgb(data, dilated);
    }
    
    // 3. FFT Band-Pass for Guilloché Patterns
    if (preprocessParams.fftEnable && preprocessParams.fftHigh > preprocessParams.fftLow) {
        console.log('Applying FFT band-pass...');
        const gray = rgbToGray(data);
        const filtered = applyFFTBandPass(gray, width, height, preprocessParams.fftLow, preprocessParams.fftHigh);
        grayToRgb(data, filtered);
    }
    
    // ===== Phase 1.0: Basic Enhancements =====
    
    // 4. Invert (if enabled)
    if (preprocessParams.invert) {
        applyInvert(data);
    }
    
    // 5. Equalize histogram (if enabled)
    if (preprocessParams.equalize) {
        applyHistogramEqualization(data, width, height);
    }
    
    // 6. CLAHE contrast enhancement
    if (preprocessParams.contrast > 0) {
        applyCLAHE(data, width, height, preprocessParams.contrast);
    }
    
    // 7. Local contrast enhancement
    if (preprocessParams.local > 0) {
        applyLocalContrast(data, width, height, preprocessParams.local);
    }
    
    // 8. High-pass filter for fine details
    if (preprocessParams.highpass > 0) {
        applyHighPass(data, width, height, preprocessParams.highpass);
    }
    
    // 9. Edge enhancement (Sobel)
    if (preprocessParams.edge > 0) {
        applySobelEdge(data, width, height, preprocessParams.edge);
    }
    
    // 10. Unsharp mask sharpening
    if (preprocessParams.sharpen > 1.0) {
        applySharpen(data, width, height, preprocessParams.sharpen);
    }
    
    // 11. Gamma correction
    if (preprocessParams.gamma !== 1.0) {
        applyGamma(data, preprocessParams.gamma);
    }
    
    // 12. Denoising (last)
    if (preprocessParams.denoise > 0) {
        applyDenoise(data, width, height, preprocessParams.denoise);
    }
    
    // Put processed data back
    ctx.putImageData(imageData, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.95);
}

function applyInvert(data) {
    for (let i = 0; i < data.length; i += 4) {
        data[i] = 255 - data[i];
        data[i + 1] = 255 - data[i + 1];
        data[i + 2] = 255 - data[i + 2];
    }
}

function applyHistogramEqualization(data, width, height) {
    // Convert to grayscale for histogram
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
        gray[i / 4] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    
    // Calculate histogram
    const hist = new Array(256).fill(0);
    for (let i = 0; i < gray.length; i++) {
        hist[gray[i]]++;
    }
    
    // Calculate CDF
    const cdf = new Array(256);
    cdf[0] = hist[0];
    for (let i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    
    // Normalize CDF
    const cdfMin = cdf.find(v => v > 0) || 0;
    const scale = 255 / (gray.length - cdfMin);
    const lut = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        lut[i] = Math.round((cdf[i] - cdfMin) * scale);
    }
    
    // Apply to each channel
    for (let i = 0; i < data.length; i += 4) {
        const y = lut[gray[i / 4]];
        const factor = y / (gray[i / 4] || 1);
        data[i] = Math.min(255, Math.round(data[i] * factor));
        data[i + 1] = Math.min(255, Math.round(data[i + 1] * factor));
        data[i + 2] = Math.min(255, Math.round(data[i + 2] * factor));
    }
}

function applyLocalContrast(data, width, height, strength) {
    const original = new Uint8ClampedArray(data);
    const radius = 3;
    
    for (let y = radius; y < height - radius; y++) {
        for (let x = radius; x < width - radius; x++) {
            const idx = (y * width + x) * 4;
            
            // Calculate local mean
            let localSum = [0, 0, 0];
            let count = 0;
            
            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    const nidx = ((y + dy) * width + (x + dx)) * 4;
                    localSum[0] += original[nidx];
                    localSum[1] += original[nidx + 1];
                    localSum[2] += original[nidx + 2];
                    count++;
                }
            }
            
            const localMean = [localSum[0] / count, localSum[1] / count, localSum[2] / count];
            
            // Enhance difference from local mean
            for (let c = 0; c < 3; c++) {
                const diff = original[idx + c] - localMean[c];
                const enhanced = localMean[c] + diff * (1 + strength * 0.3);
                data[idx + c] = Math.max(0, Math.min(255, enhanced));
            }
        }
    }
}

function applyHighPass(data, width, height, strength) {
    const original = new Uint8ClampedArray(data);
    
    // High-pass kernel (Laplacian approximation)
    const kernel = [
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    ];
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = (y * width + x) * 4;
            
            for (let c = 0; c < 3; c++) {
                let sum = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const nidx = ((y + ky) * width + (x + kx)) * 4;
                        sum += original[nidx + c] * kernel[(ky + 1) * 3 + (kx + 1)];
                    }
                }
                
                // Blend with original
                const highPass = Math.max(0, Math.min(255, sum));
                data[idx + c] = original[idx + c] * (1 - strength * 0.3) + highPass * (strength * 0.3);
            }
        }
    }
}

function applySobelEdge(data, width, height, strength) {
    const original = new Uint8ClampedArray(data);
    const gray = new Float32Array(width * height);
    
    // Convert to grayscale
    for (let i = 0; i < data.length; i += 4) {
        gray[i / 4] = 0.299 * original[i] + 0.587 * original[i + 1] + 0.114 * original[i + 2];
    }
    
    // Sobel operators
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
    
    const edgeMagnitude = new Float32Array(width * height);
    let maxMagnitude = 0;
    
    // Calculate edge magnitude
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let gx = 0, gy = 0;
            
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = (y + ky) * width + (x + kx);
                    const kidx = (ky + 1) * 3 + (kx + 1);
                    gx += gray[idx] * sobelX[kidx];
                    gy += gray[idx] * sobelY[kidx];
                }
            }
            
            const magnitude = Math.sqrt(gx * gx + gy * gy);
            edgeMagnitude[y * width + x] = magnitude;
            maxMagnitude = Math.max(maxMagnitude, magnitude);
        }
    }
    
    // Normalize and blend with original
    const scale = maxMagnitude > 0 ? 255 / maxMagnitude : 0;
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = (y * width + x) * 4;
            const edgeVal = Math.min(255, edgeMagnitude[y * width + x] * scale);
            
            // Blend edge detection with original
            const blend = strength * 0.3;
            for (let c = 0; c < 3; c++) {
                data[idx + c] = original[idx + c] * (1 - blend) + edgeVal * blend;
            }
        }
    }
}

// CLAHE-like contrast enhancement
function applyCLAHE(data, width, height, clipLimit) {
    // Convert to grayscale for histogram analysis
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
        gray[i / 4] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    
    // Simple adaptive histogram equalization (tile-based)
    const tileSize = 32;
    const tilesX = Math.ceil(width / tileSize);
    const tilesY = Math.ceil(height / tileSize);
    
    for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
            const x0 = tx * tileSize;
            const y0 = ty * tileSize;
            const x1 = Math.min(x0 + tileSize, width);
            const y1 = Math.min(y0 + tileSize, height);
            
            // Calculate histogram for this tile
            const hist = new Array(256).fill(0);
            let totalPixels = 0;
            
            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) {
                    const idx = y * width + x;
                    hist[gray[idx]]++;
                    totalPixels++;
                }
            }
            
            // Clip histogram
            if (clipLimit > 0) {
                const clip = Math.round(clipLimit * totalPixels / 256);
                let excess = 0;
                for (let i = 0; i < 256; i++) {
                    if (hist[i] > clip) {
                        excess += hist[i] - clip;
                        hist[i] = clip;
                    }
                }
                // Redistribute excess
                const redistribution = Math.floor(excess / 256);
                for (let i = 0; i < 256; i++) {
                    hist[i] += redistribution;
                }
            }
            
            // Calculate CDF
            const cdf = new Array(256);
            cdf[0] = hist[0];
            for (let i = 1; i < 256; i++) {
                cdf[i] = cdf[i - 1] + hist[i];
            }
            
            // Normalize CDF
            const cdfMin = cdf.find(v => v > 0) || 0;
            const scale = 255 / (totalPixels - cdfMin);
            
            // Apply to tile
            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) {
                    const idx = y * width + x;
                    const pixelIdx = idx * 4;
                    const newVal = Math.round((cdf[gray[idx]] - cdfMin) * scale);
                    
                    // Apply to RGB channels proportionally
                    const factor = newVal / (gray[idx] || 1);
                    data[pixelIdx] = Math.min(255, data[pixelIdx] * factor);
                    data[pixelIdx + 1] = Math.min(255, data[pixelIdx + 1] * factor);
                    data[pixelIdx + 2] = Math.min(255, data[pixelIdx + 2] * factor);
                }
            }
        }
    }
}

// Unsharp mask sharpening
function applySharpen(data, width, height, amount) {
    const original = new Uint8ClampedArray(data);
    const kernel = [
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    ];
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = (y * width + x) * 4;
            
            for (let c = 0; c < 3; c++) {
                let sum = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const kidx = ((y + ky) * width + (x + kx)) * 4;
                        sum += original[kidx + c] * kernel[(ky + 1) * 3 + (kx + 1)];
                    }
                }
                // Blend original with sharpened based on amount
                const sharpened = Math.max(0, Math.min(255, sum));
                data[idx + c] = original[idx + c] * (2 - amount) + sharpened * (amount - 1);
            }
        }
    }
}

// Gamma correction
function applyGamma(data, gamma) {
    // Precompute gamma lookup table
    const table = new Uint8Array(256);
    const invGamma = 1.0 / gamma;
    for (let i = 0; i < 256; i++) {
        table[i] = Math.round(Math.pow(i / 255.0, invGamma) * 255);
    }
    
    // Apply to all pixels
    for (let i = 0; i < data.length; i += 4) {
        data[i] = table[data[i]];
        data[i + 1] = table[data[i + 1]];
        data[i + 2] = table[data[i + 2]];
    }
}

// Simple box blur denoising
function applyDenoise(data, width, height, strength) {
    if (strength <= 0) return;
    
    const original = new Uint8ClampedArray(data);
    const radius = Math.min(2, strength);
    
    for (let y = radius; y < height - radius; y++) {
        for (let x = radius; x < width - radius; x++) {
            const idx = (y * width + x) * 4;
            
            for (let c = 0; c < 3; c++) {
                let sum = 0;
                let count = 0;
                
                for (let dy = -radius; dy <= radius; dy++) {
                    for (let dx = -radius; dx <= radius; dx++) {
                        const nidx = ((y + dy) * width + (x + dx)) * 4;
                        sum += original[nidx + c];
                        count++;
                    }
                }
                
                const blurred = sum / count;
                // Blend based on strength
                const blend = Math.min(1, strength / 10);
                data[idx + c] = original[idx + c] * (1 - blend) + blurred * blend;
            }
        }
    }
}

// ========== OPENCV.JS PREPROCESSING ==========

async function preprocessWithOpenCV(img) {
    if (!opencvLoaded || typeof cv === 'undefined') {
        throw new Error('OpenCV.js not loaded');
    }
    
    // Create Mat from image
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    let src = cv.imread(canvas);
    let dst = new cv.Mat();
    
    try {
        // Convert to LAB for CLAHE
        if (preprocessParams.contrast > 0) {
            cv.cvtColor(src, dst, cv.COLOR_BGR2Lab);
            let labPlanes = new cv.MatVector();
            cv.split(dst, labPlanes);
            let clahe = cv.createCLAHE(preprocessParams.contrast, new cv.Size(8, 8));
            clahe.apply(labPlanes.get(0), labPlanes.get(0));
            cv.merge(labPlanes, dst);
            cv.cvtColor(dst, src, cv.COLOR_Lab2BGR);
            labPlanes.delete();
            clahe.delete();
        }
        
        // Sharpen
        if (preprocessParams.sharpen > 1.0) {
            let blurred = new cv.Mat();
            cv.GaussianBlur(src, blurred, new cv.Size(0, 0), 3);
            cv.addWeighted(src, preprocessParams.sharpen, blurred, -0.5, 0, dst);
            src.delete();
            src = dst.clone();
            blurred.delete();
        }
        
        // Gamma correction
        if (preprocessParams.gamma !== 1.0) {
            const invGamma = 1.0 / preprocessParams.gamma;
            const table = new Uint8Array(256);
            for (let i = 0; i < 256; i++) {
                table[i] = Math.pow(i / 255.0, invGamma) * 255;
            }
            // Apply LUT
            let lut = cv.matFromArray(1, 256, cv.CV_8U, table);
            cv.LUT(src, lut, dst);
            src.delete();
            src = dst.clone();
            lut.delete();
        }
        
        // Denoise
        if (preprocessParams.denoise > 0) {
            cv.fastNlMeansDenoisingColored(src, dst, preprocessParams.denoise, preprocessParams.denoise, 7, 21);
            src.delete();
            src = dst.clone();
        }
        
        // Output to canvas
        cv.imshow(canvas, src);
        return canvas.toDataURL('image/jpeg', 0.95);
        
    } finally {
        src.delete();
        dst.delete();
    }
}

async function savePreprocessedImage() {
    const preprocessedImg = document.getElementById('preprocess-image');
    if (!preprocessedImg.src || preprocessedImg.style.display === 'none') {
        alert('No preprocessed image to save. Enable preprocessing and apply first.');
        return;
    }
    
    if (!reviewImages[reviewIndex]) {
        alert('No image loaded!');
        return;
    }
    
    try {
        // Convert data URL to blob
        const response = await fetch(preprocessedImg.src);
        const blob = await response.blob();
        const base64 = await new Promise(resolve => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(blob);
        });
        
        // Send to server
        const imgPath = reviewImages[reviewIndex].path;
        const saveResponse = await fetch('/api/preprocess/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                original_path: imgPath,
                image_data: base64,
                params: preprocessParams,
                engine: preprocessEngine
            })
        });
        
        const result = await saveResponse.json();
        if (result.success) {
            alert('✅ Preprocessed image saved! Location: ' + result.save_path);
        } else {
            alert('Save failed: ' + result.error);
        }
    } catch (e) {
        alert('Error saving: ' + e.message);
    }
}

// Track annotation state to prevent race conditions
let pendingAnnotationSave = false;

document.getElementById('review-image').addEventListener('load', () => {
    console.log('Image loaded, annotationMode:', annotationMode, 'preprocessMode:', preprocessMode);
    
    // Handle annotation mode
    if (annotationMode) {
        updateAnnotationCanvasSize();
        
        // IMPORTANT: Don't clear currentAnnotation if we just drew a box!
        // Only load from storage if we don't have an unsaved annotation
        if (!currentAnnotation) {
            loadCurrentAnnotation();
        } else {
            console.log('Preserving unsaved annotation:', currentAnnotation);
            redrawAnnotation();
        }
    }
    
    // Handle preprocessing mode - apply to new image after it loads
    if (preprocessMode) {
        console.log('Auto-applying preprocessing to new image');
        setTimeout(() => applyPreprocessing(), 100);
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Only work when review interface is visible
    if (document.getElementById('review-interface').style.display === 'none') return;
    
    switch(e.code) {
        case 'Space':
            e.preventDefault();
            markCorrect();
            break;
        case 'ArrowLeft':
            prevImage();
            break;
        case 'ArrowRight':
            nextImage();
            break;
        case 'KeyB':
            moveImage('cnie_back');
            break;
        case 'KeyF':
            moveImage('cnie_front');
            break;
        case 'Delete':
            deleteImage();
            break;
    }
});
</script>'''
    
    return render_template_string(render_page('Manual Review', 'manual', content, scripts))


@app.route('/stats')
def stats_page():
    content = '''
<div class="card">
    <div class="card-header">
        <div class="card-title">📊 Dataset Overview</div>
        <select class="form-select" id="stats-dataset" onchange="loadDetailedStats()" style="width: 250px;">
            <option value="combined" selected>📊 Combined (All Datasets)</option>
            <option value="v8_stage2_clean">v8_stage2_clean (Original)</option>
            <option value="v10_manual_capture">v10_manual_capture (New Captures)</option>
        </select>
    </div>
    
    <!-- Summary Cards -->
    <div class="stats-grid" id="summary-cards">
        <div class="stat-card">
            <div class="stat-value" id="stat-total">-</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-front-total">-</div>
            <div class="stat-label">Front Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-back-total">-</div>
            <div class="stat-label">Back Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="stat-nocard-total">-</div>
            <div class="stat-label">No-Card Images</div>
        </div>
    </div>
</div>

<!-- Detailed Breakdown -->
<div class="card">
    <div class="card-header">
        <div class="card-title">🔍 Detailed Breakdown (with Mislabel Detection)</div>
    </div>
    <div id="detailed-breakdown">
        <p class="text-center">Loading detailed statistics...</p>
    </div>
</div>

<!-- Advisor -->
<div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
    <div class="card-header" style="border-bottom: 1px solid rgba(255,255,255,0.2);">
        <div class="card-title" style="color: white;">🎯 Capture Advisor</div>
    </div>
    <div id="capture-advisor">
        <p>Calculating recommendations...</p>
    </div>
</div>

<!-- Auto Compensation -->
<div class="card">
    <div class="card-header">
        <div class="card-title">⚖️ Automatic Split Compensation</div>
    </div>
    <div id="compensation-plan">
        <p class="text-center">Loading compensation analysis...</p>
    </div>
</div>
'''
    
    scripts = '''<script>
const TARGETS = {
    front: { total: 500, train: 350, val: 75, test: 75 },
    back: { total: 500, train: 350, val: 75, test: 75 },
    no_card: { total: 250, train: 175, val: 37, test: 38 }
};

async function loadDetailedStats() {
    const dataset = document.getElementById('stats-dataset').value;
    
    try {
        const response = await fetch(`/api/stats/detailed?dataset=${dataset}`);
        const data = await response.json();
        
        // Update summary
        document.getElementById('stat-total').textContent = data.total;
        document.getElementById('stat-front-total').textContent = data.front;
        document.getElementById('stat-back-total').textContent = data.back;
        document.getElementById('stat-nocard-total').textContent = data.no_card;
        
        // Build detailed breakdown
        let breakdownHtml = '<div style="overflow-x: auto;">';
        breakdownHtml += '<table style="width: 100%; border-collapse: collapse;">';
        breakdownHtml += '<thead><tr style="background: #f5f5f5;">';
        breakdownHtml += '<th style="padding: 12px; text-align: left;">Category</th>';
        breakdownHtml += '<th style="padding: 12px; text-align: center;">Total</th>';
        breakdownHtml += '<th style="padding: 12px; text-align: center;">Expected Front</th>';
        breakdownHtml += '<th style="padding: 12px; text-align: center;">Actual Front</th>';
        breakdownHtml += '<th style="padding: 12px; text-align: center;">Expected Back</th>';
        breakdownHtml += '<th style="padding: 12px; text-align: center;">Actual Back</th>';
        breakdownHtml += '<th style="padding: 12px; text-align: center;">Mislabels</th>';
        breakdownHtml += '</tr></thead><tbody>';
        
        for (const [split, info] of Object.entries(data.by_split)) {
            const frontFolder = info.cnie_front || {};
            const backFolder = info.cnie_back || {};
            
            const frontTotal = (frontFolder.total || 0);
            const backTotal = (backFolder.total || 0);
            const frontInFront = frontFolder.front || 0;
            const backInFront = frontFolder.back || 0;
            const frontInBack = backFolder.front || 0;
            const backInBack = backFolder.back || 0;
            
            const mislabels = (backInFront > 0 || frontInBack > 0);
            
            breakdownHtml += `<tr style="border-bottom: 1px solid #eee; ${mislabels ? 'background: #fff3cd;' : ''}">`;
            breakdownHtml += `<td style="padding: 12px; font-weight: 600;">${split}</td>`;
            breakdownHtml += `<td style="padding: 12px; text-align: center;">${frontTotal + backTotal}</td>`;
            breakdownHtml += `<td style="padding: 12px; text-align: center; color: #666;">${frontTotal}</td>`;
            breakdownHtml += `<td style="padding: 12px; text-align: center; color: ${backInFront > 0 ? '#e74c3c' : '#27ae60'}; font-weight: ${backInFront > 0 ? 'bold' : 'normal'};">${frontInFront}${backInFront > 0 ? ' <span style="color: #e74c3c;">(+' + backInFront + ' wrong!)</span>' : ''}</td>`;
            breakdownHtml += `<td style="padding: 12px; text-align: center; color: #666;">${backTotal}</td>`;
            breakdownHtml += `<td style="padding: 12px; text-align: center; color: ${frontInBack > 0 ? '#e74c3c' : '#27ae60'}; font-weight: ${frontInBack > 0 ? 'bold' : 'normal'};">${backInBack}${frontInBack > 0 ? ' <span style="color: #e74c3c;">(+' + frontInBack + ' wrong!)</span>' : ''}</td>`;
            breakdownHtml += `<td style="padding: 12px; text-align: center;">${mislabels ? '<span style="color: #e74c3c; font-weight: bold;">⚠️ ' + (backInFront + frontInBack) + ' mislabels</span>' : '<span style="color: #27ae60;">✓ Clean</span>'}</td>`;
            breakdownHtml += '</tr>';
        }
        
        breakdownHtml += '</tbody></table></div>';
        
        if (data.mislabel_count > 0) {
            breakdownHtml += `<div class="alert alert-warning" style="margin-top: 15px;">
                <strong>⚠️ Found ${data.mislabel_count} mislabeled images!</strong><br>
                These images are in the wrong folder and will hurt model accuracy.
                <a href="/manual" style="color: #856404; text-decoration: underline;">Go to Manual Review to fix them</a>
            </div>`;
        }
        
        document.getElementById('detailed-breakdown').innerHTML = breakdownHtml;
        
        // Build advisor
        buildAdvisor(data);
        
        // Build compensation
        buildCompensation(data);
        
    } catch (e) {
        console.error('Failed to load stats:', e);
        document.getElementById('detailed-breakdown').innerHTML = '<p class="text-center" style="color: #e74c3c;">Error loading statistics</p>';
    }
}

function buildAdvisor(data) {
    let adviceHtml = '';
    const needs = [];
    
    // FIXED overall targets - not relative to current totals
    // This prevents the "never-ending target" problem
    // Total = train + val + test = 350 + 75 + 75 = 500 per class
    const OVERALL_TARGETS = {
        front: 500,   // Fixed target for total front images
        back: 500,    // Fixed target for total back images
        no_card: 250  // Fixed target for total no-card images (175 + 37 + 38)
    };
    
    const frontTotal = data.front;
    const backTotal = data.back;
    const noCardTotal = data.no_card;
    
    // Check overall balance with FIXED targets
    if (frontTotal < OVERALL_TARGETS.front) {
        const needCount = OVERALL_TARGETS.front - frontTotal;
        needs.push({
            class: 'front',
            split: '',
            count: needCount,
            icon: '📸',
            text: `Need ${needCount} more FRONT images (target: ${OVERALL_TARGETS.front})`,
            priority: 'high'
        });
    }
    
    if (backTotal < OVERALL_TARGETS.back) {
        const needCount = OVERALL_TARGETS.back - backTotal;
        needs.push({
            class: 'back', 
            split: '',
            count: needCount,
            icon: '📸',
            text: `Need ${needCount} more BACK images (target: ${OVERALL_TARGETS.back})`,
            priority: 'high'
        });
    }
    
    if (noCardTotal < OVERALL_TARGETS.no_card) {
        const needCount = OVERALL_TARGETS.no_card - noCardTotal;
        needs.push({
            class: 'no_card', 
            split: '',
            count: needCount,
            icon: '📸',
            text: `Need ${needCount} more NO-CARD images (target: ${OVERALL_TARGETS.no_card})`,
            priority: 'medium'
        });
    }
    
    // Check per-split targets
    for (const [split, info] of Object.entries(data.by_split)) {
        if (split === 'all') continue; // Skip flat structure
        
        const frontTotal = (info.cnie_front?.total || 0);
        const backTotal = (info.cnie_back?.total || 0);
        const targetF = TARGETS.front[split] || 75;
        const targetB = TARGETS.back[split] || 75;
        
        if (frontTotal < targetF) {
            const needCount = targetF - frontTotal;
            needs.push({
                class: 'front',
                split: split,
                count: needCount,
                icon: '🎯',
                text: `${split}: Need ${needCount} more FRONT`,
                priority: frontTotal < targetF * 0.5 ? 'high' : 'medium'
            });
        }
        if (backTotal < targetB) {
            const needCount = targetB - backTotal;
            needs.push({
                class: 'back',
                split: split,
                count: needCount,
                icon: '🎯',
                text: `${split}: Need ${needCount} more BACK`,
                priority: backTotal < targetB * 0.5 ? 'high' : 'medium'
            });
        }
    }
    
    if (needs.length === 0) {
        adviceHtml = '<h3 style="font-size: 24px; margin-bottom: 10px;">🎉 Perfect Balance!</h3><p>All targets met. Great job!</p>';
    } else {
        adviceHtml = '<h3 style="font-size: 24px; margin-bottom: 15px;">📋 Capture Recommendations</h3>';
        adviceHtml += '<div style="display: grid; gap: 10px;">';
        
        // Sort by priority
        needs.sort((a, b) => (a.priority === 'high' ? -1 : 1));
        
        needs.forEach(need => {
            const bg = need.priority === 'high' ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.1)';
            // Create clickable link if split is specified
            const clickAction = need.split ? 
                `href="/capture?split=${need.split}&class=${need.class}&count=${need.count}"` : 
                `href="/capture?class=${need.class}&count=${need.count}"`;
            
            adviceHtml += `<a ${clickAction} style="background: ${bg}; padding: 12px; border-radius: 8px; display: flex; align-items: center; gap: 10px; color: white; text-decoration: none; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">`;
            adviceHtml += `<span style="font-size: 24px;">${need.icon}</span>`;
            adviceHtml += `<span style="font-size: 16px; flex: 1;">${need.text}</span>`;
            adviceHtml += `<span style="background: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 4px; font-size: 12px;">Click to Capture →</span>`;
            if (need.priority === 'high') {
                adviceHtml += `<span style="background: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">HIGH</span>`;
            }
            adviceHtml += '</a>';
        });
        
        adviceHtml += '</div>';
        adviceHtml += `<div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2);">
            <a href="/capture" style="color: white; text-decoration: underline; font-weight: bold;">Or Start General Capture →</a>
        </div>`;
    }
    
    document.getElementById('capture-advisor').innerHTML = adviceHtml;
}

function buildCompensation(data) {
    // Use FIXED targets instead of relative percentages
    // This prevents the "never-ending target" problem
    const FIXED_TARGETS = {
        front: { train: 350, val: 75, test: 75 },
        back: { train: 350, val: 75, test: 75 }
    };
    
    let compHtml = '<table style="width: 100%; color: #333;">';
    compHtml += '<thead><tr style="background: #f5f5f5;"><th style="padding: 10px;">Split</th><th style="padding: 10px;">Current Front</th><th style="padding: 10px;">Target Front</th><th style="padding: 10px;">Current Back</th><th style="padding: 10px;">Target Back</th><th style="padding: 10px;">Actions</th></tr></thead><tbody>';
    
    const splits = [
        { name: 'train', currentF: data.by_split.train?.cnie_front?.total || 0, targetF: FIXED_TARGETS.front.train, currentB: data.by_split.train?.cnie_back?.total || 0, targetB: FIXED_TARGETS.back.train },
        { name: 'val', currentF: data.by_split.val?.cnie_front?.total || 0, targetF: FIXED_TARGETS.front.val, currentB: data.by_split.val?.cnie_back?.total || 0, targetB: FIXED_TARGETS.back.val },
        { name: 'test', currentF: data.by_split.test?.cnie_front?.total || 0, targetF: FIXED_TARGETS.front.test, currentB: data.by_split.test?.cnie_back?.total || 0, targetB: FIXED_TARGETS.back.test }
    ];
    
    // Store imbalances for action buttons
    const imbalances = [];
    
    splits.forEach(split => {
        const frontDiff = split.currentF - split.targetF;
        const backDiff = split.currentB - split.targetB;
        
        // Build action buttons for this split
        let actions = '';
        
        // Front actions
        if (frontDiff > 10) {
            // Excess - can move out
            const moveCount = Math.round(frontDiff);
            actions += `<button class="btn btn-sm btn-warning" onclick="showMoveModal('${split.name}', 'front', ${moveCount})" style="margin: 2px; padding: 4px 8px; font-size: 12px;">📤 Move ${moveCount} Front</button><br>`;
            imbalances.push({split: split.name, cls: 'front', count: moveCount, type: 'excess'});
        } else if (frontDiff < -10) {
            // Shortage - capture more
            const needCount = Math.round(-frontDiff);
            actions += `<a href="/capture?split=${split.name}&class=front&count=${needCount}" class="btn btn-sm btn-success" style="margin: 2px; padding: 4px 8px; font-size: 12px;">📸 Capture ${needCount} Front</a><br>`;
            imbalances.push({split: split.name, cls: 'front', count: needCount, type: 'shortage'});
        }
        
        // Back actions
        if (backDiff > 10) {
            // Excess - can move out
            const moveCount = Math.round(backDiff);
            actions += `<button class="btn btn-sm btn-warning" onclick="showMoveModal('${split.name}', 'back', ${moveCount})" style="margin: 2px; padding: 4px 8px; font-size: 12px;">📤 Move ${moveCount} Back</button><br>`;
            imbalances.push({split: split.name, cls: 'back', count: moveCount, type: 'excess'});
        } else if (backDiff < -10) {
            // Shortage - capture more
            const needCount = Math.round(-backDiff);
            actions += `<a href="/capture?split=${split.name}&class=back&count=${needCount}" class="btn btn-sm btn-success" style="margin: 2px; padding: 4px 8px; font-size: 12px;">📸 Capture ${needCount} Back</a><br>`;
            imbalances.push({split: split.name, cls: 'back', count: needCount, type: 'shortage'});
        }
        
        if (!actions) {
            actions = '<span style="color: #27ae60;">✓ Balanced</span>';
        }
        
        compHtml += `<tr style="border-bottom: 1px solid #eee;">`;
        compHtml += `<td style="padding: 10px; font-weight: 600;">${split.name}</td>`;
        compHtml += `<td style="padding: 10px; text-align: center; ${Math.abs(frontDiff) > 10 ? 'color: #e74c3c;' : 'color: #27ae60;'}">${split.currentF} ${frontDiff > 10 ? '(+' + Math.round(frontDiff) + ' excess)' : frontDiff < -10 ? '(' + Math.round(frontDiff) + ' shortage)' : '✓'}</td>`;
        compHtml += `<td style="padding: 10px; text-align: center; color: #666;">~${split.targetF}</td>`;
        compHtml += `<td style="padding: 10px; text-align: center; ${Math.abs(backDiff) > 10 ? 'color: #e74c3c;' : 'color: #27ae60;'}">${split.currentB} ${backDiff > 10 ? '(+' + Math.round(backDiff) + ' excess)' : backDiff < -10 ? '(' + Math.round(backDiff) + ' shortage)' : '✓'}</td>`;
        compHtml += `<td style="padding: 10px; text-align: center; color: #666;">~${split.targetB}</td>`;
        compHtml += `<td style="padding: 10px; text-align: center;">${actions}</td>`;
        compHtml += `</tr>`;
    });
    
    compHtml += '</tbody></table>';
    
    // Add modal for moving images
    compHtml += `
    <div id="moveModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; justify-content: center; align-items: center;">
        <div style="background: white; padding: 30px; border-radius: 12px; max-width: 400px; width: 90%;">
            <h3 style="margin-bottom: 20px;">📤 Move Images</h3>
            <p id="moveModalText" style="margin-bottom: 20px;"></p>
            <div style="margin-bottom: 20px;">
                <label style="display: block; margin-bottom: 8px; font-weight: 600;">Destination Split:</label>
                <select id="moveDestSplit" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px;">
                    <option value="train">train</option>
                    <option value="val">val</option>
                    <option value="test">test</option>
                </select>
            </div>
            <div style="margin-bottom: 20px;">
                <label style="display: block; margin-bottom: 8px; font-weight: 600;">Number of Images:</label>
                <input type="number" id="moveCount" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px;" min="1" max="100">
            </div>
            <div style="display: flex; gap: 10px;">
                <button onclick="executeMove()" style="flex: 1; padding: 12px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">Move Images</button>
                <button onclick="closeMoveModal()" style="flex: 1; padding: 12px; background: #95a5a6; color: white; border: none; border-radius: 6px; cursor: pointer;">Cancel</button>
            </div>
            <div id="moveStatus" style="margin-top: 15px; padding: 10px; border-radius: 6px; display: none;"></div>
        </div>
    </div>
    `;
    
    compHtml += '<div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px;">';
    compHtml += '<p style="margin-bottom: 10px;"><strong>Auto-Balance Recommendations:</strong></p>';
    
    // Generate smart recommendations
    const excesses = imbalances.filter(i => i.type === 'excess');
    const shortages = imbalances.filter(i => i.type === 'shortage');
    
    if (excesses.length === 0 && shortages.length === 0) {
        compHtml += '<p style="color: #27ae60;">✓ All splits are well balanced!</p>';
    } else {
        compHtml += '<div style="display: grid; gap: 10px;">';
        
        // Show move recommendations
        excesses.forEach(excess => {
            // Find matching shortage
            const matchShortage = shortages.find(s => s.cls === excess.cls && s.split !== excess.split);
            if (matchShortage) {
                const moveCount = Math.min(excess.count, matchShortage.count);
                compHtml += `<div style="padding: 12px; background: #fff3cd; border-radius: 6px; border-left: 4px solid #f39c12;">`;
                compHtml += `📤 Move <strong>${moveCount} ${excess.cls}</strong> images from <strong>${excess.split}</strong> to <strong>${matchShortage.split}</strong> `;
                compHtml += `<button onclick="quickMove('${excess.split}', '${matchShortage.split}', '${excess.cls}', ${moveCount})" class="btn btn-sm btn-warning" style="margin-left: 10px; padding: 4px 12px;">Move Now</button>`;
                compHtml += `</div>`;
            }
        });
        
        // Show remaining shortages that need capture
        shortages.forEach(shortage => {
            const hasMatchingExcess = excesses.some(e => e.cls === shortage.cls);
            if (!hasMatchingExcess) {
                compHtml += `<div style="padding: 12px; background: #d4edda; border-radius: 6px; border-left: 4px solid #27ae60;">`;
                compHtml += `📸 Need to capture <strong>${shortage.count} ${shortage.cls}</strong> images for <strong>${shortage.split}</strong> `;
                compHtml += `<a href="/capture?split=${shortage.split}&class=${shortage.cls}&count=${shortage.count}" class="btn btn-sm btn-success" style="margin-left: 10px; padding: 4px 12px; text-decoration: none; color: white;">Start Capture</a>`;
                compHtml += `</div>`;
            }
        });
        
        compHtml += '</div>';
    }
    
    compHtml += '</div>';
    
    document.getElementById('compensation-plan').innerHTML = compHtml;
}

// Move modal functions
let moveSourceSplit = '';
let moveClass = '';
let moveMaxCount = 0;

function showMoveModal(sourceSplit, cls, maxCount) {
    moveSourceSplit = sourceSplit;
    moveClass = cls;
    moveMaxCount = maxCount;
    
    document.getElementById('moveModalText').innerHTML = 
        `Move images from <strong>${sourceSplit}/${cls}</strong> to another split.<br>` +
        `Available: <strong>${maxCount}</strong> images`;
    document.getElementById('moveCount').value = Math.min(maxCount, 10);
    document.getElementById('moveCount').max = maxCount;
    
    // Disable source split in dropdown
    const destSelect = document.getElementById('moveDestSplit');
    for (let opt of destSelect.options) {
        opt.disabled = (opt.value === sourceSplit);
    }
    // Select first non-disabled option
    for (let opt of destSelect.options) {
        if (!opt.disabled) {
            destSelect.value = opt.value;
            break;
        }
    }
    
    document.getElementById('moveModal').style.display = 'flex';
    document.getElementById('moveStatus').style.display = 'none';
}

function closeMoveModal() {
    document.getElementById('moveModal').style.display = 'none';
}

async function executeMove() {
    const destSplit = document.getElementById('moveDestSplit').value;
    const count = parseInt(document.getElementById('moveCount').value);
    
    if (!destSplit || destSplit === moveSourceSplit) {
        showMoveStatus('Please select a different destination split', 'error');
        return;
    }
    
    if (!count || count < 1 || count > moveMaxCount) {
        showMoveStatus(`Please enter a number between 1 and ${moveMaxCount}`, 'error');
        return;
    }
    
    await performMove(moveSourceSplit, destSplit, moveClass, count);
}

async function quickMove(source, dest, cls, count) {
    if (confirm(`Move ${count} ${cls} images from ${source} to ${dest}?`)) {
        await performMove(source, dest, cls, count);
    }
}

async function performMove(sourceSplit, destSplit, cls, count) {
    const statusEl = document.getElementById('moveStatus') || document.getElementById('compensation-plan');
    showMoveStatus('Moving images...', 'info');
    
    try {
        const response = await fetch('/api/move_between_splits', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                source_split: sourceSplit,
                dest_split: destSplit,
                class: cls,
                count: count,
                dataset: 'combined'
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showMoveStatus(`✅ Successfully moved ${result.moved} images!`, 'success');
            setTimeout(() => {
                closeMoveModal();
                loadDetailedStats(); // Refresh stats
            }, 1500);
        } else {
            showMoveStatus('❌ Error: ' + result.error, 'error');
        }
    } catch (e) {
        showMoveStatus('❌ Network error: ' + e.message, 'error');
    }
}

function showMoveStatus(message, type) {
    const statusEl = document.getElementById('moveStatus');
    if (!statusEl) return;
    
    statusEl.textContent = message;
    statusEl.style.display = 'block';
    statusEl.style.background = type === 'error' ? '#f8d7da' : type === 'success' ? '#d4edda' : '#e2e3e5';
    statusEl.style.color = type === 'error' ? '#721c24' : type === 'success' ? '#155724' : '#383d41';
}

// Load on page load
loadDetailedStats();
</script>'''
    
    return render_template_string(render_page('Statistics', 'stats', content, scripts))


@app.route('/cleaner')
def cleaner_page():
    content = '<div class="card"><h3>Dataset Cleaner</h3><p>Automatic label verification.</p></div>'
    return render_template_string(render_page('Dataset Cleaner', 'cleaner', content))


@app.route('/train')
def train_page():
    content = '''
<div class="card">
    <div class="card-header">
        <div class="card-title">🧠 Model Training (Colab GPU)</div>
    </div>
    
    <div id="training-status" style="padding: 20px; background: #f8f9fa; border-radius: 8px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div id="status-icon" style="font-size: 48px;">⏳</div>
            <div>
                <div style="font-size: 18px; font-weight: 600;" id="status-text">Ready to Train</div>
                <div style="color: #666; margin-top: 5px;" id="status-detail">Configure parameters and start training on Colab GPU</div>
            </div>
        </div>
    </div>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <!-- Training Configuration -->
        <div>
            <h4 style="margin-bottom: 15px;">⚙️ Configuration</h4>
            
            <div style="margin-bottom: 15px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #f39c12;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">🔗 Colab SSH Hostname:</label>
                <input type="text" id="colab-host" placeholder="xxx.trycloudflare.com" class="form-control" style="width: 100%; margin-bottom: 8px;">
                <small style="color: #856404;">
                    From your Colab notebook: <code>!cloudflared tunnel --url ssh://localhost:22</code>
                </small>
            </div>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Dataset:</label>
                <select id="train-dataset" class="form-select" style="width: 100%;">
                    <option value="combined" selected>📊 Combined (All Datasets)</option>
                    <option value="v8_stage2_clean">v8_stage2_clean (Original)</option>
                    <option value="v10_manual_capture">v10_manual_capture (New Captures)</option>
                </select>
            </div>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Model Architecture:</label>
                <select id="train-model" class="form-select" style="width: 100%;">
                    <option value="efficientnet_b0" selected>EfficientNet-B0 (Recommended)</option>
                    <option value="efficientnet_b3">EfficientNet-B3 (More Accurate)</option>
                    <option value="mobilenet_v3">MobileNet-V3 (Faster)</option>
                </select>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: 600;">Epochs:</label>
                    <input type="number" id="train-epochs" value="50" min="10" max="200" class="form-control" style="width: 100%;">
                </div>
                
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 5px; font-weight: 600;">Batch Size:</label>
                    <input type="number" id="train-batch" value="32" min="8" max="128" class="form-control" style="width: 100%;">
                </div>
            </div>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Learning Rate:</label>
                <select id="train-lr" class="form-select" style="width: 100%;">
                    <option value="0.001">0.001 (Fast)</option>
                    <option value="0.0001" selected>0.0001 (Balanced)</option>
                    <option value="0.00001">0.00001 (Stable)</option>
                </select>
            </div>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Classes:</label>
                <div style="display: flex; gap: 15px; padding: 10px; background: #f0f0f0; border-radius: 6px;">
                    <label style="display: flex; align-items: center; gap: 5px;">
                        <input type="checkbox" id="class-front" checked> Front
                    </label>
                    <label style="display: flex; align-items: center; gap: 5px;">
                        <input type="checkbox" id="class-back" checked> Back
                    </label>
                    <label style="display: flex; align-items: center; gap: 5px;">
                        <input type="checkbox" id="class-nocard"> No-Card
                    </label>
                </div>
            </div>
            
            <button onclick="startColabTraining()" id="btn-start-train" class="btn btn-primary" style="width: 100%; padding: 15px; font-size: 16px; margin-bottom: 10px;">
                🚀 Deploy & Train on Colab GPU
            </button>
            
            <button onclick="cleanAndRedeploy()" id="btn-clean-redeploy" class="btn btn-warning" style="width: 100%; padding: 12px; font-size: 14px;">
                🧹 Clean & Redeploy (if previous failed)
            </button>
            
            <div style="margin-top: 15px; padding: 10px; background: #fff3cd; border-radius: 6px; font-size: 13px; color: #856404;">
                <strong>Tip:</strong> Use "Clean & Redeploy" if training failed or you want to restart fresh. This will remove old files on Colab before deploying.
            </div>
            
            <!-- Download Model Section -->
            <div style="margin-top: 20px; padding: 15px; background: #d4edda; border-radius: 8px; border: 1px solid #c3e6cb;">
                <h5 style="margin-bottom: 10px; color: #155724;">📥 Download Trained Model</h5>
                <p style="font-size: 12px; color: #155724; margin-bottom: 10px;">
                    If training completed but download failed, or you want to download a previously trained model from Colab.
                </p>
                <button onclick="manualDownloadModel()" class="btn btn-success" style="width: 100%; padding: 10px;">
                    📥 Download Model from Colab
                </button>
                <div id="manual-download-status" style="margin-top: 10px; font-size: 12px;"></div>
            </div>
        </div>
        
        <!-- Training Progress -->
        <div>
            <h4 style="margin-bottom: 15px;">📊 Progress</h4>
            
            <div id="progress-panel" style="background: #f8f9fa; padding: 15px; border-radius: 8px; min-height: 300px;">
                <div id="progress-placeholder" style="text-align: center; color: #999; padding-top: 100px;">
                    <div style="font-size: 48px; margin-bottom: 10px;">📈</div>
                    <div>Training metrics will appear here</div>
                </div>
                
                <div id="progress-content" style="display: none;">
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>Epoch Progress</span>
                            <span id="epoch-text">1 / 50</span>
                        </div>
                        <div style="background: #ddd; height: 20px; border-radius: 10px; overflow: hidden;">
                            <div id="epoch-bar" style="background: #3498db; height: 100%; width: 0%; transition: width 0.3s;"></div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 12px; color: #666;">Train Loss</div>
                            <div id="train-loss" style="font-size: 24px; font-weight: 600; color: #e74c3c;">--</div>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 12px; color: #666;">Val Accuracy</div>
                            <div id="val-acc" style="font-size: 24px; font-weight: 600; color: #27ae60;">--</div>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 12px; color: #666;">Train Acc</div>
                            <div id="train-acc" style="font-size: 24px; font-weight: 600; color: #3498db;">--</div>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 12px; color: #666;">Best Val Acc</div>
                            <div id="best-val-acc" style="font-size: 24px; font-weight: 600; color: #9b59b6;">--</div>
                        </div>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #666; margin-bottom: 10px;">Training Log:</div>
                        <div id="train-log" style="font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; background: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 4px;">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Previous Models -->
<div class="card">
    <div class="card-header">
        <div class="card-title">📦 Trained Models</div>
    </div>
    <div id="models-list">
        <p class="text-center" style="color: #999;">Loading models...</p>
    </div>
</div>
'''
    
    scripts = '''<script>
let trainingActive = false;
let trainingLogs = [];

async function cleanAndRedeploy() {
    // Get Colab hostname
    const colabHost = document.getElementById('colab-host').value.trim();
    if (!colabHost) {
        alert('Please enter the Colab SSH hostname!\\n\\nGet it from your Colab notebook by running:\\n!cloudflared tunnel --url ssh://localhost:22');
        return;
    }
    
    if (!confirm('This will clean up old files on Colab and redeploy.\\nContinue?')) {
        return;
    }
    
    // Reset UI
    document.getElementById('btn-clean-redeploy').disabled = true;
    document.getElementById('btn-clean-redeploy').textContent = '🧹 Cleaning...';
    document.getElementById('status-icon').textContent = '🧹';
    document.getElementById('status-text').textContent = 'Cleaning Colab...';
    document.getElementById('status-detail').textContent = 'Removing old files...';
    
    try {
        const response = await fetch('/api/train/colab/clean', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ host: colabHost })
        });
        
        const result = await response.json();
        
        // Reset button state
        document.getElementById('btn-clean-redeploy').disabled = false;
        document.getElementById('btn-clean-redeploy').textContent = '🧹 Clean & Redeploy (if previous failed)';
        
        if (result.success) {
            document.getElementById('status-detail').textContent = 'Cleanup complete! Starting fresh deploy...';
            // Now start normal training
            setTimeout(() => startColabTraining(), 500);
        } else {
            alert('Cleanup failed: ' + result.error + '\\n\\nTrying deploy anyway...');
            startColabTraining();
        }
    } catch (e) {
        // Reset button state
        document.getElementById('btn-clean-redeploy').disabled = false;
        document.getElementById('btn-clean-redeploy').textContent = '🧹 Clean & Redeploy (if previous failed)';
        
        alert('Cleanup error: ' + e.message + '\\n\\nTrying deploy anyway...');
        startColabTraining();
    }
}

async function startColabTraining() {
    if (trainingActive) {
        alert('Training is already in progress!');
        return;
    }
    
    // Get Colab hostname
    const colabHost = document.getElementById('colab-host').value.trim();
    if (!colabHost) {
        alert('Please enter the Colab SSH hostname!\\\\n\\\\nGet it from your Colab notebook by running:\\\\n!cloudflared tunnel --url ssh://localhost:22');
        return;
    }
    
    // Get configuration
    const config = {
        colab_host: colabHost,
        dataset: document.getElementById('train-dataset').value,
        model: document.getElementById('train-model').value,
        epochs: parseInt(document.getElementById('train-epochs').value),
        batch_size: parseInt(document.getElementById('train-batch').value),
        learning_rate: parseFloat(document.getElementById('train-lr').value),
        classes: []
    };
    
    if (document.getElementById('class-front').checked) config.classes.push('front');
    if (document.getElementById('class-back').checked) config.classes.push('back');
    if (document.getElementById('class-nocard').checked) config.classes.push('no_card');
    
    if (config.classes.length < 2) {
        alert('Please select at least 2 classes!');
        return;
    }
    
    // Update UI
    trainingActive = true;
    document.getElementById('status-icon').textContent = '🔥';
    document.getElementById('status-text').textContent = 'Deploying to Colab GPU...';
    document.getElementById('status-detail').textContent = 'Preparing dataset and uploading...';
    document.getElementById('btn-start-train').disabled = true;
    document.getElementById('btn-start-train').textContent = '⏳ Deploying...';
    
    document.getElementById('progress-placeholder').style.display = 'none';
    document.getElementById('progress-content').style.display = 'block';
    trainingLogs = ['Connecting to Colab GPU...'];
    updateLogDisplay();
    
    try {
        const response = await fetch('/api/train/colab/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('status-text').textContent = 'Training on Colab GPU...';
            document.getElementById('status-detail').textContent = `Connected to ${colabHost}`;
            trainingLogs.push('✅ Dataset uploaded successfully');
            trainingLogs.push('✅ Training started on GPU');
            updateLogDisplay();
            pollColabTrainingStatus();
        } else {
            showTrainingError(result.error);
        }
    } catch (e) {
        showTrainingError('Failed to deploy: ' + e.message);
    }
}

function updateLogDisplay() {
    const logDiv = document.getElementById('train-log');
    if (!logDiv) return;
    
    logDiv.innerHTML = trainingLogs.map(log => `<div>${log}</div>`).join('');
    logDiv.scrollTop = logDiv.scrollHeight;
}

async function pollColabTrainingStatus() {
    if (!trainingActive) return;
    
    try {
        const colabHost = document.getElementById('colab-host').value.trim();
        const response = await fetch(`/api/train/colab/status?host=${encodeURIComponent(colabHost)}`);
        const status = await response.json();
        
        if (status.running) {
            // Update progress
            if (status.epoch > 0) {
                document.getElementById('epoch-text').textContent = `${status.epoch} / ${status.total_epochs}`;
                document.getElementById('epoch-bar').style.width = (status.epoch / status.total_epochs * 100) + '%';
            }
            
            if (status.train_loss !== null) {
                document.getElementById('train-loss').textContent = status.train_loss.toFixed(4);
            }
            if (status.train_acc !== null) {
                document.getElementById('train-acc').textContent = status.train_acc.toFixed(1) + '%';
            }
            if (status.val_acc !== null) {
                document.getElementById('val-acc').textContent = status.val_acc.toFixed(1) + '%';
            }
            if (status.best_val_acc !== null) {
                document.getElementById('best-val-acc').textContent = status.best_val_acc.toFixed(1) + '%';
            }
            
            // Update logs
            if (status.logs && status.logs.length > trainingLogs.length) {
                for (let i = trainingLogs.length; i < status.logs.length; i++) {
                    trainingLogs.push(status.logs[i]);
                }
                updateLogDisplay();
            }
            
            document.getElementById('status-detail').textContent = status.message || 'Training on GPU...';
            
            setTimeout(pollColabTrainingStatus, 2000);
        } else {
            // Training complete
            if (status.error) {
                showTrainingError(status.error);
            } else {
                trainingComplete(status);
                // Auto-download model
                downloadColabModel();
            }
        }
    } catch (e) {
        console.error('Error polling status:', e);
        setTimeout(pollColabTrainingStatus, 5000);
    }
}

async function downloadColabModel() {
    const colabHost = document.getElementById('colab-host').value.trim();
    
    document.getElementById('status-text').textContent = 'Downloading Model...';
    document.getElementById('status-detail').textContent = 'Transferring from Colab...';
    
    try {
        const response = await fetch('/api/train/colab/download', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ host: colabHost })
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('status-icon').textContent = '✅';
            document.getElementById('status-text').textContent = 'Training Complete!';
            document.getElementById('status-detail').textContent = `Model saved: ${result.model_name}`;
            loadModelsList();
        } else {
            showDownloadFailed(result.error, colabHost);
        }
    } catch (e) {
        showDownloadFailed(e.message, colabHost);
    }
}

let downloadPollInterval = null;

async function manualDownloadModel() {
    const colabHost = document.getElementById('colab-host').value.trim();
    const statusDiv = document.getElementById('manual-download-status');
    
    if (!colabHost) {
        statusDiv.innerHTML = '<span style="color: #e74c3c;">❌ Please enter the Colab SSH hostname first!</span>';
        return;
    }
    
    // Clear any existing poll
    if (downloadPollInterval) {
        clearInterval(downloadPollInterval);
    }
    
    // Show initial progress UI
    statusDiv.innerHTML = `
        <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-weight: 600;">📥 Download Progress</span>
                <span id="download-percent">0%</span>
            </div>
            <div style="background: #ddd; height: 20px; border-radius: 10px; overflow: hidden;">
                <div id="download-progress-bar" style="background: #27ae60; height: 100%; width: 0%; transition: width 0.5s;"></div>
            </div>
            <div id="download-message" style="margin-top: 8px; font-size: 12px; color: #666;">Initializing...</div>
            <div id="download-log" style="margin-top: 10px; font-family: monospace; font-size: 11px; max-height: 100px; overflow-y: auto; background: #1e1e1e; color: #d4d4d4; padding: 8px; border-radius: 4px;"></div>
        </div>
    `;
    
    // Start download
    try {
        const response = await fetch('/api/train/colab/download', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ host: colabHost })
        });
        
        const result = await response.json();
        
        if (!result.success) {
            document.getElementById('download-message').innerHTML = `<span style="color: #e74c3c;">❌ ${result.error}</span>`;
            return;
        }
        
        // Start polling for progress
        pollDownloadProgress();
        downloadPollInterval = setInterval(pollDownloadProgress, 1000);
        
    } catch (e) {
        document.getElementById('download-message').innerHTML = `<span style="color: #e74c3c;">❌ Error: ${e.message}</span>`;
    }
}

async function pollDownloadProgress() {
    try {
        const response = await fetch('/api/train/colab/download/status');
        const status = await response.json();
        
        // Update progress bar
        document.getElementById('download-progress-bar').style.width = status.progress + '%';
        document.getElementById('download-percent').textContent = status.progress + '%';
        document.getElementById('download-message').textContent = status.message;
        
        // Add to log
        const logDiv = document.getElementById('download-log');
        if (status.message && !logDiv.innerHTML.includes(status.message)) {
            const line = document.createElement('div');
            line.textContent = `[${new Date().toLocaleTimeString()}] ${status.message}`;
            logDiv.appendChild(line);
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        // Check if completed
        if (status.completed) {
            clearInterval(downloadPollInterval);
            downloadPollInterval = null;
            
            if (status.error) {
                document.getElementById('download-message').innerHTML = `
                    <span style="color: #e74c3c;">❌ ${status.error}</span>
                    <div style="margin-top: 10px;">
                        <strong>Manual download:</strong><br>
                        <code style="font-size: 10px; word-break: break-all;">
                            scp -i ~/.ssh/id_colab root@${document.getElementById('colab-host').value}:/content/cnie_classifier_best.pth ~/retin-verify/training_data/models/
                        </code>
                    </div>
                `;
            } else {
                document.getElementById('download-message').innerHTML = `
                    <span style="color: #27ae60;">✅ Download complete!</span><br>
                    <strong>Model:</strong> ${status.model_name}<br>
                    <strong>Path:</strong> ${status.model_path}
                `;
                loadModelsList();
            }
        }
    } catch (e) {
        console.error('Poll error:', e);
    }
}

function showDownloadFailed(error, colabHost) {
    document.getElementById('status-icon').textContent = '⚠️';
    document.getElementById('status-text').textContent = 'Download Failed';
    
    // Add manual download instructions
    const manualDownloadHtml = `
        <div style="margin-top: 15px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #f39c12;">
            <p><strong>⚠️ Automatic download failed:</strong> ${error}</p>
            <p style="margin-top: 10px;"><strong>Manual download:</strong></p>
            <p>Run this command in your terminal:</p>
            <code style="display: block; background: #f5f5f5; padding: 10px; border-radius: 4px; margin: 10px 0; word-break: break-all;">
                scp -i ~/.ssh/id_colab root@${colabHost}:/content/cnie_classifier_best.pth ~/retin-verify/training_data/models/
            </code>
            <p style="margin-top: 10px;">Or click the <strong>"Download Model from Colab"</strong> button in the configuration panel.</p>
            <button onclick="loadModelsList()" class="btn btn-sm btn-primary" style="margin-top: 10px;">
                🔄 Refresh Models List
            </button>
        </div>
    `;
    
    document.getElementById('status-detail').innerHTML = 'See manual download instructions below:';
    document.getElementById('status-detail').insertAdjacentHTML('afterend', manualDownloadHtml);
}

async function pollTrainingStatus() {
    if (!trainingActive) return;
    
    try {
        const response = await fetch('/api/train/status');
        const status = await response.json();
        
        if (status.running) {
            // Update progress
            document.getElementById('epoch-text').textContent = `${status.epoch} / ${status.total_epochs}`;
            document.getElementById('epoch-bar').style.width = (status.epoch / status.total_epochs * 100) + '%';
            
            if (status.train_loss !== null) {
                document.getElementById('train-loss').textContent = status.train_loss.toFixed(4);
            }
            if (status.train_acc !== null) {
                document.getElementById('train-acc').textContent = status.train_acc.toFixed(1) + '%';
            }
            if (status.val_acc !== null) {
                document.getElementById('val-acc').textContent = status.val_acc.toFixed(1) + '%';
            }
            if (status.best_val_acc !== null) {
                document.getElementById('best-val-acc').textContent = status.best_val_acc.toFixed(1) + '%';
            }
            
            // Update logs
            if (status.logs && status.logs.length > trainingLogs.length) {
                const logDiv = document.getElementById('train-log');
                for (let i = trainingLogs.length; i < status.logs.length; i++) {
                    const line = document.createElement('div');
                    line.textContent = status.logs[i];
                    logDiv.appendChild(line);
                }
                trainingLogs = status.logs;
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            document.getElementById('status-detail').textContent = status.message || 'Training...';
            
            setTimeout(pollTrainingStatus, 1000);
        } else {
            // Training complete
            trainingComplete(status);
        }
    } catch (e) {
        console.error('Error polling status:', e);
        setTimeout(pollTrainingStatus, 2000);
    }
}

function trainingComplete(status) {
    trainingActive = false;
    document.getElementById('btn-start-train').disabled = false;
    document.getElementById('btn-start-train').textContent = '🚀 Start Training';
    
    if (status.error) {
        showTrainingError(status.error);
    } else {
        document.getElementById('status-icon').textContent = '✅';
        document.getElementById('status-text').textContent = 'Training Complete!';
        document.getElementById('status-detail').textContent = `Best validation accuracy: ${status.best_val_acc?.toFixed(1) || '--'}%`;
        loadModelsList();
    }
}

function showTrainingError(error) {
    trainingActive = false;
    document.getElementById('status-icon').textContent = '❌';
    document.getElementById('status-text').textContent = 'Training Failed';
    document.getElementById('status-detail').textContent = error;
    document.getElementById('btn-start-train').disabled = false;
    document.getElementById('btn-start-train').textContent = '🚀 Retry Training';
}

async function loadModelsList() {
    try {
        const response = await fetch('/api/train/models');
        const models = await response.json();
        
        let html = '<table style="width: 100%; border-collapse: collapse;">';
        html += '<thead><tr style="background: #f5f5f5;"><th style="padding: 12px;">Model</th><th style="padding: 12px;">Accuracy</th><th style="padding: 12px;">Date</th><th style="padding: 12px;">Actions</th></tr></thead><tbody>';
        
        models.forEach(model => {
            html += `<tr style="border-bottom: 1px solid #eee;">`;
            html += `<td style="padding: 12px;">${model.name}</td>`;
            html += `<td style="padding: 12px; text-align: center;">${model.accuracy ? model.accuracy.toFixed(1) + '%' : 'N/A'}</td>`;
            html += `<td style="padding: 12px; text-align: center;">${model.date}</td>`;
            html += `<td style="padding: 12px; text-align: center;">`;
            html += `<button onclick="deployModel('${model.name}')" class="btn btn-sm btn-success" style="margin-right: 5px;">Deploy</button>`;
            html += `<button onclick="deleteModel('${model.name}')" class="btn btn-sm btn-danger">Delete</button>`;
            html += `</td></tr>`;
        });
        
        html += '</tbody></table>';
        
        if (models.length === 0) {
            html = '<p style="text-align: center; color: #999; padding: 20px;">No trained models yet</p>';
        }
        
        document.getElementById('models-list').innerHTML = html;
    } catch (e) {
        document.getElementById('models-list').innerHTML = '<p style="text-align: center; color: #e74c3c;">Failed to load models</p>';
    }
}

async function deployModel(name) {
    if (!confirm(`Deploy model "${name}" as active model?`)) return;
    
    try {
        const response = await fetch('/api/train/deploy', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_name: name})
        });
        const result = await response.json();
        
        if (result.success) {
            alert('Model deployed successfully!');
            loadModelsList();
        } else {
            alert('Failed to deploy: ' + result.error);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function deleteModel(name) {
    if (!confirm(`Delete model "${name}"?`)) return;
    
    try {
        const response = await fetch('/api/train/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_name: name})
        });
        const result = await response.json();
        
        if (result.success) {
            loadModelsList();
        } else {
            alert('Failed to delete: ' + result.error);
        }
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// Load models on page load
loadModelsList();
</script>'''
    
    return render_template_string(render_page('Train Model', 'train', content, scripts))


@app.route('/evaluate')
def evaluate_page():
    content = '''
<div class="card">
    <div class="card-header">
        <div class="card-title">🎯 Real-Time Model Evaluation</div>
    </div>
    
    <div style="display: grid; grid-template-columns: 300px 1fr; gap: 20px;">
        <!-- Configuration Panel -->
        <div>
            <h4 style="margin-bottom: 15px;">⚙️ Configuration</h4>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Select Model:</label>
                <select id="eval-model" class="form-select" style="width: 100%;" onchange="loadModel()">
                    <option value="">Loading models...</option>
                </select>
                <div id="model-info" style="margin-top: 8px; font-size: 12px; color: #666;"></div>
            </div>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: 600;">Confidence Threshold:</label>
                <input type="range" id="confidence-threshold" min="50" max="99" value="80" 
                       style="width: 100%;" oninput="updateThresholdDisplay()">
                <div style="text-align: center; font-size: 12px; color: #666;">
                    <span id="threshold-value">80</span>%
                </div>
            </div>
            
            <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin-bottom: 15px;">
                <div style="font-size: 12px; color: #666; margin-bottom: 8px;">📊 Current Stats:</div>
                <div id="eval-stats" style="font-size: 14px;">
                    <div>Model: <span id="stat-model-name">-</span></div>
                    <div>Predictions: <span id="stat-predictions">0</span></div>
                    <div>Avg Confidence: <span id="stat-avg-conf">-</span></div>
                </div>
            </div>
            
            <button onclick="resetStats()" class="btn btn-sm" style="width: 100%;">
                🔄 Reset Stats
            </button>
            
            <div style="margin-top: 20px; padding: 10px; background: #e3f2fd; border-radius: 6px; font-size: 12px;">
                <strong>💡 How to use:</strong>
                <ol style="margin: 8px 0; padding-left: 16px;">
                    <li>Select a trained model</li>
                    <li>Allow camera access</li>
                    <li>Show your CNIE card</li>
                    <li>See real-time predictions!</li>
                </ol>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background: #fff3e0; border-radius: 6px; font-size: 11px;">
                <strong>🚫 No-Card Detection:</strong>
                <p style="margin: 5px 0;">When confidence is below <strong>70%</strong>, the system shows <strong>"NOT CNIE"</strong> (gray bar). This helps detect when no card is present.</p>
            </div>
        </div>
        
        <!-- Camera & Results -->
        <div>
            <div class="video-wrapper" style="position: relative; max-width: 640px; margin: 0 auto;">
                <video id="eval-video" autoplay playsinline style="width: 100%; border-radius: 8px;"></video>
                <canvas id="eval-canvas" style="display: none;"></canvas>
                
                <!-- Prediction Overlay -->
                <div id="prediction-overlay" style="position: absolute; bottom: 20px; left: 20px; right: 20px; 
                            background: rgba(0,0,0,0.8); color: white; padding: 15px; border-radius: 12px;
                            display: none;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 24px; font-weight: bold;" id="pred-label">-</div>
                            <div style="font-size: 14px; opacity: 0.8;" id="pred-confidence">Confidence: -</div>
                        </div>
                        <div style="font-size: 48px;" id="pred-icon">❓</div>
                    </div>
                    <div style="margin-top: 10px; background: rgba(255,255,255,0.2); height: 8px; border-radius: 4px; overflow: hidden;">
                        <div id="conf-bar" style="background: #4CAF50; height: 100%; width: 0%; transition: width 0.3s;"></div>
                    </div>
                </div>
                
                <!-- Status Badge -->
                <div id="eval-status" style="position: absolute; top: 20px; left: 20px; 
                            background: rgba(0,0,0,0.7); color: white; padding: 8px 16px; 
                            border-radius: 20px; font-size: 14px;">
                    ⏳ Select a model to start
                </div>
            </div>
            
            <!-- Recent Predictions -->
            <div style="margin-top: 20px;">
                <h4 style="margin-bottom: 10px;">📜 Recent Predictions</h4>
                <div id="recent-predictions" style="display: flex; gap: 10px; overflow-x: auto; padding: 10px; 
                            background: #f8f9fa; border-radius: 8px; min-height: 80px;">
                    <div style="color: #999; font-size: 12px; text-align: center; width: 100%;">
                        No predictions yet. Start camera to see results.
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
'''
    
    scripts = '''<script>
let evalVideo = document.getElementById('eval-video');
let evalCanvas = document.getElementById('eval-canvas');
let evalCtx = evalCanvas.getContext('2d');
let currentModel = null;
let isEvaluating = false;
let predictionCount = 0;
let confidenceSum = 0;
let recentPredictions = [];
let evalInterval = null;

// Load available models
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/evaluate/models');
        const models = await response.json();
        
        const select = document.getElementById('eval-model');
        select.innerHTML = '<option value="">-- Select Model --</option>';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = model.name + (model.accuracy ? ` (${model.accuracy.toFixed(1)}%)` : '');
            select.appendChild(option);
        });
    } catch (e) {
        console.error('Failed to load models:', e);
        document.getElementById('eval-model').innerHTML = '<option value="">Error loading models</option>';
    }
}

// Update threshold display
function updateThresholdDisplay() {
    document.getElementById('threshold-value').textContent = document.getElementById('confidence-threshold').value;
}

// Load selected model
async function loadModel() {
    const modelPath = document.getElementById('eval-model').value;
    if (!modelPath) {
        document.getElementById('model-info').textContent = '';
        stopEvaluation();
        return;
    }
    
    document.getElementById('eval-status').textContent = '🔄 Loading model...';
    
    try {
        const response = await fetch('/api/evaluate/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_path: modelPath})
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentModel = result.model_info;
            document.getElementById('model-info').innerHTML = `
                <div>✅ Model loaded</div>
                <div>Classes: ${result.model_info.classes.join(', ')}</div>
            `;
            document.getElementById('stat-model-name').textContent = result.model_info.name;
            document.getElementById('eval-status').textContent = '📷 Ready - Starting camera...';
            startCamera();
        } else {
            document.getElementById('eval-status').textContent = '❌ Failed to load model';
            alert('Failed to load model: ' + result.error);
        }
    } catch (e) {
        document.getElementById('eval-status').textContent = '❌ Error loading model';
        console.error(e);
    }
}

// Start camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        evalVideo.srcObject = stream;
        
        evalVideo.onloadedmetadata = () => {
            evalVideo.play();
            startEvaluation();
        };
    } catch (err) {
        document.getElementById('eval-status').textContent = '❌ Camera access denied';
        alert('Camera access required for evaluation. Please allow camera access.');
    }
}

// Start real-time evaluation
function startEvaluation() {
    if (evalInterval) clearInterval(evalInterval);
    
    isEvaluating = true;
    document.getElementById('eval-status').textContent = '🟢 Running - Show your card!';
    document.getElementById('prediction-overlay').style.display = 'block';
    
    // Run prediction every 500ms
    evalInterval = setInterval(captureAndPredict, 500);
}

// Stop evaluation
function stopEvaluation() {
    isEvaluating = false;
    if (evalInterval) {
        clearInterval(evalInterval);
        evalInterval = null;
    }
    document.getElementById('prediction-overlay').style.display = 'none';
    document.getElementById('eval-status').textContent = '⏳ Select a model to start';
}

// Capture frame and predict
async function captureAndPredict() {
    if (!isEvaluating || !evalVideo.videoWidth) return;
    
    // Draw frame to canvas
    evalCanvas.width = evalVideo.videoWidth;
    evalCanvas.height = evalVideo.videoHeight;
    evalCtx.drawImage(evalVideo, 0, 0, evalCanvas.width, evalCanvas.height);
    
    // Convert to base64
    const base64 = evalCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    
    try {
        const response = await fetch('/api/evaluate/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image: base64})
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPrediction(result);
        }
    } catch (e) {
        console.error('Prediction error:', e);
    }
}

// Display prediction result
function displayPrediction(result) {
    const threshold = parseInt(document.getElementById('confidence-threshold').value);
    const confidence = result.confidence * 100;
    
    // Define minimum threshold for valid CNIE detection
    // Below this, we treat it as "no card" / uncertain
    const MIN_VALID_CONFIDENCE = 70;  // 70% minimum to be considered a valid card
    
    let displayLabel = result.label;
    let displayIcon = '📇';
    let isUncertain = false;
    
    // Check if confidence is too low for a valid prediction
    if (confidence < MIN_VALID_CONFIDENCE) {
        displayLabel = 'NOT CNIE';
        displayIcon = '❌';
        isUncertain = true;
    } else {
        // Valid prediction - use actual label
        const icons = {
            'front': '📇',
            'back': '📷',
            'no_card': '❌'
        };
        displayIcon = icons[result.label] || '❓';
    }
    
    // Update overlay
    document.getElementById('pred-label').textContent = displayLabel;
    document.getElementById('pred-confidence').textContent = `Confidence: ${confidence.toFixed(1)}%`;
    document.getElementById('pred-icon').textContent = displayIcon;
    document.getElementById('conf-bar').style.width = confidence + '%';
    
    // Change color based on confidence and validity
    const bar = document.getElementById('conf-bar');
    if (isUncertain) {
        bar.style.background = '#9e9e9e';  // Gray for no-card/uncertain
        document.getElementById('pred-label').style.color = '#ffeb3b';  // Yellow text warning
    } else if (confidence >= threshold) {
        bar.style.background = '#4CAF50';  // Green
        document.getElementById('pred-label').style.color = '#fff';
    } else if (confidence >= threshold - 20) {
        bar.style.background = '#FF9800';  // Orange
        document.getElementById('pred-label').style.color = '#fff';
    } else {
        bar.style.background = '#f44336';  // Red
        document.getElementById('pred-label').style.color = '#fff';
    }
    
    // Only update stats for confident, valid predictions (not uncertain ones)
    if (!isUncertain && confidence >= threshold) {
        predictionCount++;
        confidenceSum += confidence;
        document.getElementById('stat-predictions').textContent = predictionCount;
        document.getElementById('stat-avg-conf').textContent = (confidenceSum / predictionCount).toFixed(1) + '%';
        
        // Add to recent predictions
        addRecentPrediction({...result, label: displayLabel}, confidence);
    }
}

// Add to recent predictions
function addRecentPrediction(result, confidence) {
    const timestamp = new Date().toLocaleTimeString();
    recentPredictions.unshift({
        label: result.label,
        confidence: confidence,
        time: timestamp
    });
    
    // Keep only last 10
    if (recentPredictions.length > 10) recentPredictions.pop();
    
    // Update display
    const container = document.getElementById('recent-predictions');
    container.innerHTML = recentPredictions.map(p => {
        // Determine icon and color based on label
        let icon = '❓';
        let color = '#FF9800';
        
        if (p.label === 'front') {
            icon = '📇';
            color = '#4CAF50';
        } else if (p.label === 'back') {
            icon = '📷';
            color = '#4CAF50';
        } else if (p.label === 'NOT CNIE' || p.label === 'no_card') {
            icon = '❌';
            color = '#9e9e9e';  // Gray for no-card
        }
        
        return `
        <div style="flex-shrink: 0; background: white; padding: 8px 12px; border-radius: 6px; 
                    border-left: 4px solid ${color};
                    min-width: 100px; text-align: center;">
            <div style="font-size: 24px; margin-bottom: 4px;">${icon}</div>
            <div style="font-weight: bold; font-size: 12px;">${p.label.toUpperCase()}</div>
            <div style="font-size: 11px; color: #666;">${p.confidence.toFixed(0)}%</div>
            <div style="font-size: 10px; color: #999;">${p.time}</div>
        </div>
        `;
    }).join('');
}

// Reset stats
function resetStats() {
    predictionCount = 0;
    confidenceSum = 0;
    recentPredictions = [];
    document.getElementById('stat-predictions').textContent = '0';
    document.getElementById('stat-avg-conf').textContent = '-';
    document.getElementById('recent-predictions').innerHTML = 
        '<div style="color: #999; font-size: 12px; text-align: center; width: 100%;">No predictions yet. Start camera to see results.</div>';
}

// Load models on page load
loadAvailableModels();
</script>'''
    
    return render_template_string(render_page('Evaluate', 'evaluate', content, scripts))


# ============ EVALUATE API ============

eval_model = None
eval_model_info = {}

@app.route('/api/evaluate/models')
def api_evaluate_models():
    """List available models for evaluation."""
    models = []
    
    # Check models directory
    if MODELS_DIR.exists():
        for model_file in sorted(MODELS_DIR.glob('*.pth'), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                import torch
                checkpoint = torch.load(model_file, map_location='cpu')
                models.append({
                    'name': model_file.name,
                    'path': str(model_file),
                    'accuracy': checkpoint.get('val_acc', 0),
                    'classes': checkpoint.get('classes', ['front', 'back']),
                    'date': datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                })
            except:
                models.append({
                    'name': model_file.name,
                    'path': str(model_file),
                    'date': datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                })
    
    return jsonify(models)


@app.route('/api/evaluate/load', methods=['POST'])
def api_evaluate_load():
    """Load a model for evaluation."""
    global eval_model, eval_model_info
    
    data = request.get_json()
    model_path = data.get('model_path')
    
    if not model_path:
        return jsonify({'success': False, 'error': 'Model path required'}), 400
    
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_b0
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get model info
        num_classes = checkpoint.get('num_classes', 2)
        classes = checkpoint.get('classes', ['front', 'back'])
        
        # Create model
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Store model
        eval_model = model
        eval_model_info = {
            'name': Path(model_path).name,
            'path': model_path,
            'classes': classes,
            'num_classes': num_classes,
            'accuracy': checkpoint.get('val_acc', 0)
        }
        
        return jsonify({
            'success': True,
            'model_info': eval_model_info
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/evaluate/predict', methods=['POST'])
def api_evaluate_predict():
    """Run prediction on image."""
    global eval_model
    
    if eval_model is None:
        return jsonify({'success': False, 'error': 'No model loaded'}), 400
    
    data = request.get_json()
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({'success': False, 'error': 'Image required'}), 400
    
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import io
        import torchvision.transforms as transforms
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = eval_model(tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        # Get label
        label = eval_model_info['classes'][pred_idx] if pred_idx < len(eval_model_info['classes']) else f'class_{pred_idx}'
        
        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence,
            'class_index': pred_idx,
            'model_name': eval_model_info['name']
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============ TRAINING API ============

training_status = {
    'running': False,
    'epoch': 0,
    'total_epochs': 0,
    'train_loss': None,
    'train_acc': None,
    'val_acc': None,
    'best_val_acc': None,
    'logs': [],
    'message': '',
    'error': None,
    'process': None
}

MODELS_DIR = DATASET_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    """Start model training."""
    global training_status
    
    if training_status['running']:
        return jsonify({'success': False, 'error': 'Training already in progress'}), 400
    
    config = request.get_json()
    
    # Reset status
    training_status = {
        'running': True,
        'epoch': 0,
        'total_epochs': config.get('epochs', 50),
        'train_loss': None,
        'train_acc': None,
        'val_acc': None,
        'best_val_acc': None,
        'logs': ['Initializing training...'],
        'message': 'Preparing dataset...',
        'error': None,
        'process': None,
        'config': config
    }
    
    # Start training in background thread
    import threading
    thread = threading.Thread(target=run_training, args=(config,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Training started'})


def run_training(config):
    """Run training in background."""
    global training_status
    
    try:
        # Check if PyTorch is available
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader
            import torchvision.transforms as transforms
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False
        
        if not TORCH_AVAILABLE:
            training_status['error'] = 'PyTorch not installed. Run: pip install torch torchvision'
            training_status['running'] = False
            return
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        training_status['logs'].append(f'Device: {device}')
        
        # Determine dataset path
        dataset_name = config.get('dataset', 'combined')
        if dataset_name == 'combined':
            # Use both datasets - prefer split structure from v10
            train_dirs = [
                str(DATASET_DIR / 'v10_manual_capture' / 'train'),
                str(DATASET_DIR / 'v8_stage2_clean' / 'train')
            ]
            val_dirs = [
                str(DATASET_DIR / 'v10_manual_capture' / 'val'),
                str(DATASET_DIR / 'v8_stage2_clean' / 'val')
            ]
        else:
            train_dirs = [str(DATASET_DIR / dataset_name / 'train')]
            val_dirs = [str(DATASET_DIR / dataset_name / 'val')]
        
        # Setup transforms
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create dataset class
        classes = config.get('classes', ['front', 'back'])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        training_status['logs'].append(f'Classes: {classes}')
        
        # Simple dataset loader
        class SimpleDataset(Dataset):
            def __init__(self, data_dirs, transform, class_to_idx):
                self.transform = transform
                self.samples = []
                self.class_to_idx = class_to_idx
                
                for data_dir in data_dirs:
                    data_path = Path(data_dir)
                    if not data_path.exists():
                        continue
                    
                    for cls_name, idx in class_to_idx.items():
                        # Try different folder naming conventions
                        for folder_name in [cls_name, f'cnie_{cls_name}', f'{cls_name}_cnie']:
                            class_dir = data_path / folder_name
                            if class_dir.exists():
                                for img_path in class_dir.glob('*.jpg'):
                                    self.samples.append((img_path, idx))
                                break
                
                # Log counts
                counts = [0] * len(class_to_idx)
                for _, label in self.samples:
                    counts[label] += 1
                
                for cls_name, idx in class_to_idx.items():
                    training_status['logs'].append(f'  {cls_name}: {counts[idx]} images')
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label
        
        train_dataset = SimpleDataset(train_dirs, train_transform, class_to_idx)
        val_dataset = SimpleDataset(val_dirs, val_transform, class_to_idx)
        
        if len(train_dataset) == 0:
            training_status['error'] = 'No training images found!'
            training_status['running'] = False
            return
        
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=0)
        
        training_status['logs'].append(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')
        
        # Create model
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
        model = model.to(device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.0001))
        criterion = nn.CrossEntropyLoss()
        
        num_epochs = config.get('epochs', 50)
        best_val_acc = 0.0
        
        training_status['total_epochs'] = num_epochs
        
        for epoch in range(1, num_epochs + 1):
            if not training_status['running']:
                break
            
            training_status['epoch'] = epoch
            training_status['message'] = f'Training epoch {epoch}/{num_epochs}...'
            
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                if not training_status['running']:
                    break
                
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100.0 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    if not training_status['running']:
                        break
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            
            # Update status
            training_status['train_loss'] = avg_train_loss
            training_status['train_acc'] = train_acc
            training_status['val_acc'] = val_acc
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                training_status['best_val_acc'] = best_val_acc
                
                # Save best model
                model_name = f"cnie_classifier_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                model_path = MODELS_DIR / model_name
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'classes': classes,
                    'num_classes': len(classes)
                }, model_path)
                training_status['latest_model'] = str(model_path)
            
            log_msg = f'Epoch {epoch}/{num_epochs} - Loss: {avg_train_loss:.4f}, Train: {train_acc:.1f}%, Val: {val_acc:.1f}% (Best: {best_val_acc:.1f}%)'
            training_status['logs'].append(log_msg)
            training_status['message'] = log_msg
        
        training_status['message'] = 'Training complete!'
        training_status['logs'].append(f'Training complete! Best val accuracy: {best_val_acc:.1f}%')
        
    except Exception as e:
        import traceback
        training_status['error'] = str(e)
        training_status['logs'].append(f'Error: {str(e)}')
        training_status['logs'].append(traceback.format_exc())
    
    finally:
        training_status['running'] = False


@app.route('/api/train/status')
def api_train_status():
    """Get training status."""
    return jsonify(training_status)


@app.route('/api/train/models')
def api_train_models():
    """List trained models."""
    models = []
    
    if MODELS_DIR.exists():
        for model_file in sorted(MODELS_DIR.glob('*.pth'), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                import torch
                checkpoint = torch.load(model_file, map_location='cpu')
                models.append({
                    'name': model_file.name,
                    'path': str(model_file),
                    'accuracy': checkpoint.get('val_acc', 0),
                    'train_acc': checkpoint.get('train_acc', 0),
                    'epoch': checkpoint.get('epoch', 0),
                    'date': datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                    'classes': checkpoint.get('classes', [])
                })
            except:
                models.append({
                    'name': model_file.name,
                    'path': str(model_file),
                    'date': datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                })
    
    return jsonify(models)


@app.route('/api/train/deploy', methods=['POST'])
def api_train_deploy():
    """Deploy a model as active."""
    data = request.get_json()
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'success': False, 'error': 'Model name required'}), 400
    
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    
    try:
        # Create symlink or copy to active model
        active_path = MODELS_DIR / 'active_model.pth'
        if active_path.exists():
            active_path.unlink()
        
        import shutil
        shutil.copy(str(model_path), str(active_path))
        
        return jsonify({'success': True, 'message': f'Model {model_name} deployed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train/delete', methods=['POST'])
def api_train_delete():
    """Delete a model."""
    data = request.get_json()
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'success': False, 'error': 'Model name required'}), 400
    
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    
    try:
        model_path.unlink()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============ COLAB REMOTE TRAINING API ============

colab_training_status = {
    'running': False,
    'epoch': 0,
    'total_epochs': 0,
    'train_loss': None,
    'train_acc': None,
    'val_acc': None,
    'best_val_acc': None,
    'logs': [],
    'message': '',
    'error': None,
    'colab_host': None,
    'process': None
}

# Download progress tracking
colab_download_status = {
    'downloading': False,
    'progress': 0,  # 0-100
    'message': '',
    'error': None,
    'model_name': None,
    'model_path': None,
    'completed': False
}

SCRIPT_DIR = Path(__file__).parent.parent / 'scripts'

@app.route('/api/train/colab/start', methods=['POST'])
def api_train_colab_start():
    """Start training on Colab GPU via SSH."""
    global colab_training_status
    
    if colab_training_status['running']:
        return jsonify({'success': False, 'error': 'Training already in progress'}), 400
    
    config = request.get_json()
    colab_host = config.get('colab_host')
    
    if not colab_host:
        return jsonify({'success': False, 'error': 'Colab hostname required'}), 400
    
    # Reset status
    colab_training_status = {
        'running': True,
        'epoch': 0,
        'total_epochs': config.get('epochs', 50),
        'train_loss': None,
        'train_acc': None,
        'val_acc': None,
        'best_val_acc': None,
        'logs': ['Initializing Colab training...'],
        'message': 'Preparing dataset tarball...',
        'error': None,
        'colab_host': colab_host,
        'process': None
    }
    
    # Start deployment in background thread
    import threading
    thread = threading.Thread(target=run_colab_training, args=(config,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Deployment started'})


def run_colab_training(config):
    """Run Colab deployment and training."""
    global colab_training_status
    
    try:
        import subprocess
        import time
        
        colab_host = config['colab_host']
        dataset = config.get('dataset', 'combined')
        epochs = config.get('epochs', 50)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.0001)
        classes = config.get('classes', ['front', 'back'])
        
        # Step 1: Create dataset tarball
        colab_training_status['logs'].append('Creating dataset tarball...')
        colab_training_status['message'] = 'Creating dataset tarball...'
        
        dataset_dir = DATASET_DIR / dataset
        if dataset == 'combined':
            # Create combined tarball from both datasets
            temp_dir = DATASET_DIR / 'temp_combined'
            temp_dir.mkdir(exist_ok=True)
            
            for split in ['train', 'val', 'test']:
                split_dir = temp_dir / split
                split_dir.mkdir(exist_ok=True)
                
                for cls in classes:
                    cls_dir = split_dir / cls
                    cls_dir.mkdir(exist_ok=True)
                    
                    # Copy from both datasets
                    for ds_name in ['v8_stage2_clean', 'v10_manual_capture']:
                        src_dir = DATASET_DIR / ds_name / split / f'cnie_{cls}'
                        if not src_dir.exists():
                            src_dir = DATASET_DIR / ds_name / split / cls
                        
                        if src_dir.exists():
                            for img in src_dir.glob('*.jpg'):
                                shutil.copy2(img, cls_dir / img.name)
            
            dataset_tar = DATASET_DIR / f'training_data_{int(time.time())}.tar.gz'
            subprocess.run(['tar', '-czf', str(dataset_tar), '-C', str(temp_dir), '.'], check=True)
            
            # Cleanup temp dir
            shutil.rmtree(temp_dir)
        else:
            dataset_tar = DATASET_DIR / f'{dataset}.tar.gz'
            if not dataset_tar.exists():
                subprocess.run(['tar', '-czf', str(dataset_tar), '-C', str(dataset_dir), '.'], check=True)
        
        colab_training_status['logs'].append(f'✅ Dataset tarball created: {dataset_tar.stat().st_size / 1024 / 1024:.1f} MB')
        
        # Step 2: Create training script
        script_content = f'''#!/usr/bin/env python3
"""Colab Training Script - Auto-generated"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from pathlib import Path
import json
import time

# Configuration
DATA_DIR = "/content/data"
EPOCHS = {epochs}
BATCH_SIZE = {batch_size}
LEARNING_RATE = {learning_rate}
CLASSES = {classes}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("CNIE CLASSIFIER TRAINING ON COLAB GPU")
print("="*70)
print(f"Device: {{DEVICE}}")
if torch.cuda.is_available():
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")

class CNIEDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {{cls: i for i, cls in enumerate(CLASSES)}}
        
        corrupted = []
        for cls_name, idx in self.class_to_idx.items():
            for class_dir_name in [cls_name, f'cnie_{{cls_name}}']:
                class_dir = self.root / class_dir_name
                if class_dir.exists():
                    for img_path in class_dir.glob("*.jpg"):
                        # Validate image before adding
                        try:
                            with Image.open(img_path) as img:
                                img.verify()  # Check if image is valid
                            self.samples.append((img_path, idx))
                        except Exception as e:
                            corrupted.append(img_path.name)
        
        if corrupted:
            print(f"  ⚠️  Skipped {{len(corrupted)}} corrupted images")
        
        print(f"  Loaded {{len(self.samples)}} images from {{root}}")
        class_counts = [0] * len(CLASSES)
        for _, label in self.samples:
            class_counts[label] += 1
        for cls_name, idx in self.class_to_idx.items():
            print(f"    {{cls_name}}: {{class_counts[idx]}}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Warning: Failed to load {{img_path}}: {{e}}")
            # Return a blank image instead of crashing
            image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                image = self.transform(image)
            return image, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = CNIEDataset(f"{{DATA_DIR}}/train", train_transform)
val_dataset = CNIEDataset(f"{{DATA_DIR}}/val", val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Create model
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_acc = 0.0
training_log = []

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100.0 * train_correct / train_total
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100.0 * val_correct / val_total
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({{
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'classes': CLASSES
        }}, '/content/cnie_classifier_best.pth')
    
    log_msg = f'Epoch {{epoch}}/{{EPOCHS}} - Loss: {{train_loss/len(train_loader):.4f}}, Train: {{train_acc:.1f}}%, Val: {{val_acc:.1f}}% (Best: {{best_val_acc:.1f}}%)'
    print(log_msg, flush=True)
    training_log.append(log_msg)
    
    # Flush stdout to ensure logs are written immediately
    import sys
    sys.stdout.flush()
    
    scheduler.step(val_acc)

# Save training log
with open('/content/training_log.json', 'w') as f:
    json.dump({{'log': training_log, 'best_acc': best_val_acc}}, f)

print(f"\\nTraining complete! Best validation accuracy: {{best_val_acc:.1f}}%", flush=True)
print(f"Model saved to: /content/cnie_classifier_best.pth", flush=True)
'''
        
        script_path = DATASET_DIR / f'train_colab_{int(time.time())}.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        colab_training_status['logs'].append('✅ Training script created')
        
        # Step 3: Deploy directly via SSH commands
        colab_training_status['logs'].append(f'🚀 Deploying to {colab_host}...')
        colab_training_status['message'] = f'Uploading to {colab_host}...'
        
        SSH_KEY = os.path.expanduser('~/.ssh/id_colab')
        
        def run_ssh_cmd(cmd, timeout=60):
            """Run SSH command with key."""
            full_cmd = ['ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no', 
                        '-o', 'ConnectTimeout=10', '-o', 'PasswordAuthentication=no',
                        f'root@{colab_host}'] + cmd
            result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
            return result
        
        def run_scp(local, remote, timeout=120):
            """Copy file via SCP."""
            full_cmd = ['scp', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no',
                        '-o', 'ConnectTimeout=10', '-o', 'PasswordAuthentication=no',
                        local, f'root@{colab_host}:{remote}']
            result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
            return result
        
        def check_ssh_key_deployed():
            """Check if SSH key is already deployed."""
            result = run_ssh_cmd(['echo', 'test'])
            return result.returncode == 0
        
        try:
            # Test connection first with key
            colab_training_status['logs'].append('Testing SSH connection...')
            
            if not check_ssh_key_deployed():
                # SSH key not deployed - show instructions
                colab_training_status['logs'].append('❌ SSH key not deployed to Colab')
                colab_training_status['logs'].append('')
                colab_training_status['logs'].append('🔑 FIRST TIME SETUP REQUIRED:')
                colab_training_status['logs'].append('Run this command in your terminal:')
                colab_training_status['logs'].append('')
                colab_training_status['logs'].append(f'  cat ~/.ssh/id_colab.pub | ssh -o StrictHostKeyChecking=no root@{colab_host} "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"')
                colab_training_status['logs'].append('')
                colab_training_status['logs'].append('Password: retinrix')
                colab_training_status['logs'].append('')
                colab_training_status['logs'].append('Then click Deploy & Train again.')
                
                colab_training_status['error'] = 'SSH key not deployed. See instructions above.'
                colab_training_status['running'] = False
                return
            
            colab_training_status['logs'].append('✅ SSH connection OK')
            
            # Create data directory
            colab_training_status['logs'].append('Creating remote directories...')
            run_ssh_cmd(['mkdir', '-p', '/content/data'])
            
            # Upload dataset
            colab_training_status['logs'].append(f'Uploading dataset ({dataset_tar.stat().st_size / 1024 / 1024:.1f} MB)...')
            result = run_scp(str(dataset_tar), '/content/data.tar.gz')
            if result.returncode != 0:
                raise Exception(f'Upload failed: {result.stderr}')
            colab_training_status['logs'].append('✅ Dataset uploaded')
            
            # Extract dataset
            colab_training_status['logs'].append('Extracting dataset...')
            result = run_ssh_cmd(['tar', '-xzf', '/content/data.tar.gz', '-C', '/content/data'])
            if result.returncode != 0:
                raise Exception(f'Extract failed: {result.stderr}')
            colab_training_status['logs'].append('✅ Dataset extracted')
            
            # Upload training script
            colab_training_status['logs'].append('Uploading training script...')
            result = run_scp(str(script_path), '/content/train.py')
            if result.returncode != 0:
                raise Exception(f'Script upload failed: {result.stderr}')
            if result.returncode != 0:
                raise Exception(f'Script upload failed: {result.stderr}')
            colab_training_status['logs'].append('✅ Script uploaded')
            
            # Start training in background
            colab_training_status['logs'].append('🚀 Starting training on GPU...')
            result = run_ssh_cmd(['nohup', 'python3', '/content/train.py', '>', '/content/train_output.log', '2>&1', '&'])
            if result.returncode != 0:
                raise Exception(f'Training start failed: {result.stderr}')
            
            colab_training_status['logs'].append('✅ Training started on GPU!')
            colab_training_status['message'] = 'Training in progress on GPU...'
            
        except Exception as e:
            colab_training_status['error'] = str(e)
            colab_training_status['logs'].append(f'❌ Error: {str(e)}')
            colab_training_status['running'] = False
            return
        
    except Exception as e:
        import traceback
        colab_training_status['error'] = str(e)
        colab_training_status['logs'].append(f'Error: {str(e)}')
        colab_training_status['logs'].append(traceback.format_exc())


@app.route('/api/train/colab/status')
def api_train_colab_status():
    """Get Colab training status."""
    try:
        host = request.args.get('host')
        
        if not host or host != colab_training_status.get('colab_host'):
            return jsonify({'running': False, 'error': 'No active training for this host'})
        
        # Try to fetch status from Colab via SSH
        if colab_training_status['running'] and not colab_training_status.get('error'):
            try:
                import subprocess
                SSH_KEY = os.path.expanduser('~/.ssh/id_colab')
                
                # Try to get training log
                result = subprocess.run(
                    ['ssh', '-i', SSH_KEY, '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no', 
                     f'root@{host}', 'cat /content/train_output.log 2>/dev/null || echo "Waiting for training to start..."'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output:
                        lines = output.split('\n')
                        # Show last 30 lines
                        colab_training_status['logs'] = lines[-30:] if len(lines) > 30 else lines
                        
                        # Check if training is still running
                        ps_result = subprocess.run(
                            ['ssh', '-i', SSH_KEY, '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
                             f'root@{host}', 'ps aux | grep train.py | grep -v grep'],
                            capture_output=True, text=True, timeout=5
                        )
                        
                        # If process not found and we have "Training complete" in logs
                        if ps_result.returncode != 0 and any('complete' in line.lower() or 'best' in line.lower() for line in lines[-5:]):
                            colab_training_status['running'] = False
                            colab_training_status['message'] = 'Training complete!'
                            
                            # Try to get final accuracy
                            for line in reversed(lines):
                                if 'Val:' in line or 'Best:' in line:
                                    try:
                                        # Extract number before %
                                        import re
                                        match = re.search(r'(\d+\.?\d*)%', line)
                                        if match:
                                            colab_training_status['best_val_acc'] = float(match.group(1))
                                    except:
                                        pass
                                    break
                        
                        # Parse epoch info from latest line
                        if lines:
                            latest = lines[-1]
                            if 'Epoch' in latest:
                                try:
                                    import re
                                    match = re.search(r'Epoch (\d+)/(\d+)', latest)
                                    if match:
                                        colab_training_status['epoch'] = int(match.group(1))
                                        colab_training_status['total_epochs'] = int(match.group(2))
                                    
                                    # Extract metrics
                                    loss_match = re.search(r'Loss: ([\d.]+)', latest)
                                    if loss_match:
                                        colab_training_status['train_loss'] = float(loss_match.group(1))
                                    
                                    train_match = re.search(r'Train: ([\d.]+)%', latest)
                                    if train_match:
                                        colab_training_status['train_acc'] = float(train_match.group(1))
                                    
                                    val_match = re.search(r'Val: ([\d.]+)%', latest)
                                    if val_match:
                                        colab_training_status['val_acc'] = float(val_match.group(1))
                                    
                                    best_match = re.search(r'Best: ([\d.]+)%', latest)
                                    if best_match:
                                        colab_training_status['best_val_acc'] = float(best_match.group(1))
                                except:
                                    pass
                                    
            except Exception as e:
                # SSH failed - training might still be starting
                pass
        
        return jsonify(colab_training_status)
    except Exception as e:
        import traceback
        return jsonify({'running': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/train/colab/clean', methods=['POST'])
def api_train_colab_clean():
    """Clean up old files on Colab before redeploying."""
    data = request.get_json()
    host = data.get('host')
    
    if not host:
        return jsonify({'success': False, 'error': 'Host required'}), 400
    
    try:
        import subprocess
        SSH_KEY = os.path.expanduser('~/.ssh/id_colab')
        
        # Kill any running training processes
        subprocess.run(
            ['ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10',
             f'root@{host}', 'pkill -f train.py || true'],
            capture_output=True, timeout=10
        )
        
        # Remove old files
        result = subprocess.run(
            ['ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10',
             f'root@{host}', 'rm -rf /content/data /content/train.py /content/data.tar.gz /content/train_output.log /content/*.pth /content/*.json || true'],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Colab cleaned successfully'})
        else:
            return jsonify({'success': False, 'error': result.stderr}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def download_model_in_background(host, model_name, local_path):
    """Download model in background thread with progress tracking."""
    global colab_download_status
    import subprocess
    import os as os_module
    
    SSH_KEY = os_module.path.expanduser('~/.ssh/id_colab')
    
    try:
        # Update status
        colab_download_status.update({
            'downloading': True,
            'progress': 5,
            'message': 'Checking model on Colab...',
            'error': None,
            'completed': False
        })
        
        # Check if model exists on Colab
        check_result = subprocess.run(
            ['ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10',
             f'root@{host}', 'ls -lh /content/cnie_classifier_best.pth'],
            capture_output=True, text=True, timeout=15
        )
        
        if check_result.returncode != 0:
            colab_download_status.update({
                'downloading': False,
                'error': 'Model file not found on Colab. Training may not have completed yet.',
                'completed': True
            })
            return
        
        # Get file size for progress estimation
        colab_download_status.update({
            'progress': 10,
            'message': 'Starting download...'
        })
        
        # Download model with SSH key using Popen for real-time progress
        cmd = ['scp', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=30',
               f'root@{host}:/content/cnie_classifier_best.pth', str(local_path)]
        
        colab_download_status.update({
            'progress': 20,
            'message': 'Downloading model file...'
        })
        
        # Run download with Popen to allow timeout handling
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5 minutes
            
            if process.returncode != 0:
                colab_download_status.update({
                    'downloading': False,
                    'error': f'Download failed: {stderr}',
                    'completed': True
                })
                return
                
        except subprocess.TimeoutExpired:
            process.kill()
            colab_download_status.update({
                'downloading': False,
                'error': 'Download timed out after 5 minutes.',
                'completed': True
            })
            return
        
        # Verify file was downloaded
        if not local_path.exists():
            colab_download_status.update({
                'downloading': False,
                'error': 'Download appeared to succeed but file not found locally',
                'completed': True
            })
            return
        
        # Success!
        file_size_mb = round(local_path.stat().st_size / 1024 / 1024, 2)
        colab_download_status.update({
            'downloading': False,
            'progress': 100,
            'message': f'Download complete! ({file_size_mb} MB)',
            'model_name': model_name,
            'model_path': str(local_path),
            'completed': True
        })
        
        # Also try to download training log (optional)
        try:
            log_path = MODELS_DIR / f'{model_name}.log.json'
            subprocess.run(
                ['scp', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no',
                 f'root@{host}:/content/training_log.json', str(log_path)],
                capture_output=True, timeout=30
            )
        except:
            pass
            
    except Exception as e:
        colab_download_status.update({
            'downloading': False,
            'error': str(e),
            'completed': True
        })


@app.route('/api/train/colab/download', methods=['POST'])
def api_train_colab_download():
    """Start downloading trained model from Colab."""
    global colab_download_status
    data = request.get_json()
    host = data.get('host')
    
    if not host:
        return jsonify({'success': False, 'error': 'Host required'}), 400
    
    import os as os_module
    SSH_KEY = os_module.path.expanduser('~/.ssh/id_colab')
    
    # Check if SSH key exists
    if not os_module.path.exists(SSH_KEY):
        return jsonify({'success': False, 'error': f'SSH key not found: {SSH_KEY}'}), 500
    
    # Reset status
    colab_download_status = {
        'downloading': True,
        'progress': 0,
        'message': 'Initializing download...',
        'error': None,
        'model_name': None,
        'model_path': None,
        'completed': False
    }
    
    # Generate model name
    model_name = f'cnie_colab_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    local_path = MODELS_DIR / model_name
    
    # Start download in background thread
    import threading
    thread = threading.Thread(target=download_model_in_background, args=(host, model_name, local_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Download started', 'model_name': model_name})


@app.route('/api/train/colab/download/status')
def api_train_colab_download_status():
    """Get download progress status."""
    return jsonify(colab_download_status)


if __name__ == '__main__':
    print("="*70)
    print("CNIE Dataset Tools - Web Application")
    print("="*70)
    print()
    print("Open your browser to: http://localhost:5000")
    print()
    print("Camera Capture Instructions:")
    print("  1. Go to 'Smart Capture' page")
    print("  2. Browser will ask for camera permission - CLICK ALLOW")
    print("  3. Click FRONT/BACK/NO-CARD button")
    print("  4. Show your card to camera")
    print("  5. Move the card slightly to trigger motion detection")
    print("  6. Images auto-upload when motion detected!")
    print()
    print("="*70)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
