#!/usr/bin/env python3
"""
Dataset Cleaner - Web Interface
Simple HTTP server for manual dataset cleaning via browser.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import base64

DATASET_DIR = None
CURRENT_IMAGES = []
CURRENT_INDEX = 0
CURRENT_SPLIT = ""
CURRENT_CLASS = ""
MOVED_COUNT = 0


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global CURRENT_IMAGES, CURRENT_INDEX
        
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        
        if path == '/':
            self.send_html(self.get_main_page())
        elif path == '/api/load':
            self.handle_load(query)
        elif path == '/api/next':
            self.handle_next()
        elif path == '/api/prev':
            self.handle_prev()
        elif path == '/api/move':
            self.handle_move(query)
        elif path == '/api/delete':
            self.handle_delete()
        elif path == '/api/current_image':
            self.serve_current_image()
        elif path == '/api/stats':
            self.send_json(self.get_stats())
        else:
            self.send_error(404)
    
    def send_html(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def get_main_page(self):
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Dataset Cleaner - Web Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; }
        .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .main { display: flex; gap: 20px; }
        .image-panel { background: white; padding: 15px; border-radius: 8px; flex: 1; }
        .actions-panel { background: white; padding: 15px; border-radius: 8px; width: 250px; }
        #image-display { max-width: 100%; max-height: 600px; border: 2px solid #ddd; }
        .btn { display: block; width: 100%; padding: 15px; margin: 10px 0; font-size: 16px; 
               border: none; border-radius: 5px; cursor: pointer; }
        .btn-correct { background: #4CAF50; color: white; }
        .btn-move { background: #2196F3; color: white; }
        .btn-delete { background: #f44336; color: white; }
        .btn-nav { background: #9E9E9E; color: white; display: inline-block; width: auto; padding: 10px 20px; }
        .info { margin: 10px 0; padding: 10px; background: #e3f2fd; border-radius: 5px; }
        .stats { background: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px; }
        select, input { padding: 8px; margin: 5px; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧹 CNIE Dataset Cleaner - Web Interface</h1>
        
        <div class="controls">
            <label>Dataset Directory:</label>
            <input type="text" id="dataset-path" value="/home/retinrix/retin-verify/training_data/v8_stage2_clean" size="60">
            <button onclick="loadDataset()">Load Dataset</button>
            <br><br>
            
            <label>Split:</label>
            <select id="split">
                <option value="train">train</option>
                <option value="val">val</option>
                <option value="test">test</option>
            </select>
            
            <label>Class:</label>
            <select id="class">
                <option value="cnie_front">cnie_front</option>
                <option value="cnie_back">cnie_back</option>
            </select>
            
            <button onclick="loadFolder()">Load Folder</button>
            
            <div id="status" class="stats"></div>
        </div>
        
        <div class="main">
            <div class="image-panel">
                <img id="image-display" src="/api/current_image" alt="No image loaded">
                <div id="image-info" class="info"></div>
                <div>
                    <button class="btn btn-nav" onclick="firstImage()">⏮ First</button>
                    <button class="btn btn-nav" onclick="prevImage()">◀ Prev</button>
                    <button class="btn btn-nav" onclick="nextImage()">Next ▶</button>
                    <button class="btn btn-nav" onclick="lastImage()">Last ⏭</button>
                    <input type="number" id="jump-to" placeholder="#" style="width: 60px;">
                    <button class="btn btn-nav" onclick="jumpTo()">Go</button>
                </div>
            </div>
            
            <div class="actions-panel">
                <h3>Actions</h3>
                <button class="btn btn-correct" onclick="markCorrect()">✓ CORRECT (Next)</button>
                <hr>
                <button class="btn btn-move" onclick="moveImage('cnie_back')">→ Move to BACK</button>
                <button class="btn btn-move" onclick="moveImage('cnie_front')">→ Move to FRONT</button>
                <hr>
                <button class="btn btn-delete" onclick="deleteImage()">✗ DELETE</button>
            </div>
        </div>
    </div>
    
    <script>
        function updateStatus() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').innerHTML = 
                        `Folder: ${data.folder} | Image ${data.current} of ${data.total} | Moved: ${data.moved}`;
                    document.getElementById('image-info').innerHTML = 
                        `<b>File:</b> ${data.filename}<br><b>Path:</b> ${data.path}`;
                });
        }
        
        function refreshImage() {
            document.getElementById('image-display').src = '/api/current_image?' + Date.now();
            updateStatus();
        }
        
        function loadDataset() {
            const path = document.getElementById('dataset-path').value;
            fetch('/api/load?path=' + encodeURIComponent(path))
                .then(r => r.json())
                .then(data => alert(data.message));
        }
        
        function loadFolder() {
            const split = document.getElementById('split').value;
            const cls = document.getElementById('class').value;
            fetch('/api/load?split=' + split + '&class=' + cls)
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    refreshImage();
                });
        }
        
        function nextImage() {
            fetch('/api/next').then(refreshImage);
        }
        
        function prevImage() {
            fetch('/api/prev').then(refreshImage);
        }
        
        function firstImage() {
            fetch('/api/load?index=0').then(refreshImage);
        }
        
        function lastImage() {
            fetch('/api/load?index=-1').then(refreshImage);
        }
        
        function jumpTo() {
            const idx = document.getElementById('jump-to').value - 1;
            fetch('/api/load?index=' + idx).then(refreshImage);
        }
        
        function markCorrect() {
            nextImage();
        }
        
        function moveImage(target) {
            if (!confirm('Move to ' + target + '?')) return;
            fetch('/api/move?to=' + target)
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    refreshImage();
                });
        }
        
        function deleteImage() {
            if (!confirm('Are you sure you want to DELETE this image?')) return;
            fetch('/api/delete')
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    refreshImage();
                });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') nextImage();
            if (e.key === 'ArrowLeft') prevImage();
            if (e.key === 'c') markCorrect();
            if (e.key === 'b') moveImage('cnie_back');
            if (e.key === 'f') moveImage('cnie_front');
            if (e.key === 'Delete') deleteImage();
        });
        
        updateStatus();
    </script>
</body>
</html>'''
    
    def handle_load(self, query):
        global DATASET_DIR, CURRENT_IMAGES, CURRENT_INDEX, CURRENT_SPLIT, CURRENT_CLASS
        
        if 'path' in query:
            DATASET_DIR = Path(query['path'][0])
            if not DATASET_DIR.exists():
                self.send_json({'error': 'Path not found'})
                return
            self.send_json({'message': f'Dataset loaded: {DATASET_DIR}'})
            return
        
        if not DATASET_DIR:
            self.send_json({'error': 'Load dataset first'})
            return
        
        if 'split' in query and 'class' in query:
            CURRENT_SPLIT = query['split'][0]
            CURRENT_CLASS = query['class'][0]
            folder = DATASET_DIR / CURRENT_SPLIT / CURRENT_CLASS
            CURRENT_IMAGES = sorted(list(folder.glob('*.jpg')))
            CURRENT_INDEX = 0
            self.send_json({
                'message': f'Loaded {len(CURRENT_IMAGES)} images from {CURRENT_SPLIT}/{CURRENT_CLASS}'
            })
            return
        
        if 'index' in query:
            idx = int(query['index'][0])
            if idx == -1:
                CURRENT_INDEX = len(CURRENT_IMAGES) - 1
            else:
                CURRENT_INDEX = max(0, min(idx, len(CURRENT_IMAGES) - 1))
            self.send_json({'message': f'Jumped to image {CURRENT_INDEX + 1}'})
    
    def handle_next(self):
        global CURRENT_INDEX
        if CURRENT_IMAGES and CURRENT_INDEX < len(CURRENT_IMAGES) - 1:
            CURRENT_INDEX += 1
        self.send_json({'index': CURRENT_INDEX})
    
    def handle_prev(self):
        global CURRENT_INDEX
        if CURRENT_IMAGES and CURRENT_INDEX > 0:
            CURRENT_INDEX -= 1
        self.send_json({'index': CURRENT_INDEX})
    
    def handle_move(self, query):
        global CURRENT_IMAGES, CURRENT_INDEX, MOVED_COUNT
        
        if not CURRENT_IMAGES or CURRENT_INDEX >= len(CURRENT_IMAGES):
            self.send_json({'error': 'No image loaded'})
            return
        
        target = query.get('to', [''])[0]
        if target not in ['cnie_front', 'cnie_back']:
            self.send_json({'error': 'Invalid target'})
            return
        
        img_path = CURRENT_IMAGES[CURRENT_INDEX]
        
        if target == CURRENT_CLASS:
            self.send_json({'message': 'Already in this folder'})
            return
        
        # Move
        dest_dir = DATASET_DIR / CURRENT_SPLIT / target
        dest_dir.mkdir(exist_ok=True)
        dest_path = dest_dir / img_path.name
        
        if dest_path.exists():
            dest_path.unlink()
        
        shutil.move(str(img_path), str(dest_path))
        
        # Remove from list
        CURRENT_IMAGES.pop(CURRENT_INDEX)
        MOVED_COUNT += 1
        
        if CURRENT_INDEX >= len(CURRENT_IMAGES):
            CURRENT_INDEX = max(0, len(CURRENT_IMAGES) - 1)
        
        self.send_json({'message': f'Moved to {target}'})
    
    def handle_delete(self):
        global CURRENT_IMAGES, CURRENT_INDEX
        
        if not CURRENT_IMAGES or CURRENT_INDEX >= len(CURRENT_IMAGES):
            self.send_json({'error': 'No image loaded'})
            return
        
        img_path = CURRENT_IMAGES[CURRENT_INDEX]
        img_path.unlink()
        CURRENT_IMAGES.pop(CURRENT_INDEX)
        
        if CURRENT_INDEX >= len(CURRENT_IMAGES):
            CURRENT_INDEX = max(0, len(CURRENT_IMAGES) - 1)
        
        self.send_json({'message': 'Deleted'})
    
    def serve_current_image(self):
        if not CURRENT_IMAGES or CURRENT_INDEX >= len(CURRENT_IMAGES):
            self.send_error(404)
            return
        
        img_path = CURRENT_IMAGES[CURRENT_INDEX]
        try:
            with open(img_path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.end_headers()
            self.wfile.write(data)
        except:
            self.send_error(404)
    
    def get_stats(self):
        if not CURRENT_IMAGES:
            return {
                'folder': f'{CURRENT_SPLIT}/{CURRENT_CLASS}' if CURRENT_SPLIT else 'Not loaded',
                'current': 0,
                'total': 0,
                'moved': MOVED_COUNT,
                'filename': 'No image',
                'path': ''
            }
        
        img_path = CURRENT_IMAGES[CURRENT_INDEX] if CURRENT_INDEX < len(CURRENT_IMAGES) else None
        return {
            'folder': f'{CURRENT_SPLIT}/{CURRENT_CLASS}',
            'current': CURRENT_INDEX + 1,
            'total': len(CURRENT_IMAGES),
            'moved': MOVED_COUNT,
            'filename': img_path.name if img_path else 'No image',
            'path': str(img_path) if img_path else ''
        }


def main():
    port = 8080
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"="*60)
    print(f"Dataset Cleaner Web Server")
    print(f"="*60)
    print(f"Open in your browser: http://localhost:{port}")
    print(f"")
    print(f"Keyboard shortcuts:")
    print(f"  → / ←     - Next/Prev image")
    print(f"  C         - Mark correct (next)")
    print(f"  B         - Move to BACK")
    print(f"  F         - Move to FRONT")
    print(f"  Delete    - Delete image")
    print(f"="*60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")


if __name__ == '__main__':
    main()
