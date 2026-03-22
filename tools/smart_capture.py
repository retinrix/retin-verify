#!/usr/bin/env python3
"""
Smart Dataset Capture Tool
- Real-time camera feed with auto-detection
- Detects card orientation (front/back/no-card) automatically
- Auto-captures frames when card is stable
- Shows dataset stats and advises which class needs more images
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Dataset paths
DATASET_DIR = Path.home() / "retin-verify" / "training_data" / "v10_manual_capture"


class CardDetector:
    """Detects card orientation in real-time."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.chip_template = self._create_chip_template()
        self.stability_counter = 0
        self.last_detection = None
        self.stability_threshold = 5  # frames of stable detection
        
    def _create_chip_template(self):
        """Create chip template for back detection."""
        template = np.ones((80, 60, 3), dtype=np.uint8) * 200
        center = (30, 40)
        axes = (22, 32)
        cv2.ellipse(template, center, axes, 0, 0, 360, (100, 100, 100), -1)
        for y in range(20, 61, 10):
            cv2.line(template, (10, y), (50, y), (180, 180, 180), 1)
        for x in [15, 30, 45]:
            cv2.line(template, (x, 20), (x, 60), (180, 180, 180), 1)
        cv2.rectangle(template, (20, 30), (40, 50), (160, 160, 160), -1)
        return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    def detect_orientation(self, frame):
        """
        Detect card orientation from frame.
        Returns: (orientation, confidence, details)
        orientation: 'front', 'back', 'no_card', 'uncertain'
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face (indicator of front)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        has_face = len(faces) > 0
        face_conf = min(len(faces) * 0.3 + 0.5, 1.0) if has_face else 0.0
        
        # Detect chip (indicator of back)
        max_chip_conf = 0.0
        for scale in [0.7, 0.85, 1.0, 1.15, 1.3]:
            new_w = int(self.chip_template.shape[1] * scale)
            new_h = int(self.chip_template.shape[0] * scale)
            resized = cv2.resize(self.chip_template, (new_w, new_h))
            if gray.shape[0] >= resized.shape[0] and gray.shape[1] >= resized.shape[1]:
                result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                _, local_max, _, _ = cv2.minMaxLoc(result)
                max_chip_conf = max(max_chip_conf, local_max)
        has_chip = max_chip_conf > 0.55
        
        # Detect if there's a card at all (basic edge detection)
        edges = cv2.Canny(gray, 50, 150)
        card_detected = np.sum(edges > 0) > (edges.size * 0.05)  # At least 5% edges
        
        # Determine orientation
        if has_face and not has_chip:
            orientation = 'front'
            confidence = face_conf
        elif has_chip and not has_face:
            orientation = 'back'
            confidence = max_chip_conf
        elif has_face and has_chip:
            orientation = 'uncertain'
            confidence = 0.5
        elif card_detected:
            orientation = 'no_card'  # Card present but unclear orientation
            confidence = 0.3
        else:
            orientation = 'no_card'
            confidence = 1.0
        
        details = {
            'faces': len(faces),
            'face_conf': face_conf,
            'chip_conf': max_chip_conf,
            'has_card': card_detected
        }
        
        return orientation, confidence, details, faces
    
    def check_stability(self, orientation, confidence):
        """Check if detection is stable enough for capture."""
        if orientation == self.last_detection and confidence > 0.6:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
        
        self.last_detection = orientation
        
        is_stable = self.stability_counter >= self.stability_threshold
        return is_stable, self.stability_counter
    
    def reset_stability(self):
        """Reset stability counter after capture."""
        self.stability_counter = 0
        self.last_detection = None


class DatasetStats:
    """Track and advise on dataset statistics."""
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.counts = {'front': 0, 'back': 0, 'no_card': 0}
        self.target_balance = {'front': 300, 'back': 300, 'no_card': 150}
        self.update_counts()
    
    def update_counts(self):
        """Update current image counts."""
        self.counts = {'front': 0, 'back': 0, 'no_card': 0}
        
        for cls in ['front', 'back', 'no_card']:
            folder = self.dataset_dir / cls
            if folder.exists():
                self.counts[cls] = len(list(folder.glob('*.jpg')))
    
    def get_advice(self):
        """Get advice on which class needs more images."""
        advice = []
        
        # Check balance between front and back
        front_back_diff = abs(self.counts['front'] - self.counts['back'])
        if front_back_diff > 50:
            if self.counts['front'] < self.counts['back']:
                advice.append("⚠️  FRONT needs more images!")
            else:
                advice.append("⚠️  BACK needs more images!")
        
        # Check targets
        for cls, count in self.counts.items():
            target = self.target_balance[cls]
            if count < target:
                needed = target - count
                advice.append(f"📸 {cls.upper()}: {count}/{target} (need {needed} more)")
            else:
                advice.append(f"✅ {cls.upper()}: {count}/{target} (complete!)")
        
        # Overall status
        total = sum(self.counts.values())
        target_total = sum(self.target_balance.values())
        progress = (total / target_total) * 100
        
        return '\n'.join(advice), progress, total
    
    def get_priority_class(self):
        """Get the class that most needs images."""
        deficits = {}
        for cls, count in self.counts.items():
            deficit = self.target_balance[cls] - count
            deficits[cls] = deficit
        
        return max(deficits, key=deficits.get)


def draw_overlay(frame, orientation, confidence, is_stable, stability, stats_info, fps=0):
    """Draw information overlay on frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "SMART CARD CAPTURE", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Current detection
    color = (0, 255, 0) if is_stable else (0, 165, 255)
    status_text = f"Detected: {orientation.upper()} ({confidence:.2f})"
    if is_stable:
        status_text += " [STABLE - CAPTURING]"
    else:
        status_text += f" [stabilizing {stability}/{5}]"
    
    cv2.putText(frame, status_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Stats
    y_offset = 90
    for line in stats_info.split('\n'):
        cv2.putText(frame, line, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Instructions at bottom
    cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "Move card to auto-capture | Q:Quit | R:Reset counter | S:Stats", 
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


def main():
    """Main capture loop."""
    print("="*70)
    print("SMART CARD CAPTURE TOOL")
    print("="*70)
    print()
    print("This tool will:")
    print("1. Start your camera")
    print("2. Auto-detect card orientation (front/back)")
    print("3. Automatically capture when stable")
    print("4. Show you which class needs more images")
    print()
    print("Controls:")
    print("  - Just MOVE the card in front of camera")
    print("  - Hold it steady when detection is stable")
    print("  - Q: Quit")
    print("  - R: Reset capture counter for current class")
    print("  - S: Show detailed stats")
    print("="*70)
    print()
    
    # Setup
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for cls in ['front', 'back', 'no_card']:
        (DATASET_DIR / cls).mkdir(exist_ok=True)
    
    detector = CardDetector()
    stats = DatasetStats(DATASET_DIR)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Make sure you have a webcam connected.")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Camera started! Show your cards...")
    print()
    
    capture_counts = {'front': 0, 'back': 0, 'no_card': 0}
    last_capture_time = 0
    capture_cooldown = 1.0  # seconds between captures
    
    fps_history = []
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        last_time = current_time
        
        # Detect orientation
        orientation, confidence, details, faces = detector.detect_orientation(frame)
        
        # Check stability
        is_stable, stability_count = detector.check_stability(orientation, confidence)
        
        # Get stats advice
        advice, progress, total = stats.get_advice()
        priority = stats.get_priority_class()
        
        # Draw face rectangles if front detected
        if orientation == 'front' and len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Auto-capture if stable and not on cooldown
        if is_stable and orientation in ['front', 'back', 'no_card']:
            if current_time - last_capture_time > capture_cooldown:
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{orientation}_{timestamp}.jpg"
                filepath = DATASET_DIR / orientation / filename
                
                cv2.imwrite(str(filepath), frame)
                capture_counts[orientation] += 1
                last_capture_time = current_time
                
                # Reset stability to avoid double capture
                detector.reset_stability()
                
                # Update stats
                stats.update_counts()
                
                print(f"✓ Captured {orientation}: {filename}")
        
        # Draw overlay
        frame = draw_overlay(frame, orientation, confidence, is_stable, 
                            stability_count, advice, avg_fps)
        
        # Show frame
        cv2.imshow('Smart Card Capture', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_stability()
            print("Reset stability counter")
        elif key == ord('s'):
            print("\n" + "="*50)
            print("DETAILED STATS")
            print("="*50)
            print(advice)
            print(f"\nCaptures this session:")
            for cls, count in capture_counts.items():
                print(f"  {cls}: {count}")
            print("="*50 + "\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*70)
    print("CAPTURE SESSION COMPLETE")
    print("="*70)
    print(f"\nImages saved to: {DATASET_DIR}")
    print("\nCaptures this session:")
    for cls, count in capture_counts.items():
        print(f"  {cls}: {count}")
    
    stats.update_counts()
    print("\nFinal dataset counts:")
    for cls, count in stats.counts.items():
        print(f"  {cls}: {count}")
    print("="*70)


if __name__ == '__main__':
    main()
