#!/usr/bin/env python3
"""
Dataset Cleaner UI Tool
Automatically flags potential mislabels using face and chip detection.
Allows manual review and correction of flagged images.
"""

import os
import sys
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk


class ChipDetector:
    """Chip detection using template matching."""
    
    def __init__(self):
        self.template = self._create_template()
        self.threshold = 0.55
        
    def _create_template(self):
        """Create synthetic CNIE chip template."""
        template = np.ones((80, 60, 3), dtype=np.uint8) * 200
        center = (30, 40)
        axes = (22, 32)
        cv2.ellipse(template, center, axes, 0, 0, 360, (150, 150, 150), 2)
        cv2.ellipse(template, center, axes, 0, 0, 360, (100, 100, 100), -1)
        
        # Globe pattern lines
        for y in range(20, 61, 10):
            cv2.line(template, (10, y), (50, y), (180, 180, 180), 1)
        for x in [15, 30, 45]:
            cv2.line(template, (x, 20), (x, 60), (180, 180, 180), 1)
        
        # Central contact area
        cv2.rectangle(template, (20, 30), (40, 50), (160, 160, 160), -1)
        cv2.rectangle(template, (20, 30), (40, 50), (120, 120, 120), 1)
        
        return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    def detect(self, image_path):
        """Detect chip in image. Returns (detected: bool, confidence: float)."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, 0.0
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            max_val = 0.0
            
            # Multi-scale template matching
            for scale in [0.7, 0.85, 1.0, 1.15, 1.3]:
                new_w = int(self.template.shape[1] * scale)
                new_h = int(self.template.shape[0] * scale)
                resized = cv2.resize(self.template, (new_w, new_h))
                
                if gray.shape[0] >= resized.shape[0] and gray.shape[1] >= resized.shape[1]:
                    result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                    _, local_max, _, _ = cv2.minMaxLoc(result)
                    max_val = max(max_val, local_max)
            
            return max_val > self.threshold, max_val
        except Exception as e:
            print(f"Chip detection error: {e}")
            return False, 0.0


class FaceDetector:
    """Face detection using OpenCV Haar cascade."""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            self.cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.cascade = None
            print(f"Warning: Haar cascade not found at {cascade_path}")
    
    def detect(self, image_path):
        """Detect face in image. Returns (detected: bool, confidence: float, regions: list)."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, 0.0, []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if self.cascade is None:
                return False, 0.0, []
            
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
            )
            
            detected = len(faces) > 0
            confidence = min(len(faces) * 0.3 + 0.4, 1.0) if detected else 0.0
            
            return detected, confidence, faces.tolist()
        except Exception as e:
            print(f"Face detection error: {e}")
            return False, 0.0, []


class DatasetCleaner:
    """Main dataset cleaning logic."""
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.face_detector = FaceDetector()
        self.chip_detector = ChipDetector()
        
        self.flagged_images = []  # List of dicts with image info
        self.stats = defaultdict(int)
        
    def scan_dataset(self, progress_callback=None):
        """Scan dataset and flag potential mislabels."""
        self.flagged_images = []
        self.stats = defaultdict(int)
        
        # Expected structure: dataset_dir/{train,val,test}/{cnie_front,cnie_back}/*.jpg
        total_images = 0
        for split in ['train', 'val', 'test']:
            for class_name in ['cnie_front', 'cnie_back']:
                class_dir = self.dataset_dir / split / class_name
                if class_dir.exists():
                    total_images += len(list(class_dir.glob('*.jpg')))
        
        processed = 0
        for split in ['train', 'val', 'test']:
            for class_name in ['cnie_front', 'cnie_back']:
                class_dir = self.dataset_dir / split / class_name
                if not class_dir.exists():
                    continue
                
                is_front = 'front' in class_name
                
                for img_path in class_dir.glob('*.jpg'):
                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_images, str(img_path))
                    
                    result = self._analyze_image(img_path, is_front)
                    self.stats['total'] += 1
                    
                    if result['flagged']:
                        self.flagged_images.append(result)
                        self.stats['flagged'] += 1
                        if is_front:
                            self.stats['front_no_face'] += 1
                        else:
                            self.stats['back_no_chip'] += 1
                    else:
                        self.stats['verified'] += 1
                        if is_front:
                            self.stats['front_with_face'] += 1
                        else:
                            self.stats['back_with_chip'] += 1
        
        return self.stats
    
    def _analyze_image(self, img_path, is_front):
        """Analyze a single image and return flag status."""
        result = {
            'path': str(img_path),
            'name': img_path.name,
            'original_label': 'front' if is_front else 'back',
            'split': img_path.parent.parent.name,
            'flagged': False,
            'reason': '',
            'face_detected': False,
            'face_confidence': 0.0,
            'face_regions': [],
            'chip_detected': False,
            'chip_confidence': 0.0,
            'suggested_label': None
        }
        
        # Run both detectors
        face_detected, face_conf, face_regions = self.face_detector.detect(img_path)
        chip_detected, chip_conf = self.chip_detector.detect(img_path)
        
        result['face_detected'] = face_detected
        result['face_confidence'] = face_conf
        result['face_regions'] = face_regions
        result['chip_detected'] = chip_detected
        result['chip_confidence'] = chip_conf
        
        # Flag logic
        if is_front:
            # Front should have face
            if not face_detected:
                result['flagged'] = True
                result['reason'] = 'Front image: No face detected'
                # Suggest back if chip detected
                if chip_detected and chip_conf > 0.6:
                    result['suggested_label'] = 'back'
        else:
            # Back should have chip
            if not chip_detected:
                result['flagged'] = True
                result['reason'] = 'Back image: No chip detected'
                # Suggest front if face detected
                if face_detected and face_conf > 0.5:
                    result['suggested_label'] = 'front'
        
        # Edge case: both detected or neither detected
        if face_detected and chip_detected:
            result['notes'] = 'Both face and chip detected'
        elif not face_detected and not chip_detected:
            result['notes'] = 'Neither face nor chip detected - ambiguous'
        
        return result


class DatasetCleanerUI:
    """Tkinter UI for dataset cleaning."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CNIE Dataset Cleaner")
        self.root.geometry("1400x900")
        
        self.cleaner = None
        self.current_index = 0
        self.current_image_tk = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components."""
        # Top frame - Dataset selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill='x')
        
        ttk.Label(top_frame, text="Dataset Directory:").pack(side='left')
        self.path_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.path_var, width=60).pack(side='left', padx=5)
        ttk.Button(top_frame, text="Browse", command=self.browse_dataset).pack(side='left')
        ttk.Button(top_frame, text="Scan Dataset", command=self.scan_dataset).pack(side='left', padx=10)
        
        # Progress frame
        self.progress_frame = ttk.Frame(self.root, padding="10")
        self.progress_frame.pack(fill='x')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x')
        self.status_label = ttk.Label(self.progress_frame, text="Ready")
        self.status_label.pack()
        
        # Main content frame
        content_frame = ttk.Frame(self.root, padding="10")
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - Image viewer
        left_frame = ttk.LabelFrame(content_frame, text="Image Preview", padding="10")
        left_frame.pack(side='left', fill='both', expand=True)
        
        self.canvas = tk.Canvas(left_frame, bg='gray', width=600, height=500)
        self.canvas.pack(fill='both', expand=True)
        
        self.image_info_label = ttk.Label(left_frame, text="No image loaded")
        self.image_info_label.pack(pady=5)
        
        # Right panel - Controls and list
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding="10")
        stats_frame.pack(fill='x', pady=5)
        self.stats_text = tk.Text(stats_frame, height=8, width=40)
        self.stats_text.pack(fill='x')
        
        # Flagged images list
        list_frame = ttk.LabelFrame(right_frame, text="Flagged Images", padding="10")
        list_frame.pack(fill='both', expand=True, pady=5)
        
        # Treeview for flagged images
        columns = ('name', 'original', 'reason', 'suggested')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        self.tree.heading('name', text='Image')
        self.tree.heading('original', text='Original')
        self.tree.heading('reason', text='Reason')
        self.tree.heading('suggested', text='Suggested')
        self.tree.column('name', width=150)
        self.tree.column('original', width=60)
        self.tree.column('reason', width=150)
        self.tree.column('suggested', width=60)
        self.tree.pack(fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self.on_select_image)
        
        # Action buttons
        action_frame = ttk.LabelFrame(right_frame, text="Actions", padding="10")
        action_frame.pack(fill='x', pady=5)
        
        ttk.Button(action_frame, text="✓ Keep Original", command=self.keep_original).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="→ Relabel to Front", command=lambda: self.relabel('front')).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="→ Relabel to Back", command=lambda: self.relabel('back')).pack(fill='x', pady=2)
        ttk.Button(action_frame, text="✗ Exclude Image", command=self.exclude_image).pack(fill='x', pady=2)
        
        # Navigation
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill='x', pady=5)
        ttk.Button(nav_frame, text="← Previous", command=self.prev_image).pack(side='left')
        ttk.Button(nav_frame, text="Next →", command=self.next_image).pack(side='right')
        
        # Export buttons
        export_frame = ttk.Frame(right_frame)
        export_frame.pack(fill='x', pady=5)
        ttk.Button(export_frame, text="Export Cleaned Dataset", command=self.export_dataset).pack(side='left', padx=2)
        ttk.Button(export_frame, text="Export Report", command=self.export_report).pack(side='left', padx=2)
        
    def browse_dataset(self):
        """Browse for dataset directory."""
        path = filedialog.askdirectory(title="Select Dataset Directory")
        if path:
            self.path_var.set(path)
    
    def scan_dataset(self):
        """Scan dataset and flag potential mislabels."""
        dataset_dir = self.path_var.get()
        if not dataset_dir or not os.path.exists(dataset_dir):
            messagebox.showerror("Error", "Please select a valid dataset directory")
            return
        
        # Check if this looks like the root dataset directory
        expected_splits = ['train', 'val', 'test']
        found_splits = [d for d in expected_splits if (Path(dataset_dir) / d).exists()]
        
        if len(found_splits) == 0:
            messagebox.showwarning(
                "Wrong Directory Selected",
                "Please select the ROOT dataset directory that contains train/val/test folders.\n\n"
                f"Current selection: {dataset_dir}\n\n"
                "Expected structure:\n"
                "  dataset_dir/\n"
                "    ├── train/\n"
                "    │   ├── cnie_front/\n"
                "    │   └── cnie_back/\n"
                "    ├── val/\n"
                "    └── test/"
            )
            return
        
        if len(found_splits) < 3:
            if not messagebox.askyesno(
                "Partial Dataset Detected",
                f"Only found {len(found_splits)} split(s): {', '.join(found_splits)}\n\n"
                "Continue anyway?"
            ):
                return
        
        self.cleaner = DatasetCleaner(dataset_dir)
        
        def progress_callback(processed, total, current_file):
            percent = (processed / total) * 100
            self.progress_var.set(percent)
            self.status_label.config(text=f"Processing {processed}/{total}: {os.path.basename(current_file)}")
            self.root.update_idletasks()
        
        # Run scan
        self.status_label.config(text="Scanning dataset...")
        stats = self.cleaner.scan_dataset(progress_callback)
        
        # Update stats display
        stats_text = f"""Total Images: {stats['total']}
Verified Clean: {stats['verified']}
Flagged: {stats['flagged']}

Front with Face: {stats.get('front_with_face', 0)}
Front without Face: {stats.get('front_no_face', 0)}
Back with Chip: {stats.get('back_with_chip', 0)}
Back without Chip: {stats.get('back_no_chip', 0)}
        """
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats_text)
        
        # Populate treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for i, img_info in enumerate(self.cleaner.flagged_images):
            self.tree.insert('', 'end', iid=str(i), values=(
                img_info['name'][:30],
                img_info['original_label'],
                img_info['reason'][:25],
                img_info.get('suggested_label', '-')
            ))
        
        self.status_label.config(text=f"Scan complete. {stats['flagged']} images flagged for review.")
        
        if self.cleaner.flagged_images:
            self.current_index = 0
            self.show_image(0)
    
    def on_select_image(self, event):
        """Handle image selection from treeview."""
        selection = self.tree.selection()
        if selection:
            self.current_index = int(selection[0])
            self.show_image(self.current_index)
    
    def show_image(self, index):
        """Display image with annotations."""
        if not self.cleaner or index >= len(self.cleaner.flagged_images):
            return
        
        img_info = self.cleaner.flagged_images[index]
        
        # Load and prepare image
        img = cv2.imread(img_info['path'])
        if img is None:
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw face detection boxes
        if img_info['face_detected']:
            for (x, y, w, h) in img_info['face_regions']:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"Face ({img_info['face_confidence']:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw chip detection indicator
        if img_info['chip_detected']:
            h, w = img.shape[:2]
            cv2.putText(img, f"Chip: {img_info['chip_confidence']:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Resize to fit canvas
        canvas_w = 580
        canvas_h = 480
        img_h, img_w = img.shape[:2]
        scale = min(canvas_w/img_w, canvas_h/img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Convert to PIL and then Tkinter
        pil_img = Image.fromarray(img_resized)
        self.current_image_tk = ImageTk.PhotoImage(pil_img)
        
        # Update canvas
        self.canvas.delete('all')
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=self.current_image_tk)
        
        # Update info label
        info = f"""Image: {img_info['name']}
Original Label: {img_info['original_label']}
Reason: {img_info['reason']}
Face Detected: {img_info['face_detected']} (conf: {img_info['face_confidence']:.2f})
Chip Detected: {img_info['chip_detected']} (conf: {img_info['chip_confidence']:.2f})
Suggested Label: {img_info.get('suggested_label', 'None')}
Action: {img_info.get('action', 'pending')}
Path: {img_info['path']}
        """
        self.image_info_label.config(text=info)
        
        # Select in treeview
        self.tree.selection_set(str(index))
        self.tree.see(str(index))
    
    def keep_original(self):
        """Mark current image as verified (keep original label)."""
        if self.cleaner and self.current_index < len(self.cleaner.flagged_images):
            self.cleaner.flagged_images[self.current_index]['action'] = 'keep'
            self.next_image()
    
    def relabel(self, new_label):
        """Relabel current image."""
        if self.cleaner and self.current_index < len(self.cleaner.flagged_images):
            self.cleaner.flagged_images[self.current_index]['action'] = f'relabel_to_{new_label}'
            self.cleaner.flagged_images[self.current_index]['new_label'] = new_label
            self.next_image()
    
    def exclude_image(self):
        """Mark current image for exclusion."""
        if self.cleaner and self.current_index < len(self.cleaner.flagged_images):
            self.cleaner.flagged_images[self.current_index]['action'] = 'exclude'
            self.next_image()
    
    def next_image(self):
        """Show next flagged image."""
        if self.cleaner and self.current_index < len(self.cleaner.flagged_images) - 1:
            self.current_index += 1
            self.show_image(self.current_index)
    
    def prev_image(self):
        """Show previous flagged image."""
        if self.cleaner and self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.current_index)
    
    def export_dataset(self):
        """Export cleaned dataset."""
        if not self.cleaner:
            messagebox.showerror("Error", "No dataset scanned")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        output_path = Path(output_dir) / f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        stats = {'kept': 0, 'relabeled': 0, 'excluded': 0, 'pending': 0}
        
        # Process all images
        all_images = []
        
        # Add verified (non-flagged) images
        for split in ['train', 'val', 'test']:
            for class_name in ['cnie_front', 'cnie_back']:
                class_dir = self.cleaner.dataset_dir / split / class_name
                if not class_dir.exists():
                    continue
                for img_path in class_dir.glob('*.jpg'):
                    # Check if this image was flagged
                    is_flagged = any(f['path'] == str(img_path) for f in self.cleaner.flagged_images)
                    if not is_flagged:
                        all_images.append({
                            'src': img_path,
                            'split': split,
                            'label': class_name,
                            'action': 'keep'
                        })
                        stats['kept'] += 1
        
        # Add flagged images with their actions
        for img_info in self.cleaner.flagged_images:
            action = img_info.get('action', 'pending')
            
            if action == 'exclude':
                stats['excluded'] += 1
                continue
            
            if action == 'pending':
                stats['pending'] += 1
                continue
            
            if action == 'keep':
                new_label = img_info['original_label']
                new_label_dir = 'cnie_front' if new_label == 'front' else 'cnie_back'
                stats['kept'] += 1
            elif action.startswith('relabel_to_'):
                new_label = img_info['new_label']
                new_label_dir = 'cnie_front' if new_label == 'front' else 'cnie_back'
                stats['relabeled'] += 1
            else:
                continue
            
            all_images.append({
                'src': Path(img_info['path']),
                'split': img_info['split'],
                'label': new_label_dir,
                'action': action
            })
        
        # Copy images
        for img in all_images:
            dest = output_path / img['split'] / img['label'] / img['src'].name
            shutil.copy2(img['src'], dest)
        
        # Save metadata
        metadata = {
            'export_date': datetime.now().isoformat(),
            'original_dataset': str(self.cleaner.dataset_dir),
            'statistics': stats,
            'flagged_images': self.cleaner.flagged_images
        }
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        messagebox.showinfo("Export Complete", 
            f"Dataset exported to:\n{output_path}\n\n"
            f"Kept: {stats['kept']}\n"
            f"Relabeled: {stats['relabeled']}\n"
            f"Excluded: {stats['excluded']}\n"
            f"Pending: {stats['pending']}")
    
    def export_report(self):
        """Export cleaning report as JSON."""
        if not self.cleaner:
            messagebox.showerror("Error", "No dataset scanned")
            return
        
        output_file = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"dataset_cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if not output_file:
            return
        
        report = {
            'scan_date': datetime.now().isoformat(),
            'dataset_dir': str(self.cleaner.dataset_dir),
            'statistics': dict(self.cleaner.stats),
            'flagged_images': self.cleaner.flagged_images
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        messagebox.showinfo("Export Complete", f"Report saved to:\n{output_file}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = DatasetCleanerUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
