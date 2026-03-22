#!/usr/bin/env python3
"""
Dataset Cleaner UI Tool v2
- Automatic scan mode with face/chip detection
- Manual scan mode: browse folder by folder, manually correct labels
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


class DatasetCleanerV2:
    """Main dataset cleaning logic."""
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.current_images = []  # List of image paths for manual mode
        self.current_index = 0
        self.current_split = ''
        self.current_class = ''
        
    def get_images_in_folder(self, split, class_name):
        """Get all images in a specific folder."""
        folder = self.dataset_dir / split / class_name
        if not folder.exists():
            return []
        return sorted(list(folder.glob('*.jpg')))
    
    def move_image(self, img_path, from_class, to_class):
        """Move image from one class folder to another."""
        try:
            dest_dir = img_path.parent.parent / to_class
            dest_dir.mkdir(exist_ok=True)
            dest_path = dest_dir / img_path.name
            
            # If file exists at destination, remove it first
            if dest_path.exists():
                dest_path.unlink()
            
            shutil.move(str(img_path), str(dest_path))
            return dest_path
        except Exception as e:
            print(f"Error moving image: {e}")
            return None


class DatasetCleanerUIV2:
    """Tkinter UI for dataset cleaning with manual mode."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CNIE Dataset Cleaner v2 - Manual & Auto Mode")
        self.root.geometry("1600x1000")
        
        self.cleaner = None
        self.current_image_tk = None
        self.manual_images = []
        self.manual_index = 0
        self.manual_split = ''
        self.manual_class = ''
        self.moved_count = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components."""
        # Top frame - Dataset selection and mode switch
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill='x')
        
        ttk.Label(top_frame, text="Dataset Directory:").pack(side='left')
        self.path_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.path_var, width=50).pack(side='left', padx=5)
        ttk.Button(top_frame, text="Browse", command=self.browse_dataset).pack(side='left')
        ttk.Button(top_frame, text="Load Dataset", command=self.load_dataset).pack(side='left', padx=5)
        
        # Mode selection frame
        mode_frame = ttk.LabelFrame(self.root, text="Mode Selection", padding="10")
        mode_frame.pack(fill='x', padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value="manual")
        ttk.Radiobutton(mode_frame, text="🔍 Manual Scan Mode", variable=self.mode_var, 
                        value="manual", command=self.switch_mode).pack(side='left', padx=20)
        ttk.Radiobutton(mode_frame, text="🤖 Auto Scan Mode", variable=self.mode_var, 
                        value="auto", command=self.switch_mode).pack(side='left', padx=20)
        
        # Manual mode frame
        self.manual_frame = ttk.LabelFrame(self.root, text="Manual Review", padding="10")
        self.manual_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Folder selection for manual mode
        folder_frame = ttk.Frame(self.manual_frame)
        folder_frame.pack(fill='x', pady=5)
        
        ttk.Label(folder_frame, text="Select Folder to Review:").pack(side='left')
        self.split_var = tk.StringVar(value="train")
        ttk.Combobox(folder_frame, textvariable=self.split_var, values=["train", "val", "test"], 
                     width=10, state="readonly").pack(side='left', padx=5)
        
        self.class_var = tk.StringVar(value="cnie_front")
        ttk.Combobox(folder_frame, textvariable=self.class_var, 
                     values=["cnie_front", "cnie_back"], width=15, state="readonly").pack(side='left', padx=5)
        
        ttk.Button(folder_frame, text="Load Folder", command=self.load_manual_folder).pack(side='left', padx=10)
        
        self.folder_stats_label = ttk.Label(folder_frame, text="")
        self.folder_stats_label.pack(side='left', padx=20)
        
        # Main content area for manual mode
        manual_content = ttk.Frame(self.manual_frame)
        manual_content.pack(fill='both', expand=True, pady=10)
        
        # Left: Image viewer
        left_frame = ttk.LabelFrame(manual_content, text="Image Viewer", padding="10")
        left_frame.pack(side='left', fill='both', expand=True)
        
        self.manual_canvas = tk.Canvas(left_frame, bg='gray', width=700, height=600)
        self.manual_canvas.pack(fill='both', expand=True)
        
        self.manual_info_label = ttk.Label(left_frame, text="No image loaded", justify='left')
        self.manual_info_label.pack(pady=5)
        
        # Center: Action buttons
        action_frame = ttk.LabelFrame(manual_content, text="Actions", padding="10")
        action_frame.pack(side='left', fill='y', padx=10)
        
        ttk.Button(action_frame, text="✓ CORRECT (Next)", 
                   command=self.mark_correct).pack(fill='x', pady=5)
        ttk.Separator(action_frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Label(action_frame, text="Move to:").pack()
        ttk.Button(action_frame, text="→ Move to BACK", 
                   command=lambda: self.move_image("cnie_back")).pack(fill='x', pady=5)
        ttk.Button(action_frame, text="→ Move to FRONT", 
                   command=lambda: self.move_image("cnie_front")).pack(fill='x', pady=5)
        
        ttk.Separator(action_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Button(action_frame, text="✗ DELETE Image", 
                   command=self.delete_image).pack(fill='x', pady=5)
        
        # Right: Progress and list
        right_frame = ttk.Frame(manual_content)
        right_frame.pack(side='right', fill='y', padx=5)
        
        # Progress stats
        progress_frame = ttk.LabelFrame(right_frame, text="Progress", padding="10")
        progress_frame.pack(fill='x', pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="0 / 0")
        self.progress_label.pack()
        
        self.moved_label = ttk.Label(progress_frame, text="Moved: 0")
        self.moved_label.pack()
        
        # Navigation
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill='x', pady=10)
        ttk.Button(nav_frame, text="⏮ First", command=self.first_image).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_image).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_image).pack(side='left', padx=2)
        ttk.Button(nav_frame, text="Last ⏭", command=self.last_image).pack(side='left', padx=2)
        
        # Jump to index
        jump_frame = ttk.Frame(right_frame)
        jump_frame.pack(fill='x', pady=5)
        ttk.Label(jump_frame, text="Jump to:").pack(side='left')
        self.jump_var = tk.StringVar()
        ttk.Entry(jump_frame, textvariable=self.jump_var, width=8).pack(side='left', padx=2)
        ttk.Button(jump_frame, text="Go", command=self.jump_to).pack(side='left', padx=2)
        
        # Auto mode frame (initially hidden)
        self.auto_frame = ttk.LabelFrame(self.root, text="Auto Scan (Face & Chip Detection)", padding="10")
        
        # Auto mode content
        auto_content = ttk.Frame(self.auto_frame)
        auto_content.pack(fill='both', expand=True)
        
        # Auto controls
        auto_ctrl = ttk.Frame(auto_content)
        auto_ctrl.pack(fill='x')
        
        ttk.Button(auto_ctrl, text="Start Auto Scan", command=self.start_auto_scan).pack(side='left', padx=5)
        self.auto_progress = ttk.Progressbar(auto_ctrl, length=400, mode='determinate')
        self.auto_progress.pack(side='left', padx=10)
        self.auto_status = ttk.Label(auto_ctrl, text="Ready")
        self.auto_status.pack(side='left')
        
        # Auto results
        self.auto_results = ttk.Frame(auto_content)
        self.auto_results.pack(fill='both', expand=True, pady=10)
        
        # Stats text
        self.auto_stats_text = tk.Text(self.auto_results, height=10, width=40)
        self.auto_stats_text.pack(side='left', fill='both', expand=True)
        
        # Treeview for flagged images
        columns = ('name', 'folder', 'reason', 'suggested')
        self.auto_tree = ttk.Treeview(self.auto_results, columns=columns, show='headings', height=10)
        self.auto_tree.heading('name', text='Image')
        self.auto_tree.heading('folder', text='Folder')
        self.auto_tree.heading('reason', text='Reason')
        self.auto_tree.heading('suggested', text='Suggested')
        self.auto_tree.pack(side='right', fill='both', expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief='sunken')
        self.status_bar.pack(fill='x', side='bottom')
        
        # Show manual mode by default
        self.switch_mode()
    
    def switch_mode(self):
        """Switch between manual and auto mode."""
        mode = self.mode_var.get()
        if mode == "manual":
            self.manual_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.auto_frame.pack_forget()
        else:
            self.manual_frame.pack_forget()
            self.auto_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    def browse_dataset(self):
        """Browse for dataset directory."""
        path = filedialog.askdirectory(title="Select Dataset Directory (root with train/val/test)")
        if path:
            self.path_var.set(path)
    
    def load_dataset(self):
        """Load the dataset."""
        dataset_dir = self.path_var.get()
        if not dataset_dir or not os.path.exists(dataset_dir):
            messagebox.showerror("Error", "Please select a valid dataset directory")
            return
        
        # Validate structure
        expected_splits = ['train', 'val', 'test']
        found_splits = [d for d in expected_splits if (Path(dataset_dir) / d).exists()]
        
        if len(found_splits) == 0:
            messagebox.showerror("Invalid Dataset", 
                "No train/val/test folders found. Please select the root dataset directory.")
            return
        
        self.cleaner = DatasetCleanerV2(dataset_dir)
        self.status_bar.config(text=f"Dataset loaded: {dataset_dir} ({len(found_splits)} splits)")
        messagebox.showinfo("Success", f"Dataset loaded successfully!\nFound {len(found_splits)} splits.")
    
    # ==================== MANUAL MODE ====================
    
    def load_manual_folder(self):
        """Load images from selected folder for manual review."""
        if not self.cleaner:
            messagebox.showerror("Error", "Please load dataset first")
            return
        
        split = self.split_var.get()
        class_name = self.class_var.get()
        
        self.manual_split = split
        self.manual_class = class_name
        self.manual_images = self.cleaner.get_images_in_folder(split, class_name)
        self.manual_index = 0
        self.moved_count = 0
        
        total = len(self.manual_images)
        self.folder_stats_label.config(text=f"Total: {total} images")
        
        if total == 0:
            messagebox.showinfo("Info", f"No images found in {split}/{class_name}")
            return
        
        self.update_manual_display()
        self.status_bar.config(text=f"Loaded {total} images from {split}/{class_name}")
    
    def update_manual_display(self):
        """Update the manual mode display."""
        if not self.manual_images or self.manual_index >= len(self.manual_images):
            self.manual_canvas.delete('all')
            self.manual_info_label.config(text="No more images in this folder")
            return
        
        img_path = self.manual_images[self.manual_index]
        
        # Load and display image
        try:
            img = Image.open(img_path)
            img.thumbnail((680, 580))
            self.current_image_tk = ImageTk.PhotoImage(img)
            
            self.manual_canvas.delete('all')
            self.manual_canvas.create_image(350, 300, image=self.current_image_tk)
            
            # Update info
            info = f"""File: {img_path.name}
Path: {img_path}
Current Folder: {self.manual_split}/{self.manual_class}
Image {self.manual_index + 1} of {len(self.manual_images)}
Size: {img.size}
            """
            self.manual_info_label.config(text=info)
            
            # Update progress
            self.progress_label.config(text=f"{self.manual_index + 1} / {len(self.manual_images)}")
            self.moved_label.config(text=f"Moved this session: {self.moved_count}")
            
        except Exception as e:
            self.manual_info_label.config(text=f"Error loading image: {e}")
    
    def mark_correct(self):
        """Mark current image as correct and move to next."""
        self.next_image()
    
    def move_image(self, target_class):
        """Move current image to target class folder."""
        if not self.manual_images or self.manual_index >= len(self.manual_images):
            return
        
        img_path = self.manual_images[self.manual_index]
        current_class = self.manual_class
        
        if target_class == current_class:
            messagebox.showinfo("Info", "Image is already in this folder")
            return
        
        # Confirm
        if not messagebox.askyesno("Confirm Move", 
            f"Move {img_path.name}\nfrom {current_class}\nto {target_class}?"):
            return
        
        # Move image
        new_path = self.cleaner.move_image(img_path, current_class, target_class)
        
        if new_path:
            # Remove from current list
            self.manual_images.pop(self.manual_index)
            self.moved_count += 1
            
            # Adjust index if at end
            if self.manual_index >= len(self.manual_images):
                self.manual_index = max(0, len(self.manual_images) - 1)
            
            self.update_manual_display()
            self.status_bar.config(text=f"Moved to {target_class}")
        else:
            messagebox.showerror("Error", "Failed to move image")
    
    def delete_image(self):
        """Delete current image."""
        if not self.manual_images or self.manual_index >= len(self.manual_images):
            return
        
        img_path = self.manual_images[self.manual_index]
        
        if not messagebox.askyesno("Confirm Delete", 
            f"Are you sure you want to DELETE\n{img_path.name}?"):
            return
        
        try:
            img_path.unlink()
            self.manual_images.pop(self.manual_index)
            
            if self.manual_index >= len(self.manual_images):
                self.manual_index = max(0, len(self.manual_images) - 1)
            
            self.update_manual_display()
            self.status_bar.config(text=f"Deleted {img_path.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete: {e}")
    
    def next_image(self):
        """Go to next image."""
        if self.manual_images and self.manual_index < len(self.manual_images) - 1:
            self.manual_index += 1
            self.update_manual_display()
    
    def prev_image(self):
        """Go to previous image."""
        if self.manual_images and self.manual_index > 0:
            self.manual_index -= 1
            self.update_manual_display()
    
    def first_image(self):
        """Go to first image."""
        if self.manual_images:
            self.manual_index = 0
            self.update_manual_display()
    
    def last_image(self):
        """Go to last image."""
        if self.manual_images:
            self.manual_index = len(self.manual_images) - 1
            self.update_manual_display()
    
    def jump_to(self):
        """Jump to specific index."""
        try:
            idx = int(self.jump_var.get()) - 1  # 1-based to 0-based
            if 0 <= idx < len(self.manual_images):
                self.manual_index = idx
                self.update_manual_display()
            else:
                messagebox.showerror("Error", f"Index must be between 1 and {len(self.manual_images)}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    # ==================== AUTO MODE ====================
    
    def start_auto_scan(self):
        """Start automatic scan with face and chip detection."""
        if not self.cleaner:
            messagebox.showerror("Error", "Please load dataset first")
            return
        
        # Simple face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        flagged = []
        stats = {'total': 0, 'flagged': 0}
        
        total_images = 0
        for split in ['train', 'val', 'test']:
            for cls in ['cnie_front', 'cnie_back']:
                folder = self.cleaner.dataset_dir / split / cls
                if folder.exists():
                    total_images += len(list(folder.glob('*.jpg')))
        
        processed = 0
        for split in ['train', 'val', 'test']:
            for class_name in ['cnie_front', 'cnie_back']:
                folder = self.cleaner.dataset_dir / split / class_name
                if not folder.exists():
                    continue
                
                is_front = 'front' in class_name
                
                for img_path in folder.glob('*.jpg'):
                    processed += 1
                    stats['total'] += 1
                    
                    # Update progress
                    self.auto_progress['value'] = (processed / total_images) * 100
                    self.auto_status.config(text=f"Scanning {processed}/{total_images}: {img_path.name}")
                    self.root.update_idletasks()
                    
                    # Detect face
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
                    has_face = len(faces) > 0
                    
                    # Simple flag logic
                    flagged_img = False
                    reason = ""
                    suggested = ""
                    
                    if is_front and not has_face:
                        flagged_img = True
                        reason = "Front: No face detected"
                        suggested = "back?"
                    elif not is_front and has_face:
                        flagged_img = True
                        reason = "Back: Face detected"
                        suggested = "front?"
                    
                    if flagged_img:
                        stats['flagged'] += 1
                        flagged.append({
                            'path': str(img_path),
                            'name': img_path.name,
                            'folder': f"{split}/{class_name}",
                            'reason': reason,
                            'suggested': suggested
                        })
        
        # Display results
        self.auto_stats_text.delete('1.0', tk.END)
        self.auto_stats_text.insert('1.0', f"""Auto Scan Complete

Total Images: {stats['total']}
Flagged: {stats['flagged']}
Clean: {stats['total'] - stats['flagged']}

Review flagged images in the table.
""")
        
        # Populate tree
        for item in self.auto_tree.get_children():
            self.auto_tree.delete(item)
        
        for img_info in flagged:
            self.auto_tree.insert('', 'end', values=(
                img_info['name'][:40],
                img_info['folder'],
                img_info['reason'],
                img_info['suggested']
            ))
        
        self.auto_status.config(text=f"Scan complete. {stats['flagged']} images flagged.")


def main():
    root = tk.Tk()
    app = DatasetCleanerUIV2(root)
    root.mainloop()


if __name__ == '__main__':
    main()
