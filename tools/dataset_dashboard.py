#!/usr/bin/env python3
"""
Dataset Dashboard
- Shows current dataset statistics
- Visualizes class balance
- Advises on which classes need more data
- Can analyze multiple datasets for comparison
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def scan_dataset(dataset_dir):
    """Scan dataset and return detailed statistics."""
    dataset_path = Path(dataset_dir)
    stats = {
        'total': 0,
        'by_class': defaultdict(int),
        'by_split': defaultdict(lambda: defaultdict(int)),
        'suspicious': []  # Potentially mislabeled
    }
    
    # Try to load face detector for checking
    face_cascade = None
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    except:
        pass
    
    # Scan for images
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = Path(root) / file
                stats['total'] += 1
                
                # Try to determine class from path
                parts = path.parts
                cls = None
                for p in parts:
                    if 'front' in p.lower():
                        cls = 'front'
                        break
                    elif 'back' in p.lower():
                        cls = 'back'
                        break
                    elif 'no_card' in p.lower() or 'nocard' in p.lower():
                        cls = 'no_card'
                        break
                
                if cls:
                    stats['by_class'][cls] += 1
                    
                    # Check split
                    for split in ['train', 'val', 'test']:
                        if split in parts:
                            stats['by_split'][split][cls] += 1
                            break
    
    return stats


def analyze_balance(stats):
    """Analyze class balance and provide recommendations."""
    front = stats['by_class'].get('front', 0)
    back = stats['by_class'].get('back', 0)
    no_card = stats['by_class'].get('no_card', 0)
    total = front + back + no_card
    
    analysis = {
        'total': total,
        'front': front,
        'back': back,
        'no_card': no_card,
        'front_pct': (front / total * 100) if total > 0 else 0,
        'back_pct': (back / total * 100) if total > 0 else 0,
        'no_card_pct': (no_card / total * 100) if total > 0 else 0,
        'imbalance_score': 0,
        'recommendations': []
    }
    
    # Calculate imbalance
    if front > 0 and back > 0:
        ratio = max(front, back) / min(front, back)
        analysis['imbalance_score'] = ratio
        
        if ratio > 1.5:
            if front > back:
                analysis['recommendations'].append(
                    f"⚠️  IMBALANCED: Need {front - back} MORE BACK images (ratio {ratio:.1f}:1)"
                )
            else:
                analysis['recommendations'].append(
                    f"⚠️  IMBALANCED: Need {back - front} MORE FRONT images (ratio {ratio:.1f}:1)"
                )
        else:
            analysis['recommendations'].append(f"✅ Front/Back balance is good (ratio {ratio:.1f}:1)")
    
    # Check minimum counts
    min_recommended = 200
    for cls, count in [('front', front), ('back', back), ('no_card', no_card)]:
        if count < min_recommended:
            needed = min_recommended - count
            analysis['recommendations'].append(
                f"📸 {cls.upper()}: Only {count} images, need {needed} more"
            )
        else:
            analysis['recommendations'].append(
                f"✅ {cls.upper()}: {count} images (sufficient)"
            )
    
    return analysis


def visualize_ascii(stats, analysis):
    """Create ASCII bar chart."""
    max_val = max(stats['by_class'].values()) if stats['by_class'] else 1
    
    lines = []
    lines.append("="*70)
    lines.append("DATASET STATISTICS DASHBOARD")
    lines.append("="*70)
    lines.append("")
    lines.append(f"Total Images: {analysis['total']}")
    lines.append("")
    
    # Bar chart
    bar_width = 50
    for cls in ['front', 'back', 'no_card']:
        count = stats['by_class'].get(cls, 0)
        pct = analysis.get(f'{cls}_pct', 0)
        bar_len = int((count / max(max_val, 1)) * bar_width)
        bar = '█' * bar_len
        lines.append(f"{cls:10} │{bar:<{bar_width}}│ {count:4} ({pct:5.1f}%)")
    
    lines.append("")
    lines.append("-"*70)
    
    # Split breakdown
    if stats['by_split']:
        lines.append("Split Breakdown:")
        for split in ['train', 'val', 'test']:
            if split in stats['by_split']:
                split_stats = stats['by_split'][split]
                f = split_stats.get('front', 0)
                b = split_stats.get('back', 0)
                n = split_stats.get('no_card', 0)
                lines.append(f"  {split:8}: Front={f:3} Back={b:3} NoCard={n:3}")
        lines.append("")
        lines.append("-"*70)
    
    # Recommendations
    lines.append("RECOMMENDATIONS:")
    lines.append("")
    for rec in analysis['recommendations']:
        lines.append(f"  {rec}")
    
    lines.append("")
    lines.append("="*70)
    
    return '\n'.join(lines)


def compare_datasets(dataset_paths):
    """Compare multiple datasets."""
    print("="*70)
    print("DATASET COMPARISON")
    print("="*70)
    print("")
    print(f"{'Dataset':<40} {'Front':>8} {'Back':>8} {'NoCard':>8} {'Total':>8}")
    print("-"*70)
    
    for path in dataset_paths:
        name = Path(path).name[:38]
        stats = scan_dataset(path)
        analysis = analyze_balance(stats)
        
        print(f"{name:<40} {analysis['front']:>8} {analysis['back']:>8} "
              f"{analysis['no_card']:>8} {analysis['total']:>8}")
    
    print("="*70)


def generate_report(dataset_dir, output_file=None):
    """Generate detailed JSON report."""
    stats = scan_dataset(dataset_dir)
    analysis = analyze_balance(stats)
    
    report = {
        'scan_date': datetime.now().isoformat(),
        'dataset_dir': str(dataset_dir),
        'statistics': {
            'total': analysis['total'],
            'by_class': dict(stats['by_class']),
            'by_split': {k: dict(v) for k, v in stats['by_split'].items()},
            'percentages': {
                'front': analysis['front_pct'],
                'back': analysis['back_pct'],
                'no_card': analysis['no_card_pct']
            }
        },
        'analysis': {
            'imbalance_score': analysis['imbalance_score'],
            'recommendations': analysis['recommendations']
        }
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")
    
    return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Statistics Dashboard')
    parser.add_argument('datasets', nargs='+', help='Dataset directory(s) to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare multiple datasets')
    parser.add_argument('--report', help='Output JSON report file')
    parser.add_argument('--watch', action='store_true', help='Watch mode (refresh every 5s)')
    
    args = parser.parse_args()
    
    if args.compare and len(args.datasets) > 1:
        compare_datasets(args.datasets)
        return
    
    dataset_dir = args.datasets[0]
    
    if args.watch:
        # Watch mode - continuously update
        import time
        import os
        
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            
            stats = scan_dataset(dataset_dir)
            analysis = analyze_balance(stats)
            print(visualize_ascii(stats, analysis))
            
            if args.report:
                generate_report(dataset_dir, args.report)
            
            print("\n[Watching... Press Ctrl+C to exit]")
            time.sleep(5)
    else:
        # Single scan
        stats = scan_dataset(dataset_dir)
        analysis = analyze_balance(stats)
        print(visualize_ascii(stats, analysis))
        
        if args.report:
            generate_report(dataset_dir, args.report)


if __name__ == '__main__':
    main()
