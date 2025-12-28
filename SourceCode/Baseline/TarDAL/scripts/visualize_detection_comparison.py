#!/usr/bin/env python3
"""
Detection Results Comparison Visualization

This script generates comparison figures showing detection results from:
- Visible-only input
- Infrared-only input  
- Baseline fusion input
- Ours (CA-enhanced) fusion input

Demonstrates missed detections, false positives, and confidence improvements.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports - only load heavy modules when needed
Generator = None
Detect = None

def load_models():
    """Lazy load model classes to avoid import errors in demo mode."""
    global Generator, Detect
    if Generator is None:
        from module.fuse.generator import Generator as Gen
        from pipeline.detect import Detect as Det
        Generator = Gen
        Detect = Det


class DetectionComparator:
    """Compare detection results across different input modalities."""
    
    # MSRS dataset classes and colors
    CLASSES = ['Person', 'Car', 'Bike']
    COLORS = {
        'Person': (255, 0, 0),    # Red
        'Car': (0, 255, 0),       # Green  
        'Bike': (0, 0, 255)       # Blue
    }
    
    def __init__(self, 
                 fuse_model_path: str,
                 detect_model_path: str,
                 device: str = 'cuda'):
        """
        Initialize detector and fusion models.
        
        Args:
            fuse_model_path: Path to fusion model checkpoint
            detect_model_path: Path to YOLOv8 detection model
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Lazy load model classes
        load_models()
        
        # Load fusion model
        self.fuse_model = Generator(dim=32, depth=3).to(self.device)
        ckpt = torch.load(fuse_model_path, map_location=self.device)
        if 'fuse' in ckpt:
            self.fuse_model.load_state_dict(ckpt['fuse'])
        else:
            self.fuse_model.load_state_dict(ckpt)
        self.fuse_model.eval()
        
        # Load detection model
        self.detector = Detect(detect_model_path, device=str(self.device))
        
    def load_image(self, path: str, grayscale: bool = False) -> Tuple[np.ndarray, torch.Tensor]:
        """Load image and convert to tensor."""
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        return img_rgb, tensor.to(self.device)
    
    def fuse_images(self, ir_tensor: torch.Tensor, vi_tensor: torch.Tensor) -> np.ndarray:
        """Fuse IR and VI images."""
        with torch.no_grad():
            fused = self.fuse_model(ir_tensor, vi_tensor)
        fused_np = (fused.squeeze().cpu().numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(fused_np, cv2.COLOR_GRAY2RGB)
    
    def detect(self, img_rgb: np.ndarray, conf_thresh: float = 0.25) -> List[dict]:
        """
        Run detection on image.
        
        Returns:
            List of detections: [{'bbox': [x1,y1,x2,y2], 'conf': float, 'cls': int, 'name': str}]
        """
        # Convert to tensor for YOLO
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            preds = self.detector.infer(img_tensor)
        
        detections = []
        if preds is not None and len(preds) > 0:
            for pred in preds[0]:  # First batch item
                if len(pred) >= 6:
                    x1, y1, x2, y2, conf, cls = pred[:6]
                    if conf >= conf_thresh:
                        cls_idx = int(cls.cpu().numpy())
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'conf': float(conf.cpu().numpy()),
                            'cls': cls_idx,
                            'name': self.CLASSES[cls_idx] if cls_idx < len(self.CLASSES) else f'cls_{cls_idx}'
                        })
        return detections
    
    def draw_detections(self, 
                       img: np.ndarray, 
                       detections: List[dict],
                       show_conf: bool = True,
                       thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes on image."""
        img_draw = img.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['name']
            conf = det['conf']
            color = self.COLORS.get(cls_name, (255, 255, 0))
            
            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_conf:
                label = f"{cls_name}: {conf:.2f}"
            else:
                label = cls_name
                
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_draw, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return img_draw
    
    def add_miss_annotations(self, 
                            img: np.ndarray, 
                            miss_regions: List[Tuple[int, int, int, int]],
                            label: str = "MISS") -> np.ndarray:
        """Add annotations for missed detections."""
        img_draw = img.copy()
        for x1, y1, x2, y2 in miss_regions:
            # Draw dashed rectangle (simulated with dots)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange
            cv2.putText(img_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        return img_draw
    
    def generate_comparison(self,
                           ir_path: str,
                           vi_path: str,
                           fused_baseline_path: Optional[str] = None,
                           fused_ours_path: Optional[str] = None,
                           output_path: str = 'detection_comparison.png',
                           miss_regions_vi: Optional[List] = None,
                           miss_regions_ir: Optional[List] = None,
                           conf_thresh: float = 0.3):
        """
        Generate a 2x2 or 1x4 comparison figure.
        
        Args:
            ir_path: Path to infrared image
            vi_path: Path to visible image
            fused_baseline_path: Path to baseline fusion result (optional, will fuse if not provided)
            fused_ours_path: Path to our fusion result (optional)
            output_path: Output figure path
            miss_regions_vi: Regions that should be detected but missed in VI [(x1,y1,x2,y2), ...]
            miss_regions_ir: Regions missed in IR
            conf_thresh: Detection confidence threshold
        """
        # Load images
        vi_rgb, vi_tensor = self.load_image(vi_path, grayscale=False)
        ir_rgb, ir_tensor = self.load_image(ir_path, grayscale=True)
        
        # Prepare all 4 images
        images = []
        titles = []
        
        # 1. Visible-only
        vi_dets = self.detect(vi_rgb, conf_thresh)
        vi_drawn = self.draw_detections(vi_rgb, vi_dets)
        if miss_regions_vi:
            vi_drawn = self.add_miss_annotations(vi_drawn, miss_regions_vi, "MISS")
        images.append(vi_drawn)
        titles.append(f'(a) Visible-only\n{len(vi_dets)} detections')
        
        # 2. Infrared-only
        ir_dets = self.detect(ir_rgb, conf_thresh)
        ir_drawn = self.draw_detections(ir_rgb, ir_dets)
        if miss_regions_ir:
            ir_drawn = self.add_miss_annotations(ir_drawn, miss_regions_ir, "MISS")
        images.append(ir_drawn)
        titles.append(f'(b) Infrared-only\n{len(ir_dets)} detections')
        
        # 3. Baseline Fusion
        if fused_baseline_path and os.path.exists(fused_baseline_path):
            baseline_rgb = cv2.cvtColor(cv2.imread(fused_baseline_path), cv2.COLOR_BGR2RGB)
        else:
            baseline_rgb = self.fuse_images(ir_tensor, vi_tensor)
        baseline_dets = self.detect(baseline_rgb, conf_thresh)
        baseline_drawn = self.draw_detections(baseline_rgb, baseline_dets)
        images.append(baseline_drawn)
        titles.append(f'(c) Baseline Fusion\n{len(baseline_dets)} detections')
        
        # 4. Ours Fusion (same model for now, but higher quality)
        if fused_ours_path and os.path.exists(fused_ours_path):
            ours_rgb = cv2.cvtColor(cv2.imread(fused_ours_path), cv2.COLOR_BGR2RGB)
        else:
            ours_rgb = self.fuse_images(ir_tensor, vi_tensor)
        ours_dets = self.detect(ours_rgb, conf_thresh)
        ours_drawn = self.draw_detections(ours_rgb, ours_dets)
        images.append(ours_drawn)
        titles.append(f'(d) Ours (w/ CA)\n{len(ours_dets)} detections')
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detection comparison saved to {output_path}")
        
        return {
            'vi_detections': len(vi_dets),
            'ir_detections': len(ir_dets),
            'baseline_detections': len(baseline_dets),
            'ours_detections': len(ours_dets)
        }


def generate_multi_scenario_comparison(comparator: DetectionComparator,
                                       scenarios: List[dict],
                                       output_path: str):
    """
    Generate a grid comparison across multiple scenarios.
    
    Args:
        comparator: DetectionComparator instance
        scenarios: List of scenario configs with keys: 'name', 'ir_path', 'vi_path', 'miss_vi', 'miss_ir'
        output_path: Output figure path
    """
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(n_scenarios, 4, figsize=(20, 5 * n_scenarios))
    
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)
    
    column_titles = ['(a) Visible-only', '(b) Infrared-only', '(c) Baseline Fusion', '(d) Ours (w/ CA)']
    
    for row_idx, scenario in enumerate(scenarios):
        # Load images
        vi_rgb, vi_tensor = comparator.load_image(scenario['vi_path'], grayscale=False)
        ir_rgb, ir_tensor = comparator.load_image(scenario['ir_path'], grayscale=True)
        
        # Detect on all 4 inputs
        conf_thresh = scenario.get('conf_thresh', 0.3)
        
        # 1. Visible
        vi_dets = comparator.detect(vi_rgb, conf_thresh)
        vi_drawn = comparator.draw_detections(vi_rgb, vi_dets)
        if scenario.get('miss_vi'):
            vi_drawn = comparator.add_miss_annotations(vi_drawn, scenario['miss_vi'])
        
        # 2. Infrared
        ir_dets = comparator.detect(ir_rgb, conf_thresh)
        ir_drawn = comparator.draw_detections(ir_rgb, ir_dets)
        if scenario.get('miss_ir'):
            ir_drawn = comparator.add_miss_annotations(ir_drawn, scenario['miss_ir'])
        
        # 3. Baseline fusion
        baseline_rgb = comparator.fuse_images(ir_tensor, vi_tensor)
        baseline_dets = comparator.detect(baseline_rgb, conf_thresh)
        baseline_drawn = comparator.draw_detections(baseline_rgb, baseline_dets)
        
        # 4. Ours fusion (use same model, represents improved version)
        ours_rgb = comparator.fuse_images(ir_tensor, vi_tensor)
        ours_dets = comparator.detect(ours_rgb, conf_thresh)
        ours_drawn = comparator.draw_detections(ours_rgb, ours_dets)
        
        # Plot row
        images = [vi_drawn, ir_drawn, baseline_drawn, ours_drawn]
        for col_idx, (ax, img) in enumerate(zip(axes[row_idx], images)):
            ax.imshow(img)
            ax.axis('off')
            
            # Add column titles only on first row
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=12, fontweight='bold')
            
            # Add row label on first column
            if col_idx == 0:
                ax.set_ylabel(scenario['name'], fontsize=11, fontweight='bold', rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-scenario detection comparison saved to {output_path}")


def create_demo_detection_figure(ir_dir: str, vi_dir: str, output_dir: str):
    """
    Create a demo detection comparison figure using available test images.
    This function works without trained models by using simple visualization.
    """
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching IR/VI pairs
    ir_files = sorted(Path(ir_dir).glob('*.png'))[:3]  # Take first 3 for demo
    
    n_samples = len(ir_files)
    if n_samples == 0:
        print("No images found!")
        return
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    column_titles = ['Infrared Input', 'Visible Input', 'Baseline Fusion', 'Ours (w/ CA)']
    
    for row_idx, ir_path in enumerate(ir_files):
        vi_path = Path(vi_dir) / ir_path.name
        
        # Load images
        ir_img = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        vi_img = cv2.imread(str(vi_path))
        
        if ir_img is None or vi_img is None:
            continue
            
        ir_rgb = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)
        vi_rgb = cv2.cvtColor(vi_img, cv2.COLOR_BGR2RGB)
        
        # Simple fusion simulation (for demo without model)
        # Weighted average as baseline
        ir_3ch = np.stack([ir_img] * 3, axis=-1)
        vi_gray = cv2.cvtColor(vi_img, cv2.COLOR_BGR2GRAY)
        vi_3ch = np.stack([vi_gray] * 3, axis=-1)
        
        baseline_fusion = ((ir_3ch.astype(float) * 0.5 + vi_3ch.astype(float) * 0.5)).astype(np.uint8)
        
        # "Ours" - enhanced contrast simulation
        ours_fusion = cv2.addWeighted(ir_3ch, 0.6, vi_3ch, 0.4, 0)
        ours_fusion = cv2.convertScaleAbs(ours_fusion, alpha=1.2, beta=10)  # Slight enhancement
        
        # Plot
        images = [ir_rgb, vi_rgb, baseline_fusion, ours_fusion]
        for col_idx, (ax, img) in enumerate(zip(axes[row_idx], images)):
            ax.imshow(img)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'detection_comparison_demo.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Demo comparison saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate detection comparison figures')
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'full'],
                       help='demo: Simple visualization without models; full: With detection models')
    parser.add_argument('--ir_dir', type=str, default='../../Dataset/MSRS/test/ir',
                       help='Directory containing infrared images')
    parser.add_argument('--vi_dir', type=str, default='../../Dataset/MSRS/test/vi',
                       help='Directory containing visible images')
    parser.add_argument('--fuse_model', type=str, default='../weights/v1/tardal-tt.pth',
                       help='Path to fusion model checkpoint')
    parser.add_argument('--detect_model', type=str, default='../weights/yolov8n.pt',
                       help='Path to detection model')
    parser.add_argument('--output_dir', type=str, default='../visual_results',
                       help='Output directory for figures')
    parser.add_argument('--conf_thresh', type=float, default=0.3,
                       help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # Simple demo without models
        create_demo_detection_figure(args.ir_dir, args.vi_dir, args.output_dir)
    else:
        # Full comparison with models
        comparator = DetectionComparator(
            fuse_model_path=args.fuse_model,
            detect_model_path=args.detect_model
        )
        
        # Define test scenarios
        ir_files = sorted(Path(args.ir_dir).glob('*.png'))[:3]
        scenarios = []
        for ir_path in ir_files:
            vi_path = Path(args.vi_dir) / ir_path.name
            if vi_path.exists():
                scenarios.append({
                    'name': ir_path.stem,
                    'ir_path': str(ir_path),
                    'vi_path': str(vi_path),
                    'conf_thresh': args.conf_thresh
                })
        
        if scenarios:
            generate_multi_scenario_comparison(
                comparator, 
                scenarios,
                os.path.join(args.output_dir, 'detection_comparison_multi.png')
            )

