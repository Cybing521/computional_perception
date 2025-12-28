
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv
import argparse

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def draw_zoom_in(img, x, y, w, h, zoom_scale=2, color=(0, 0, 255), thickness=2):
    """
    Draw a zoom-in patch on the image corner.
    img: Source image
    x, y, w, h: Region of Interest (ROI) to zoom
    zoom_scale: Resize scale
    """
    h_img, w_img = img.shape[:2]
    
    # Extract ROI
    roi = img[y:y+h, x:x+w]
    zoom_w, zoom_h = int(w * zoom_scale), int(h * zoom_scale)
    roi_zoomed = cv2.resize(roi, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
    
    # Draw rectangle on original ROI
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    
    # Place zoomed patch (bottom-right by default, can be adjusted)
    # Adding a white border
    roi_zoomed = cv2.copyMakeBorder(roi_zoomed, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    # Check placement
    paste_y = h_img - zoom_h - 10
    paste_x = w_img - zoom_w - 10
    
    if paste_y < 0: paste_y = 0
    if paste_x < 0: paste_x = 0
    
    img[paste_y:paste_y+roi_zoomed.shape[0], paste_x:paste_x+roi_zoomed.shape[1]] = roi_zoomed
    
    # Draw arrow (optional, simplified here)
    cv2.line(img, (x+w//2, y+h), (paste_x, paste_y), color, 1)
    
    return img

def plot_comparison_matrix(image_names, src_ir_dir, src_vi_dir, methods_dirs, output_path, roi=None):
    """
    Generate a grid comparison: IR | VI | Method1 | Method2 | Ours
    methods_dirs: dict {'MethodName': 'path/to/images'}
    """
    num_methods = len(methods_dirs)
    cols = 2 + num_methods
    rows = len(image_names)
    
    plt.figure(figsize=(4 * cols, 4 * rows))
    
    for i, name in enumerate(image_names):
        # 1. IR
        ir_path = os.path.join(src_ir_dir, name)
        ir_img = cv2.imread(ir_path)
        if ir_img is None: continue
        if roi: ir_img = draw_zoom_in(ir_img.copy(), *roi)
        
        plt.subplot(rows, cols, i * cols + 1)
        plt.imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
        plt.title('Infrared')
        plt.axis('off')
        
        # 2. VI
        vi_path = os.path.join(src_vi_dir, name)
        vi_img = cv2.imread(vi_path)
        if roi: vi_img = draw_zoom_in(vi_img.copy(), *roi)
        
        plt.subplot(rows, cols, i * cols + 2)
        plt.imshow(cv2.cvtColor(vi_img, cv2.COLOR_BGR2RGB))
        plt.title('Visible')
        plt.axis('off')
        
        # 3. Methods
        for j, (method_name, dir_path) in enumerate(methods_dirs.items()):
            img_path = os.path.join(dir_path, name)
            if not os.path.exists(img_path):
                # Try finding with suffix variants (e.g. 0100N.png vs 0100.png)
                # This is a basic fallback
                img_bg = np.zeros_like(vi_img) 
                img = img_bg
            else:
                img = cv2.imread(img_path)
                
            if roi and img is not None: 
                img = draw_zoom_in(img.copy(), *roi)
                
            plt.subplot(rows, cols, i * cols + 3 + j)
            if img is not None:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(method_name)
            plt.axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comparison matrix saved to {output_path}")

def plot_training_curves(log_file, output_dir):
    """
    Parse a custom log file (assuming specific format) or standard YOLO CSV.
    This is a placeholder for standard YOLO results.csv parsing.
    """
    # Assuming YOLO style results.csv
    # epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    epochs = []
    map50 = []
    map5095 = []
    losses = []
    
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # skip header
        for row in reader:
            try:
                # Adjust indices based on actual CSV format of YOLOv8/v5
                # Usually: index 0 is epoch. mAP50 is often around index 6 or 10.
                # Here we assume a standard YOLOv5/v8 results.csv layout
                # Identify columns by header in real app
                
                # Simple heuristic search for columns
                idx_epoch = 0
                idx_map50 = [i for i, h in enumerate(header) if 'mAP50' in h or 'map50' in h][0]
                idx_map95 = [i for i, h in enumerate(header) if 'mAP50-95' in h or 'map95' in h][0]
                idx_loss = [i for i, h in enumerate(header) if 'val/box_loss' in h or 'val/loss' in h]
                
                epochs.append(float(row[idx_epoch]))
                map50.append(float(row[idx_map50]))
                map5095.append(float(row[idx_map95]))
                
            except Exception as e:
                continue

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, map50, label='mAP@50', linewidth=2)
    plt.plot(epochs, map5095, label='mAP@50:95', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    print(f"Training curves saved to {output_dir}")

def plot_attention_heatmap(model_path, img_ir, img_vi, output_dir):
    """
    Load model, run inference, and plot Coordinate Attention heatmaps.
    Requires torch and the specific model definition.
    """
    import torch
    from module.fuse.generator import Generator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = Generator(dim=32, depth=3).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # Handle different checkpoint formats
    if 'fuse' in checkpoint:
        # Full checkpoint with fuse, disc, detect keys
        model.load_state_dict(checkpoint['fuse'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Preprocess
    def load_tensor(img_path):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (640, 480)) # Assume standard size
        return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        
    t_ir = load_tensor(img_ir)
    t_vi = load_tensor(img_vi)
    
    with torch.no_grad():
        _ = model(t_ir, t_vi)
        
    # Retrieve attention maps
    if hasattr(model.att, 'last_attn_h'):
        att_h = model.att.last_attn_h.squeeze().mean(dim=0).numpy() # (H, 1) -> (H,)
        att_w = model.att.last_attn_w.squeeze().mean(dim=0).numpy() # (1, W) -> (W,)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Original Image (Fused or VI)
        src = cv2.imread(img_vi)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 2, 1)
        plt.imshow(src)
        plt.title("Source Image")
        plt.axis('off')
        
        # Attention H (Vertical) - plot as bar on side
        plt.subplot(2, 2, 2)
        plt.plot(att_h, np.arange(len(att_h))[::-1])
        plt.title("Vertical Attention (Y-axis)")
        plt.xlabel("Weight")
        plt.ylabel("Height")
        
        # Attention W (Horizontal) - plot as bar below
        plt.subplot(2, 2, 3)
        plt.plot(np.arange(len(att_w)), att_w)
        plt.title("Horizontal Attention (X-axis)")
        plt.xlabel("Width")
        plt.ylabel("Weight")
        plt.gca().invert_yaxis()

        # Combined Overlay
        attn_map = att_h[:, None] * att_w[None, :]
        attn_map = cv2.resize(attn_map, (src.shape[1], src.shape[0]))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(src, 0.6, heatmap, 0.4, 0)
        
        plt.subplot(2, 2, 4)
        plt.imshow(overlay)
        plt.title("Attention Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_analysis.png'))
        print(f"Attention analysis saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='matrix', choices=['matrix', 'curve', 'attention'])
    parser.add_argument('--ir_dir', type=str, help='Infrared images dir or file')
    parser.add_argument('--vi_dir', type=str, help='Visible images dir or file')
    parser.add_argument('--ours_dir', type=str, help='Our fused images dir')
    parser.add_argument('--baseline_dir', type=str, help='Baseline fused images dir')
    parser.add_argument('--output_dir', type=str, default='visual_results')
    parser.add_argument('--log_file', type=str, help='Path to results.csv')
    parser.add_argument('--model_path', type=str, help='Path to .pth checkpoint')
    parser.add_argument('--roi', type=str, help='x,y,w,h e.g., "100,200,50,50"')
    parser.add_argument('--img_names', type=str, help='Comma separated image names e.g. "001.png,002.png"')
    
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    
    if args.mode == 'matrix':
        img_names = args.img_names.split(',') if args.img_names else []
        methods = {
            'Baseline': args.baseline_dir,
            'Ours (CA)': args.ours_dir
        }
        roi = [int(x) for x in args.roi.split(',')] if args.roi else None
        
        plot_comparison_matrix(img_names, args.ir_dir, args.vi_dir, methods, 
                               os.path.join(args.output_dir, 'comparison_matrix.png'), roi)
                               
    elif args.mode == 'curve':
        plot_training_curves(args.log_file, args.output_dir)
        
    elif args.mode == 'attention':
        plot_attention_heatmap(args.model_path, args.ir_dir, args.vi_dir, args.output_dir)
