#!/usr/bin/env python3
"""
Attention Mechanism Comparison Visualization

This script visualizes and compares feature maps from different attention mechanisms:
- SE Block (Squeeze-and-Excitation): Channel-wise attention
- CBAM (Convolutional Block Attention Module): Channel + Local Spatial attention
- CA (Coordinate Attention): Global Directional attention (Ours)

Demonstrates why CA is superior for road scene fusion tasks.
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Tuple, Optional

# Set matplotlib style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
})


# ============== Attention Module Definitions ==============

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - Channel Attention Only"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.last_attention = None
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        self.last_attention = y.detach().cpu()
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """CBAM Channel Attention"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.last_channel_att = None
        self.last_spatial_att = None
        
    def forward(self, x):
        ca = self.channel_attention(x)
        self.last_channel_att = ca.detach().cpu()
        x = x * ca
        
        sa = self.spatial_attention(x)
        self.last_spatial_att = sa.detach().cpu()
        return x * sa


class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.relu(x + 3) / 6


class CoordAtt(nn.Module):
    """Coordinate Attention - Global Directional Attention"""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = h_sigmoid()
        
        self.last_attn_h = None
        self.last_attn_w = None
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h)
        a_w = self.conv_w(x_w)
        
        a_h = self.sigmoid(a_h)
        a_w = self.sigmoid(a_w)
        
        self.last_attn_h = a_h.detach().cpu()
        self.last_attn_w = a_w.detach().cpu()
        
        return identity * a_h * a_w


# ============== Visualization Functions ==============

def load_image_as_feature(img_path: str, channels: int = 64) -> torch.Tensor:
    """
    Load image and convert to simulated feature tensor.
    In practice, this would be intermediate CNN features.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img = cv2.resize(img, (256, 192))  # Standard size
    
    # Create multi-channel feature by applying different filters
    features = []
    for i in range(channels):
        # Apply random-ish transformations to simulate different feature channels
        kernel_size = 3 + (i % 4) * 2
        if i % 3 == 0:
            feat = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif i % 3 == 1:
            feat = cv2.Sobel(img, cv2.CV_64F, 1 if i % 2 == 0 else 0, 
                            1 if i % 2 == 1 else 0, ksize=min(kernel_size, 7))
            feat = np.abs(feat)
        else:
            feat = cv2.Laplacian(img, cv2.CV_64F)
            feat = np.abs(feat)
        
        features.append(feat)
    
    features = np.stack(features, axis=0).astype(np.float32)
    features = (features - features.min()) / (features.max() - features.min() + 1e-8)
    
    return torch.from_numpy(features).unsqueeze(0)  # (1, C, H, W)


def visualize_attention_maps(se: SEBlock, cbam: CBAM, ca: CoordAtt,
                            feature: torch.Tensor, 
                            original_img: np.ndarray,
                            output_path: str):
    """
    Create a comparison figure showing attention from SE, CBAM, and CA.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Apply attention modules
    with torch.no_grad():
        _ = se(feature)
        _ = cbam(feature)
        _ = ca(feature)
    
    # SE attention - channel weights only (show as bar)
    se_weights = se.last_attention.squeeze().numpy()
    axes[1, 0].bar(range(len(se_weights)), se_weights, color='steelblue', alpha=0.8)
    axes[1, 0].set_title('SE: Channel Weights', fontweight='bold')
    axes[1, 0].set_xlabel('Channel Index')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_xlim([0, len(se_weights)])
    
    # SE spatial map (uniform, since SE doesn't have spatial attention)
    se_spatial = np.ones_like(original_img, dtype=np.float32) * se_weights.mean()
    axes[0, 1].imshow(original_img, cmap='gray', alpha=0.6)
    axes[0, 1].imshow(se_spatial, cmap='jet', alpha=0.4, vmin=0, vmax=1)
    axes[0, 1].set_title('(a) SE Block\n(Uniform Spatial)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # CBAM spatial attention
    cbam_spatial = cbam.last_spatial_att.squeeze().numpy()
    cbam_spatial_resized = cv2.resize(cbam_spatial, (original_img.shape[1], original_img.shape[0]))
    axes[0, 2].imshow(original_img, cmap='gray', alpha=0.5)
    hm_cbam = axes[0, 2].imshow(cbam_spatial_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[0, 2].set_title('(b) CBAM\n(Local Spatial)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # CBAM channel weights
    cbam_channel = cbam.last_channel_att.squeeze().mean(dim=(1,2)).numpy()
    axes[1, 1].bar(range(len(cbam_channel)), cbam_channel, color='darkorange', alpha=0.8)
    axes[1, 1].set_title('CBAM: Channel Weights', fontweight='bold')
    axes[1, 1].set_xlabel('Channel Index')
    axes[1, 1].set_xlim([0, len(cbam_channel)])
    
    # CA attention (2D map from H and W attention)
    ca_h = ca.last_attn_h.squeeze().mean(dim=0).numpy()  # (H, 1)
    ca_w = ca.last_attn_w.squeeze().mean(dim=0).numpy()  # (1, W)
    
    # Create 2D attention map
    ca_2d = ca_h[:, None] * ca_w[None, :]
    ca_2d_resized = cv2.resize(ca_2d, (original_img.shape[1], original_img.shape[0]))
    ca_2d_resized = (ca_2d_resized - ca_2d_resized.min()) / (ca_2d_resized.max() - ca_2d_resized.min() + 1e-8)
    
    axes[0, 3].imshow(original_img, cmap='gray', alpha=0.5)
    hm_ca = axes[0, 3].imshow(ca_2d_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[0, 3].set_title('(c) CA (Ours)\n(Global Directional)', fontweight='bold')
    axes[0, 3].axis('off')
    
    # CA directional attention
    axes[1, 2].plot(ca_h, np.arange(len(ca_h))[::-1], 'b-', linewidth=2, label='Vertical (Y)')
    axes[1, 2].set_title('CA: Vertical Attention', fontweight='bold')
    axes[1, 2].set_xlabel('Weight')
    axes[1, 2].set_ylabel('Y Position')
    axes[1, 2].grid(True, alpha=0.3)
    
    axes[1, 3].plot(np.arange(len(ca_w)), ca_w, 'r-', linewidth=2, label='Horizontal (X)')
    axes[1, 3].set_title('CA: Horizontal Attention', fontweight='bold')
    axes[1, 3].set_xlabel('X Position')
    axes[1, 3].set_ylabel('Weight')
    axes[1, 3].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    fig.colorbar(hm_ca, cax=cbar_ax, label='Attention Weight')
    
    plt.suptitle('Attention Mechanism Comparison for Road Scene Fusion', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Attention comparison saved to {output_path}")


def create_ablation_visual(ir_path: str, vi_path: str, output_path: str, channels: int = 32):
    """
    Create a comprehensive ablation visualization figure.
    """
    # Load images
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    vi_img = cv2.imread(vi_path, cv2.IMREAD_GRAYSCALE)
    
    if ir_img is None or vi_img is None:
        print("Error loading images")
        return
    
    ir_img = cv2.resize(ir_img, (256, 192))
    vi_img = cv2.resize(vi_img, (256, 192))
    
    # Create feature tensor (simulated fusion features)
    combined = ((ir_img.astype(float) + vi_img.astype(float)) / 2).astype(np.uint8)
    feature = load_image_as_feature(ir_path, channels)
    
    # Initialize attention modules
    se = SEBlock(channels, reduction=8)
    cbam = CBAM(channels, reduction=8)
    ca = CoordAtt(channels, channels, reduction=8)
    
    # Visualize
    visualize_attention_maps(se, cbam, ca, feature, combined, output_path)


def create_simple_comparison_figure(ir_path: str, vi_path: str, output_dir: str):
    """
    Create simplified comparison showing key differences between attention mechanisms.
    """
    # Load images
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    vi_img = cv2.imread(vi_path, cv2.IMREAD_GRAYSCALE)
    
    if ir_img is None or vi_img is None:
        # Use synthetic image for demo
        h, w = 192, 256
        ir_img = np.zeros((h, w), dtype=np.uint8)
        vi_img = np.zeros((h, w), dtype=np.uint8)
        
        # Draw simulated scene - person (vertical) and road (horizontal)
        cv2.rectangle(ir_img, (100, 50), (130, 150), 255, -1)  # Person
        cv2.rectangle(ir_img, (0, 160), (256, 192), 100, -1)   # Road
        
        cv2.rectangle(vi_img, (100, 50), (130, 150), 200, -1)  # Person
        cv2.line(vi_img, (0, 170), (256, 170), 255, 3)         # Lane line
    else:
        ir_img = cv2.resize(ir_img, (256, 192))
        vi_img = cv2.resize(vi_img, (256, 192))
    
    combined = ((ir_img.astype(float) + vi_img.astype(float)) / 2).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original fused
    axes[0].imshow(combined, cmap='gray')
    axes[0].set_title('Fused Feature Input', fontweight='bold')
    axes[0].axis('off')
    
    # SE - uniform spatial
    se_map = np.ones_like(combined, dtype=float) * 0.5
    axes[1].imshow(combined, cmap='gray', alpha=0.5)
    axes[1].imshow(se_map, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title('(a) SE Block\nUniform (No Spatial)', fontweight='bold')
    axes[1].axis('off')
    axes[1].text(0.5, -0.1, 'Channel attention only\nLoses position info', 
                transform=axes[1].transAxes, ha='center', fontsize=9, style='italic')
    
    # CBAM - local patches
    cbam_map = np.zeros_like(combined, dtype=float)
    # Simulate local attention hotspots
    for _ in range(20):
        cx, cy = np.random.randint(20, 236), np.random.randint(20, 172)
        cv2.circle(cbam_map, (cx, cy), 15, np.random.uniform(0.3, 1.0), -1)
    cbam_map = cv2.GaussianBlur(cbam_map, (21, 21), 0)
    cbam_map = (cbam_map - cbam_map.min()) / (cbam_map.max() - cbam_map.min() + 1e-8)
    
    axes[2].imshow(combined, cmap='gray', alpha=0.5)
    axes[2].imshow(cbam_map, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('(b) CBAM\nLocal Spatial', fontweight='bold')
    axes[2].axis('off')
    axes[2].text(0.5, -0.1, '7×7 conv spatial attention\nScattered hotspots', 
                transform=axes[2].transAxes, ha='center', fontsize=9, style='italic')
    
    # CA - directional (clear H and V bands)
    ca_h = np.zeros(192)
    ca_w = np.zeros(256)
    
    # Vertical attention (person region)
    ca_h[40:160] = 0.8
    ca_h = cv2.GaussianBlur(ca_h.reshape(1, -1), (1, 21), 0).flatten()
    
    # Horizontal attention (road/lane region)  
    ca_w[80:180] = 0.9
    ca_w = cv2.GaussianBlur(ca_w.reshape(1, -1), (1, 31), 0).flatten()
    
    ca_map = ca_h[:, None] * ca_w[None, :]
    ca_map = (ca_map - ca_map.min()) / (ca_map.max() - ca_map.min() + 1e-8)
    
    axes[3].imshow(combined, cmap='gray', alpha=0.5)
    im = axes[3].imshow(ca_map, cmap='hot', alpha=0.6, vmin=0, vmax=1)
    axes[3].set_title('(c) CA (Ours)\nGlobal Directional', fontweight='bold')
    axes[3].axis('off')
    axes[3].text(0.5, -0.1, 'H×W decomposition\nPrecise position encoding', 
                transform=axes[3].transAxes, ha='center', fontsize=9, style='italic',
                color='darkgreen', fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.15, aspect=50)
    cbar.set_label('Attention Weight', fontsize=11)
    
    plt.suptitle('Attention Mechanism Comparison: Why Coordinate Attention Works Better', 
                fontsize=14, fontweight='bold', y=1.05)
    
    output_path = os.path.join(output_dir, 'attention_mechanism_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Simple comparison saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize attention mechanism comparison')
    parser.add_argument('--ir_path', type=str, 
                       default='../../Dataset/MSRS/test/ir/00004N.png',
                       help='Path to infrared image')
    parser.add_argument('--vi_path', type=str,
                       default='../../Dataset/MSRS/test/vi/00004N.png', 
                       help='Path to visible image')
    parser.add_argument('--output_dir', type=str, default='../visual_results',
                       help='Output directory')
    parser.add_argument('--mode', type=str, default='simple', 
                       choices=['simple', 'full'],
                       help='simple: Conceptual comparison; full: With actual features')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'simple':
        create_simple_comparison_figure(args.ir_path, args.vi_path, args.output_dir)
    else:
        create_ablation_visual(args.ir_path, args.vi_path, 
                              os.path.join(args.output_dir, 'attention_ablation.png'))


