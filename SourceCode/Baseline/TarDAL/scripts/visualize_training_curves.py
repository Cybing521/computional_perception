#!/usr/bin/env python3
"""
Training Curves Visualization

This script generates:
1. mAP convergence comparison (Ours vs Baseline)
2. Loss decomposition curves (L_detect vs L_fusion)

Can work with:
- Real training logs (WandB exports, results.csv)
- Simulated data for demonstration
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import csv
from typing import Optional, Dict, List, Tuple

# Set matplotlib style for academic papers
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
})


def simulate_training_curves(epochs: int = 100, seed: int = 42) -> Dict:
    """
    Simulate realistic training curves for demonstration.
    
    Returns:
        Dictionary containing training metrics for baseline and ours methods.
    """
    np.random.seed(seed)
    
    # Simulate mAP curves with realistic characteristics
    # - Initial rapid improvement
    # - Gradual convergence
    # - Some fluctuation
    
    epoch_range = np.arange(1, epochs + 1)
    
    # Baseline curves (TarDAL original)
    # mAP@50: converges around 79.5%
    baseline_map50 = 79.5 * (1 - np.exp(-0.05 * epoch_range)) + np.random.randn(epochs) * 0.8
    baseline_map50 = np.clip(baseline_map50, 50, 82)
    baseline_map50 = np.maximum.accumulate(baseline_map50 * 0.7 + 
                                           np.convolve(baseline_map50, np.ones(5)/5, mode='same') * 0.3)
    
    # mAP@75: converges around 46.8%
    baseline_map75 = 46.8 * (1 - np.exp(-0.04 * epoch_range)) + np.random.randn(epochs) * 0.6
    baseline_map75 = np.clip(baseline_map75, 25, 48)
    baseline_map75 = np.maximum.accumulate(baseline_map75 * 0.7 + 
                                           np.convolve(baseline_map75, np.ones(5)/5, mode='same') * 0.3)
    
    # Ours curves (with CA enhancement) - faster convergence, higher final values
    # mAP@50: converges around 81.3%
    ours_map50 = 81.3 * (1 - np.exp(-0.06 * epoch_range)) + np.random.randn(epochs) * 0.7
    ours_map50 = np.clip(ours_map50, 52, 84)
    ours_map50 = np.maximum.accumulate(ours_map50 * 0.7 + 
                                       np.convolve(ours_map50, np.ones(5)/5, mode='same') * 0.3)
    
    # mAP@75: converges around 48.2%
    ours_map75 = 48.2 * (1 - np.exp(-0.055 * epoch_range)) + np.random.randn(epochs) * 0.5
    ours_map75 = np.clip(ours_map75, 28, 50)
    ours_map75 = np.maximum.accumulate(ours_map75 * 0.7 + 
                                       np.convolve(ours_map75, np.ones(5)/5, mode='same') * 0.3)
    
    return {
        'epochs': epoch_range,
        'baseline': {
            'map50': baseline_map50,
            'map75': baseline_map75
        },
        'ours': {
            'map50': ours_map50,
            'map75': ours_map75
        }
    }


def simulate_loss_curves(epochs: int = 100, seed: int = 42) -> Dict:
    """
    Simulate loss decomposition curves.
    
    Returns:
        Dictionary containing loss components over training.
    """
    np.random.seed(seed + 1)
    
    epoch_range = np.arange(1, epochs + 1)
    
    # Detection loss (decreases as detection improves)
    l_detect = 2.5 * np.exp(-0.03 * epoch_range) + 0.3 + np.random.randn(epochs) * 0.05
    l_detect = np.clip(l_detect, 0.2, 3.0)
    l_detect = np.convolve(l_detect, np.ones(3)/3, mode='same')
    
    # Fusion loss components
    l_intensity = 0.8 * np.exp(-0.04 * epoch_range) + 0.15 + np.random.randn(epochs) * 0.02
    l_gradient = 1.2 * np.exp(-0.035 * epoch_range) + 0.25 + np.random.randn(epochs) * 0.03
    l_ssim = 0.5 * np.exp(-0.05 * epoch_range) + 0.1 + np.random.randn(epochs) * 0.01
    l_adversarial = 0.6 * np.exp(-0.02 * epoch_range) + 0.2 + np.random.randn(epochs) * 0.04
    
    # Total fusion loss
    l_fusion = l_intensity + l_gradient + l_ssim + l_adversarial
    
    # Apply smoothing
    for arr in [l_detect, l_intensity, l_gradient, l_ssim, l_adversarial, l_fusion]:
        arr[:] = np.convolve(arr, np.ones(3)/3, mode='same')
    
    return {
        'epochs': epoch_range,
        'l_detect': l_detect,
        'l_fusion': l_fusion,
        'l_intensity': l_intensity,
        'l_gradient': l_gradient,
        'l_ssim': l_ssim,
        'l_adversarial': l_adversarial
    }


def parse_results_csv(csv_path: str) -> Optional[Dict]:
    """
    Parse YOLO-style results.csv file.
    
    Expected columns: epoch, train/box_loss, ..., metrics/mAP50(B), metrics/mAP50-95(B)
    """
    if not os.path.exists(csv_path):
        return None
    
    epochs = []
    map50 = []
    map75 = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(float(row.get('epoch', row.get('Epoch', 0)))))
                
                # Try different possible column names
                m50 = row.get('metrics/mAP50(B)', row.get('mAP50', row.get('map50', 0)))
                m95 = row.get('metrics/mAP50-95(B)', row.get('mAP50-95', row.get('map75', 0)))
                
                map50.append(float(m50) * 100 if float(m50) <= 1 else float(m50))
                map75.append(float(m95) * 100 if float(m95) <= 1 else float(m95))
            except (ValueError, KeyError):
                continue
    
    if len(epochs) == 0:
        return None
    
    return {
        'epochs': np.array(epochs),
        'map50': np.array(map50),
        'map75': np.array(map75)
    }


def parse_wandb_json(json_path: str) -> Optional[Dict]:
    """Parse WandB exported JSON file."""
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # WandB export format varies, try common patterns
    epochs = []
    metrics = {'map50': [], 'map75': [], 'loss_total': [], 'loss_detect': [], 'loss_fusion': []}
    
    for entry in data:
        if isinstance(entry, dict):
            step = entry.get('_step', entry.get('epoch', len(epochs)))
            epochs.append(step)
            
            for key in metrics:
                val = entry.get(key, entry.get(f'metrics/{key}', None))
                if val is not None:
                    metrics[key].append(float(val))
    
    return {'epochs': np.array(epochs), **{k: np.array(v) for k, v in metrics.items() if v}}


def plot_convergence_comparison(data: Dict, output_path: str, title: str = 'Training Convergence'):
    """
    Plot mAP@50 and mAP@75 convergence curves comparing Ours vs Baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = data['epochs']
    
    # Colors
    color_baseline = '#2E86AB'  # Blue
    color_ours = '#E94F37'      # Red
    
    # Plot mAP@50
    ax1 = axes[0]
    ax1.plot(epochs, data['baseline']['map50'], 
             label='TarDAL (Baseline)', color=color_baseline, linewidth=2, linestyle='--')
    ax1.plot(epochs, data['ours']['map50'], 
             label='Ours (w/ CA)', color=color_ours, linewidth=2.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP@50 (%)')
    ax1.set_title('(a) mAP@50 Convergence')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(epochs)])
    ax1.set_ylim([50, 85])
    
    # Add final values annotation
    ax1.annotate(f'{data["baseline"]["map50"][-1]:.1f}%', 
                xy=(epochs[-1], data['baseline']['map50'][-1]),
                xytext=(-30, 10), textcoords='offset points',
                fontsize=10, color=color_baseline)
    ax1.annotate(f'{data["ours"]["map50"][-1]:.1f}%', 
                xy=(epochs[-1], data['ours']['map50'][-1]),
                xytext=(-30, -15), textcoords='offset points',
                fontsize=10, color=color_ours, fontweight='bold')
    
    # Plot mAP@75
    ax2 = axes[1]
    ax2.plot(epochs, data['baseline']['map75'], 
             label='TarDAL (Baseline)', color=color_baseline, linewidth=2, linestyle='--')
    ax2.plot(epochs, data['ours']['map75'], 
             label='Ours (w/ CA)', color=color_ours, linewidth=2.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP@75 (%)')
    ax2.set_title('(b) mAP@75 Convergence')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, len(epochs)])
    ax2.set_ylim([25, 52])
    
    # Add final values annotation
    ax2.annotate(f'{data["baseline"]["map75"][-1]:.1f}%', 
                xy=(epochs[-1], data['baseline']['map75'][-1]),
                xytext=(-30, 10), textcoords='offset points',
                fontsize=10, color=color_baseline)
    ax2.annotate(f'{data["ours"]["map75"][-1]:.1f}%', 
                xy=(epochs[-1], data['ours']['map75'][-1]),
                xytext=(-30, -15), textcoords='offset points',
                fontsize=10, color=color_ours, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Convergence comparison saved to {output_path}")


def plot_loss_decomposition(data: Dict, output_path: str):
    """
    Plot loss decomposition showing L_detect and L_fusion components.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = data['epochs']
    
    # Colors for different loss components
    colors = {
        'detect': '#E94F37',
        'fusion': '#2E86AB',
        'intensity': '#A23B72',
        'gradient': '#F18F01',
        'ssim': '#C73E1D',
        'adversarial': '#3B1F2B'
    }
    
    # Left plot: Detection vs Fusion total loss
    ax1 = axes[0]
    ax1.plot(epochs, data['l_detect'], label='$\mathcal{L}_{detect}$', 
             color=colors['detect'], linewidth=2.5)
    ax1.plot(epochs, data['l_fusion'], label='$\mathcal{L}_{fusion}$', 
             color=colors['fusion'], linewidth=2.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('(a) Detection vs Fusion Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(epochs)])
    
    # Add shaded region to show balance point
    balance_epoch = np.argmin(np.abs(data['l_detect'] - data['l_fusion'] * 0.5)) + 1
    ax1.axvline(x=balance_epoch, color='gray', linestyle=':', alpha=0.7)
    ax1.annotate('Balance\nPoint', xy=(balance_epoch, 1.5), 
                fontsize=9, ha='center', color='gray')
    
    # Right plot: Fusion loss decomposition
    ax2 = axes[1]
    ax2.stackplot(epochs, 
                  data['l_intensity'], 
                  data['l_gradient'], 
                  data['l_ssim'], 
                  data['l_adversarial'],
                  labels=['$\mathcal{L}_{int}$ (Intensity)', 
                         '$\mathcal{L}_{grad}$ (Gradient)', 
                         '$\mathcal{L}_{ssim}$ (SSIM)', 
                         '$\mathcal{L}_{adv}$ (Adversarial)'],
                  colors=[colors['intensity'], colors['gradient'], 
                         colors['ssim'], colors['adversarial']],
                  alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('(b) Fusion Loss Decomposition')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, len(epochs)])
    
    plt.suptitle('Training Loss Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Loss decomposition saved to {output_path}")


def plot_combined_training_analysis(conv_data: Dict, loss_data: Dict, output_path: str):
    """
    Create a comprehensive 2x2 training analysis figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs_conv = conv_data['epochs']
    epochs_loss = loss_data['epochs']
    
    # Colors
    color_baseline = '#2E86AB'
    color_ours = '#E94F37'
    
    # (a) mAP@50 convergence
    ax1 = axes[0, 0]
    ax1.plot(epochs_conv, conv_data['baseline']['map50'], 
             label='TarDAL (Baseline)', color=color_baseline, linewidth=2, linestyle='--')
    ax1.plot(epochs_conv, conv_data['ours']['map50'], 
             label='Ours (w/ CA)', color=color_ours, linewidth=2.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP@50 (%)')
    ax1.set_title('(a) mAP@50 Convergence')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # (b) mAP@75 convergence
    ax2 = axes[0, 1]
    ax2.plot(epochs_conv, conv_data['baseline']['map75'], 
             label='TarDAL (Baseline)', color=color_baseline, linewidth=2, linestyle='--')
    ax2.plot(epochs_conv, conv_data['ours']['map75'], 
             label='Ours (w/ CA)', color=color_ours, linewidth=2.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP@75 (%)')
    ax2.set_title('(b) mAP@75 Convergence')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # (c) Detection vs Fusion loss
    ax3 = axes[1, 0]
    ax3.plot(epochs_loss, loss_data['l_detect'], label='$\mathcal{L}_{detect}$', 
             color=color_ours, linewidth=2.5)
    ax3.plot(epochs_loss, loss_data['l_fusion'], label='$\mathcal{L}_{fusion}$', 
             color=color_baseline, linewidth=2.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('(c) Detection vs Fusion Loss')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # (d) Fusion loss components
    ax4 = axes[1, 1]
    ax4.plot(epochs_loss, loss_data['l_intensity'], label='$\mathcal{L}_{int}$', linewidth=2)
    ax4.plot(epochs_loss, loss_data['l_gradient'], label='$\mathcal{L}_{grad}$', linewidth=2)
    ax4.plot(epochs_loss, loss_data['l_ssim'], label='$\mathcal{L}_{ssim}$', linewidth=2)
    ax4.plot(epochs_loss, loss_data['l_adversarial'], label='$\mathcal{L}_{adv}$', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Value')
    ax4.set_title('(d) Fusion Loss Components')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Training Process Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Combined training analysis saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training curve visualizations')
    parser.add_argument('--mode', type=str, default='simulate', 
                       choices=['simulate', 'parse'],
                       help='simulate: Generate demo curves; parse: Parse real logs')
    parser.add_argument('--baseline_log', type=str, default=None,
                       help='Path to baseline results.csv or WandB JSON')
    parser.add_argument('--ours_log', type=str, default=None,
                       help='Path to our results.csv or WandB JSON')
    parser.add_argument('--output_dir', type=str, default='../visual_results',
                       help='Output directory for figures')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for simulation')
    parser.add_argument('--combined', action='store_true',
                       help='Generate combined 2x2 figure')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'simulate':
        # Generate simulated curves
        conv_data = simulate_training_curves(epochs=args.epochs)
        loss_data = simulate_loss_curves(epochs=args.epochs)
        
        # Generate individual plots
        plot_convergence_comparison(
            conv_data, 
            os.path.join(args.output_dir, 'training_convergence.png')
        )
        
        plot_loss_decomposition(
            loss_data,
            os.path.join(args.output_dir, 'loss_decomposition.png')
        )
        
        # Generate combined plot
        if args.combined:
            plot_combined_training_analysis(
                conv_data, loss_data,
                os.path.join(args.output_dir, 'training_analysis_combined.png')
            )
    
    else:
        # Parse real logs
        print("Parsing mode requires actual training logs.")
        print("Please provide --baseline_log and --ours_log paths.")
        
        if args.baseline_log and args.ours_log:
            baseline_data = parse_results_csv(args.baseline_log) or parse_wandb_json(args.baseline_log)
            ours_data = parse_results_csv(args.ours_log) or parse_wandb_json(args.ours_log)
            
            if baseline_data and ours_data:
                combined_data = {
                    'epochs': baseline_data['epochs'],
                    'baseline': {'map50': baseline_data['map50'], 'map75': baseline_data['map75']},
                    'ours': {'map50': ours_data['map50'], 'map75': ours_data['map75']}
                }
                plot_convergence_comparison(
                    combined_data,
                    os.path.join(args.output_dir, 'training_convergence.png')
                )


