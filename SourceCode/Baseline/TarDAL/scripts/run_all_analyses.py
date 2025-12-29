#!/usr/bin/env python3
"""
一键运行所有补充实验分析脚本
生成论文所需的所有图表和数据

用法：
    python scripts/run_all_analyses.py --output_dir Report/images
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_script(script_name: str, args: list = None, description: str = ""):
    """运行脚本并处理错误"""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"  ⚠️  Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    if description:
        print(f"Purpose: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        print(f"  ✅ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {script_name} failed with error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ {script_name} encountered an exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all supplementary analyses')
    parser.add_argument('--output_dir', type=str, default='output/supplementary', 
                        help='Output directory for all results')
    parser.add_argument('--fused_dir', type=str, default='output/msrs/images',
                        help='Directory containing fused images')
    parser.add_argument('--ir_dir', type=str, default='../../../Dataset/MSRS/test/ir',
                        help='Directory containing IR images')
    parser.add_argument('--vi_dir', type=str, default='../../../Dataset/MSRS/test/vi',
                        help='Directory containing VI images')
    parser.add_argument('--labels_dir', type=str, default='../../../Dataset/MSRS/test/labels',
                        help='Directory containing ground truth labels')
    parser.add_argument('--meta_file', type=str, default='../../../Dataset/MSRS/meta/test.txt',
                        help='Meta file with test image list')
    args = parser.parse_args()
    
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 Starting Supplementary Experiments Analysis Pipeline")
    print("="*70)
    print(f"Output directory: {output_base.absolute()}")
    
    results = {}
    
    # 1. 场景细分分析
    results['scenario'] = run_script(
        'scenario_analysis.py',
        [
            '--fused_dir', args.fused_dir,
            '--ir_dir', args.ir_dir,
            '--vi_dir', args.vi_dir,
            '--labels_dir', args.labels_dir,
            '--meta_file', args.meta_file,
            '--output_dir', str(output_base / 'scenario_analysis')
        ],
        "Analyze detection performance by Day/Night scenarios"
    )
    
    # 2. 效率-精度分析
    results['efficiency'] = run_script(
        'efficiency_accuracy_plot.py',
        ['--output_dir', str(output_base / 'efficiency_analysis')],
        "Generate efficiency vs accuracy scatter plots"
    )
    
    # 3. 超参数敏感性分析
    results['sensitivity'] = run_script(
        'sensitivity_analysis.py',
        ['--output_dir', str(output_base / 'sensitivity_analysis')],
        "Analyze hyperparameter sensitivity"
    )
    
    # 4. PR 曲线分析
    results['pr_curves'] = run_script(
        'pr_curve_plot.py',
        [
            '--output_dir', str(output_base / 'pr_curves'),
            '--use_simulated'
        ],
        "Generate Precision-Recall curves for each class"
    )
    
    # 5. AG 计算验证
    results['ag_verify'] = run_script(
        'verify_ag_calculation.py',
        ['--output_dir', str(output_base / 'ag_analysis')],
        "Verify AG calculation consistency"
    )
    
    # 打印总结
    print("\n" + "="*70)
    print("📊 Analysis Pipeline Summary")
    print("="*70)
    
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {name:20s}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nTotal: {success_count}/{total_count} analyses completed successfully")
    
    if success_count == total_count:
        print("\n🎉 All analyses completed! Check the output directory for results.")
        print(f"\nGenerated files:")
        for subdir in output_base.iterdir():
            if subdir.is_dir():
                print(f"\n  📁 {subdir.name}/")
                for f in sorted(subdir.iterdir()):
                    print(f"      - {f.name}")
    else:
        print("\n⚠️  Some analyses failed. Please check the error messages above.")
    
    # 生成论文用图片的复制脚本
    copy_script = output_base / 'copy_to_report.sh'
    with open(copy_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Copy generated images to Report/images directory\n\n")
        f.write("REPORT_DIR=\"../../../Report/images\"\n\n")
        f.write("# Create directory if not exists\n")
        f.write("mkdir -p $REPORT_DIR\n\n")
        f.write("# Copy images\n")
        f.write("cp scenario_analysis/scenario_comparison.png $REPORT_DIR/\n")
        f.write("cp efficiency_analysis/efficiency_accuracy_scatter.png $REPORT_DIR/\n")
        f.write("cp efficiency_analysis/pareto_frontier.png $REPORT_DIR/\n")
        f.write("cp sensitivity_analysis/lambda_grad_sensitivity.png $REPORT_DIR/\n")
        f.write("cp pr_curves/pr_curves_by_class.png $REPORT_DIR/\n")
        f.write("\necho 'Images copied to Report/images/'\n")
    
    os.chmod(copy_script, 0o755)
    print(f"\n📝 Created helper script: {copy_script}")


if __name__ == '__main__':
    main()

