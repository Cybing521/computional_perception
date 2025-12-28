"""
图像融合质量评估脚本
计算 SSIM, PSNR, EN(熵), MI(互信息), SF(空间频率), AG(平均梯度), VIF 等指标
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import sobel
import warnings
warnings.filterwarnings('ignore')


def rgb2gray(img):
    """RGB转灰度"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def entropy(img):
    """计算图像熵 (EN)"""
    img = rgb2gray(img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def mutual_information(img1, img2):
    """计算互信息 (MI)"""
    img1 = rgb2gray(img1).flatten()
    img2 = rgb2gray(img2).flatten()
    
    hist_2d, _, _ = np.histogram2d(img1, img2, bins=256)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))


def spatial_frequency(img):
    """计算空间频率 (SF)"""
    img = rgb2gray(img).astype(np.float64)
    
    # 行频率
    RF = np.sqrt(np.mean(np.diff(img, axis=1) ** 2))
    # 列频率
    CF = np.sqrt(np.mean(np.diff(img, axis=0) ** 2))
    
    return np.sqrt(RF ** 2 + CF ** 2)


def average_gradient(img):
    """计算平均梯度 (AG)"""
    img = rgb2gray(img).astype(np.float64)
    
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))


def standard_deviation(img):
    """计算标准差 (SD)"""
    img = rgb2gray(img).astype(np.float64)
    return np.std(img)


def edge_intensity(img):
    """计算边缘强度 (EI)"""
    img = rgb2gray(img).astype(np.float64)
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    return np.mean(np.sqrt(sobelx ** 2 + sobely ** 2))


def calculate_ssim_with_source(fused, ir, vi):
    """计算融合图像与源图像的SSIM"""
    fused_gray = rgb2gray(fused)
    ir_gray = rgb2gray(ir)
    vi_gray = rgb2gray(vi)
    
    ssim_ir = ssim(fused_gray, ir_gray, data_range=255)
    ssim_vi = ssim(fused_gray, vi_gray, data_range=255)
    
    return ssim_ir, ssim_vi, (ssim_ir + ssim_vi) / 2


def calculate_psnr_with_source(fused, ir, vi):
    """计算融合图像与源图像的PSNR"""
    fused_gray = rgb2gray(fused)
    ir_gray = rgb2gray(ir)
    vi_gray = rgb2gray(vi)
    
    psnr_ir = psnr(ir_gray, fused_gray, data_range=255)
    psnr_vi = psnr(vi_gray, fused_gray, data_range=255)
    
    return psnr_ir, psnr_vi, (psnr_ir + psnr_vi) / 2


def Qabf(fused, ir, vi):
    """计算 Qabf 指标 (基于梯度的融合质量)"""
    fused = rgb2gray(fused).astype(np.float64)
    ir = rgb2gray(ir).astype(np.float64)
    vi = rgb2gray(vi).astype(np.float64)
    
    # Sobel算子
    gx_ir = cv2.Sobel(ir, cv2.CV_64F, 1, 0, ksize=3)
    gy_ir = cv2.Sobel(ir, cv2.CV_64F, 0, 1, ksize=3)
    gx_vi = cv2.Sobel(vi, cv2.CV_64F, 1, 0, ksize=3)
    gy_vi = cv2.Sobel(vi, cv2.CV_64F, 0, 1, ksize=3)
    gx_f = cv2.Sobel(fused, cv2.CV_64F, 1, 0, ksize=3)
    gy_f = cv2.Sobel(fused, cv2.CV_64F, 0, 1, ksize=3)
    
    # 梯度幅值
    g_ir = np.sqrt(gx_ir ** 2 + gy_ir ** 2)
    g_vi = np.sqrt(gx_vi ** 2 + gy_vi ** 2)
    g_f = np.sqrt(gx_f ** 2 + gy_f ** 2)
    
    # 简化的Qabf计算
    eps = 1e-10
    Qaf = np.sum(g_ir * g_f) / (np.sum(g_ir ** 2) + eps)
    Qbf = np.sum(g_vi * g_f) / (np.sum(g_vi ** 2) + eps)
    
    # 权重
    w_ir = g_ir / (g_ir + g_vi + eps)
    w_vi = g_vi / (g_ir + g_vi + eps)
    
    Qabf_val = np.mean(w_ir * Qaf + w_vi * Qbf)
    
    return min(Qabf_val, 1.0)


def evaluate_single_image(fused_path, ir_path, vi_path):
    """评估单张融合图像"""
    fused = cv2.imread(str(fused_path))
    ir = cv2.imread(str(ir_path))
    vi = cv2.imread(str(vi_path))
    
    if fused is None or ir is None or vi is None:
        return None
    
    # 确保尺寸一致
    h, w = ir.shape[:2]
    fused = cv2.resize(fused, (w, h))
    vi = cv2.resize(vi, (w, h))
    
    metrics = {}
    
    # SSIM
    ssim_ir, ssim_vi, ssim_avg = calculate_ssim_with_source(fused, ir, vi)
    metrics['SSIM_IR'] = ssim_ir
    metrics['SSIM_VI'] = ssim_vi
    metrics['SSIM_AVG'] = ssim_avg
    
    # PSNR
    psnr_ir, psnr_vi, psnr_avg = calculate_psnr_with_source(fused, ir, vi)
    metrics['PSNR_IR'] = psnr_ir
    metrics['PSNR_VI'] = psnr_vi
    metrics['PSNR_AVG'] = psnr_avg
    
    # 无参考指标
    metrics['EN'] = entropy(fused)  # 熵
    metrics['SF'] = spatial_frequency(fused)  # 空间频率
    metrics['AG'] = average_gradient(fused)  # 平均梯度
    metrics['SD'] = standard_deviation(fused)  # 标准差
    metrics['EI'] = edge_intensity(fused)  # 边缘强度
    
    # 融合质量指标
    metrics['MI_IR'] = mutual_information(fused, ir)  # 与IR的互信息
    metrics['MI_VI'] = mutual_information(fused, vi)  # 与VI的互信息
    metrics['MI_AVG'] = (metrics['MI_IR'] + metrics['MI_VI']) / 2
    metrics['Qabf'] = Qabf(fused, ir, vi)  # Qabf指标
    
    return metrics


def main():
    # 路径配置
    # 路径配置
    current_dir = Path(__file__).parent.resolve()
    fused_dir = current_dir / 'output/msrs/images'
    ir_dir = current_dir.parent.parent / 'Dataset/MSRS/test/ir'
    vi_dir = current_dir.parent.parent / 'Dataset/MSRS/test/vi'
    
    # 获取所有融合图像
    fused_images = sorted(fused_dir.glob('*.png'))
    
    print(f"找到 {len(fused_images)} 张融合图像")
    print("开始计算质量指标...\n")
    
    all_metrics = []
    
    for fused_path in tqdm(fused_images, desc="评估进度"):
        name = fused_path.name
        ir_path = ir_dir / name
        vi_path = vi_dir / name
        
        if not ir_path.exists() or not vi_path.exists():
            continue
        
        metrics = evaluate_single_image(fused_path, ir_path, vi_path)
        if metrics:
            metrics['name'] = name
            all_metrics.append(metrics)
    
    # 计算平均值
    if all_metrics:
        print("\n" + "=" * 60)
        print("图像融合质量评估结果")
        print("=" * 60)
        
        metric_names = ['SSIM_IR', 'SSIM_VI', 'SSIM_AVG', 'PSNR_IR', 'PSNR_VI', 'PSNR_AVG',
                        'EN', 'SF', 'AG', 'SD', 'EI', 'MI_IR', 'MI_VI', 'MI_AVG', 'Qabf']
        
        print(f"\n{'指标':<12} {'平均值':<12} {'最小值':<12} {'最大值':<12}")
        print("-" * 50)
        
        results = {}
        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            avg_val = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            results[metric] = {'avg': float(avg_val), 'min': float(min_val), 'max': float(max_val)}
            print(f"{metric:<12} {avg_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
        
        print("\n" + "=" * 60)
        print("指标说明:")
        print("-" * 60)
        print("SSIM: 结构相似性 (越高越好, 范围0-1)")
        print("PSNR: 峰值信噪比 (越高越好, 单位dB)")
        print("EN:   熵 (越高表示信息量越大)")
        print("SF:   空间频率 (越高表示细节越丰富)")
        print("AG:   平均梯度 (越高表示清晰度越好)")
        print("SD:   标准差 (越高表示对比度越好)")
        print("EI:   边缘强度 (越高表示边缘越清晰)")
        print("MI:   互信息 (越高表示保留源图像信息越多)")
        print("Qabf: 融合质量指标 (越高越好, 范围0-1)")
        print("=" * 60)
        
        # 保存结果
        import json
        result_path = fused_dir.parent / 'evaluation_results.json'
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n评估结果已保存到: {result_path}")


if __name__ == '__main__':
    main()




