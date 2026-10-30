"""基于纯 OpenCV 的车位特征提取。

设计要点
--------
空车位 = 一片相对均匀的沥青/水泥地面，只有轻微纹理；
有车位 = 车身、车窗、轮胎、车灯、阴影带来的强边缘、强对比、强角点。

因此每个归一化 ROI 上提取 6 个互补的灰度/色彩统计量：

=========  ==================  ==============================================
特征        含义                区分度来源
=========  ==================  ==============================================
edge        Canny 边缘像素占比   车体轮廓与车窗产生大量边缘
contrast    Otsu 最大类间方差    车与地面之间的明暗双峰
corner      Shi-Tomasi 角点密度  车灯、后视镜、车牌、车窗直角
std         灰度标准差           车身明暗反差远大于平整地面
sat         HSV 饱和度离散度     彩色车漆/塑料件的颜色聚集
bgdiff      与空位基准图之差     最强证据，需要一张“空场”参考帧
=========  ==================  ==============================================

全部为经典图像处理方法，不依赖任何深度学习框架。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

FEATURE_KEYS: Tuple[str, ...] = ("edge", "contrast", "corner", "std", "sat", "bgdiff")


# --------------------------------------------------------------------------
# 基础算子
# --------------------------------------------------------------------------
def auto_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    """根据中值亮度自适应确定 Canny 双阈值。

    OpenCV 官方推荐做法：固定阈值在不同光照下会失效，用中值做基准更稳。
    """
    v = float(np.median(gray))
    low = int(max(0.0, (1.0 - sigma) * v))
    high = int(min(255.0, (1.0 + sigma) * v))
    if high <= low:
        high = min(255, low + 20)
    return low, high


def otsu_between_class_variance(gray: np.ndarray, bins: int = 64) -> float:
    """计算直方图的最大类间方差（归一化到 0~0.25）。

    为空车位时直方图近似单峰，类间方差很小；有车时“暗车身 + 亮地面”
    形成双峰，类间方差显著增大。这是最经典的无监督可分性度量。
    """
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    centers = (np.arange(bins) + 0.5) * (256.0 / bins)
    cum_p = np.cumsum(p)
    cum_m = np.cumsum(p * centers)
    mu_total = cum_m[-1]
    best = 0.0
    for t in range(bins - 1):
        w0 = cum_p[t]
        w1 = 1.0 - w0
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        mu0 = cum_m[t] / w0
        mu1 = (mu_total - cum_m[t]) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
        if between > best:
            best = between
    return float(best) / (255.0 * 255.0)


def edge_density(gray: np.ndarray, sigma: float = 0.33) -> float:
    """Canny 边缘像素占 ROI 的比例。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lo, hi = auto_canny_thresholds(blur, sigma)
    edges = cv2.Canny(blur, lo, hi, L2gradient=True)
    return float(np.count_nonzero(edges)) / float(gray.size)


def corner_density(gray: np.ndarray, min_eig: float = 32.0,
                   block: int = 5, ksize: int = 3) -> float:
    """每 1000 像素内“强角点”的个数。

    这里没有用 ``cv2.goodFeaturesToTrack``：它的 ``qualityLevel`` 是相对
    当前图像最大特征值的比例，画面越平整反而越容易凑出一堆弱角点（饱和），
    导致空车位与有车位的角点密度关系被颠倒。改用 ``cornerMinEigenVal`` +
    **绝对阈值** + 连通域计数，物理含义明确、对光照更稳定。
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    eig = cv2.cornerMinEigenVal(blur, blockSize=block, ksize=ksize)
    mask = (eig > float(min_eig)).astype(np.uint8)
    n, _ = cv2.connectedComponents(mask, connectivity=8)
    return max(0, n - 1) * 1000.0 / float(gray.size)


def saturation_std(bgr: np.ndarray) -> float:
    """HSV 饱和度通道的标准差，反映颜色信息的丰富程度。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].std())


def background_diff(gray: np.ndarray, ref_gray: np.ndarray) -> float:
    """与空位基准图的平均绝对差。"""
    if ref_gray is None or ref_gray.shape[:2] != gray.shape[:2]:
        return 0.0
    small_a = cv2.GaussianBlur(gray, (5, 5), 0)
    small_b = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    return float(cv2.absdiff(small_a, small_b).mean())


# --------------------------------------------------------------------------
# 单 ROI 特征向量
# --------------------------------------------------------------------------
def extract_features(
    roi_bgr: np.ndarray,
    ref_roi: Optional[np.ndarray] = None,
    sigma: float = 0.33,
    ref_brightness_ratio: float = 1.0,
) -> Dict[str, float]:
    """对单个车位 ROI 提取全部原始特征。

    参数
    ----
    roi_bgr : 透视矫正后的车位小图（BGR）
    ref_roi : 同一车位在“空场基准图”上的 ROI，可为 None
    ref_brightness_ratio : 当前帧/基准帧的亮度比，用于光照补偿
    """
    if roi_bgr.size == 0:
        return {k: 0.0 for k in FEATURE_KEYS}

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    feats: Dict[str, float] = {
        "edge": edge_density(gray, sigma),
        "contrast": otsu_between_class_variance(gray),
        "corner": corner_density(gray),
        "std": float(gray.std()),
        "sat": saturation_std(roi_bgr),
        "bgdiff": 0.0,
    }

    if ref_roi is not None and ref_roi.size:
        ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)
        k = float(ref_brightness_ratio)
        if abs(k - 1.0) > 0.02:  # 只有明显的光照变化才做补偿，避免引入噪声
            k = min(1.35, max(0.74, k))
            ref_gray = np.clip(ref_gray.astype(np.float32) * k, 0, 255).astype(np.uint8)
        feats["bgdiff"] = background_diff(gray, ref_gray)

    return feats


def normalize_features(
    raw: Dict[str, float],
    ranges: Dict[str, Sequence[float]],
) -> Dict[str, float]:
    """把原始特征按配置区间线性归一化到 [0, 1]。"""
    out: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        val = float(raw.get(key, 0.0))
        lo, hi = ranges.get(key, (0.0, 1.0))
        span = float(hi) - float(lo)
        if abs(span) < 1e-9:
            out[key] = 0.0
        else:
            out[key] = float(np.clip((val - float(lo)) / span, 0.0, 1.0))
    return out


def feature_stats(rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """统计一组特征样本的均值/分位数，用于标定归一化区间与写报告。"""
    stats: Dict[str, Dict[str, float]] = {}
    for key in FEATURE_KEYS:
        vals = np.array([float(r.get(key, 0.0)) for r in rows], dtype=np.float64)
        if vals.size == 0:
            stats[key] = {"mean": 0.0, "min": 0.0, "max": 0.0, "p10": 0.0, "p90": 0.0}
            continue
        stats[key] = {
            "mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
            "p02": float(np.percentile(vals, 2)),
            "p98": float(np.percentile(vals, 98)),
        }
    return stats
