"""多特征融合的车位占用检测器（纯 OpenCV）。

流程
----
1. 取帧、可选的光照归一化（与空位基准图比较）
2. 逐车位透视矫正 -> 归一化 ROI
3. 提取 6 个原始特征
4. 按标定区间归一化到 [0,1]，按权重加权求和得到占用得分 score
5. score >= threshold 判定为“占用”

另外提供 :meth:`SlotDetector.calibrate`，用带标签的样本自动标定
归一化区间、权重档位与判定阈值（只用 numpy 做网格搜索，不依赖任何 ML 库）。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DetectorConfig, WEIGHTS_DEFAULT, WEIGHTS_WITH_REF
from .features import FEATURE_KEYS, extract_features, normalize_features
from .slots import Slot, SlotExtractor, SlotSet


@dataclass
class Detection:
    """一帧的检测输出。"""

    scores: List[float] = field(default_factory=list)
    raw: List[Dict[str, float]] = field(default_factory=list)
    norm: List[Dict[str, float]] = field(default_factory=list)
    rois: List[np.ndarray] = field(default_factory=list)
    brightness_ratio: float = 1.0

    def occupied(self, threshold: float) -> List[bool]:
        return [s >= threshold for s in self.scores]

    def summary(self) -> str:
        if not self.scores:
            return "无车位"
        arr = np.asarray(self.scores)
        return f"得分 min={arr.min():.2f} 均值={arr.mean():.2f} max={arr.max():.2f}"


def score_features(norm: Dict[str, float], weights: Dict[str, float]) -> float:
    """加权融合归一化特征。权重为 0 的特征自动跳过并重新归一化。"""
    num = 0.0
    den = 0.0
    for key in FEATURE_KEYS:
        w = float(weights.get(key, 0.0))
        if w <= 0:
            continue
        num += w * float(norm.get(key, 0.0))
        den += w
    return float(num / den) if den > 0 else 0.0


class SlotDetector:
    """针对一组固定车位的检测器。"""

    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.extractor = SlotExtractor(
            self.cfg.roi_w, self.cfg.roi_h, self.cfg.inner_margin
        )
        self._ref_image: Optional[np.ndarray] = None
        self._ref_rois: List[np.ndarray] = []
        self._ref_brightness: float = 0.0
        self._slot_ids: List[int] = []

    # ---------------- 基准图 ----------------
    @property
    def has_reference(self) -> bool:
        return self._ref_image is not None

    def set_reference(self, image: np.ndarray, slots: SlotSet) -> None:
        """设置空车位基准图（停车场全空时拍的一张背景图）。"""
        self._ref_image = image.copy()
        self._ref_rois = [self.extractor.extract(image, s) for s in slots]
        self._ref_brightness = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        self._slot_ids = [s.index for s in slots]

    def clear_reference(self) -> None:
        self._ref_image = None
        self._ref_rois = []
        self._ref_brightness = 0.0

    def _frame_brightness(self, image: np.ndarray) -> float:
        small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

    # ---------------- 主检测 ----------------
    def detect(self, image: np.ndarray, slots: SlotSet,
               keep_rois: bool = False) -> Detection:
        cfg = self.cfg
        res = Detection()

        use_ref = bool(
            cfg.use_reference and self.has_reference and len(self._ref_rois) == len(slots)
        )
        ratio = 1.0
        if use_ref:
            ref_br = self._ref_brightness
            cur_br = self._frame_brightness(image) if cfg.normalize_light else ref_br
            if ref_br > 1e-3:
                ratio = cur_br / ref_br
        res.brightness_ratio = ratio

        weights = dict(cfg.weights)
        if not use_ref:
            weights["bgdiff"] = 0.0  # 没有基准图时禁用该通道

        raws: List[Dict[str, float]] = []
        for i, slot in enumerate(slots):
            roi = self.extractor.extract(image, slot)
            ref_roi = self._ref_rois[i] if use_ref else None
            f = extract_features(roi, ref_roi, cfg.canny_sigma, ratio)
            raws.append(f)
            if keep_rois:
                res.rois.append(roi.copy())

        # 背景差的相对化：减去全场地面的中位数，抑制整体光照变化 / 雨雾影响
        if use_ref and raws:
            abs_diffs = np.array([r.get("bgdiff", 0.0) for r in raws], dtype=np.float64)
            base = float(np.median(abs_diffs))
            for r in raws:
                r["bgdiff_abs"] = r.get("bgdiff", 0.0)
                r["bgdiff"] = max(0.0, r.get("bgdiff", 0.0) - base)

        for r in raws:
            n = normalize_features(r, cfg.ranges)
            res.raw.append(r)
            res.norm.append(n)
            res.scores.append(score_features(n, weights))

        return res

    # ---------------- 自动标定 ----------------
    def calibrate(
        self,
        free_rows: Sequence[Dict[str, float]],
        busy_rows: Sequence[Dict[str, float]],
        use_ref: bool = False,
        fit_weights: bool = True,
    ) -> Dict[str, object]:
        """用“空车位 / 有车”两类样本自动标定参数。

        步骤
        ----
        1. 每个特征的归一化区间取 (空位 P85, 有车 P15)；若两类不可分则自动放宽
        2. 在若干候选权重档位中网格搜索，选训练集准确率最高的组合
        3. 在得分轴上扫描阈值，选 F1 最大的那个（同时兼顾准确率）

        返回标定报告（准确率、阈值、各特征统计），并就地更新 ``self.cfg``。
        """
        keys = [k for k in FEATURE_KEYS if not (k == "bgdiff" and not use_ref)]
        ranges: Dict[str, Tuple[float, float]] = {}
        report: Dict[str, object] = {}

        for k in keys:
            f_vals = np.array([float(r.get(k, 0.0)) for r in free_rows], dtype=np.float64)
            b_vals = np.array([float(r.get(k, 0.0)) for r in busy_rows], dtype=np.float64)
            if f_vals.size == 0 or b_vals.size == 0:
                ranges[k] = self.cfg.ranges.get(k, (0.0, 1.0))
                continue
            lo = float(np.percentile(f_vals, 85))
            hi = float(np.percentile(b_vals, 15))
            if hi <= lo:  # 两类在该特征上重叠，退化为分位数中段
                lo = float(np.percentile(f_vals, 70))
                hi = float(np.percentile(b_vals, 30))
            if hi <= lo:
                hi = lo + max(1e-6, abs(lo) * 0.25 + 1e-6)
            ranges[k] = (lo, hi)
            report[k] = {
                "free_mean": float(f_vals.mean()),
                "busy_mean": float(b_vals.mean()),
                "lo": lo,
                "hi": hi,
            }

        # 候选权重档位
        candidates: List[Dict[str, float]] = []
        base = dict(WEIGHTS_WITH_REF if use_ref else WEIGHTS_DEFAULT)
        candidates.append({k: base.get(k, 0.0) for k in FEATURE_KEYS})
        candidates.append({k: (1.0 if k in keys else 0.0) for k in FEATURE_KEYS})
        for k in keys:  # 单特征 + 主特征加强
            cand = {kk: (0.15 if kk in keys else 0.0) for kk in FEATURE_KEYS}
            cand[k] = 0.85 if k == "bgdiff" else 0.50
            candidates.append(cand)

        best = None
        for weights in candidates:
            nf = [normalize_features(r, ranges) for r in free_rows]
            nb = [normalize_features(r, ranges) for r in busy_rows]
            sf = np.array([score_features(n, weights) for n in nf], dtype=np.float64)
            sb = np.array([score_features(n, weights) for n in nb], dtype=np.float64)
            th, metrics = _best_threshold(sf, sb)
            if best is None or metrics["f1"] > best[0]["f1"] + 1e-6:
                best = (metrics, weights, th, sf, sb)

        assert best is not None
        metrics, weights, threshold, _, _ = best
        self.cfg.ranges.update({k: tuple(v) for k, v in ranges.items()})
        if fit_weights:
            self.cfg.weights.update(weights)
        self.cfg.threshold = float(threshold)
        if use_ref:
            self.cfg.weights["bgdiff"] = max(self.cfg.weights.get("bgdiff", 0.0), 0.0)
        else:
            self.cfg.weights["bgdiff"] = 0.0

        report.update(
            {
                "threshold": float(threshold),
                "weights": {k: float(v) for k, v in self.cfg.weights.items()},
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "n_free": len(free_rows),
                "n_busy": len(busy_rows),
            }
        )
        return report


def _best_threshold(scores_free: np.ndarray, scores_busy: np.ndarray,
                    grid: int = 200) -> Tuple[float, Dict[str, float]]:
    """在 [0,1] 上扫描阈值，返回 F1 最大且准确率不低于最优的点。"""
    if scores_free.size == 0 and scores_busy.size == 0:
        return 0.5, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_th, best_metrics = 0.5, {"accuracy": -1.0, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for th in np.linspace(0.02, 0.98, grid):
        tp = int((scores_busy >= th).sum())
        fp = int((scores_free >= th).sum())
        fn = int((scores_busy < th).sum())
        tn = int((scores_free < th).sum())
        total = tp + fp + fn + tn
        if total == 0:
            continue
        acc = (tp + tn) / total
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if (f1, acc) > (best_metrics["f1"], best_metrics["accuracy"]):
            best_th = float(th)
            best_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # 若两类在得分轴上完全分开，取分隔带中点，使判定阈值拥有最大裕度
    if scores_free.size and scores_busy.size:
        hi_free = float(scores_free.max())
        lo_busy = float(scores_busy.min())
        if hi_free < lo_busy:
            best_th = (hi_free + lo_busy) / 2.0
    return best_th, best_metrics
