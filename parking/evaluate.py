"""检测精度评估工具。

思路：先跑一遍视频，把**每个车位每一帧的原始特征**缓存下来；之后无论改阈值、
改权重还是改滤波参数，都可以在内存里瞬间重算指标，不用反复解码视频。

指标包括：准确率 / 精确率 / 召回率 / F1，分“单帧检测”和“系统输出（含时间滤波）”
两套统计；并给出阈值扫描曲线与逐车位明细，方便定位问题。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DetectorConfig, SystemConfig, TrackerConfig
from .detector import SlotDetector, score_features
from .features import normalize_features
from .slots import SlotSet
from .tracker import SlotTracker


@dataclass
class RawCache:
    """一遍视频扫描后的原始特征缓存。"""

    raw: List[List[Dict[str, float]]] = field(default_factory=list)  # 每帧每车位
    gt: List[List[int]] = field(default_factory=list)                # 0/1/2（2 表示忽略）
    fps: float = 25.0

    def __len__(self) -> int:
        return len(self.raw)


def collect(path: str, slots: SlotSet, detector: SlotDetector,
            gt: Optional[List[List[int]]] = None,
            max_frames: int = 0, progress: bool = True,
            stride: int = 1,
            transform=None) -> RawCache:
    """跑一遍视频，缓存每帧每车位的原始特征。

    ``transform`` 可以传入一个 ``frame -> frame`` 的函数，用来模拟真实的
    画质退化（噪声、模糊、压缩、光照变化等），用于鲁棒性测试。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cache = RawCache(fps=float(fps))
    idx = 0
    used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (idx % stride) != 0:
            idx += 1
            continue
        work = transform(frame) if transform is not None else frame
        det = detector.detect(work, slots)
        cache.raw.append(det.raw)
        if gt is not None and idx < len(gt):
            cache.gt.append(gt[idx])
        else:
            cache.gt.append([-1] * len(slots))
        idx += 1
        used += 1
        if progress and used % 50 == 0:
            print(f"  扫描 {idx}/{total or '?'} 帧 ...", end="\r")
        if max_frames and used >= max_frames:
            break
    cap.release()
    if progress:
        print(f"  扫描完成 {len(cache.raw)} 帧" + " " * 20)
    return cache


# --------------------------------------------------------------------------
# 鲁棒性测试：对画面做退化处理后再检测
# --------------------------------------------------------------------------
def _noise(sigma: float):
    def fn(frame):
        rng = np.random.default_rng(1234)
        noise = rng.normal(0, sigma, frame.shape).astype(np.float32)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return fn


def _blur(ksize: int):
    def fn(frame):
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)
    return fn


def _motion_blur(length: int = 13):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    def fn(frame):
        return cv2.filter2D(frame, -1, k)
    return fn


def _jpeg(quality: int = 28):
    def fn(frame):
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else frame
    return fn


def _brightness(scale: float, bias: float = 0.0):
    def fn(frame):
        return np.clip(frame.astype(np.float32) * scale + bias, 0, 255).astype(np.uint8)
    return fn


def _lowres(scale: float = 0.45):
    def fn(frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(8, int(w * scale)), max(8, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return fn


def _occlusion(ratio: float = 0.18):
    """随机在车位上盖一块深色矩形，模拟行人/购物车/立柱遮挡。"""
    def fn(frame):
        rng = np.random.default_rng(99)
        h, w = frame.shape[:2]
        out = frame.copy()
        for _ in range(int(w * h / 90000)):
            x = int(rng.integers(0, max(1, w - 60)))
            y = int(rng.integers(0, max(1, h - 60)))
            bw = int(rng.integers(30, 70))
            bh = int(rng.integers(40, 120))
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (48, 50, 54), -1)
        return out
    return fn


DEGRADATIONS = [
    ("原始画面", None),
    ("高斯噪声 σ=12", _noise(12)),
    ("高斯模糊 7x7", _blur(7)),
    ("运动模糊 13px", _motion_blur(13)),
    ("JPEG 质量 28", _jpeg(28)),
    ("亮度 -40%", _brightness(0.6)),
    ("亮度 +30%", _brightness(1.3, 12)),
    ("半分辨率", _lowres(0.45)),
    ("随机遮挡", _occlusion(0.18)),
]


def run_robustness(path: str, slots: SlotSet, detector: SlotDetector,
                   gt: List[List[int]], tracker_cfg=None, stride: int = 2,
                   max_frames: int = 0, progress: bool = True) -> List[Dict[str, object]]:
    """逐项退化条件下评估，返回结果表。"""
    rows: List[Dict[str, object]] = []
    for name, fn in DEGRADATIONS:
        if progress:
            print(f"[鲁棒性] {name} ...")
        cache = collect(path, slots, detector, gt=gt, max_frames=max_frames,
                        stride=stride, transform=fn, progress=False)
        m = evaluate_cache(cache, detector.cfg, tracker_cfg)
        rows.append({
            "condition": name,
            "raw_accuracy": m["raw"]["accuracy"],
            "sys_accuracy": m["system"]["accuracy"],
            "sys_precision": m["system"]["precision"],
            "sys_recall": m["system"]["recall"],
            "sys_f1": m["system"]["f1"],
            "samples": m["system"]["samples"],
        })
    return rows


def print_robustness(rows: List[Dict[str, object]]) -> None:
    print()
    print("=" * 78)
    print("  鲁棒性测试（对画面施加退化后的精度）")
    print("=" * 78)
    print(f"  {'测试条件':<16}{'单帧准确率':>12}{'系统准确率':>12}{'精确率':>10}"
          f"{'召回率':>10}{'F1':>8}")
    print("-" * 78)
    for r in rows:
        print(f"  {r['condition']:<16}{r['raw_accuracy']:>12.4f}{r['sys_accuracy']:>12.4f}"
              f"{r['sys_precision']:>10.4f}{r['sys_recall']:>10.4f}{r['sys_f1']:>8.4f}")
    print("=" * 78)



# --------------------------------------------------------------------------
# 指标计算
# --------------------------------------------------------------------------
@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        p = self.tp + self.fp
        return self.tp / p if p else 0.0

    @property
    def recall(self) -> float:
        r = self.tp + self.fn
        return self.tp / r if r else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def add(self, pred: bool, truth: bool) -> None:
        if pred and truth:
            self.tp += 1
        elif pred and not truth:
            self.fp += 1
        elif not pred and not truth:
            self.tn += 1
        else:
            self.fn += 1

    def as_dict(self) -> dict:
        return {
            "samples": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
        }


def evaluate_cache(cache: RawCache, cfg: DetectorConfig,
                   tracker_cfg: Optional[TrackerConfig] = None,
                   ignore: int = 2) -> Dict[str, object]:
    """用缓存的特征重算指标（可任意调整阈值/权重）。"""
    keys = [k for k in cfg.ranges if k in ("edge", "contrast", "corner", "std", "sat", "bgdiff")]
    weights = dict(cfg.weights)
    if not weights.get("bgdiff"):
        weights["bgdiff"] = 0.0

    raw_m = Metrics()
    sys_m = Metrics()
    tracker = SlotTracker(tracker_cfg or TrackerConfig(), n_slots=len(cache.raw[0]) if cache.raw else 0)
    per_slot = [Metrics() for _ in range(len(cache.raw[0]) if cache.raw else 0)]
    misses: List[Tuple[int, int, float]] = []

    for f_idx, frame_features in enumerate(cache.raw):
        scores = []
        for s_idx, raw in enumerate(frame_features):
            norm = normalize_features(raw, cfg.ranges)
            scores.append(score_features(norm, weights))
        states = tracker.update(scores, tracker_cfg)
        truth_row = cache.gt[f_idx] if f_idx < len(cache.gt) else [-1] * len(scores)
        for s_idx, score in enumerate(scores):
            truth = truth_row[s_idx] if s_idx < len(truth_row) else -1
            if truth == ignore or truth < 0:
                continue
            pred_raw = score >= cfg.threshold
            raw_m.add(pred_raw, truth == 1)
            pred_sys = bool(states[s_idx])
            sys_m.add(pred_sys, truth == 1)
            per_slot[s_idx].add(pred_sys, truth == 1)
            if pred_sys != (truth == 1):
                misses.append((f_idx, s_idx + 1, round(scores[s_idx], 3)))

    return {
        "raw": raw_m.as_dict(),
        "system": sys_m.as_dict(),
        "per_slot": [m.as_dict() for m in per_slot],
        "misses": misses[:200],
        "threshold": cfg.threshold,
    }


def sweep_threshold(cache: RawCache, cfg: DetectorConfig,
                    lo: float = 0.20, hi: float = 0.85, steps: int = 27) -> List[Dict[str, float]]:
    """单帧检测的阈值扫描曲线。"""
    curve: List[Dict[str, float]] = []
    for th in np.linspace(lo, hi, steps):
        c = DetectorConfig.from_dict(cfg.to_dict())
        c.threshold = float(th)
        m = evaluate_cache(cache, c, TrackerConfig(window=1, stable=1, enter=0.5, leave=0.5))["raw"]
        curve.append({"threshold": round(float(th), 3), "accuracy": m["accuracy"],
                      "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})
    return curve


# --------------------------------------------------------------------------
# 标定（用真值自动调参）
# --------------------------------------------------------------------------
def calibrate_from_cache(cache: RawCache, detector: SlotDetector,
                         use_ref: bool = True, max_per_class: int = 4000) -> Dict[str, object]:
    """用缓存里的“真值已确定帧”作为训练样本，自动标定区间/权重/阈值。"""
    free_rows: List[Dict[str, float]] = []
    busy_rows: List[Dict[str, float]] = []
    for f_idx, frame_features in enumerate(cache.raw):
        truth_row = cache.gt[f_idx] if f_idx < len(cache.gt) else []
        for s_idx, raw in enumerate(frame_features):
            if s_idx >= len(truth_row):
                continue
            t = truth_row[s_idx]
            if t == 0:
                free_rows.append(raw)
            elif t == 1:
                busy_rows.append(raw)
    if len(free_rows) > max_per_class:
        step = len(free_rows) // max_per_class + 1
        free_rows = free_rows[::step]
    if len(busy_rows) > max_per_class:
        step = len(busy_rows) // max_per_class + 1
        busy_rows = busy_rows[::step]
    report = detector.calibrate(free_rows, busy_rows, use_ref=use_ref)
    report["avg_lo"] = 0
    return report


# --------------------------------------------------------------------------
# 数据集装载
# --------------------------------------------------------------------------
def load_ground_truth(path: str) -> List[List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("gt", [])


def print_report(report: Dict[str, object], title: str = "评估结果") -> None:
    raw = report["raw"]
    sysm = report["system"]
    print()
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)
    print(f"  判定阈值        : {report['threshold']:.3f}")
    print(f"  样本数          : {sysm['samples']}  (帧数 × 车位数)")
    print("-" * 66)
    print(f"  {'指标':<10}{'单帧检测':>14}{'系统输出(含滤波)':>22}")
    for name, key in (("准确率", "accuracy"), ("精确率", "precision"),
                      ("召回率", "recall"), ("F1", "f1")):
        print(f"  {name:<10}{raw[key]:>14.4f}{sysm[key]:>22.4f}")
    print("-" * 66)
    print(f"  混淆矩阵(系统) : TP={sysm['tp']}  FP={sysm['fp']}  "
          f"TN={sysm['tn']}  FN={sysm['fn']}")
    per_slot = report.get("per_slot") or []
    if per_slot:
        accs = [m["accuracy"] for m in per_slot if m["samples"] > 0]
        if accs:
            print(f"  逐车位准确率   : 最低 {min(accs):.3f} / 平均 {np.mean(accs):.3f} / "
                  f"最高 {max(accs):.3f}")
    print("=" * 66)
