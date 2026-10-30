"""运行报告生成：占用率曲线图、指标 JSON、文字小结。"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .text import TextBatch

PANEL_BG = (30, 31, 36)
GRID = (62, 64, 72)
ACCENT = (120, 200, 255)
ACCENT2 = (110, 210, 140)


def draw_curve(history: Sequence[Tuple[float, int]], total: int, size: Tuple[int, int] = (1000, 420),
               title: str = "占用车位数随时间变化") -> np.ndarray:
    """用 OpenCV 画一张占用率曲线图（不依赖 matplotlib）。"""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = PANEL_BG

    left, right, top, bottom = 90, w - 40, 66, h - 62
    plot_w, plot_h = right - left, bottom - top

    tb = TextBatch()
    tb.add((left, 22), title, 20, (235, 238, 244))

    if not history:
        tb.add((left, top + 40), "无数据", 16, (180, 184, 192))
        return tb.flush(img)

    t_max = max(history[-1][0], 1e-3)
    y_max = max(1, total)

    # 网格与刻度
    for i in range(5):
        y = int(top + plot_h * i / 4)
        cv2.line(img, (left, y), (right, y), GRID, 1, cv2.LINE_AA)
        val = int(round(y_max * (1 - i / 4)))
        tb.add((left - 12, y), str(val), 13, (168, 172, 180), "rm")
    for i in range(6):
        x = int(left + plot_w * i / 5)
        cv2.line(img, (x, top), (x, bottom), GRID, 1, cv2.LINE_AA)
        tb.add((x, bottom + 10), f"{t_max * i / 5:.0f}s", 13, (168, 172, 180), "ct")

    cv2.rectangle(img, (left, top), (right, bottom), (96, 100, 110), 1)

    # 曲线 + 填充
    pts = []
    for t, occ in history:
        x = left + int(plot_w * min(1.0, t / t_max))
        y = bottom - int(plot_h * min(1.0, occ / y_max))
        pts.append((x, y))
    if len(pts) >= 2:
        poly = np.array(pts, np.int32)
        # 填充区域按首尾两点垂直下落到横轴闭合成多边形
        fill = np.vstack([poly, [[poly[-1][0], bottom]], [[poly[0][0], bottom]]])
        shade = img.copy()
        cv2.fillPoly(shade, [fill], (60, 84, 110))
        cv2.addWeighted(shade, 0.55, img, 0.45, 0, dst=img)
        cv2.polylines(img, [poly], False, ACCENT, 2, cv2.LINE_AA)

    # 统计信息
    occ_vals = [c for _, c in history]
    avg = float(np.mean(occ_vals)) if occ_vals else 0.0
    peak = int(max(occ_vals)) if occ_vals else 0
    tb.add((left, bottom + 32),
           f"平均 {avg:.1f} 辆 ({avg / y_max * 100:.0f}%)    峰值 {peak} 辆    总车位 {total}",
           14, (190, 196, 206))
    return tb.flush(img)


def save_metrics(path: str, metrics: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def save_summary(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
