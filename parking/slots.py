"""车位几何模型：定义、透视矫正、序列化。

一个车位用一个四点多边形（逆时针或顺时针均可，程序自动排序）描述，
因此既支持正射俯视图的矩形车位，也支持摄像头斜视角下的梯形车位。

核心函数 :func:`warp_slot` 会把任意四边形透视矫正成固定尺寸的正立小图，
后续所有特征都在这个归一化 ROI 上计算，与车位在画面中的角度、远近无关。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """把四个点排成 左上、右上、右下、左下 的顺序。

    使用 x+y 与 y-x 的极值定位，对凸四边形稳定可靠。
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("车位需要 4 个顶点，得到 %d 个" % pts.shape[0])
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmin(d))]
    bl = pts[int(np.argmax(d))]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_size(quad: np.ndarray) -> Tuple[float, float]:
    """返回四边形两条平均边长 (宽, 高)。"""
    q = order_quad(quad)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
    return float(w), float(h)


def orient_portrait(quad: np.ndarray) -> np.ndarray:
    """让归一化 ROI 始终“竖着”：长边成为 ROI 的高度。

    车位大多是竖向长方形（2.4m x 5.3m）。若把横向车位也按同样顺序
    切出来，ROI 会变成扁的，特征分布随之改变。这里统一旋转顶点顺序，
    使长边对应 ROI 的纵向，保证同一套阈值适用于所有车位。
    """
    q = order_quad(quad)
    w, h = quad_size(q)
    if w > h:  # 宽大于高 -> 顺时针滚动一位，长边变高边
        q = np.roll(q, -1, axis=0)
    return q


def shrink_quad(quad: np.ndarray, ratio: float) -> np.ndarray:
    """按比例向内收缩四边形（逐边朝质心收缩）。

    收缩的目的是避开车位白色边界线：边界线本身对比度极高，
    若不裁掉，空车位也会得到很强的边缘/方差响应。
    """
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    c = q.mean(axis=0, keepdims=True)
    return (q + (c - q) * float(ratio)).astype(np.float32)


@dataclass
class Slot:
    """一个车位。"""

    index: int
    points: List[Point]
    label: str = ""

    def __post_init__(self) -> None:
        self.points = [[float(x), float(y)] for x, y in self.points]

    # ---------------- 几何 ----------------
    @property
    def quad(self) -> np.ndarray:
        return order_quad(self.points)

    @property
    def portrait_quad(self) -> np.ndarray:
        return orient_portrait(self.points)

    def center(self) -> Tuple[int, int]:
        c = np.asarray(self.points, dtype=np.float32).mean(axis=0)
        return int(round(c[0])), int(round(c[1]))

    def area(self) -> float:
        return float(cv2.contourArea(np.asarray(self.points, np.float32)))

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(
            np.asarray(self.points, np.float32), (float(x), float(y)), False
        ) >= 0

    def box(self) -> Tuple[int, int, int, int]:
        q = np.asarray(self.points, np.float32)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        return int(x0), int(y0), int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict:
        d = {"id": self.index, "points": [[round(x, 2), round(y, 2)] for x, y in self.points]}
        if self.label:
            d["label"] = self.label
        return d

    @classmethod
    def from_dict(cls, d: dict, fallback_index: int = 0) -> "Slot":
        idx = d.get("id", d.get("index", fallback_index))
        pts = d.get("points") or d.get("polygon")
        if pts is None:
            raise ValueError("车位缺少 points 字段")
        return cls(index=int(idx), points=[tuple(p[:2]) for p in pts], label=d.get("label", ""))


# --------------------------------------------------------------------------
# ROI 提取
# --------------------------------------------------------------------------
class SlotExtractor:
    """把车位的四边形区域透视矫正成归一化 ROI，并自动做内缩裁剪。"""

    def __init__(self, roi_w: int = 96, roi_h: int = 192, inner_margin: float = 0.12,
                 cache: bool = True):
        self.roi_w = int(roi_w)
        self.roi_h = int(roi_h)
        self.inner_margin = float(inner_margin)
        self._cache: dict = {}
        self._use_cache = cache

    def inner_box(self) -> Tuple[int, int, int, int]:
        """归一化 ROI 中真正参与计算的内缩区域 (x, y, w, h)。"""
        dx = int(round(self.roi_w * self.inner_margin))
        dy = int(round(self.roi_h * self.inner_margin))
        return dx, dy, self.roi_w - 2 * dx, self.roi_h - 2 * dy

    def matrix(self, slot: Slot) -> np.ndarray:
        """车位 -> 归一化 ROI 的透视变换矩阵（带缓存）。"""
        key = id(slot)
        if self._use_cache and key in self._cache:
            return self._cache[key]
        src = orient_portrait(slot.points)
        dst = np.array(
            [[0, 0], [self.roi_w - 1, 0], [self.roi_w - 1, self.roi_h - 1], [0, self.roi_h - 1]],
            dtype=np.float32,
        )
        # 直接对内缩后的四边形做变换，等于“矫正 + 裁剪”一步完成
        src_inner = shrink_quad(src, self.inner_margin)
        m = cv2.getPerspectiveTransform(src_inner, dst)
        if self._use_cache:
            self._cache[key] = m
        return m

    def extract(self, image: np.ndarray, slot: Slot) -> np.ndarray:
        """返回该车位的归一化 ROI（BGR，roi_w x roi_h）。"""
        m = self.matrix(slot)
        return cv2.warpPerspective(
            image, m, (self.roi_w, self.roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def inner(self, roi: np.ndarray) -> np.ndarray:
        """从完整归一化 ROI 中取内缩区域（若 ROI 已内缩则直接返回）。"""
        h, w = roi.shape[:2]
        if (w, h) == (self.roi_w, self.roi_h):
            return roi
        return roi

    def polygon_in_image(self, slot: Slot) -> np.ndarray:
        """把归一化 ROI 的四个角反投影回原图，用于调试可视化。"""
        m_inv = np.linalg.inv(self.matrix(slot))
        corners = np.array(
            [[[0, 0]], [[self.roi_w - 1, 0]], [[self.roi_w - 1, self.roi_h - 1]], [[0, self.roi_h - 1]]],
            dtype=np.float32,
        )
        return cv2.perspectiveTransform(corners, m_inv).reshape(-1, 2)


class SlotSet:
    """车位集合：负责保存/加载、查找、采样等。"""

    def __init__(self, slots: Iterable[Slot], image_size: Optional[Tuple[int, int]] = None,
                 reference: Optional[str] = None, meta: Optional[dict] = None):
        self.slots: List[Slot] = list(slots)
        self.image_size = image_size
        self.reference = reference
        self.meta = meta or {}

    def __len__(self) -> int:
        return len(self.slots)

    def __iter__(self):
        return iter(self.slots)

    def __getitem__(self, i: int) -> Slot:
        return self.slots[i]

    def renumber(self) -> None:
        for i, s in enumerate(self.slots, start=1):
            s.index = i

    def draw_polygons(self, image: np.ndarray) -> np.ndarray:
        """调试用：把车位框画到图上。"""
        out = image.copy()
        for s in self.slots:
            pts = np.asarray(s.points, np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 255), 2)
            cx, cy = s.center()
            cv2.putText(out, str(s.index), (cx - 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return out

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "image_size": list(self.image_size) if self.image_size else None,
            "reference": self.reference,
            "meta": self.meta,
            "slots": [s.to_dict() for s in self.slots],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SlotSet":
        raw = d.get("slots") or []
        slots = [Slot.from_dict(s, i + 1) for i, s in enumerate(raw)]
        size = d.get("image_size")
        return cls(
            slots,
            image_size=tuple(size) if size else None,
            reference=d.get("reference"),
            meta=d.get("meta") or {},
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SlotSet":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def make_grid_slots(x0: float, y0: float, bay_w: float, bay_h: float,
                    cols: int, rows: int, gap_x: float = 0.0, gap_y: float = 0.0,
                    vertical: bool = True) -> List[Slot]:
    """按网格批量生成车位（方便没有现成标注时快速起一个布局）。"""
    slots: List[Slot] = []
    idx = 1
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * (bay_w + gap_x)
            y = y0 + r * (bay_h + gap_y)
            if vertical:
                pts = [(x, y), (x + bay_w, y), (x + bay_w, y + bay_h), (x, y + bay_h)]
            else:
                pts = [(x, y), (x + bay_h, y), (x + bay_h, y + bay_w), (x, y + bay_w)]
            slots.append(Slot(index=idx, points=pts))
            idx += 1
    return slots
