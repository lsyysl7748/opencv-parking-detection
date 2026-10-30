"""中文文字绘制工具。

OpenCV 自带的 ``cv2.putText`` 只支持 ASCII（Hershey 字体），无法显示中文。
本模块把一帧内需要显示的所有文字收集起来，最后用 Pillow + 系统中文字体
一次性绘制，既保证了中文正常显示，也避免了逐条转换带来的开销。

若环境中没有 Pillow 或找不到中文字体，会自动退化为 cv2.putText（ASCII）。
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    r"C:\Windows\Fonts\Deng.ttf",      # 等线
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

try:  # Pillow 为可选依赖
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


class _FontCache:
    """按字号缓存字体对象。"""

    def __init__(self) -> None:
        self.path: Optional[str] = self._find_font()
        self._cache: dict = {}

    @staticmethod
    def _find_font() -> Optional[str]:
        for p in _FONT_CANDIDATES:
            if os.path.exists(p):
                return p
        return None

    def get(self, size: int):
        if not _HAS_PIL or self.path is None:
            return None
        size = max(8, int(size))
        if size not in self._cache:
            try:
                self._cache[size] = ImageFont.truetype(self.path, size)
            except Exception:
                self._cache[size] = ImageFont.load_default()
        return self._cache[size]


_FONTS = _FontCache()


def font_available() -> bool:
    return bool(_HAS_PIL and _FONTS.path)


def text_size(text: str, size: int) -> Tuple[int, int]:
    """返回文字在给定字号下的 (宽, 高)。"""
    font = _FONTS.get(size)
    if font is None:
        scale = size / 30.0
        w = int(len(text) * 11 * scale)
        return w, int(size * 1.2)
    box = font.getbbox(text)
    return int(box[2] - box[0]), int(box[3] - box[1] + size * 0.25)


class TextBatch:
    """一帧的文字批处理器。

    用法::

        batch = TextBatch()
        batch.add((20, 20), "空闲车位 12", 18, (120, 220, 140))
        frame = batch.flush(frame)
    """

    def __init__(self) -> None:
        self.items: List[Tuple[Tuple[float, float], str, int,
                                Tuple[int, int, int], str]] = []

    def add(self, xy: Tuple[float, float], text: str, size: int = 16,
            color_bgr: Tuple[int, int, int] = (240, 240, 240),
            anchor: str = "lt") -> None:
        if text is None or text == "":
            return
        self.items.append(((float(xy[0]), float(xy[1])), str(text), int(size),
                           tuple(int(c) for c in color_bgr), anchor))

    def __len__(self) -> int:
        return len(self.items)

    def flush(self, frame: np.ndarray) -> np.ndarray:
        if not self.items:
            return frame
        if not font_available():
            return self._flush_cv(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for (x, y), text, size, color, anchor in self.items:
            font = _FONTS.get(size)
            bbox = font.getbbox(text) if font is not None else (0, 0, 0, 0)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            ax, ay = _anchor_offset(anchor, w, h)
            tx, ty = x + ax - bbox[0], y + ay - bbox[1]
            draw.text((tx, ty), text, font=font, fill=(color[2], color[1], color[0]))
        self.items.clear()
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _flush_cv(self, frame: np.ndarray) -> np.ndarray:
        for (x, y), text, size, color, anchor in self.items:
            ascii_text = text.encode("ascii", "ignore").decode("ascii") or "?"
            scale = max(0.3, size / 28.0)
            (w, h), _ = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            ax, ay = _anchor_offset(anchor, w, h)
            cv2.putText(frame, ascii_text, (int(x + ax), int(y + ay + h)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        self.items.clear()
        return frame


def _anchor_offset(anchor: str, w: int, h: int) -> Tuple[int, int]:
    """把左上角坐标换算成给定锚点的偏移。"""
    ax = 0
    ay = 0
    if "c" in anchor[0]:
        ax = -w // 2
    elif "r" in anchor[0]:
        ax = -w
    if "m" in anchor[1]:
        ay = -h // 2
    elif "b" in anchor[1]:
        ay = -h
    return ax, ay


def put_text(frame: np.ndarray, text: str, xy: Tuple[float, float], size: int = 16,
             color_bgr: Tuple[int, int, int] = (240, 240, 240), anchor: str = "lt"
             ) -> np.ndarray:
    """一次性绘制单行文字（内部仍走批量接口）。"""
    batch = TextBatch()
    batch.add(xy, text, size, color_bgr, anchor)
    return batch.flush(frame)


def draw_panel(frame: np.ndarray, rect: Sequence[int], color_bgr: Tuple[int, int, int],
               alpha: float = 0.72, radius: int = 10) -> np.ndarray:
    """绘制半透明圆角面板（就地修改并返回画面）。"""
    x, y, w, h = [int(v) for v in rect]
    overlay = frame.copy()
    _rounded_rect_fill(overlay, (x, y, w, h), color_bgr, radius)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def _rounded_rect_fill(img: np.ndarray, rect: Sequence[int], color: Tuple[int, int, int],
                       radius: int) -> None:
    x, y, w, h = rect
    r = int(max(0, min(radius, min(w, h) // 2)))
    cv2.rectangle(img, (x + r, y), (x + w - r - 1, y + h - 1), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w - 1, y + h - r - 1), color, -1)
    for cx, cy in ((x + r, y + r), (x + w - r - 1, y + r),
                   (x + r, y + h - r - 1), (x + w - r - 1, y + h - r - 1)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def draw_bar(frame: np.ndarray, rect: Sequence[int], ratio: float,
             fill_bgr: Tuple[int, int, int], track_bgr: Tuple[int, int, int] = (64, 64, 70),
             radius: int = 6) -> np.ndarray:
    """水平进度条（占用率）。"""
    x, y, w, h = [int(v) for v in rect]
    _rounded_rect_fill(frame, (x, y, w, h), track_bgr, radius)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    fw = int(round(w * ratio))
    if fw >= 2:
        _rounded_rect_fill(frame, (x, y, max(fw, 2), h), fill_bgr, min(radius, fw // 2))
    return frame
