"""检测结果可视化渲染。

负责把“车位 + 状态 + 统计”画成一张可以直接给人看的监控画面：

* 空车位 / 占用车位的半透明色块与描边
* 车位编号标签、单帧得分小色条
* 左上角信息面板：总车位、空闲、占用、利用率进度条、耗时与帧率
* 左下角进出事件滚动条
* 右下角占用率历史曲线

所有英文/数字由 OpenCV 绘制，中文文字统一交给 :mod:`parking.text`（Pillow）。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import SystemConfig, ViewConfig
from .detector import Detection
from .manager import FrameStats
from .slots import Slot, SlotSet
from .text import TextBatch, draw_bar, draw_panel


def _mix(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def _ramp(t: float, low: Tuple[int, int, int], mid: Tuple[int, int, int],
          high: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """三色渐变：0 -> low，0.5 -> mid，1 -> high。"""
    t = float(np.clip(t, 0.0, 1.0))
    return _mix(low, mid, t * 2) if t <= 0.5 else _mix(mid, high, (t - 0.5) * 2)


def _time_str(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


class Renderer:
    """把检测结果渲染成可视化画面。"""

    def __init__(self, view: Optional[ViewConfig] = None):
        self.view = view or ViewConfig()
        self._history: Deque[int] = deque(maxlen=240)

    # ------------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        slots: SlotSet,
        states: Sequence[bool],
        det: Optional[Detection] = None,
        stats: Optional[FrameStats] = None,
        events: Optional[Sequence] = None,
        time_s: float = 0.0,
        title: str = "停车场车位实时监测",
        timestamp: Optional[str] = None,
    ) -> np.ndarray:
        view = self.view
        out = frame.copy()
        h, w = out.shape[:2]
        scores = det.scores if det is not None else [0.0] * len(slots)

        # ---------- 车位色块（一次 alpha 混合完成） ----------
        overlay = out.copy()
        for slot, occ in zip(slots, states):
            color = view.busy_color if occ else view.free_color
            cv2.fillPoly(overlay, [np.asarray(slot.points, np.int32)], color)
        cv2.addWeighted(overlay, view.fill_alpha, out, 1 - view.fill_alpha, 0, dst=out)

        # ---------- 车位描边 ----------
        for slot, occ in zip(slots, states):
            color = view.busy_color if occ else view.free_color
            cv2.polylines(out, [np.asarray(slot.points, np.int32)], True, color, 2, cv2.LINE_AA)

        # ---------- 面板背景 ----------
        panel_w = 408
        panel_h = 146
        draw_panel(out, (16, 16, panel_w, panel_h), view.panel_color, 0.74, 12)
        if view.show_history:
            draw_panel(out, (w - 292, h - 108, 276, 92), view.panel_color, 0.72, 12)
        if events:
            n = min(len(events), 4)
            draw_panel(out, (16, h - 44 - n * 24, 236, 34 + n * 24), view.panel_color, 0.72, 12)

        # ---------- 文字与数值 ----------
        tb = TextBatch()
        txt = view.text_color
        muted = (170, 178, 190)

        tb.add((36, 30), title, 18, txt)
        tb.add((panel_w - 4, 32), timestamp or _time_str(time_s), 13, muted, "rt")
        tb.add((36, 96), "空闲", 14, muted)

        free_n = sum(1 for s in states if not s)
        busy_n = sum(1 for s in states if s)
        total = max(1, len(states))
        rate = busy_n / total

        tb.add((80, 82), str(free_n), 30, view.free_color)
        tb.add((168, 96), "占用", 14, muted)
        tb.add((212, 82), str(busy_n), 30, view.busy_color)
        tb.add((296, 96), "总车位", 14, muted)
        tb.add((352, 88), str(len(states)), 20, txt)
        tb.add((36, 136), f"利用率 {rate * 100:.0f}%", 13, muted)

        if stats is not None:
            tb.add((panel_w - 4, 136),
                   f"{stats.elapsed_ms:.1f} ms/帧 · {stats.fps:.0f} FPS", 13, muted, "rt")

        # 利用率进度条（绿 -> 黄 -> 红 渐变）
        bar_color = _ramp(rate, view.free_color, (60, 200, 235), view.busy_color)
        draw_bar(out, (150, 122, 240, 12), rate, bar_color, radius=6)

        # ---------- 车位编号 ----------
        if view.show_index:
            for slot, occ, sc in zip(slots, states, scores):
                x, y = _label_pos(slot, h)
                color = view.busy_color if occ else view.free_color
                _draw_chip(out, tb, (x, y), str(slot.index), color, sc, view.show_score)

        # ---------- 事件滚动条 ----------
        if events:
            tb.add((30, h - 30 - min(len(events), 4) * 24), "车辆进出", 13, muted)
            for i, ev in enumerate(list(events)[:4]):
                y = h - 12 - (min(len(events), 4) - i) * 24 + 4
                label = f"#{ev.slot_index:02d}  {ev.kind}"
                color = view.busy_color if ev.occupied else view.free_color
                tb.add((36, y), label, 14, color)
                tb.add((150, y), ev.stamp(), 13, muted)

        # ---------- 占比历史曲线 ----------
        if view.show_history:
            self._history.append(busy_n)
            self._draw_history(out, (w - 292, h - 108, 276, 92), total, tb, muted)

        # ---------- 图例 ----------
        _draw_legend(out, tb, (w - 200, 24), view)

        return tb.flush(out)

    # ------------------------------------------------------------------
    def _draw_history(self, frame: np.ndarray, rect: Sequence[int], total: int,
                      tb: TextBatch, muted) -> None:
        x, y, w, h = [int(v) for v in rect]
        tb.add((x + 14, y + 8), "占用数曲线", 12, muted)
        gx0, gy0 = x + 14, y + 62
        gw, gh = w - 28, 26
        cv2.line(frame, (gx0, gy0), (gx0 + gw, gy0), (80, 84, 92), 1, cv2.LINE_AA)
        cv2.line(frame, (gx0, gy0 - gh), (gx0, gy0), (80, 84, 92), 1, cv2.LINE_AA)
        pts = list(self._history)
        if len(pts) < 2:
            return
        n = len(pts)
        step = gw / max(1, n - 1)
        arr = np.array(
            [[int(gx0 + i * step), int(gy0 - (total and p / total or 0) * gh)] for i, p in enumerate(pts)],
            np.int32,
        )
        cv2.polylines(frame, [arr], False, (120, 200, 255), 2, cv2.LINE_AA)
        last = arr[-1]
        cv2.circle(frame, (int(last[0]), int(last[1])), 3, (140, 220, 255), -1, cv2.LINE_AA)


def _draw_chip(frame: np.ndarray, tb: TextBatch, xy: Tuple[int, int], text: str,
               color: Tuple[int, int, int], score: float, show_score: bool) -> None:
    """车位编号小标签：深色底 + 状态色文字 + 可选得分条。"""
    x, y = int(xy[0]), int(xy[1])
    width = 30 if len(text) < 3 else 38
    height = 22
    cv2.rectangle(frame, (x, y), (x + width, y + height), (26, 26, 30), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)
    tb.add((x + width / 2, y + 11), text, 13, color, "cm")
    if show_score:
        bar_w = width - 4
        fill = int(round(bar_w * float(np.clip(score, 0, 1))))
        cv2.rectangle(frame, (x + 2, y + height - 4), (x + 2 + bar_w, y + height - 2),
                      (52, 54, 60), -1)
        if fill > 0:
            cv2.rectangle(frame, (x + 2, y + height - 4), (x + 2 + fill, y + height - 2),
                          color, -1)


def _label_pos(slot: Slot, image_h: int) -> Tuple[int, int]:
    """给编号标签选一个不遮挡车身、且在车位内的位置（贴近入口一侧）。"""
    q = np.asarray(slot.points, np.float32)
    cx, cy = slot.center()
    top_y = float(q[:, 1].min())
    bottom_y = float(q[:, 1].max())
    if cy < image_h * 0.5:
        y = bottom_y - 24          # 上排车位：标签放在入口一侧（下方）
    else:
        y = top_y + 2              # 下排车位：标签放在上方
    return int(cx - 15), int(y)


def _draw_legend(frame: np.ndarray, tb: TextBatch, org: Tuple[int, int], view: ViewConfig) -> None:
    x, y = int(org[0]), int(org[1])
    for i, (label, color) in enumerate((("空闲", view.free_color), ("占用", view.busy_color))):
        px = x + i * 84
        cv2.rectangle(frame, (px, y), (px + 18, y + 18), color, -1)
        tb.add((px + 24, y + 2), label, 14, view.text_color)
