"""车位标注工具（OpenCV 窗口交互）。

用法::

    python main.py annotate --image data/lot_reference.png --out data/slots.json

操作说明（窗口内也会显示英文提示）::

    鼠标左键  依次点 4 个角点 -> 自动生成一个车位（顶点顺序自动纠正）
    鼠标右键  撤销当前未完成的点
    u         撤销上一个车位
    d         删除光标所在的车位
    g         以网格方式批量生成车位（需先用 --bay-w/--bay-h/--cols/--rows 指定）
    s         保存并退出
    q / Esc   不保存退出

窗口按屏幕大小自动缩放，鼠标坐标会换算回原图尺寸。
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .slots import Slot, SlotSet, make_grid_slots, order_quad
from .sources import read_image

WINDOW = "Parking Slot Annotator"
MAX_VIEW_W, MAX_VIEW_H = 1280, 760
BAR_H = 62  # 顶部提示条高度


class SlotAnnotator:
    def __init__(self, image: np.ndarray, slots: Optional[SlotSet] = None):
        self.image = image
        self.slots: List[Slot] = list(slots.slots) if slots else []
        self.pending: List[Tuple[int, int]] = []
        self.h, self.w = image.shape[:2]
        # 留出顶部提示条，保证窗口整体不超过屏幕
        self.scale = min(1.0, MAX_VIEW_W / self.w, (MAX_VIEW_H - BAR_H) / self.h)
        self.mouse: Tuple[int, int] = (0, 0)
        self.message = ""
        self._grid = {"bay_w": 100, "bay_h": 210, "cols": 10, "rows": 2,
                      "x0": 110, "y0": 56, "gap_y": 190}

    # ---------------- 坐标换算 ----------------
    def to_original(self, x: int, y: int) -> Tuple[int, int]:
        if self.scale <= 0:
            return x, y
        return (int(round(x / self.scale)), int(round(y / self.scale)))

    # ---------------- 鼠标回调 ----------------
    def on_mouse(self, event: int, x: int, y: int, flags: int, param=None) -> None:
        ox, oy = self.to_original(x, y)
        self.mouse = (ox, oy)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending.append((ox, oy))
            if len(self.pending) == 4:
                pts = order_quad(self.pending)
                self.slots.append(Slot(index=len(self.slots) + 1, points=[tuple(p) for p in pts]))
                self.pending = []
                self.message = "added"
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.pending:
                self.pending.pop()
                self.message = "undo point"
            else:
                # 没有待定点时，右键删除光标下的车位
                for i in range(len(self.slots) - 1, -1, -1):
                    if self.slots[i].contains(ox, oy):
                        self.slots.pop(i)
                        self.message = "deleted"
                        break

    # ---------------- 绘制 ----------------
    def draw(self) -> np.ndarray:
        view = self.image.copy()
        for i, s in enumerate(self.slots, start=1):
            s.index = i
            pts = np.asarray(s.points, np.int32)
            cv2.polylines(view, [pts], True, (70, 220, 120), 2, cv2.LINE_AA)
            c = pts.mean(axis=0).astype(int)
            cv2.putText(view, str(i), (int(c[0]) - 8, int(c[1]) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 220, 120), 2, cv2.LINE_AA)
        # 当前未闭合的点
        for i, p in enumerate(self.pending):
            cv2.circle(view, p, 5, (60, 180, 240), -1, cv2.LINE_AA)
            if i > 0:
                cv2.line(view, self.pending[i - 1], p, (60, 180, 240), 2, cv2.LINE_AA)
        if len(self.pending) == 3:
            cv2.line(view, self.pending[-1], self.pending[0], (60, 180, 240), 1, cv2.LINE_AA)
        cv2.circle(view, self.mouse, 3, (240, 240, 60), -1, cv2.LINE_AA)

        if self.scale < 1.0:
            view = cv2.resize(view, (int(self.w * self.scale), int(self.h * self.scale)),
                              interpolation=cv2.INTER_AREA)

        # 顶部提示条
        bar_h = BAR_H
        bar = np.zeros((bar_h, view.shape[1], 3), np.uint8)
        bar[:] = (32, 32, 36)
        lines = [
            "L-click x4 = add slot   R-click = undo point / delete slot   U = undo slot",
            "G = grid  S = save   Q/ESC = quit      (image %dx%d)" % (self.w, self.h),
        ]
        for i, t in enumerate(lines):
            cv2.putText(bar, t, (12, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (225, 228, 235), 1, cv2.LINE_AA)
        status = f"slots: {len(self.slots)}   pending: {len(self.pending)}   {self.message}"
        cv2.putText(bar, status, (view.shape[1] - 330, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (120, 210, 140), 1, cv2.LINE_AA)
        return np.vstack([bar, view])

    # ---------------- 交互主循环 ----------------
    def run(self) -> List[Slot]:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            view = self.draw()
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and self.slots:
                self.slots.pop()
                self.message = "undo slot"
            elif key == ord("s"):
                break
            elif key == ord("g"):
                self._grid_slots()
                self.message = "grid generated"
        cv2.destroyWindow(WINDOW)
        for i, s in enumerate(self.slots, start=1):
            s.index = i
        return self.slots

    def _grid_slots(self) -> None:
        g = self._grid
        rows = []
        for r in range(g["rows"]):
            base = make_grid_slots(g["x0"], g["y0"] + r * g["gap_y"], g["bay_w"], g["bay_h"],
                                   g["cols"], 1)
            rows.extend(base)
        for i, s in enumerate(rows, start=len(self.slots) + 1):
            s.index = i
        self.slots.extend(rows)


def annotate(image_path: str, out_path: Optional[str] = None,
             slots: Optional[SlotSet] = None) -> Optional[SlotSet]:
    """打开标注窗口，返回标注结果并保存。"""
    image = read_image(image_path)
    if image is None:
        raise SystemExit(f"无法读取图片: {image_path}")
    tool = SlotAnnotator(image, slots)
    result = tool.run()
    h, w = image.shape[:2]
    slotset = SlotSet(result, image_size=(w, h),
                      meta={"source": os.path.basename(image_path)})
    if out_path:
        slotset.save(out_path)
        print(f"已保存 {len(result)} 个车位 -> {out_path}")
    return slotset


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="车位标注工具")
    ap.add_argument("--image", required=True, help="参考图（可在视频首帧上标注）")
    ap.add_argument("--out", default="data/slots.json")
    ap.add_argument("--from", dest="base", default=None, help="在已有标注基础上继续编辑")
    args = ap.parse_args(argv)

    base = SlotSet.load(args.base) if args.base and os.path.exists(args.base) else None
    annotate(args.image, args.out, base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
