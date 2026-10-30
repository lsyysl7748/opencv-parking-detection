"""停车场状态管理与统计报表。

把“检测 -> 平滑”的结果组织成业务层的停车场状态：
总车位、占用数、空闲数、利用率、进出事件流水、占用率历史曲线，
并支持导出 CSV 报表与 JSON 快照。
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from collections import deque

from .slots import Slot, SlotSet


@dataclass
class Event:
    """一次车位状态变化。"""

    time_s: float
    slot_index: int
    occupied: bool
    source: str = "video"

    @property
    def kind(self) -> str:
        return "驶入" if self.occupied else "驶离"

    def label(self) -> str:
        return f"#{self.slot_index:02d} {self.kind}"

    def stamp(self) -> str:
        t = max(0.0, float(self.time_s))
        m, s = divmod(t, 60)
        return f"{int(m):02d}:{s:05.2f}"


@dataclass
class FrameStats:
    """某一帧的统计结果。"""

    total: int = 0
    occupied: int = 0
    free: int = 0
    rate: float = 0.0
    elapsed_ms: float = 0.0
    fps: float = 0.0


class ParkingManager:
    """停车场业务状态。"""

    def __init__(self, slots: Optional[SlotSet] = None, max_history: int = 8000):
        self.slots: Optional[SlotSet] = slots
        self.events: List[Event] = []
        self.history: Deque[Tuple[float, int]] = deque(maxlen=max_history)
        self._fps_smooth: float = 0.0
        self._t0 = time.time()
        self._prev_states: Optional[List[bool]] = None

    # ---------------- 状态 ----------------
    def bind_slots(self, slots: SlotSet) -> None:
        self.slots = slots

    @property
    def total(self) -> int:
        return len(self.slots) if self.slots else 0

    def update(self, occupied: Sequence[bool], time_s: float,
               elapsed_ms: float = 0.0) -> FrameStats:
        """提交一帧的占用状态；状态变化时自动记录事件。"""
        total = len(occupied)
        occ = sum(1 for v in occupied if v)
        rate = (occ / total) if total else 0.0

        # 用上一帧的状态做比较；第一帧只登记不产生事件，
        # 否则会把“视频开始时已经停着的车”误报成驶入。
        prev = self._prev_states
        if prev is not None and len(prev) == len(occupied):
            for i, flag in enumerate(occupied, start=1):
                if prev[i - 1] != bool(flag):
                    self.events.append(Event(time_s=time_s, slot_index=i,
                                             occupied=bool(flag)))
        self._prev_states = [bool(v) for v in occupied]
        self.history.append((time_s, occ))

        if elapsed_ms > 0:
            inst = 1000.0 / max(elapsed_ms, 1e-3)
            self._fps_smooth = inst if self._fps_smooth == 0 else \
                self._fps_smooth * 0.85 + inst * 0.15

        return FrameStats(
            total=total,
            occupied=occ,
            free=total - occ,
            rate=rate,
            elapsed_ms=elapsed_ms,
            fps=self._fps_smooth,
        )

    def recent_events(self, n: int = 8) -> List[Event]:
        return list(self.events[-n:])[::-1]

    def reset(self) -> None:
        self.events.clear()
        self.history.clear()
        self._fps_smooth = 0.0
        self._t0 = time.time()
        self._prev_states = None

    # ---------------- 导出 ----------------
    def save_snapshot(self, path: str, occupied: Sequence[bool],
                      extra: Optional[dict] = None) -> str:
        """导出当前车位状态为 JSON。"""
        from .config import save_json

        data = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(occupied),
            "occupied": sum(1 for v in occupied if v),
            "free": sum(1 for v in occupied if not v),
            "slots": [
                {"id": i, "occupied": bool(v)} for i, v in enumerate(occupied, start=1)
            ],
            "recent_events": [
                {"time": e.stamp(), "slot": e.slot_index, "kind": e.kind}
                for e in self.recent_events(20)
            ],
            "extra": extra or {},
        }
        save_json(path, data)
        return path

    def export_csv(self, path: str) -> str:
        """导出 CSV 报表：事件流水 + 占用率历史。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["记录类型", "时间(秒)", "时间(hh:mm:ss)", "车位", "事件/占用数", "说明"])
            for e in self.events:
                w.writerow(["事件", f"{e.time_s:.2f}", e.stamp(), f"#{e.slot_index}",
                            "占用" if e.occupied else "空闲", e.kind])
            w.writerow([])
            w.writerow(["占用率历史", "时间(秒)", "", "", "占用车位数", ""])
            for t, occ in self.history:
                w.writerow(["占用率", f"{t:.2f}", _fmt_time(t), "", occ,
                            f"{occ}/{self.total}" if self.total else ""])
            w.writerow([])
            w.writerow(["汇总", "", "", "", "", ""])
            w.writerow(["总车位", self.total, "", "", "", ""])
            occ_now = self.history[-1][1] if self.history else 0
            w.writerow(["当前占用", occ_now, "", "", "", ""])
            w.writerow(["驶入次数", sum(1 for e in self.events if e.occupied), "", "", "", ""])
            w.writerow(["驶离次数", sum(1 for e in self.events if not e.occupied), "", "", "", ""])
        return path

    def summary_text(self) -> str:
        """文字版运行小结（用于命令行输出与报告）。"""
        n_in = sum(1 for e in self.events if e.occupied)
        n_out = sum(1 for e in self.events if not e.occupied)
        span = self.history[-1][0] if self.history else 0.0
        occ_vals = [c for _, c in self.history]
        avg_rate = (sum(occ_vals) / len(occ_vals) / self.total * 100.0) if (occ_vals and self.total) else 0.0
        peak = max(occ_vals) if occ_vals else 0
        lines = [
            f"总车位数      : {self.total}",
            f"分析时长      : {span:.2f} 秒",
            f"平均占用率    : {avg_rate:.1f}%",
            f"峰值占用      : {peak} 辆",
            f"驶入事件      : {n_in} 次",
            f"驶离事件      : {n_out} 次",
        ]
        return "\n".join(lines)


def _fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"
