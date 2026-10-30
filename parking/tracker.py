"""时间维平滑与车位状态机。

单帧得分不可避免会有抖动（车灯反光、行人遮挡、摄像头噪声）。
这里用两级滤波：

1. **滑动窗口中值**——抑制孤立尖峰，比均值更能保留“进出”时的阶跃
2. **迟滞 + 稳定计数**——进入用较高的 ``enter`` 门限、退出用较低的 ``leave``
   门限，且需连续满足 ``stable`` 帧才真正切换，彻底消除临界值附近的状态闪烁
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence

import numpy as np

from .config import TrackerConfig


@dataclass
class SlotState:
    """单个车位的实时状态。"""

    index: int
    occupied: bool = True
    score: float = 0.0        # 当帧原始得分
    smooth: float = 0.0       # 平滑后的得分
    pending: int = 0          # 连续满足切换条件的帧数
    frames_occupied: int = 0
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=15))

    @property
    def confidence(self) -> float:
        """置信度：离判定边界越远越可信。"""
        return float(min(1.0, abs(self.smooth - 0.5) * 2.0))

    def reset(self, window: int = 15) -> None:
        self.history = deque(maxlen=window)
        self.pending = 0
        self.score = 0.0
        self.smooth = 0.0


class SlotTracker:
    """对整组车位做时间维平滑，输出稳定的占用状态。"""

    def __init__(self, cfg: Optional[TrackerConfig] = None, n_slots: int = 0):
        self.cfg = cfg or TrackerConfig()
        self.states: List[SlotState] = []
        self.resize(n_slots)

    # ---------------- 初始化 ----------------
    def resize(self, n_slots: int) -> None:
        if len(self.states) == n_slots:
            return
        self.states = [
            SlotState(index=i + 1, occupied=False, history=deque(maxlen=self.cfg.window))
            for i in range(n_slots)
        ]

    def reset_with(self, occupied: Sequence[bool]) -> None:
        """用给定初始状态重置（例如视频开头已知哪些车位有车）。"""
        self.resize(len(occupied))
        for st, occ in zip(self.states, occupied):
            st.occupied = bool(occ)
            st.pending = 0
            st.history.clear()
            st.history.extend([1.0 if occ else 0.0] * min(3, self.cfg.window))

    # ---------------- 主更新 ----------------
    def update(self, scores: Sequence[float], cfg: Optional[TrackerConfig] = None) -> List[bool]:
        """输入当帧各车位得分，返回平滑后的占用状态。"""
        cfg = cfg or self.cfg
        if cfg.window != self.cfg.window:
            self.cfg.window = cfg.window
            for st in self.states:
                st.history = deque(st.history, maxlen=cfg.window)
        self.cfg = cfg
        self.resize(len(scores))

        for st, raw_score in zip(self.states, scores):
            s = float(np.clip(raw_score, 0.0, 1.0))
            st.score = s
            st.history.append(s)
            window = np.fromiter(st.history, dtype=np.float64, count=len(st.history))
            st.smooth = float(np.median(window))

            threshold = cfg.leave if st.occupied else cfg.enter
            need = st.smooth <= threshold if st.occupied else st.smooth >= threshold
            st.pending = st.pending + 1 if need else 0

            # 前几帧窗口还没填满，降低稳定帧数要求，避免开头反应迟钝
            warmup = max(1, cfg.stable - (cfg.window - len(st.history)))
            if st.pending >= min(cfg.stable, warmup):
                st.occupied = not st.occupied
                st.pending = 0
                st.frames_occupied = 0

            st.frames_occupied = st.frames_occupied + 1 if st.occupied else 0

        return [st.occupied for st in self.states]

    # ---------------- 查询 ----------------
    @property
    def occupied_flags(self) -> List[bool]:
        return [st.occupied for st in self.states]

    def count(self) -> int:
        return sum(1 for st in self.states if st.occupied)

    def smoothed_scores(self) -> List[float]:
        return [st.smooth for st in self.states]

    def confidence_of(self, index: int) -> float:
        for st in self.states:
            if st.index == index:
                return st.confidence
        return 0.0
