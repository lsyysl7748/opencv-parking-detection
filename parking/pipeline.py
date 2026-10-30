"""检测流水线：把检测器、跟踪器、业务状态与渲染串起来。

命令行模式、批处理模式和图形界面共用同一套流程，保证三种入口的结果一致：

    读帧 -> 逐车位特征 -> 多特征融合评分 -> 时间维平滑 -> 统计/事件 -> 渲染
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import SystemConfig
from .detector import Detection, SlotDetector
from .manager import FrameStats, ParkingManager
from .render import Renderer
from .slots import SlotSet
from .tracker import SlotTracker


@dataclass
class AnalysisResult:
    """一帧的完整分析结果。"""

    states: List[bool] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    detection: Optional[Detection] = None
    stats: Optional[FrameStats] = None
    time_s: float = 0.0

    @property
    def free_count(self) -> int:
        return sum(1 for s in self.states if not s)

    @property
    def busy_count(self) -> int:
        return sum(1 for s in self.states if s)

    def free_indices(self) -> List[int]:
        return [i + 1 for i, s in enumerate(self.states) if not s]


class ParkingAnalyzer:
    """面向一组固定车位的完整分析器。"""

    def __init__(self, slots: SlotSet, config: Optional[SystemConfig] = None,
                 reference: Optional[np.ndarray] = None, title: str = "停车场车位实时监测"):
        self.config = config or SystemConfig()
        self.title = title
        self.detector = SlotDetector(self.config.detector)
        self.tracker = SlotTracker(self.config.tracker, len(slots))
        self.manager = ParkingManager(slots)
        self.renderer = Renderer(self.config.view)
        self.frame_index = 0
        self.slots = slots
        self.reference = reference
        if reference is not None:
            self.detector.set_reference(reference, slots)

    # ---------------- 配置 ----------------
    def rebuild(self, slots: Optional[SlotSet] = None) -> None:
        """车位或参数变化后重建内部组件。"""
        if slots is not None and slots is not self.slots:
            self.slots = slots
            self.manager.bind_slots(slots)
        self.detector.cfg = self.config.detector
        self.detector.extractor = type(self.detector.extractor)(
            self.config.detector.roi_w, self.config.detector.roi_h,
            self.config.detector.inner_margin,
        )
        if self.reference is not None:
            self.detector.set_reference(self.reference, self.slots)
        self.tracker = SlotTracker(self.config.tracker, len(self.slots))
        self.manager.reset()

    def set_slot_set(self, slots: SlotSet) -> None:
        self.slots = slots
        self.manager.bind_slots(slots)
        self.tracker = SlotTracker(self.config.tracker, len(slots))
        if self.reference is not None:
            self.detector.set_reference(self.reference, slots)
        else:
            self.detector.clear_reference()

    def set_reference(self, image: np.ndarray) -> None:
        self.reference = image.copy()
        self.detector.set_reference(self.reference, self.slots)
        self.config.detector.use_reference = True

    def clear_reference(self) -> None:
        self.reference = None
        self.detector.clear_reference()
        self.config.detector.use_reference = False

    # ---------------- 主流程 ----------------
    def size_mismatch(self, frame: np.ndarray) -> Optional[str]:
        """检查画面尺寸与车位标注时的尺寸是否一致，返回提示文字或 None。"""
        if not self.slots.image_size:
            return None
        w, h = self.slots.image_size
        fh, fw = frame.shape[:2]
        if abs(fw - w) > 2 or abs(fh - h) > 2:
            return (f"车位标注尺寸 {w}x{h} 与当前画面 {fw}x{fh} 不一致，"
                    f"车位框可能对不上，建议在画面上重新标注")
        return None

    def analyze(self, frame: np.ndarray, time_s: Optional[float] = None,
                keep_rois: bool = False) -> AnalysisResult:
        t0 = time.perf_counter()
        det = self.detector.detect(frame, self.slots, keep_rois=keep_rois)
        states = self.tracker.update(det.scores, self.config.tracker)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if time_s is None:
            time_s = self.frame_index / 25.0
        stats = self.manager.update(states, time_s, elapsed)
        self.frame_index += 1

        return AnalysisResult(states=list(states), scores=list(det.scores),
                              detection=det, stats=stats, time_s=time_s)

    def render(self, frame: np.ndarray, result: AnalysisResult,
               timestamp: Optional[str] = None) -> np.ndarray:
        return self.renderer.render(
            frame,
            self.slots,
            result.states,
            det=result.detection,
            stats=result.stats,
            events=self.manager.recent_events(4),
            time_s=result.time_s,
            title=self.title,
            timestamp=timestamp,
        )

    def process(self, frame: np.ndarray, time_s: Optional[float] = None,
                draw: bool = True, timestamp: Optional[str] = None) -> Tuple[np.ndarray, AnalysisResult]:
        """一步完成“分析 + 渲染”，返回 (可视化画面, 结果)。"""
        result = self.analyze(frame, time_s)
        if not draw:
            return frame, result
        return self.render(frame, result, timestamp=timestamp), result
