"""配置与参数持久化。

本模块集中管理整个系统的可调参数：
    * ROI 归一化尺寸与内缩比例
    * 各特征的归一化区间 (lo, hi) 与融合权重
    * 判定阈值、时间平滑窗口、事件稳定性
    * 可视化配色

所有参数都可以写入 JSON 并在界面上实时调整，方便现场标定。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

# 项目根目录（parking 包的上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")

# 特征名称 -> 中文名（界面与报告里用）
FEATURE_NAMES: Dict[str, str] = {
    "edge": "边缘密度",
    "contrast": "Otsu类间方差",
    "corner": "角点密度",
    "std": "灰度标准差",
    "sat": "饱和度离散度",
    "bgdiff": "背景差分",
}

# 特征默认归一化区间：把原始特征值线性映射到 [0, 1]
# lo 附近的取值视为“空车位”，hi 附近的取值视为“有车”
# （这些数值是在演示数据集上统计得到的先验值；换摄像头后用
#   `python main.py evaluate --calibrate --save-config` 可重新标定）
FEATURE_RANGE_DEFAULT: Dict[str, Tuple[float, float]] = {
    "edge": (0.0030, 0.0450),
    "contrast": (0.0008, 0.0100),
    "corner": (0.05, 1.20),
    "std": (10.0, 32.0),
    "sat": (5.0, 24.0),
    "bgdiff": (2.0, 15.0),
}

# 特征融合权重（背景差只有在提供空位基准图时才会启用）
WEIGHTS_DEFAULT: Dict[str, float] = {
    "edge": 0.30,
    "contrast": 0.26,
    "corner": 0.20,
    "std": 0.14,
    "sat": 0.10,
    "bgdiff": 0.0,
}

# 提供空位基准图时的推荐权重
WEIGHTS_WITH_REF: Dict[str, float] = {
    "edge": 0.16,
    "contrast": 0.12,
    "corner": 0.10,
    "std": 0.08,
    "sat": 0.04,
    "bgdiff": 0.50,
}


@dataclass
class DetectorConfig:
    """单车位检测参数。"""

    roi_w: int = 96          # 归一化 ROI 宽（像素）
    roi_h: int = 192         # 归一化 ROI 高（像素）
    inner_margin: float = 0.08  # ROI 四周内缩比例，避开白色车位线本身
    canny_sigma: float = 0.33   # 自适应 Canny 阈值系数
    threshold: float = 0.50     # 综合得分判定阈值
    ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(FEATURE_RANGE_DEFAULT)
    )
    weights: Dict[str, float] = field(default_factory=lambda: dict(WEIGHTS_DEFAULT))
    use_reference: bool = True  # 有基准图时是否启用背景差分
    normalize_light: bool = True  # 基准图按当前帧亮度做光照补偿

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ranges"] = {k: list(v) for k, v in self.ranges.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorConfig":
        cfg = cls()
        for key, value in (d or {}).items():
            if key == "ranges":
                cfg.ranges.update({k: tuple(v) for k, v in value.items()})
            elif key == "weights":
                cfg.weights.update(value)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


@dataclass
class TrackerConfig:
    """时间维度平滑参数：抑制车辆进出、反光、噪声造成的抖动。"""

    window: int = 15        # 滑动窗口帧数
    enter: float = 0.56     # 进入“占用”状态的门限（迟滞上界）
    leave: float = 0.44     # 退出“占用”状态的门限（迟滞下界）
    stable: int = 3         # 状态需要连续满足的帧数才真正切换

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ViewConfig:
    """可视化配色与显示选项。"""

    free_color: Tuple[int, int, int] = (92, 200, 110)      # BGR 空闲-绿
    busy_color: Tuple[int, int, int] = (66, 72, 226)       # BGR 占用-红
    edge_color: Tuple[int, int, int] = (235, 235, 235)
    panel_color: Tuple[int, int, int] = (34, 34, 40)
    text_color: Tuple[int, int, int] = (245, 245, 245)
    fill_alpha: float = 0.24
    show_index: bool = True
    show_score: bool = True
    show_hud: bool = True
    show_history: bool = True
    show_debug_rois: bool = False


@dataclass
class SystemConfig:
    """聚合配置，可整体序列化为 JSON。"""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    view: ViewConfig = field(default_factory=ViewConfig)

    def to_dict(self) -> dict:
        return {
            "detector": self.detector.to_dict(),
            "tracker": self.tracker.to_dict(),
            "view": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self.view).items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SystemConfig":
        cfg = cls()
        if not d:
            return cfg
        cfg.detector = DetectorConfig.from_dict(d.get("detector", {}))
        for key, value in (d.get("tracker") or {}).items():
            if hasattr(cfg.tracker, key):
                setattr(cfg.tracker, key, value)
        for key, value in (d.get("view") or {}).items():
            if hasattr(cfg.view, key):
                setattr(cfg.view, key, tuple(value) if isinstance(value, list) else value)
        return cfg


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sync_thresholds(cfg: "SystemConfig", threshold: Optional[float] = None,
                    band: float = 0.08) -> "SystemConfig":
    """让跟踪器的迟滞门限跟随检测阈值，二者始终相差 ``band``。"""
    t = float(cfg.detector.threshold if threshold is None else threshold)
    cfg.detector.threshold = t
    cfg.tracker.enter = min(0.95, t + band)
    cfg.tracker.leave = max(0.05, t - band)
    return cfg


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
