"""合成停车场演示数据生成器。

在没有真实监控录像的情况下，用纯 OpenCV 程序化地“画”出一个停车场：

* 沥青地面（多尺度噪声 + 油污 + 轮胎印）
* 两排车位 + 中间行车道 + 路缘石/绿化带
* 白色车位标线
* 带车身、车窗、后视镜、车灯、轮胎、投影的俯视小汽车
* 车辆驶入 / 驶离的滑动动画、行车道上的过路车
* 光照缓慢变化与传感器噪声

输出四样东西，供检测程序和评估脚本使用：

    lot_reference.png   空场基准图（可用于背景差分）
    slots.json          车位标注（四边形顶点）
    parking_demo.mp4    演示视频
    ground_truth.json   逐帧真值（0 空闲 / 1 占用 / 2 状态切换中，忽略）

    python -m parking.demo --out data
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .slots import Slot, SlotSet

# --------------------------------------------------------------------------
# 场景参数
# --------------------------------------------------------------------------
W, H = 1280, 720
BAY_W, BAY_H = 104, 210
COLS = 10
X0 = 110
TOP_Y = 56
BOTTOM_Y = 452

CAR_W, CAR_L = 84, 176

# 常见车漆色（BGR）
CAR_COLORS: List[Tuple[int, int, int]] = [
    (232, 232, 232),  # 白
    (188, 190, 194),  # 银
    (42, 44, 48),     # 黑
    (120, 74, 40),    # 深蓝
    (38, 38, 168),    # 红
    (96, 100, 104),   # 灰
    (60, 80, 46),     # 墨绿
    (150, 150, 152),  # 亮银
    (28, 26, 30),     # 深灰
    (170, 140, 96),   # 香槟金
]

FPS = 20
DURATION_S = 42.0
SLIDE_S = 1.2  # 车辆滑入/滑出所需时间


# --------------------------------------------------------------------------
# 基础绘图工具
# --------------------------------------------------------------------------
def _rounded_rect_mask(w: int, h: int, r: int) -> np.ndarray:
    """生成圆角矩形掩膜（0/255）。"""
    r = int(max(0, min(r, min(w, h) // 2)))
    m = np.zeros((h, w), np.uint8)
    if r <= 0:
        m[:] = 255
        return m
    cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def _fill_rounded(img: np.ndarray, x: int, y: int, w: int, h: int, r: int,
                  color: Tuple[int, int, int]) -> None:
    mask = _rounded_rect_mask(w, h, r)
    roi = img[y:y + h, x:x + w]
    if roi.shape[:2] != mask.shape[:2]:
        return
    roi[mask > 0] = color


def _alpha_blend(dst: np.ndarray, patch: np.ndarray, alpha: np.ndarray,
                 x: int, y: int) -> None:
    """把带 alpha 的小图贴到大图上（自动裁剪越界部分）。"""
    ph, pw = patch.shape[:2]
    dh, dw = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x + pw), min(dh, y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    sub = dst[y0:y1, x0:x1].astype(np.float32)
    p = patch[sy0:sy1, sx0:sx1].astype(np.float32)
    a = (alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[..., None]
    dst[y0:y1, x0:x1] = np.clip(sub * (1 - a) + p * a, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 地面与场景
# --------------------------------------------------------------------------
def make_asphalt(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """多尺度噪声沥青地面。"""
    base = np.array([82, 84, 88], np.float32)
    noise = np.zeros((h, w), np.float32)
    for cell, amp in ((8, 5.0), (20, 5.5), (56, 4.0), (150, 3.0)):
        small = rng.normal(0, 1, (max(2, h // cell + 2), max(2, w // cell + 2))).astype(np.float32)
        noise += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    img = np.repeat(base[None, None, :], h, axis=0)
    img = np.repeat(img, w, axis=1)
    img = img + noise[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 油污 / 补丁块
    layer = img.copy()
    for _ in range(26):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        ax, ay = int(rng.integers(18, 90)), int(rng.integers(12, 70))
        angle = float(rng.uniform(0, 360))
        shade = int(rng.integers(-26, 12))
        col = tuple(int(np.clip(c + shade, 0, 255)) for c in (78, 80, 84))
        cv2.ellipse(layer, (cx, cy), (ax, ay), angle, 0, 360, col, -1)
    img = cv2.addWeighted(layer, 0.35, img, 0.65, 0)

    # 裂缝
    for _ in range(14):
        pts = [(int(rng.integers(0, w)), int(rng.integers(0, h)))]
        for _ in range(int(rng.integers(3, 7))):
            px, py = pts[-1]
            pts.append((int(np.clip(px + rng.integers(-70, 70), 0, w - 1)),
                        int(np.clip(py + rng.integers(-40, 40), 0, h - 1))))
        cv2.polylines(img, [np.array(pts, np.int32)], False,
                      tuple(int(c) for c in (60, 62, 66)), 2, cv2.LINE_AA)

    return cv2.GaussianBlur(img, (3, 3), 0)


def build_scene(rng: np.random.Generator) -> Tuple[np.ndarray, List[Slot]]:
    """画出空场景 + 返回车位列表。"""
    img = make_asphalt(W, H, rng)

    # 顶部路缘石 + 绿化带
    cv2.rectangle(img, (0, 0), (W, 34), (128, 130, 132), -1)
    cv2.line(img, (0, 34), (W, 34), (150, 152, 154), 2)
    for x in range(30, W, 96):
        cv2.circle(img, (x + 20, 16), 13, (58, 92, 52), -1, cv2.LINE_AA)
        cv2.circle(img, (x + 20, 13), 9, (72, 116, 64), -1, cv2.LINE_AA)

    # 底部路缘石
    cv2.rectangle(img, (0, 688), (W, H), (122, 124, 126), -1)
    cv2.line(img, (0, 688), (W, 688), (146, 148, 150), 2)
    for x in range(10, W, 74):
        cv2.rectangle(img, (x, 694), (x + 30, 700), (108, 110, 112), -1)

    slots: List[Slot] = []
    idx = 1
    # 上排车位：车头朝上（朝路缘）
    for c in range(COLS):
        x = X0 + c * BAY_W
        slots.append(Slot(index=idx, points=[
            (x, TOP_Y), (x + BAY_W, TOP_Y), (x + BAY_W, TOP_Y + BAY_H), (x, TOP_Y + BAY_H)]))
        idx += 1
    # 下排车位：对应上排编号继续
    for c in range(COLS):
        x = X0 + c * BAY_W
        slots.append(Slot(index=idx, points=[
            (x, BOTTOM_Y), (x + BAY_W, BOTTOM_Y), (x + BAY_W, BOTTOM_Y + BAY_H), (x, BOTTOM_Y + BAY_H)]))
        idx += 1

    # 车位标线（三边 U 形 + 入口短线）
    line_col = (222, 224, 226)
    for s in slots:
        x0, y0, x1, y1 = s.points[0][0], s.points[0][1], s.points[2][0], s.points[2][1]
        is_top = abs(y0 - TOP_Y) < 1
        if is_top:
            cv2.line(img, (int(x0), int(y0)), (int(x0 + BAY_W), int(y0)), line_col, 3)
            cv2.line(img, (int(x0), int(y0)), (int(x0), int(y1)), line_col, 2)
            cv2.line(img, (int(x0 + BAY_W), int(y0)), (int(x0 + BAY_W), int(y1)), line_col, 2)
        else:
            cv2.line(img, (int(x0), int(y1)), (int(x0 + BAY_W), int(y1)), line_col, 3)
            cv2.line(img, (int(x0), int(y0)), (int(x0), int(y1)), line_col, 2)
            cv2.line(img, (int(x0 + BAY_W), int(y0)), (int(x0 + BAY_W), int(y1)), line_col, 2)

    # 车道中央虚线 + 导向箭头
    for x in range(60, W, 120):
        cv2.rectangle(img, (x, 352), (x + 56, 360), (196, 198, 200), -1)
    for x in (250, 780):
        cv2.arrowedLine(img, (x, 396), (x + 90, 396), (206, 208, 210), 3, cv2.LINE_AA, tipLength=0.35)

    # 轮胎印：大部分车位边缘有两道淡淡的使用痕迹
    marks = np.zeros((H, W), np.uint8)
    for s in slots:
        if rng.random() < 0.55:
            continue
        cx, cy = s.center()
        for dx in (-26, 24):
            cv2.line(marks, (cx + dx, cy - 60), (cx + dx, cy + 60), 255, 6, cv2.LINE_AA)
    marks = cv2.GaussianBlur(marks, (0, 0), 4).astype(np.float32) / 255.0
    img = np.clip(img.astype(np.float32) * (1.0 - 0.10 * marks[..., None]), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, slots


# --------------------------------------------------------------------------
# 小汽车
# --------------------------------------------------------------------------
@dataclass
class Car:
    """一辆车的运行时状态。"""

    slot_index: int
    color: Tuple[int, int, int]
    angle: float = 0.0
    jitter: Tuple[float, float] = (0.0, 0.0)
    t_arrive: float = -1.0
    t_leave: float = 1e9
    facing: int = -1          # -1 车头朝上 / +1 车头朝下
    initial: bool = False     # 视频开始时是否已经在位


def draw_car_layer(car_color: Tuple[int, int, int], facing: int = -1
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """生成一辆俯视汽车的 (BGR, alpha) 小图，车头朝上。

    俯视视角下的车由这几层构成：车身底色 -> 深色玻璃座舱 -> 车身同色的车顶
    -> 风挡/后窗与侧窗 -> 引擎盖缝、车灯、车牌、后视镜、轮胎。
    """
    pad = 12
    w, h = CAR_W + pad * 2, CAR_L + pad * 2
    canvas = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)

    x0, y0 = pad, pad
    # ---- 车身 ----
    body_mask = _rounded_rect_mask(CAR_W, CAR_L, 19)
    body = np.zeros((CAR_L, CAR_W, 3), np.uint8)
    body[:] = car_color
    # 前后方向的明暗渐变，避免纯色平板感
    grad = np.linspace(1.12, 0.84, CAR_L, dtype=np.float32)[:, None, None]
    body = np.clip(body.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    body[body_mask == 0] = 0

    # 车身左侧高光条（模拟顶光反射）
    hi = tuple(int(min(255, c * 1.28 + 18)) for c in car_color)
    cv2.line(body, (6, 26), (6, CAR_L - 26), hi, 2, cv2.LINE_AA)
    dark = tuple(int(c * 0.66) for c in car_color)
    cv2.line(body, (CAR_W - 7, 26), (CAR_W - 7, CAR_L - 26), dark, 2, cv2.LINE_AA)

    # ---- 玻璃座舱 ----
    glass = (32, 35, 40)
    glass_hi = (86, 92, 100)
    gx = int(CAR_W * 0.15)
    gw = CAR_W - 2 * gx
    gy = int(CAR_L * 0.255)
    gh = int(CAR_L * 0.47)
    _fill_rounded(body, gx, gy, gw, gh, 13, glass)
    # 前风挡：上窄下宽的梯形
    cv2.fillPoly(body, [np.array([
        (gx + 4, gy + 2), (gx + gw - 4, gy + 2),
        (gx + gw - 1, gy + int(gh * 0.34)), (gx + 1, gy + int(gh * 0.34))], np.int32)], glass_hi)
    # 后窗：上宽下窄
    ry = gy + int(gh * 0.66)
    cv2.fillPoly(body, [np.array([
        (gx + 2, ry), (gx + gw - 2, ry),
        (gx + gw - 5, gy + gh - 2), (gx + 5, gy + gh - 2)], np.int32)], glass_hi)
    # 车顶：车身色，压在中间，只留前后风挡与两侧侧窗露出的玻璃
    roof_col = tuple(int(c * 0.94 + 6) for c in car_color)
    ry0, ry1 = gy + int(gh * 0.34), gy + int(gh * 0.66)
    _fill_rounded(body, gx + 1, ry0, gw - 2, ry1 - ry0, 8, roof_col)
    cv2.line(body, (gx + 6, (ry0 + ry1) // 2), (gx + gw - 8, (ry0 + ry1) // 2),
             tuple(int(min(255, c * 1.18 + 12)) for c in car_color), 2, cv2.LINE_AA)
    # 玻璃斜向高光
    cv2.line(body, (gx + 8, gy + 6), (gx + gw - 16, gy + int(gh * 0.30)), (150, 160, 172), 2, cv2.LINE_AA)
    cv2.line(body, (gx + 14, gy + gh - 6), (gx + gw - 10, ry), (128, 136, 148), 1, cv2.LINE_AA)

    # ---- 引擎盖 / 后备箱分缝 ----
    seam = tuple(int(c * 0.62) for c in car_color)
    cv2.line(body, (9, gy - 9), (CAR_W - 9, gy - 9), seam, 1, cv2.LINE_AA)
    cv2.line(body, (9, gy + gh + 9), (CAR_W - 9, gy + gh + 9), seam, 1, cv2.LINE_AA)

    # ---- 车灯：车头朝上 -> 上端前大灯，下端尾灯 ----
    cv2.ellipse(body, (int(CAR_W * 0.22), 11), (10, 5), 0, 0, 360, (228, 238, 252), -1, cv2.LINE_AA)
    cv2.ellipse(body, (int(CAR_W * 0.78), 11), (10, 5), 0, 0, 360, (228, 238, 252), -1, cv2.LINE_AA)
    cv2.rectangle(body, (int(CAR_W * 0.14), CAR_L - 12), (int(CAR_W * 0.14) + 17, CAR_L - 7),
                  (44, 46, 202), -1)
    cv2.rectangle(body, (int(CAR_W * 0.66), CAR_L - 12), (int(CAR_W * 0.66) + 17, CAR_L - 7),
                  (44, 46, 202), -1)
    # 车牌
    cv2.rectangle(body, (int(CAR_W * 0.5) - 13, CAR_L - 14), (int(CAR_W * 0.5) + 13, CAR_L - 8),
                  (206, 208, 212), -1)

    # 合成到画布
    roi = canvas[y0:y0 + CAR_L, x0:x0 + CAR_W]
    roi[body_mask > 0] = body[body_mask > 0]
    alpha[y0:y0 + CAR_L, x0:x0 + CAR_W] = body_mask

    # 车轮（略微伸出车身，做出俯视的立体感）
    wheel_col = (24, 24, 26)
    for wx, wy in ((-5, int(CAR_L * 0.13)), (CAR_W - 4, int(CAR_L * 0.13)),
                   (-5, int(CAR_L * 0.70)), (CAR_W - 4, int(CAR_L * 0.70))):
        p1 = (x0 + wx, y0 + wy)
        p2 = (x0 + wx + 9, y0 + wy + 30)
        cv2.rectangle(canvas, p1, p2, wheel_col, -1)
        cv2.rectangle(alpha, p1, p2, 255, -1)

    # 后视镜
    for mx in (-6, CAR_W - 2):
        p1 = (x0 + mx, y0 + int(CAR_L * 0.40))
        p2 = (x0 + mx + 8, y0 + int(CAR_L * 0.40) + 11)
        cv2.rectangle(canvas, p1, p2, tuple(int(c * 0.75) for c in car_color), -1)
        cv2.rectangle(alpha, p1, p2, 255, -1)

    # 车头朝下时整体旋转 180°
    if facing > 0:
        canvas = cv2.rotate(canvas, cv2.ROTATE_180)
        alpha = cv2.rotate(alpha, cv2.ROTATE_180)

    # 轻微倒角柔化，去掉锯齿
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return canvas, alpha


def car_pose(car: Car, slot_lookup: Dict[int, Slot]) -> Tuple[float, float, float]:
    """返回 (cx, cy, angle_deg)。"""
    slot = slot_lookup[car.slot_index]
    cx, cy = slot.center()
    return cx + car.jitter[0], cy + car.jitter[1], car.angle


# --------------------------------------------------------------------------
# 时间轴
# --------------------------------------------------------------------------
def make_cars(slots: List[Slot], rng: random.Random) -> List[Car]:
    """安排每辆车的到达/离开时间，形成有进有出的动态场景。"""
    cars: List[Car] = []
    for s in slots:
        color = rng.choice(CAR_COLORS)
        # 上排车位朝向路缘（车头朝上），下排朝下
        facing = -1 if s.index <= COLS else 1
        car = Car(slot_index=s.index, color=color, facing=facing,
                  angle=float(rng.uniform(-2.2, 2.2)),
                  jitter=(float(rng.uniform(-3, 3)), float(rng.uniform(-4, 4))))
        r = rng.random()
        if r < 0.42:  # 视频开始时已经在位
            car.initial = True
            car.t_arrive = -1.0
            car.t_leave = float(rng.uniform(5, DURATION_S - 7)) if rng.random() < 0.6 else 1e9
        elif r < 0.88:  # 中途驶入
            car.t_arrive = float(rng.uniform(1.0, DURATION_S - 13))
            if rng.random() < 0.5:
                car.t_leave = car.t_arrive + float(rng.uniform(8, 16))
        else:  # 全程都空着
            car.t_arrive = 1e9
        cars.append(car)
    return cars


def car_state(car: Car, t: float) -> str:
    """返回 'absent' / 'arriving' / 'parked' / 'leaving'。"""
    if car.initial:
        if t < car.t_leave:
            return "parked"
        if t < car.t_leave + SLIDE_S:
            return "leaving"
        return "absent"
    if t < car.t_arrive:
        return "absent"
    if t < car.t_arrive + SLIDE_S:
        return "arriving"
    if t < car.t_leave:
        return "parked"
    if t < car.t_leave + SLIDE_S:
        return "leaving"
    return "absent"


def slide_offset(car: Car, t: float) -> float:
    """滑入/滑出时沿车位轴向的偏移量（像素）。"""
    st = car_state(car, t)
    if st == "arriving":
        k = (t - car.t_arrive) / SLIDE_S
        return float((1.0 - k) * (BAY_H * 0.95))
    if st == "leaving":
        k = (t - car.t_leave) / SLIDE_S
        return float(k * (BAY_H * 0.95))
    return 0.0


# --------------------------------------------------------------------------
# 逐帧渲染
# --------------------------------------------------------------------------
@dataclass
class Scene:
    background: np.ndarray
    slots: List[Slot]
    slotset: SlotSet
    cars: List[Car]
    cache: Dict[Tuple[Tuple[int, int, int], int], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def lookup(self) -> Dict[int, Slot]:
        return {s.index: s for s in self.slots}


def build_scene_object(seed: int = 7) -> Scene:
    rng = np.random.default_rng(seed)
    bg, slots = build_scene(rng)
    slotset = SlotSet(slots, image_size=(W, H))
    cars = make_cars(slots, random.Random(seed))
    return Scene(background=bg, slots=slots, slotset=slotset, cars=cars)


def render_frame(scene: Scene, t: float, rng: np.random.Generator,
                 traffic: Optional[List[dict]] = None) -> Tuple[np.ndarray, List[int]]:
    """渲染 t 时刻的画面，同时返回每个车位的真值标注。"""
    frame = scene.background.copy()
    slots = scene.slots
    lookup = scene.lookup()

    # 光照缓慢变化（模拟云层遮挡 / 相机自动增益）
    gain = 1.0 + 0.055 * math.sin(t * 0.6) + 0.018 * math.sin(t * 2.3)

    moving: List[Tuple[float, float, Car, float]] = []
    ground: List[Tuple[float, float, Car, float]] = []
    shadow_layer = np.zeros(frame.shape[:2], np.uint8)

    for car in scene.cars:
        st = car_state(car, t)
        if st == "absent":
            continue
        cx, cy, ang = car_pose(car, lookup)
        off = slide_offset(car, t)
        # 上排（车头朝上）从车位下方滑入，下排从上方滑入
        cy = cy - off * car.facing
        if st == "parked":
            ground.append((cx, cy, car, ang))
        else:
            moving.append((cx, cy, car, ang))

    # 影子先统一画在一层上，最后一次性模糊，开销更可控
    for cx, cy, car, ang in ground + moving:
        rect = ((cx + 6, cy + 7), (CAR_W + 2, CAR_L + 2), ang)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(shadow_layer, [box], 255)
    shadow = cv2.GaussianBlur(shadow_layer, (0, 0), 7).astype(np.float32) / 255.0
    frame = np.clip(frame.astype(np.float32) * (1.0 - 0.42 * shadow[..., None]), 0, 255).astype(np.uint8)

    # 先画停在位上的车，再画正在移动的车（避免被盖住）
    for cx, cy, car, ang in ground + moving:
        key = (car.color, car.facing)
        layer = scene.cache.get(key)
        if layer is None:
            layer = draw_car_layer(car.color, car.facing)
            scene.cache[key] = layer
        patch, alpha = layer
        ph, pw = patch.shape[:2]
        m = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), ang, 1.0)
        rot = cv2.warpAffine(patch, m, (pw, ph), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        rot_a = cv2.warpAffine(alpha, m, (pw, ph), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _alpha_blend(frame, rot, rot_a, int(round(cx - pw / 2)), int(round(cy - ph / 2)))

    # 行车道过路车
    if traffic:
        for item in traffic:
            cx = item["x"]
            cy = item["y"]
            color = item["color"]
            key = (color, 90)
            layer = scene.cache.get(key)
            if layer is None:
                base, alpha = draw_car_layer(color, -1)
                M = cv2.getRotationMatrix2D((base.shape[1] / 2.0, base.shape[0] / 2.0), 90, 1.0)
                base = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]))
                alpha = cv2.warpAffine(alpha, M, (alpha.shape[1], alpha.shape[0]))
                layer = (base, alpha)
                scene.cache[key] = layer
            patch, alpha = layer
            ph, pw = patch.shape[:2]
            _alpha_blend(frame, patch, alpha, int(round(cx - pw / 2)), int(round(cy - ph / 2)))

    # 光照 + 传感器噪声
    frame = np.clip(frame.astype(np.float32) * gain, 0, 255)
    frame += rng.normal(0, 2.6, frame.shape).astype(np.float32)
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # 真值：0 空闲 / 1 占用 / 2 状态切换中（评估时忽略，避免惩罚滤波延迟）
    gt: List[int] = []
    half = 0.8
    for s in slots:
        car = next((c for c in scene.cars if c.slot_index == s.index), None)
        if car is None:
            gt.append(0)
            continue
        states = {car_state(car, max(0.0, t - half)), car_state(car, t), car_state(car, t + half)}
        if states == {"parked"}:
            gt.append(1)
        elif states == {"absent"}:
            gt.append(0)
        elif "parked" in states and states <= {"parked", "absent"}:
            gt.append(1)
        else:
            gt.append(2)
    return frame, gt


def traffic_cars(t: float, rng: random.Random) -> List[dict]:
    """行车道上的过路车：每 11 秒左右一辆从右往左开过。"""
    out: List[dict] = []
    period = 11.0
    for k in range(int(DURATION_S // period) + 2):
        t_start = k * period + 2.0
        dt = t - t_start
        if 0 <= dt <= 6.0:
            x = W + 120 - dt * (W + 240) / 6.0
            out.append({"x": x, "y": 420, "color": CAR_COLORS[(k * 3 + 1) % len(CAR_COLORS)]})
    return out


# --------------------------------------------------------------------------
# 数据集导出
# --------------------------------------------------------------------------
def generate(out_dir: str, seed: int = 7, duration: float = DURATION_S,
             fps: int = FPS, perspective: bool = False,
             video_name: str = "parking_demo.mp4",
             slots_name: str = "slots.json",
             gt_name: str = "ground_truth.json",
             ref_name: str = "lot_reference.png") -> dict:
    """生成完整演示数据集，返回数据集信息。"""
    os.makedirs(out_dir, exist_ok=True)
    scene = build_scene_object(seed)
    rng = np.random.default_rng(seed + 1)
    traffic_rng = random.Random(seed)

    homo = None
    if perspective:
        homo = _perspective_matrix()

    # 空场基准图：临时把车辆时间表清空后渲染一张
    for car in scene.cars:
        car.t_arrive = 1e9
        car.initial = False
        car.t_leave = 1e9
    empty_bg, _ = render_frame(scene, 0.0, np.random.default_rng(seed + 98), traffic=[])
    scene = build_scene_object(seed)  # 恢复车辆时间表

    ref_img = _apply_perspective(empty_bg, homo) if homo is not None else empty_bg
    cv2.imwrite(os.path.join(out_dir, ref_name), ref_img)

    slots_out = scene.slotset
    suffix = "_persp" if perspective else ""
    if homo is not None:
        slots_out = _warp_slots(scene.slotset, homo)
    slots_out.save(os.path.join(out_dir, slots_name))
    cv2.imwrite(os.path.join(out_dir, f"slots_overlay{suffix}.png"),
                slots_out.draw_polygons(ref_img))

    n_frames = int(duration * fps)
    path = os.path.join(out_dir, video_name)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not writer.isOpened():
        raise SystemExit("无法创建视频文件，请检查 OpenCV 的 ffmpeg 支持")

    gt_frames: List[List[int]] = []
    sample_every = 40
    for i in range(n_frames):
        t = i / fps
        frame, gt = render_frame(scene, t, rng, traffic_cars(t, traffic_rng))
        if homo is not None:
            frame = _apply_perspective(frame, homo)
        writer.write(frame)
        gt_frames.append(gt)
        if i % sample_every == 0:
            print(f"  渲染 {i + 1}/{n_frames} 帧 ...", end="\r")
        if i == 120:  # 留一张有车的快照做展示
            cv2.imwrite(os.path.join(out_dir, f"scene_sample{suffix}.png"), frame)
    writer.release()
    print(f"  渲染完成 {n_frames} 帧" + " " * 20)

    with open(os.path.join(out_dir, gt_name), "w", encoding="utf-8") as f:
        json.dump(
            {
                "fps": fps,
                "frames": n_frames,
                "n_slots": len(scene.slots),
                "description": "0=空闲 1=占用 2=状态切换中(评估时忽略)",
                "gt": gt_frames,
            },
            f,
        )

    info = {
        "out_dir": os.path.abspath(out_dir),
        "video": path,
        "frames": n_frames,
        "fps": fps,
        "slots": len(scene.slots),
        "perspective": perspective,
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


# --------------------------------------------------------------------------
# 透视（模拟立杆摄像机斜视角）
# --------------------------------------------------------------------------
def _perspective_matrix() -> np.ndarray:
    """把俯视图映射成“高处斜拍”的样子。"""
    src = np.float32([[36, 20], [W - 36, 20], [W - 36, H - 20], [36, H - 20]])
    dst = np.float32([[300, 40], [1180, 96], [1272, 700], [20, 660]])
    return cv2.getPerspectiveTransform(src, dst)


def _apply_perspective(frame: np.ndarray, homo: np.ndarray) -> np.ndarray:
    canvas = np.zeros_like(frame)
    canvas[:] = (72, 74, 78)
    warp = cv2.warpPerspective(frame, homo, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), homo, (W, H))
    canvas[mask > 0] = warp[mask > 0]
    vign = _vignette(W, H)
    return np.clip(canvas.astype(np.float32) * vign[..., None], 0, 255).astype(np.uint8)


def _warp_slots(slotset: SlotSet, homo: np.ndarray) -> SlotSet:
    out: List[Slot] = []
    for s in slotset.slots:
        pts = np.asarray(s.points, np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, homo).reshape(-1, 2)
        out.append(Slot(index=s.index, points=[tuple(p) for p in warped]))
    return SlotSet(out, image_size=slotset.image_size, meta={"perspective": True})


def _vignette(w: int, h: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    return np.clip(1.05 - 0.28 * r ** 2, 0.55, 1.05)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="生成合成停车场演示数据")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data"), help="输出目录")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--perspective", action="store_true", help="同时生成斜视角版本")
    args = ap.parse_args(argv)

    print("生成正射俯视数据集 ...")
    generate(args.out, seed=args.seed, duration=args.duration, fps=args.fps)
    if args.perspective:
        print("生成透视斜视角数据集 ...")
        generate(args.out, seed=args.seed, duration=args.duration, fps=args.fps,
                 perspective=True, video_name="parking_persp.mp4",
                 slots_name="slots_persp.json", gt_name="ground_truth_persp.json",
                 ref_name="lot_reference_persp.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
