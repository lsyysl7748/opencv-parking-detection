"""统一的帧来源：视频文件 / 单张图片 / USB 摄像头 / 图片序列。

三种数据源对外暴露相同的接口，主程序和界面不需要关心底层差异。
另外提供 :class:`ThreadedSource`，在后台线程里预读帧，让界面保持流畅。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameSource:
    """帧来源基类。"""

    kind = "unknown"

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """返回 (是否成功, 帧, 相对于源起点的时间戳/秒)。"""
        raise NotImplementedError

    def release(self) -> None:
        pass

    @property
    def is_live(self) -> bool:
        return False

    @property
    def total_frames(self) -> int:
        return 0

    @property
    def fps(self) -> float:
        return 25.0

    @property
    def size(self) -> Tuple[int, int]:
        return (0, 0)

    def seek_start(self) -> None:
        pass


class VideoSource(FrameSource):
    kind = "video"

    def __init__(self, path: str):
        self.path = path
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps = 25.0
        self._total = 0
        self._idx = 0

    def open(self) -> bool:
        if not os.path.exists(self.path):
            return False
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 1e-3 else 25.0
        self._total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._idx = 0
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self.cap is None:
            return False, None, 0.0
        ok, frame = self.cap.read()
        if not ok:
            return False, None, self._idx / self._fps
        t = self._idx / self._fps
        self._idx += 1
        return True, frame, t

    def seek_start(self) -> None:
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._idx = 0

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> Tuple[int, int]:
        if self.cap is None:
            return (0, 0)
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


class CameraSource(FrameSource):
    kind = "camera"

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self._t0 = time.time()

    def open(self) -> bool:
        # Windows 下优先用 DirectShow，打开更快也更稳
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._t0 = time.time()
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self.cap is None:
            return False, None, 0.0
        ok, frame = self.cap.read()
        return ok, frame, time.time() - self._t0

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_live(self) -> bool:
        return True

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)


class ImageSource(FrameSource):
    """单张图片，可重复读取。"""

    kind = "image"

    def __init__(self, path: str):
        self.path = path
        self.image: Optional[np.ndarray] = None

    def open(self) -> bool:
        self.image = read_image(self.path)
        return self.image is not None

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self.image is None:
            return False, None, 0.0
        return True, self.image.copy(), 0.0

    @property
    def total_frames(self) -> int:
        return 1

    @property
    def size(self) -> Tuple[int, int]:
        if self.image is None:
            return (0, 0)
        h, w = self.image.shape[:2]
        return (w, h)


def read_image(path: str) -> Optional[np.ndarray]:
    """读取图片，兼容中文路径（cv2.imread 在 Windows 上不支持非 ASCII 路径）。"""
    if not path or not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def write_image(path: str, image: np.ndarray) -> bool:
    """保存图片，兼容中文路径。"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", image)
    if not ok:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    buf.tofile(path)
    return True


def open_source(kind: str, target) -> Optional[FrameSource]:
    """按类型创建数据源。"""
    if kind == "video":
        src = VideoSource(str(target))
    elif kind == "image":
        src = ImageSource(str(target))
    elif kind == "camera":
        src = CameraSource(int(target) if str(target).isdigit() else 0)
    else:
        raise ValueError(f"未知的数据源类型: {kind}")
    return src if src.open() else None


class ThreadedSource:
    """后台线程预读，主线程只取最新帧，避免界面卡顿。"""

    def __init__(self, source: FrameSource, buffer_size: int = 3):
        self.source = source
        self.queue: "queue.Queue[Tuple[bool, Optional[np.ndarray], float]]" = queue.Queue(
            maxsize=buffer_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._paused = threading.Event()
        self.loop = True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.02)
                continue
            ok, frame, t = self.source.read()
            if not ok:
                if self.source.is_live:
                    time.sleep(0.05)
                    continue
                if self.loop and self.source.total_frames > 0:
                    self.source.seek_start()
                    continue
                self.queue.put((False, None, t))
                break
            try:
                self.queue.put((True, frame, t), timeout=0.5)
            except queue.Full:
                continue

    def read(self, timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray], float]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return False, None, 0.0

    def pause(self, flag: bool = True) -> None:
        self._paused.set() if flag else self._paused.clear()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.source.release()
