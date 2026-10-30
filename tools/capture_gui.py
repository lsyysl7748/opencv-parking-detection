"""截取桌面界面截图（Windows）。用于制作文档配图 / 回归检查。

    python tools/capture_gui.py  [输出路径]

在没有可见桌面的环境里 PIL.ImageGrab 抓到的可能是黑图，
这里改用 Win32 的 PrintWindow(PW_RENDERFULLCONTENT) 直接从窗口取像素。
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import tkinter as tk  # noqa: E402

from parking.gui import ParkingApp  # noqa: E402


class Args:
    slots = os.path.join("data", "slots.json")
    reference = os.path.join("data", "lot_reference.png")
    src = os.path.join("data", "parking_demo.mp4")
    kind = "video"


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def capture(hwnd: int, out_path: str) -> None:
    from PIL import Image

    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    top = user32.GetAncestor(hwnd, 2)
    rect = wintypes.RECT()
    user32.GetWindowRect(top, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = user32.GetWindowDC(top)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow(top, mem, 3)
    bi = _BITMAPINFO()
    bi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bi.bmiHeader.biWidth = w
    bi.bmiHeader.biHeight = -h
    bi.bmiHeader.biPlanes = 1
    bi.bmiHeader.biBitCount = 32
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bi), 0)
    Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1).save(out_path)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(top, hdc)
    print("screenshot ->", out_path, f"{w}x{h}")


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join("docs", "screenshot_gui.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    root = tk.Tk()
    app = ParkingApp(root, Args())
    app.pause()
    root.update()
    for i in range(90):  # 推进到画面里有进出事件的时刻
        ok, frame, t = app.source.read(timeout=2.0)
        if not ok:
            break
        app._process_frame(frame, t)
        if i % 15 == 0:
            root.update()
    root.update()
    time.sleep(0.4)
    root.update()
    capture(int(root.winfo_id()), out)
    root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
