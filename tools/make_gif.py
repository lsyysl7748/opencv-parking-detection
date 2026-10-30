"""从演示视频生成 README 用的 GIF（带检测叠加效果）。

    python tools/make_gif.py [输出路径] --start 6 --dur 8 --fps 8 --width 640

只用于制作文档配图，不参与检测逻辑。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from parking.config import SystemConfig
from parking.pipeline import ParkingAnalyzer
from parking.slots import SlotSet
from parking.sources import read_image


def parse_args(argv):
    out = argv[0] if argv and not argv[0].startswith("--") else os.path.join("docs", "demo.gif")
    opts = {"start": 6.0, "dur": 8.0, "fps": 8.0, "width": 640.0, "colors": 64.0}
    for i, token in enumerate(argv):
        if token.startswith("--") and i + 1 < len(argv):
            key = token[2:]
            if key in opts:
                opts[key] = float(argv[i + 1])
    return out, opts


def main() -> int:
    out, opts = parse_args(sys.argv[1:])
    start, dur, fps, width = (opts["start"], opts["dur"], opts["fps"], int(opts["width"]))

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video = os.path.join(root, "data", "parking_demo.mp4")
    slots = SlotSet.load(os.path.join(root, "data", "slots.json"))
    ref_path = os.path.join(root, "data", "lot_reference.png")
    reference = read_image(ref_path) if os.path.exists(ref_path) else None

    cfg = SystemConfig()
    cfg.view.show_history = False  # GIF 里曲线意义不大，去掉更干净
    analyzer = ParkingAnalyzer(slots, cfg, reference=reference)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    first = int(start * src_fps)
    count = int(dur * fps)
    step = max(1, int(round(src_fps / fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    idx = 0
    total = int(dur * src_fps)
    while idx < total and len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        # 每一帧都走一遍分析（保证滤波器状态连续），只按帧率抽样保存
        vis, _ = analyzer.process(frame, (first + idx) / src_fps)
        if idx % step == 0:
            h, w = vis.shape[:2]
            nh = int(round(h * width / w))
            small = cv2.resize(vis, (width, nh), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("没有采到帧")

    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    quant = [Image.fromarray(f).quantize(colors=int(opts.get("colors", 64)),
                                         method=Image.MEDIANCUT) for f in frames]
    quant[0].save(out, save_all=True, append_images=quant[1:],
                  duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"GIF 已生成: {out}  {len(quant)} 帧  {width}px  {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
