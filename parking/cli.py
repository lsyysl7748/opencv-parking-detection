"""命令行入口：视频 / 图片 / 摄像头 / 标注 / 评估 / 生成演示数据。

    python main.py gui
    python main.py video --src data/parking_demo.mp4 --slots data/slots.json
    python main.py image --src data/scene_sample.png --slots data/slots.json
    python main.py camera --index 0 --slots data/slots.json
    python main.py annotate --image data/lot_reference.png --out data/slots.json
    python main.py evaluate --data data --calibrate
    python main.py demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

from . import demo as demo_mod
from .config import (ROOT, DATA_DIR, OUT_DIR, DetectorConfig, SystemConfig, load_json,
                     save_json, sync_thresholds)
from .evaluate import (calibrate_from_cache, collect, evaluate_cache, load_ground_truth,
                       print_report, print_robustness, run_robustness, sweep_threshold)
from .pipeline import ParkingAnalyzer
from .report import draw_curve, save_metrics, save_summary
from .slots import SlotSet
from .sources import CameraSource, ImageSource, ThreadedSource, VideoSource, read_image, write_image


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------
def _default_data_dir() -> str:
    return DATA_DIR


def resolve(path: str) -> str:
    """相对路径优先按当前目录解析，其次按项目根目录。"""
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), ROOT):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return p
    return os.path.abspath(path)


def load_runtime(args) -> tuple:
    """装载车位标注、参数配置与基准图。"""
    slots_path = resolve(args.slots)
    if not os.path.exists(slots_path):
        raise SystemExit(
            f"找不到车位标注文件: {slots_path}\n"
            f"请先运行:  python main.py annotate --image <参考图> --out {args.slots}"
        )
    slots = SlotSet.load(slots_path)

    cfg = SystemConfig()
    if getattr(args, "config", None) and os.path.exists(resolve(args.config)):
        cfg = SystemConfig.from_dict(load_json(resolve(args.config)))

    reference = None
    ref_path = getattr(args, "reference", None)
    if ref_path is None and slots.reference:
        ref_path = slots.reference
    if ref_path:
        p = resolve(ref_path)
        if os.path.exists(p):
            reference = read_image(p)
            if reference is not None:
                print(f"已加载空位基准图: {p}")
    return slots, cfg, reference


def build_analyzer(args, slots: SlotSet, cfg: SystemConfig, reference) -> ParkingAnalyzer:
    analyzer = ParkingAnalyzer(slots, cfg, reference=reference)
    return analyzer


def _print_stats(stats, prefix: str = "") -> None:
    if stats is None:
        return
    print(f"{prefix}总 {stats.total} | 占用 {stats.occupied} | 空闲 {stats.free} | "
          f"利用率 {stats.rate * 100:.0f}% | {stats.elapsed_ms:.1f} ms/帧 | {stats.fps:.0f} FPS")


# --------------------------------------------------------------------------
# 视频模式
# --------------------------------------------------------------------------
def cmd_video(args) -> int:
    slots, cfg, reference = load_runtime(args)
    if args.threshold is not None:
        sync_thresholds(cfg, float(args.threshold))

    src = VideoSource(resolve(args.src))
    if not src.open():
        raise SystemExit(f"无法打开视频: {args.src}")
    analyzer = build_analyzer(args, slots, cfg, reference)
    print(f"视频: {args.src}  {src.size[0]}x{src.size[1]}  {src.fps:.1f} FPS  "
          f"{src.total_frames} 帧  车位 {len(slots)} 个")
    if slots.image_size and (abs(slots.image_size[0] - src.size[0]) > 2
                             or abs(slots.image_size[1] - src.size[1]) > 2):
        print(f"警告: 车位标注尺寸 {slots.image_size[0]}x{slots.image_size[1]} "
              f"与视频 {src.size[0]}x{src.size[1]} 不一致，检测结果可能不准")

    writer = None
    if args.save:
        out_path = resolve(args.save)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 src.fps or 25.0, src.size)
        if not writer.isOpened():
            print("警告: 无法创建输出视频，将只做分析")
            writer = None
        else:
            print(f"结果视频: {out_path}")

    frame_idx = 0
    saved = 0
    paused = False
    show = not args.headless
    if show:
        cv2.namedWindow("Parking Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Recognition", 1100, 620)

    try:
        while True:
            ok, frame, t = src.read()
            if not ok:
                break
            frame_idx += 1
            if args.every > 1 and (frame_idx % args.every) != 1:
                continue
            vis, result = analyzer.process(frame, t)
            if writer is not None:
                writer.write(vis)
            if frame_idx % 20 == 1:
                _print_stats(result.stats, prefix=f"  第 {frame_idx:4d} 帧  ")

            if show:
                cv2.imshow("Parking Recognition", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                if key == ord("s"):
                    saved += 1
                    p = os.path.join(OUT_DIR, f"frame_{saved:03d}.png")
                    write_image(p, vis)
                    print(f"  已保存截图 {p}")
                while paused:
                    k2 = cv2.waitKey(30) & 0xFF
                    if k2 in (ord(" "), ord("q"), 27):
                        paused = False
                        break
            if args.limit and frame_idx >= args.limit:
                break
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print()
    print(analyzer.manager.summary_text())
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = analyzer.manager.export_csv(os.path.join(OUT_DIR, "events.csv"))
    print(f"事件报表: {csv_path}")
    return 0


# --------------------------------------------------------------------------
# 图片模式
# --------------------------------------------------------------------------
def cmd_image(args) -> int:
    slots, cfg, reference = load_runtime(args)
    image = read_image(resolve(args.src))
    if image is None:
        raise SystemExit(f"无法读取图片: {args.src}")
    analyzer = build_analyzer(args, slots, cfg, reference)
    # 单张图片没有时间维度，直接按原始得分判定，避免平滑窗口把结果拉向初始状态
    det = analyzer.detector.detect(image, slots)
    states = det.occupied(cfg.detector.threshold)
    stats = analyzer.manager.update(states, 0.0, 0.0)
    vis = analyzer.renderer.render(image, slots, states, det=det, stats=stats, time_s=0.0,
                                   title="停车场车位识别")
    out = resolve(args.save or os.path.join(OUT_DIR, "result_image.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_image(out, vis)
    _print_stats(stats)
    print(f"空闲车位: {[i + 1 for i, s in enumerate(states) if not s]}")
    print(f"结果图片: {out}")
    if not args.headless:
        cv2.namedWindow("Parking Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Recognition", 1100, 620)
        cv2.imshow("Parking Recognition", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------
# 摄像头模式
# --------------------------------------------------------------------------
def cmd_camera(args) -> int:
    slots, cfg, reference = load_runtime(args)
    src = CameraSource(args.index, args.width, args.height)
    if not src.open():
        raise SystemExit(f"无法打开摄像头 {args.index}")
    print(f"摄像头 {args.index} 已打开，按 q 退出，按 r 抓取空场基准图，按 s 保存截图")
    analyzer = build_analyzer(args, slots, cfg, reference)

    cv2.namedWindow("Parking Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parking Recognition", 1100, 620)
    n = 0
    try:
        while True:
            ok, frame, t = src.read()
            if not ok:
                print("读取摄像头失败")
                break
            vis, result = analyzer.process(frame, t)
            n += 1
            if n % 30 == 0:
                _print_stats(result.stats, prefix=f"  第 {n:5d} 帧  ")
            cv2.imshow("Parking Recognition", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                analyzer.set_reference(frame)
                print("已把当前画面设为空位基准图")
            if key == ord("s"):
                p = os.path.join(OUT_DIR, f"camera_{int(time.time())}.png")
                write_image(p, vis)
                print(f"已保存截图 {p}")
            if args.limit and n >= args.limit:
                break
    finally:
        src.release()
        cv2.destroyAllWindows()
    print()
    print(analyzer.manager.summary_text())
    return 0


# --------------------------------------------------------------------------
# 标注模式
# --------------------------------------------------------------------------
def cmd_annotate(args) -> int:
    from .annotate import annotate

    base = None
    if os.path.exists(resolve(args.out)):
        base = SlotSet.load(resolve(args.out))
        print(f"在已有标注基础上编辑（{len(base)} 个车位）")
    annotate(resolve(args.image), resolve(args.out), base)
    return 0


# --------------------------------------------------------------------------
# 评估模式
# --------------------------------------------------------------------------
def _dataset_paths(data_dir: str, args) -> tuple:
    d = resolve(data_dir)
    slots = resolve(getattr(args, "slots", None) or os.path.join(d, "slots.json"))
    gt = resolve(getattr(args, "gt", None) or os.path.join(d, "ground_truth.json"))
    video = resolve(getattr(args, "video", None) or os.path.join(d, "parking_demo.mp4"))
    ref = resolve(getattr(args, "reference", None) or os.path.join(d, "lot_reference.png"))
    return slots, gt, video, ref


def cmd_evaluate(args) -> int:
    slots_path, gt_path, video_path, ref_path = _dataset_paths(args.data, args)
    if not os.path.exists(video_path):
        raise SystemExit(f"找不到视频: {video_path}\n请先运行:  python main.py demo")
    slots = SlotSet.load(slots_path)
    gt = load_ground_truth(gt_path) if os.path.exists(gt_path) else None
    if gt is None:
        raise SystemExit(f"找不到真值文件: {gt_path}（评估需要带标注的数据集）")

    cfg = SystemConfig()
    if args.config and os.path.exists(resolve(args.config)):
        cfg = SystemConfig.from_dict(load_json(resolve(args.config)))

    reference = read_image(ref_path) if os.path.exists(ref_path) else None
    from .detector import SlotDetector

    detector = SlotDetector(cfg.detector)
    if reference is not None:
        detector.set_reference(reference, slots)

    print("扫描视频并缓存特征 ...")
    cache = collect(video_path, slots, detector, gt=gt, max_frames=args.max_frames)

    if args.calibrate:
        print("用真值自动标定参数 ...")
        # 训练/测试划分：奇数帧标定，偶数帧评估，避免自评自测
        train_raw = cache.raw[::2]
        train_gt = cache.gt[::2]
        from .evaluate import RawCache

        train_cache = RawCache(raw=train_raw, gt=train_gt, fps=cache.fps)
        report = calibrate_from_cache(train_cache, detector, use_ref=reference is not None)
        sync_thresholds(cfg, detector.cfg.threshold)
        print(f"  标定结果: 阈值={report['threshold']:.3f} 训练准确率={report['accuracy']:.4f} "
              f"F1={report['f1']:.4f}")
        from .evaluate import RawCache as RC

        cache = RC(raw=cache.raw[1::2], gt=cache.gt[1::2], fps=cache.fps)
        print(f"  在另一半帧上评估（{len(cache.raw)} 帧）")

    metrics = evaluate_cache(cache, detector.cfg, cfg.tracker)
    print_report(metrics, title=f"检测精度评估   {os.path.basename(video_path)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "video": video_path,
        "slots": len(slots),
        "threshold": detector.cfg.threshold,
        "weights": detector.cfg.weights,
        "ranges": {k: list(v) for k, v in detector.cfg.ranges.items()},
        "metrics": {"raw": metrics["raw"], "system": metrics["system"]},
    }
    if args.curve:
        curve = sweep_threshold(cache, detector.cfg)
        out["threshold_curve"] = curve
        print("\n阈值扫描 (单帧检测):")
        print("  阈值   准确率  精确率  召回率   F1")
        for row in curve:
            mark = " <-" if abs(row["threshold"] - detector.cfg.threshold) < 0.026 else ""
            print(f"  {row['threshold']:.2f}   {row['accuracy']:.3f}   {row['precision']:.3f}"
                  f"   {row['recall']:.3f}   {row['f1']:.3f}{mark}")

    save_metrics(os.path.join(OUT_DIR, "metrics.json"), out)
    print(f"\n指标已保存: {os.path.join(OUT_DIR, 'metrics.json')}")

    if args.robustness:
        rows = run_robustness(video_path, slots, detector, gt, cfg.tracker,
                              stride=max(1, args.robust_stride))
        print_robustness(rows)
        save_metrics(os.path.join(OUT_DIR, "robustness.json"), {"rows": rows})
        print(f"鲁棒性结果已保存: {os.path.join(OUT_DIR, 'robustness.json')}")

    if args.save_config:
        cfg_path = resolve(args.save_config)
        save_json(cfg_path, {"detector": detector.cfg.to_dict(),
                             "tracker": cfg.tracker.to_dict()})
        print(f"标定参数已保存: {cfg_path}")
    return 0


# --------------------------------------------------------------------------
# 生成演示数据
# --------------------------------------------------------------------------
def cmd_demo(args) -> int:
    data_dir = resolve(args.out or DATA_DIR)
    print(f"生成合成停车场演示数据 -> {data_dir}")
    demo_mod.generate(data_dir, seed=args.seed, duration=args.duration, fps=args.fps,
                      perspective=args.perspective)
    print("\n完成，可以运行:")
    print("  python main.py video --src data/parking_demo.mp4 --slots data/slots.json")
    return 0


# --------------------------------------------------------------------------
# 报告（跑一遍视频，输出曲线图 + 小结 + 截图）
# --------------------------------------------------------------------------
def cmd_report(args) -> int:
    slots_path, _, video_path, ref_path = _dataset_paths(args.data, args)
    slots = SlotSet.load(slots_path)
    cfg = SystemConfig()
    if args.config and os.path.exists(resolve(args.config)):
        cfg = SystemConfig.from_dict(load_json(resolve(args.config)))
    reference = read_image(ref_path) if os.path.exists(ref_path) else None
    analyzer = ParkingAnalyzer(slots, cfg, reference=reference)

    src = VideoSource(video_path)
    if not src.open():
        raise SystemExit(f"无法打开视频: {video_path}")
    video_size = src.size
    video_fps = src.fps
    total_frames = src.total_frames

    rep_dir = resolve(args.out or os.path.join(OUT_DIR, "report"))
    os.makedirs(rep_dir, exist_ok=True)

    frames: List[tuple] = []
    n = 0
    total_ms = 0.0
    while True:
        ok, frame, t = src.read()
        if not ok:
            break
        vis, result = analyzer.process(frame, t)
        n += 1
        if result.stats is not None:
            total_ms += result.stats.elapsed_ms
        if n in (1, int(src.total_frames * 0.25) or 2,
                 int(src.total_frames * 0.55) or 3, int(src.total_frames * 0.85) or 4):
            frames.append((n, t, vis.copy()))
        if args.limit and n >= args.limit:
            break
    src.release()

    for i, (idx, t, vis) in enumerate(frames, start=1):
        p = os.path.join(rep_dir, f"frame_{i:02d}.png")
        write_image(p, vis)
        print(f"样例帧: {p}  (第 {idx} 帧, t={t:.1f}s)")

    hist = list(analyzer.manager.history)
    curve = draw_curve(hist, len(slots))
    curve_path = os.path.join(rep_dir, "occupancy_curve.png")
    write_image(curve_path, curve)
    print(f"占用率曲线: {curve_path}")

    summary = "\n".join([
        "基于 OpenCV 的停车位识别系统 - 运行小结",
        "=" * 40,
        f"视频文件      : {os.path.basename(video_path)}",
        f"总帧数        : {n} / {total_frames or '?'}",
        f"分辨率        : {video_size[0]}x{video_size[1]}  {video_fps:.1f} FPS",
        f"车位数量      : {len(slots)}",
        f"是否使用基准图: {'是' if reference is not None else '否'}",
        f"判定阈值      : {cfg.detector.threshold:.2f}",
        f"平均耗时      : {total_ms / max(1, n):.1f} ms/帧",
        "",
        analyzer.manager.summary_text(),
        "",
        f"前 {min(10, len(analyzer.manager.events))} 条事件:",
    ])
    for e in analyzer.manager.events[:10]:
        summary += f"  {e.stamp()}  车位 #{e.slot_index:02d}  {e.kind}\n"
    summary_path = os.path.join(rep_dir, "summary.txt")
    save_summary(summary_path, summary)
    analyzer.manager.export_csv(os.path.join(rep_dir, "events.csv"))
    print(summary)
    print(f"小结: {summary_path}")
    return 0


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def cmd_gui(args) -> int:
    from .gui import run_gui

    return run_gui(args)


# --------------------------------------------------------------------------
# 解析器
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="基于 OpenCV 的停车位识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="command")

    def add_common(p, need_slots: bool = True):
        if need_slots:
            p.add_argument("--slots", default=os.path.join("data", "slots.json"),
                           help="车位标注 JSON")
        p.add_argument("--reference", default=os.path.join("data", "lot_reference.png"),
                       help="空车位基准图（可选，用于背景差分）")

    p_gui = sub.add_parser("gui", help="打开桌面可视化界面")
    add_common(p_gui)
    p_gui.add_argument("--src", default=os.path.join("data", "parking_demo.mp4"))
    p_gui.add_argument("--kind", default="video", choices=["video", "image", "camera"])
    p_gui.add_argument("--config", default=None)
    p_gui.set_defaults(func=cmd_gui)

    p_video = sub.add_parser("video", help="处理视频文件")
    p_video.add_argument("--src", required=True)
    add_common(p_video)
    p_video.add_argument("--save", default=os.path.join("output", "result_video.mp4"),
                         help="输出视频路径")
    p_video.add_argument("--headless", action="store_true", help="不显示窗口（批处理）")
    p_video.add_argument("--every", type=int, default=1, help="每 N 帧处理一次")
    p_video.add_argument("--limit", type=int, default=0, help="只处理前 N 帧")
    p_video.add_argument("--threshold", type=float, default=None)
    p_video.add_argument("--config", default=None)
    p_video.set_defaults(func=cmd_video)

    p_image = sub.add_parser("image", help="处理单张图片")
    p_image.add_argument("--src", required=True)
    add_common(p_image)
    p_image.add_argument("--save", default=os.path.join("output", "result_image.png"))
    p_image.add_argument("--headless", action="store_true")
    p_image.add_argument("--config", default=None)
    p_image.set_defaults(func=cmd_image)

    p_cam = sub.add_parser("camera", help="实时摄像头监测")
    add_common(p_cam)
    p_cam.add_argument("--index", type=int, default=0)
    p_cam.add_argument("--width", type=int, default=1280)
    p_cam.add_argument("--height", type=int, default=720)
    p_cam.add_argument("--limit", type=int, default=0)
    p_cam.add_argument("--config", default=None)
    p_cam.set_defaults(func=cmd_camera)

    p_ann = sub.add_parser("annotate", help="标注车位")
    p_ann.add_argument("--image", required=True)
    p_ann.add_argument("--out", default=os.path.join("data", "slots.json"))
    p_ann.set_defaults(func=cmd_annotate)

    p_eval = sub.add_parser("evaluate", help="用真值评估检测精度")
    p_eval.add_argument("--data", default="data")
    p_eval.add_argument("--slots", default=None)
    p_eval.add_argument("--gt", default=None)
    p_eval.add_argument("--video", default=None)
    p_eval.add_argument("--reference", default=None)
    p_eval.add_argument("--config", default=None)
    p_eval.add_argument("--save-config", default=None)
    p_eval.add_argument("--calibrate", action="store_true", help="自动标定参数")
    p_eval.add_argument("--curve", action="store_true", help="输出阈值扫描曲线")
    p_eval.add_argument("--robustness", action="store_true", help="做退化条件下的鲁棒性测试")
    p_eval.add_argument("--robust-stride", type=int, default=2, help="鲁棒性测试的采样步长")
    p_eval.add_argument("--max-frames", type=int, default=0)
    p_eval.set_defaults(func=cmd_evaluate)

    p_demo = sub.add_parser("demo", help="生成合成演示数据")
    p_demo.add_argument("--out", default="data")
    p_demo.add_argument("--seed", type=int, default=7)
    p_demo.add_argument("--duration", type=float, default=demo_mod.DURATION_S)
    p_demo.add_argument("--fps", type=int, default=demo_mod.FPS)
    p_demo.add_argument("--perspective", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_rep = sub.add_parser("report", help="生成运行报告（曲线图、截图、小结）")
    p_rep.add_argument("--data", default="data")
    p_rep.add_argument("--slots", default=None)
    p_rep.add_argument("--video", default=None)
    p_rep.add_argument("--reference", default=None)
    p_rep.add_argument("--config", default=None)
    p_rep.add_argument("--out", default=os.path.join("output", "report"))
    p_rep.add_argument("--limit", type=int, default=0)
    p_rep.set_defaults(func=cmd_report)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:  # Windows 控制台默认不是 UTF-8，这里统一一下
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "command", None):
        # 不传子命令时默认启动桌面界面：这样在 PyCharm / VS Code 里直接点“运行”
        # 就能看到效果，不用记命令行参数。（全部用法仍可 python main.py -h 查看）
        print("未指定子命令，默认启动桌面界面；查看全部用法：python main.py -h")
        args.slots = os.path.join("data", "slots.json")
        args.reference = os.path.join("data", "lot_reference.png")
        args.src = os.path.join("data", "parking_demo.mp4")
        args.kind = "video"
        try:
            return cmd_gui(args)
        except Exception as exc:  # 没有图形环境时退回命令行帮助
            print(f"\n[提示] 桌面界面启动失败：{exc}")
            print("如果当前环境没有图形界面，请改用命令行模式，例如：")
            print("  python main.py video --src data/parking_demo.mp4 --slots data/slots.json")
            print()
            ap.print_help()
            return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
