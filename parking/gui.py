"""桌面可视化主程序（Tkinter + OpenCV）。

界面分三栏：

* 左栏  数据源、车位标注、运行控制、参数调节、导出（可滚动）
* 中栏  实时检测画面（OpenCV 渲染结果）
* 右栏  统计数字、车位状态列表、车辆进出日志

视频解码放在后台线程，检测与渲染在主线程按设定帧率推进；
快捷键：空格 播放/暂停，S 截图，R 设为基准图，A 标注车位，Q 退出。
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False

from .config import (ROOT, DATA_DIR, OUT_DIR, SystemConfig, save_json,
                     sync_thresholds)
from .pipeline import AnalysisResult, ParkingAnalyzer
from .slots import SlotSet
from .sources import (CameraSource, FrameSource, ImageSource, ThreadedSource, VideoSource,
                      read_image, write_image)

FONT_UI = ("Microsoft YaHei UI", 10)
FONT_UI_B = ("Microsoft YaHei UI", 10, "bold")
FONT_TITLE = ("Microsoft YaHei UI", 13, "bold")
FONT_BIG = ("Microsoft YaHei UI", 22, "bold")
FONT_SMALL = ("Microsoft YaHei UI", 9)

CLR_BG = "#1e1f22"
CLR_PANEL = "#26272b"
CLR_CARD = "#2f3136"
CLR_TEXT = "#e8eaee"
CLR_MUTED = "#9aa1ad"
CLR_FREE = "#5cc86e"
CLR_BUSY = "#e24848"
CLR_ACCENT = "#3f8ee0"


class ParkingApp:
    """主窗口。"""

    def __init__(self, root: tk.Tk, args=None):
        self.root = root
        self.args = args
        self.cfg = SystemConfig()
        self.slots: Optional[SlotSet] = None
        self.analyzer: Optional[ParkingAnalyzer] = None
        self.source: Optional[ThreadedSource] = None
        self.source_kind = "none"
        self.source_path = ""
        self.reference_image: Optional[np.ndarray] = None
        self._slots_path = ""

        self.playing = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_vis: Optional[np.ndarray] = None
        self.last_result: Optional[AnalysisResult] = None
        self._photo = None
        self._shown_size = (0, 0)
        self._last_event_count = 0
        self._frame_ms = 16.0
        self._next_due = 0.0
        self._tick_job = None

        self._build_ui()
        self._bind_keys()
        self._load_initial()

    # ==================================================================
    # 界面搭建
    # ==================================================================
    def _build_ui(self) -> None:
        r = self.root
        r.title("基于 OpenCV 的停车位识别系统")
        r.geometry("1420x880")
        r.minsize(1180, 720)
        r.configure(bg=CLR_BG)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=CLR_BG)
        style.configure("Panel.TFrame", background=CLR_PANEL)
        style.configure("TLabel", background=CLR_BG, foreground=CLR_TEXT, font=FONT_UI)
        style.configure("Panel.TLabel", background=CLR_PANEL, foreground=CLR_TEXT, font=FONT_UI)
        style.configure("Muted.TLabel", background=CLR_PANEL, foreground=CLR_MUTED, font=FONT_SMALL)
        style.configure("Title.TLabel", background=CLR_BG, foreground=CLR_TEXT, font=FONT_TITLE)
        style.configure("TButton", font=FONT_UI, padding=(8, 5))
        style.configure("Accent.TButton", font=FONT_UI_B, padding=(10, 6))
        style.configure("TCheckbutton", background=CLR_PANEL, foreground=CLR_TEXT, font=FONT_UI)
        style.map("TCheckbutton", background=[("active", CLR_PANEL)])
        style.configure("TScale", background=CLR_PANEL)
        style.configure("Treeview", background=CLR_CARD, fieldbackground=CLR_CARD,
                        foreground=CLR_TEXT, font=FONT_SMALL, rowheight=22, borderwidth=0)
        style.configure("Treeview.Heading", font=FONT_SMALL,
                        background=CLR_PANEL, foreground=CLR_MUTED)
        style.configure("TProgressbar", background=CLR_ACCENT, troughcolor=CLR_CARD,
                        borderwidth=0, thickness=10)
        style.configure("TLabelframe", background=CLR_PANEL, foreground=CLR_MUTED,
                        font=FONT_SMALL, borderwidth=1)
        style.configure("TLabelframe.Label", background=CLR_PANEL, foreground=CLR_MUTED,
                        font=FONT_SMALL)

        # ---------- 顶部标题 ----------
        header = tk.Frame(r, bg=CLR_BG, height=52)
        header.pack(side=tk.TOP, fill=tk.X, padx=14, pady=(10, 2))
        tk.Label(header, text="基于 OpenCV 的停车位识别系统", bg=CLR_BG, fg=CLR_TEXT,
                 font=("Microsoft YaHei UI", 17, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="纯传统图像处理 · 多特征融合 · 实时占用判定",
                 bg=CLR_BG, fg=CLR_MUTED, font=FONT_UI).pack(side=tk.LEFT, padx=14, pady=(6, 0))
        self.lbl_header_status = tk.Label(header, text="就绪", bg=CLR_BG, fg=CLR_FREE,
                                          font=FONT_UI_B)
        self.lbl_header_status.pack(side=tk.RIGHT, pady=(6, 0))

        body = tk.Frame(r, bg=CLR_BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=6)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_center(body)
        self._build_right(body)
        self._build_statusbar(r)

    # ---------------- 左栏（可滚动） ----------------
    def _build_left(self, parent) -> None:
        holder = tk.Frame(parent, bg=CLR_PANEL, width=262)
        holder.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        holder.grid_propagate(False)

        canvas = tk.Canvas(holder, bg=CLR_PANEL, highlightthickness=0, width=244,
                           yscrollincrement=24)
        vsb = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=CLR_PANEL)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_inner(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas(event):
            canvas.itemconfigure(win, width=event.width)

        def on_wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"

        inner.bind("<Configure>", on_inner)
        canvas.bind("<Configure>", on_canvas)
        canvas.bind_all("<MouseWheel>", on_wheel)

        left = inner

        def section(title: str) -> tk.Frame:
            f = tk.Frame(left, bg=CLR_PANEL)
            f.pack(fill=tk.X, padx=10, pady=(6, 1))
            tk.Label(f, text=title, bg=CLR_PANEL, fg=CLR_MUTED, font=FONT_SMALL,
                     anchor="w").pack(fill=tk.X)
            return f

        f1 = section("数据源")
        for text, cmd in (("打开视频文件", self.open_video), ("打开图片", self.open_image),
                          ("打开摄像头", self.open_camera)):
            ttk.Button(f1, text=text, command=cmd).pack(fill=tk.X, pady=1)

        f2 = section("车位标注")
        ttk.Button(f2, text="加载车位标注…", command=self.load_slots).pack(fill=tk.X, pady=1)
        ttk.Button(f2, text="在画面上标注车位", command=self.annotate_slots).pack(fill=tk.X, pady=1)
        self.lbl_slots = tk.Label(f2, text="车位: 未加载", bg=CLR_PANEL, fg=CLR_MUTED,
                                  font=FONT_SMALL, anchor="w", justify="left", wraplength=225)
        self.lbl_slots.pack(fill=tk.X, pady=(3, 0))

        f3 = section("空位基准图（可选）")
        ttk.Button(f3, text="把当前画面设为基准", command=self.set_reference).pack(fill=tk.X, pady=1)
        ttk.Button(f3, text="清除基准图", command=self.clear_reference).pack(fill=tk.X, pady=1)
        self.lbl_ref = tk.Label(f3, text="基准图: 未设置", bg=CLR_PANEL, fg=CLR_MUTED,
                                font=FONT_SMALL, anchor="w", justify="left", wraplength=225)
        self.lbl_ref.pack(fill=tk.X, pady=(3, 0))

        f4 = section("运行控制")
        self.btn_play = ttk.Button(f4, text="▶  开始", style="Accent.TButton",
                                   command=self.toggle_play)
        self.btn_play.pack(fill=tk.X, pady=1)
        rowc = tk.Frame(f4, bg=CLR_PANEL)
        rowc.pack(fill=tk.X, pady=1)
        ttk.Button(rowc, text="单帧分析", command=self.step_once).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(rowc, text="重置统计", command=self.reset_stats).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        f5 = section("检测参数")
        self.var_thr = tk.DoubleVar(value=self.cfg.detector.threshold)
        self._slider(f5, "判定阈值", self.var_thr, 0.15, 0.90, self.on_threshold)
        self.var_win = tk.IntVar(value=self.cfg.tracker.window)
        self._slider(f5, "平滑窗口", self.var_win, 1, 41, self.on_window, integer=True)
        self.var_stable = tk.IntVar(value=self.cfg.tracker.stable)
        self._slider(f5, "稳定帧数", self.var_stable, 1, 10, self.on_stable, integer=True)

        f6 = section("显示选项")
        self.var_idx = tk.BooleanVar(value=True)
        self.var_score = tk.BooleanVar(value=True)
        self.var_hist = tk.BooleanVar(value=True)
        row6 = tk.Frame(f6, bg=CLR_PANEL)
        row6.pack(fill=tk.X)
        for text, var, attr in (("编号", self.var_idx, "show_index"),
                                ("得分条", self.var_score, "show_score"),
                                ("曲线", self.var_hist, "show_history")):
            cb = ttk.Checkbutton(row6, text=text, variable=var,
                                 command=lambda a=attr, v=var: self.on_view_toggle(a, v))
            cb.pack(side=tk.LEFT, padx=(0, 2))

        f7 = section("导出")
        ttk.Button(f7, text="保存当前截图", command=self.save_snapshot).pack(fill=tk.X, pady=1)
        ttk.Button(f7, text="导出事件 CSV", command=self.export_csv).pack(fill=tk.X, pady=1)
        ttk.Button(f7, text="保存参数配置", command=self.save_config).pack(fill=tk.X, pady=1)
        tk.Frame(inner, bg=CLR_PANEL, height=12).pack(fill=tk.X)

    def _slider(self, parent, label: str, var, lo, hi, callback, integer: bool = False) -> None:
        row = tk.Frame(parent, bg=CLR_PANEL)
        row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(row, text=label, bg=CLR_PANEL, fg=CLR_TEXT, font=FONT_SMALL).pack(side=tk.LEFT)
        val_lbl = tk.Label(row, text="", bg=CLR_PANEL, fg=CLR_ACCENT, font=FONT_SMALL)
        val_lbl.pack(side=tk.RIGHT)

        def fmt(v):
            if integer:
                val_lbl.config(text=str(int(float(v))))
            else:
                val_lbl.config(text=f"{float(v):.2f}")

        fmt(var.get())
        s = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL,
                      command=lambda v, c=callback, f=fmt: (f(v), c(v)))
        s.pack(fill=tk.X, pady=(0, 3))

    # ---------------- 中栏 ----------------
    def _build_center(self, parent) -> None:
        wrap = tk.Frame(parent, bg="#141517")
        wrap.grid(row=0, column=1, sticky="nsew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(wrap, bg="#141517", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.create_text(400, 260, text="请打开视频 / 图片 / 摄像头", fill=CLR_MUTED,
                                font=("Microsoft YaHei UI", 14))

        bar = tk.Frame(wrap, bg="#141517")
        bar.grid(row=1, column=0, sticky="ew")
        self.lbl_src = tk.Label(bar, text="数据源: 未打开", bg="#141517", fg=CLR_MUTED,
                                font=FONT_SMALL, anchor="w")
        self.lbl_src.pack(side=tk.LEFT, padx=8, pady=4)
        self.lbl_time = tk.Label(bar, text="", bg="#141517", fg=CLR_MUTED, font=FONT_SMALL)
        self.lbl_time.pack(side=tk.RIGHT, padx=8)

    # ---------------- 右栏 ----------------
    def _build_right(self, parent) -> None:
        right = tk.Frame(parent, bg=CLR_PANEL, width=302)
        right.grid(row=0, column=2, sticky="nse", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="实时统计", bg=CLR_PANEL, fg=CLR_MUTED, font=FONT_SMALL,
                 anchor="w").pack(fill=tk.X, padx=12, pady=(10, 4))

        cards = tk.Frame(right, bg=CLR_PANEL)
        cards.pack(fill=tk.X, padx=10)
        self.card_free = self._card(cards, "空闲", "0", CLR_FREE, 0)
        self.card_busy = self._card(cards, "占用", "0", CLR_BUSY, 1)

        row = tk.Frame(right, bg=CLR_PANEL)
        row.pack(fill=tk.X, padx=12, pady=(8, 0))
        tk.Label(row, text="总车位", bg=CLR_PANEL, fg=CLR_MUTED, font=FONT_SMALL).pack(side=tk.LEFT)
        self.lbl_total = tk.Label(row, text="--", bg=CLR_PANEL, fg=CLR_TEXT, font=FONT_UI_B)
        self.lbl_total.pack(side=tk.RIGHT)

        row2 = tk.Frame(right, bg=CLR_PANEL)
        row2.pack(fill=tk.X, padx=12, pady=(4, 0))
        tk.Label(row2, text="占用率", bg=CLR_PANEL, fg=CLR_MUTED, font=FONT_SMALL).pack(side=tk.LEFT)
        self.lbl_rate = tk.Label(row2, text="--", bg=CLR_PANEL, fg=CLR_TEXT, font=FONT_UI_B)
        self.lbl_rate.pack(side=tk.RIGHT)
        self.bar_rate = ttk.Progressbar(right, maximum=100, value=0)
        self.bar_rate.pack(fill=tk.X, padx=12, pady=(2, 8))

        row3 = tk.Frame(right, bg=CLR_PANEL)
        row3.pack(fill=tk.X, padx=12)
        self.lbl_perf = tk.Label(row3, text="-- ms/帧 · -- FPS", bg=CLR_PANEL, fg=CLR_MUTED,
                                 font=FONT_SMALL, anchor="w")
        self.lbl_perf.pack(fill=tk.X)

        tk.Label(right, text="车位状态", bg=CLR_PANEL, fg=CLR_MUTED, font=FONT_SMALL,
                 anchor="w").pack(fill=tk.X, padx=12, pady=(10, 2))
        tree_wrap = tk.Frame(right, bg=CLR_PANEL, height=250)
        tree_wrap.pack(fill=tk.X, padx=10)
        tree_wrap.pack_propagate(False)
        self.tree = ttk.Treeview(tree_wrap, columns=("id", "state", "score"),
                                 show="headings", height=10)
        for col, text, width, anchor in (("id", "车位", 60, "center"),
                                         ("state", "状态", 80, "center"),
                                         ("score", "得分", 80, "center")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor)
        self.tree.tag_configure("free", foreground=CLR_FREE)
        self.tree.tag_configure("busy", foreground=CLR_BUSY)
        sb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(right, text="车辆进出日志", bg=CLR_PANEL, fg=CLR_MUTED, font=FONT_SMALL,
                 anchor="w").pack(fill=tk.X, padx=12, pady=(10, 2))
        log_wrap = tk.Frame(right, bg=CLR_PANEL)
        log_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(log_wrap, height=7, width=26, bg=CLR_CARD, fg=CLR_TEXT,
                           font=FONT_SMALL, relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED,
                           insertbackground=CLR_TEXT)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb2 = ttk.Scrollbar(log_wrap, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=sb2.set)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.tag_configure("free", foreground=CLR_FREE)
        self.log.tag_configure("busy", foreground=CLR_BUSY)
        self.log.tag_configure("muted", foreground=CLR_MUTED)

    def _card(self, parent, caption: str, value: str, color: str, col: int) -> tk.Label:
        card = tk.Frame(parent, bg=CLR_CARD, height=78)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 6, 0))
        parent.columnconfigure(col, weight=1)
        card.pack_propagate(False)
        tk.Label(card, text=caption, bg=CLR_CARD, fg=CLR_MUTED, font=FONT_SMALL).pack(
            anchor="w", padx=10, pady=(8, 0))
        lbl = tk.Label(card, text=value, bg=CLR_CARD, fg=color, font=FONT_BIG)
        lbl.pack(anchor="w", padx=8, pady=(0, 6))
        return lbl

    # ---------------- 状态栏 ----------------
    def _build_statusbar(self, parent) -> None:
        bar = tk.Frame(parent, bg="#191a1d", height=26)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.lbl_status = tk.Label(bar, text="就绪", bg="#191a1d", fg=CLR_MUTED, font=FONT_SMALL,
                                   anchor="w")
        self.lbl_status.pack(side=tk.LEFT, padx=12, pady=3)
        tk.Label(bar, text="空格 播放/暂停   S 截图   R 设基准   A 标注   Q 退出",
                 bg="#191a1d", fg="#6d7480", font=FONT_SMALL).pack(side=tk.RIGHT, padx=12)

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<KeyPress-s>", lambda e: self.save_snapshot())
        self.root.bind("<KeyPress-r>", lambda e: self.set_reference())
        self.root.bind("<KeyPress-a>", lambda e: self.annotate_slots())
        self.root.bind("<KeyPress-q>", lambda e: self.on_close())

    # ==================================================================
    # 初始化 / 数据源
    # ==================================================================
    def _load_initial(self) -> None:
        a = self.args
        slots_path = getattr(a, "slots", None) or os.path.join(DATA_DIR, "slots.json")
        if not os.path.isabs(slots_path):
            slots_path = os.path.join(ROOT, slots_path)
        if os.path.exists(slots_path):
            self._load_slots_file(slots_path)

        ref_path = getattr(a, "reference", None) or os.path.join(DATA_DIR, "lot_reference.png")
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(ROOT, ref_path)
        if os.path.exists(ref_path):
            self.reference_image = read_image(ref_path)
            if self.reference_image is not None:
                self.lbl_ref.config(text=f"基准图: {os.path.basename(ref_path)}")

        src = getattr(a, "src", None) or os.path.join(DATA_DIR, "parking_demo.mp4")
        if not os.path.isabs(src):
            src = os.path.join(ROOT, src)
        if os.path.exists(src):
            kind = getattr(a, "kind", "video")
            self.open_source(kind, src, autoplay=True)

    def _rebuild_analyzer(self) -> None:
        if self.slots is None:
            return
        self.analyzer = ParkingAnalyzer(self.slots, self.cfg, reference=self.reference_image,
                                        title="停车场车位实时监测")
        self._fill_tree()
        name = os.path.basename(self._slots_path) if self._slots_path else "内存标注"
        self.lbl_slots.config(text=f"车位: {len(self.slots)} 个 ({name})")

    def _load_slots_file(self, path: str) -> bool:
        try:
            self.slots = SlotSet.load(path)
        except Exception as exc:
            messagebox.showerror("加载失败", f"车位标注文件无法读取:\n{exc}")
            return False
        self._slots_path = path
        self._rebuild_analyzer()
        self.set_status(f"已加载车位标注: {path}")
        return True

    def open_source(self, kind: str, path: str, autoplay: bool = False) -> None:
        self.pause()
        if self.source is not None:
            self.source.stop()
            self.source = None

        if kind == "video":
            src: FrameSource = VideoSource(path)
        elif kind == "image":
            src = ImageSource(path)
        elif kind == "camera":
            src = CameraSource(int(path) if str(path).isdigit() else 0)
        else:
            return
        if not src.open():
            messagebox.showerror("打开失败", f"无法打开数据源:\n{path}")
            return

        if isinstance(src, ImageSource):
            self.source = None
            self.last_frame = src.image.copy() if src.image is not None else None
            self.source_kind = "image"
            self.source_path = path
            self.lbl_src.config(text=f"数据源: 图片 {os.path.basename(path)}")
            self._frame_ms = 0
            self._process_frame(self.last_frame, 0.0)
            self.set_status("图片已加载，按“单帧分析”可重新检测")
            return

        self.source = ThreadedSource(src)
        self.source.start()
        self.source_kind = kind
        self.source_path = path
        fps = src.fps if not src.is_live else 25.0
        self._frame_ms = 1000.0 / max(1.0, fps)
        label = os.path.basename(path) if kind == "video" else f"摄像头 {path}"
        extra = f"{src.size[0]}x{src.size[1]}  {fps:.0f}FPS" if kind == "video" else ""
        self.lbl_src.config(text=f"数据源: {label}   {extra}")
        self.set_status(f"已打开 {label}")
        if autoplay:
            self.play()

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频文件", initialdir=os.path.join(ROOT, "data"),
            filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("所有文件", "*.*")])
        if path:
            self.open_source("video", path, autoplay=True)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图片", initialdir=os.path.join(ROOT, "data"),
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")])
        if path:
            self.open_source("image", path)

    def open_camera(self) -> None:
        if not messagebox.askokcancel("摄像头", "将打开系统默认摄像头（索引 0），继续？"):
            return
        self.open_source("camera", "0", autoplay=True)

    def load_slots(self) -> None:
        path = filedialog.askopenfilename(
            title="选择车位标注", initialdir=os.path.join(ROOT, "data"),
            filetypes=[("车位标注", "*.json"), ("所有文件", "*.*")])
        if path and self._load_slots_file(path):
            self.reset_stats()

    # ==================================================================
    # 播放循环
    # ==================================================================
    def play(self) -> None:
        if self.source is None:
            if self.last_frame is not None:
                self._process_frame(self.last_frame.copy(), 0.0)
                return
            messagebox.showinfo("提示", "请先打开视频、图片或摄像头")
            return
        self.playing = True
        self.btn_play.config(text="⏸  暂停")
        self.set_header_status("运行中", CLR_FREE)
        self._next_due = 0.0
        if self._tick_job is None:
            self._tick_job = self.root.after(4, self._tick)

    def pause(self) -> None:
        self.playing = False
        self.btn_play.config(text="▶  开始")
        self.set_header_status("已暂停", CLR_MUTED)

    def toggle_play(self) -> None:
        self.pause() if self.playing else self.play()

    def step_once(self) -> None:
        if self.source is not None:
            ok, frame, t = self.source.read(timeout=0.5)
            if ok:
                self._process_frame(frame, t)
                return
        if self.last_frame is not None:
            self._process_frame(self.last_frame.copy(), 0.0)

    def _tick(self) -> None:
        self._tick_job = None
        if not self.playing:
            return
        now = time.perf_counter()
        if self._next_due and now < self._next_due:
            self._tick_job = self.root.after(max(4, int((self._next_due - now) * 1000)),
                                             self._tick)
            return

        if self.source is None:
            self.pause()
            return
        ok, frame, t = self.source.read(timeout=0.02)
        if not ok:
            self._tick_job = self.root.after(30, self._tick)
            return
        self._process_frame(frame, t)
        self._next_due = now + self._frame_ms / 1000.0
        self._tick_job = self.root.after(4, self._tick)

    # ==================================================================
    # 单帧处理
    # ==================================================================
    def _ensure_analyzer(self) -> bool:
        if self.analyzer is None:
            if self.slots is None:
                self.set_status("还没有车位标注：请先“加载车位标注”或在画面上标注")
                return False
            self._rebuild_analyzer()
        return self.analyzer is not None

    def _process_frame(self, frame: Optional[np.ndarray], t: float) -> None:
        if frame is None:
            return
        self.last_frame = frame.copy()
        if not self._ensure_analyzer():
            self._show_plain(frame)
            return
        assert self.analyzer is not None
        warn = self.analyzer.size_mismatch(frame)
        try:
            vis, result = self.analyzer.process(frame, t)
        except Exception as exc:  # 避免一个异常把界面卡死
            self.set_status(f"检测异常: {exc}")
            self._show_plain(frame)
            return
        self.last_vis = vis
        self.last_result = result
        self._draw(vis)
        self._update_stats(result)
        self._update_events()
        if warn:
            self.set_status(warn)

    def _show_plain(self, frame: np.ndarray) -> None:
        self.last_vis = frame
        self._draw(frame)

    def _draw(self, vis: np.ndarray) -> None:
        if not _HAS_PIL:
            return
        cw = max(80, self.canvas.winfo_width())
        ch = max(60, self.canvas.winfo_height())
        h, w = vis.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self._shown_size = (nw, nh)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

    def _on_canvas_resize(self, event) -> None:
        if self.last_vis is None:
            return
        cw = max(80, self.canvas.winfo_width())
        ch = max(60, self.canvas.winfo_height())
        h, w = self.last_vis.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        if abs(nw - self._shown_size[0]) > 2 or abs(nh - self._shown_size[1]) > 2:
            self._draw(self.last_vis)

    # ---------------- 统计面板 ----------------
    def _update_stats(self, result: AnalysisResult) -> None:
        stats = result.stats
        if stats is None:
            return
        self.card_free.config(text=str(stats.free))
        self.card_busy.config(text=str(stats.occupied))
        self.lbl_total.config(text=str(stats.total))
        self.lbl_rate.config(text=f"{stats.rate * 100:.0f}%")
        self.bar_rate.config(value=stats.rate * 100)
        self.lbl_perf.config(text=f"{stats.elapsed_ms:.1f} ms/帧 · {stats.fps:.0f} FPS")
        self.lbl_time.config(text=f"t = {result.time_s:.1f} s")

        items = self.tree.get_children()
        if len(items) != len(result.states):
            self._fill_tree()
            items = self.tree.get_children()
        for i, (item, occ, sc) in enumerate(zip(items, result.states, result.scores)):
            self.tree.item(item, values=(f"#{i + 1:02d}", "占用" if occ else "空闲", f"{sc:.2f}"),
                           tags=("busy" if occ else "free",))
        self.set_header_status(f"运行中 · 空闲 {stats.free}/{stats.total}", CLR_FREE)

    def _fill_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        n = len(self.slots) if self.slots else 0
        for i in range(1, n + 1):
            self.tree.insert("", tk.END, values=(f"#{i:02d}", "空闲", "0.00"), tags=("free",))

    def _update_events(self) -> None:
        if self.analyzer is None:
            return
        events = self.analyzer.manager.events
        if len(events) == self._last_event_count:
            return
        for ev in events[self._last_event_count:]:
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, f"{ev.stamp()}  ", ("muted",))
            self.log.insert(tk.END, f"车位 #{ev.slot_index:02d}  ", ())
            self.log.insert(tk.END, f"{ev.kind}\n", ("busy" if ev.occupied else "free",))
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)
        self._last_event_count = len(events)

    # ---------------- 状态栏 ----------------
    def set_status(self, text: str) -> None:
        self.lbl_status.config(text=text)

    def set_header_status(self, text: str, color: str = CLR_FREE) -> None:
        self.lbl_header_status.config(text=text, fg=color)

    # ==================================================================
    # 交互命令
    # ==================================================================
    def on_threshold(self, value) -> None:
        sync_thresholds(self.cfg, float(value))

    def on_window(self, value) -> None:
        self.cfg.tracker.window = max(1, int(float(value)))
        if self.analyzer is not None:
            self.analyzer.tracker.cfg.window = self.cfg.tracker.window

    def on_stable(self, value) -> None:
        self.cfg.tracker.stable = max(1, int(float(value)))
        if self.analyzer is not None:
            self.analyzer.tracker.cfg.stable = self.cfg.tracker.stable

    def on_view_toggle(self, attr: str, var) -> None:
        setattr(self.cfg.view, attr, bool(var.get()))
        if self.last_vis is not None and self.last_result is not None and self.analyzer:
            vis = self.analyzer.render(self.last_frame, self.last_result)
            self.last_vis = vis
            self._draw(vis)

    def set_reference(self) -> None:
        frame = self.last_frame
        if frame is None:
            messagebox.showinfo("提示", "当前没有可用画面")
            return
        if self.analyzer is None and not self._ensure_analyzer():
            return
        assert self.analyzer is not None
        self.reference_image = frame.copy()
        self.analyzer.set_reference(self.reference_image)
        self.lbl_ref.config(text="基准图: 当前帧（已启用背景差分）")
        self.set_status("已把当前画面设为空位基准图，背景差分通道已启用")
        self.reset_stats()

    def clear_reference(self) -> None:
        self.reference_image = None
        if self.analyzer is not None:
            self.analyzer.clear_reference()
        self.lbl_ref.config(text="基准图: 未设置")
        self.set_status("已清除基准图")

    def reset_stats(self) -> None:
        if self.analyzer is not None:
            self.analyzer.manager.reset()
            self.analyzer.tracker.reset_with([False] * len(self.slots or []))
            self._last_event_count = 0
            self.log.configure(state=tk.NORMAL)
            self.log.delete("1.0", tk.END)
            self.log.configure(state=tk.DISABLED)
        self.set_status("统计已重置")

    def annotate_slots(self) -> None:
        frame = self.last_frame
        if frame is None:
            messagebox.showinfo("提示", "请先打开视频/图片，再在其画面上标注车位")
            return
        os.makedirs(DATA_DIR, exist_ok=True)
        temp_path = os.path.join(DATA_DIR, "_annotate_source.png")
        write_image(temp_path, frame)
        was_playing = self.playing
        self.pause()
        out_path = os.path.join(DATA_DIR, "slots.json")
        try:
            from .annotate import SlotAnnotator

            tool = SlotAnnotator(read_image(temp_path), self.slots)
            result = tool.run()
            if not result:
                self.set_status("未标注任何车位")
                return
            h, w = frame.shape[:2]
            slotset = SlotSet(result, image_size=(w, h))
            slotset.save(out_path)
            self.slots = slotset
            self._slots_path = out_path
            self._rebuild_analyzer()
            self.set_status(f"已保存 {len(slotset)} 个车位标注")
        except Exception as exc:
            messagebox.showerror("标注失败", f"{exc}")
        finally:
            if was_playing:
                self.play()

    def save_snapshot(self) -> None:
        if self.last_vis is None:
            messagebox.showinfo("提示", "当前没有画面")
            return
        os.makedirs(OUT_DIR, exist_ok=True)
        name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(OUT_DIR, name)
        write_image(path, self.last_vis)
        self.set_status(f"截图已保存: {path}")

    def export_csv(self) -> None:
        if self.analyzer is None:
            messagebox.showinfo("提示", "还没有分析结果")
            return
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"events_{datetime.now().strftime('%H%M%S')}.csv")
        self.analyzer.manager.export_csv(path)
        self.set_status(f"事件报表已导出: {path}")

    def save_config(self) -> None:
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "config.json")
        save_json(path, self.cfg.to_dict())
        self.set_status(f"参数配置已保存: {path}")

    def on_close(self) -> None:
        self.playing = False
        if self.source is not None:
            self.source.stop()
        self.root.destroy()


def run_gui(args=None) -> int:
    root = tk.Tk()
    app = ParkingApp(root, args)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    return 0


def main() -> int:
    return run_gui(None)
