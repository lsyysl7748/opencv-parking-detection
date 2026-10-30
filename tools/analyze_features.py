"""调参诊断脚本：比较不同 ROI 内缩比例下的特征可分性。"""
import sys, os, json
sys.path.insert(0, r"D:\deepseek\parking_system")
import numpy as np
import cv2

from parking.config import DetectorConfig, SystemConfig
from parking.detector import SlotDetector
from parking.evaluate import load_ground_truth
from parking.features import FEATURE_KEYS
from parking.slots import SlotSet, SlotExtractor

D = r"D:\deepseek\parking_system\data"
slots = SlotSet.load(os.path.join(D, "slots.json"))
gt = load_ground_truth(os.path.join(D, "ground_truth.json"))
ref = cv2.imread(os.path.join(D, "lot_reference.png"))

cap = cv2.VideoCapture(os.path.join(D, "parking_demo.mp4"))
frames = []
idx = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if idx % 10 == 0:
        frames.append((f, gt[idx]))
    idx += 1
cap.release()
print("frames sampled:", len(frames))


def auc(free, busy):
    free = np.asarray(free); busy = np.asarray(busy)
    if free.size == 0 or busy.size == 0:
        return 0.5
    allv = np.concatenate([free, busy])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # 处理并列
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    r_busy = ranks[free.size:].sum()
    n1, n0 = busy.size, free.size
    return (r_busy - n1 * (n1 + 1) / 2.0) / (n1 * n0)


for margin in (0.02, 0.04, 0.06, 0.08, 0.12):
    cfg = DetectorConfig(inner_margin=margin)
    det = SlotDetector(cfg)
    det.set_reference(ref, slots)
    free_rows, busy_rows = [], []
    for f, g in frames:
        r = det.detect(f, slots)
        for i, gg in enumerate(g):
            if gg == 0:
                free_rows.append(r.raw[i])
            elif gg == 1:
                busy_rows.append(r.raw[i])
    print(f"\n=== inner_margin={margin}  free={len(free_rows)} busy={len(busy_rows)}")
    for k in FEATURE_KEYS:
        fv = [x.get(k, 0.0) for x in free_rows]
        bv = [x.get(k, 0.0) for x in busy_rows]
        print(f"  {k:<9} free_mean={np.mean(fv):8.4f} busy_mean={np.mean(bv):8.4f} "
              f"AUC={auc(fv, bv):.3f}")
