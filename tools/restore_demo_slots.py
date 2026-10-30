"""恢复演示数据集里被覆盖的车位标注（场景是确定性的，可原样重建）。

    python tools/restore_demo_slots.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parking.demo import build_scene_object
from parking.sources import read_image, write_image


def main() -> int:
    # 1) 先备份当前（可能是手工标注的）文件
    backup_dir = os.path.join("_backup")
    os.makedirs(backup_dir, exist_ok=True)
    if os.path.exists("data/slots.json"):
        shutil.copy2("data/slots.json", os.path.join(backup_dir, "slots_before_restore.json"))
        print("已备份原文件 ->", os.path.join(backup_dir, "slots_before_restore.json"))

    # 2) 用固定的随机种子重建场景，得到原始 20 个车位
    scene = build_scene_object(7)
    scene.slotset.save("data/slots.json")
    print(f"已恢复 data/slots.json：{len(scene.slotset)} 个车位")

    # 3) 同步刷新标注可视化图
    ref = read_image("data/lot_reference.png")
    if ref is not None:
        write_image("data/slots_overlay.png", scene.slotset.draw_polygons(ref))
        print("已刷新 data/slots_overlay.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
