"""基于 OpenCV 的停车位识别系统 —— 统一入口。

    python main.py              启动桌面可视化界面（推荐）
    python main.py gui          启动桌面可视化界面
    python main.py video --src data/parking_demo.mp4 --slots data/slots.json
    python main.py demo         重新生成演示数据
    python main.py evaluate --calibrate --curve    精度评估
    python main.py -h           查看全部用法

在 PyCharm / VS Code 里直接点“运行”即可（不带参数默认打开界面）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from parking.cli import main  # noqa: E402
except ModuleNotFoundError as exc:  # 常见于 PyCharm 选错了解释器
    missing = getattr(exc, "name", "") or str(exc)
    print("=" * 72)
    print(f"启动失败：当前 Python 环境缺少依赖 `{missing}`")
    print(f"当前使用的解释器：{sys.executable}")
    if missing in ("cv2", "numpy"):
        print()
        print("解决方式（二选一）：")
        print("  A. 装依赖：")
        print(f'     "{sys.executable}" -m pip install -r requirements.txt')
        print()
        print("  B. 在 IDE 里换成一个已经装了 OpenCV 的解释器：")
        print("     PyCharm: Settings → Project → Python Interpreter → 选择/新建虚拟环境")
        print("     VS Code: Ctrl+Shift+P → Python: Select Interpreter")
        print("     （若机器上已有装好 opencv 的虚拟环境，可直接 Add → Existing 选中它的 python.exe）")
    else:
        print()
        print("请安装依赖：")
        print(f'  "{sys.executable}" -m pip install -r requirements.txt')
    print("=" * 72)
    raise SystemExit(1)

if __name__ == "__main__":
    raise SystemExit(main())
