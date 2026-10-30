"""统计项目代码规模（用于简历 / 文档中的客观数字）。"""

import glob
import os

paths = sorted(glob.glob("parking/*.py")) + ["main.py"] + sorted(glob.glob("tools/*.py"))
total = code = comment = blank = 0
for path in paths:
    if not os.path.exists(path):
        continue
    lines = open(path, encoding="utf-8").read().splitlines()
    n = len(lines)
    c = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
    b = sum(1 for l in lines if not l.strip())
    total += n
    code += c
    blank += b
    comment += n - c - b
    print(f"{path:<30}{n:>6} 行")

print("-" * 44)
print(f"文件数 {len(paths)}")
print(f"总行数 {total} | 有效代码 {code} | 注释 {comment} | 空行 {blank}")

# 演示数据规模
for name, pattern in (("演示视频", "data/*.mp4"), ("车位标注", "data/slots*.json")):
    for p in sorted(glob.glob(pattern)):
        print(f"{name}: {p}  {os.path.getsize(p) / 1024:.0f} KB")
