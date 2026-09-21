#!/usr/bin/env python3
"""
用按键时间表 + 键对错误率表 + 错误修正时间实测值组装标准产物
（2026-08-31 定稿; 2026-09-19 术语统一 + 段类重参数化; 2026-09-21 段位特征落地）

读: 按键时间表.npz     按**段角色** 6 片 (version 3) 按键时间 S(角色,…) ms
        角色 = (码长, 段位): L2i1 角点 / L3i1·L4i1 首段 / L3i2 短尾 / L4i2 中段 / L4i3 长尾
        (2026-09-21 段位特征落地: 同一 (p,a,b,n) 签名在 3 键与 4 键码中角色不同, 4-D 单表
         无法表达 → 拆片; 与错误侧 09-19 段类重参数化同因。旧版单表 F(31,30,30,31) version 2)
    键对错误率表.txt    900 键对 × 5 段类 P_err(a,b | 段类)
       (2026-09-19 段类 = (有前键, 有后键, 段位): c0 角点/c1 首段/c2 短尾/c3 中段/c4 长尾。
        旧 4 列签名 (有无前键×有无后键) 在仅 2/4 键数据上与 (码长,段位) 一一对应, 加入
        3 键后「尾段」被劈为 c2 (3键seg2) 与 c4 (4键seg3) → 旧模型拟合不足
        D=10.17/df=2/p=0.006, 新模型 D=0.14 五类饱和; 09-21 复核 p=0.0020 仍显著)
    错误修正时间-实测值.txt  标量 cost_ms (实测净成本; label 行式另含手别诊断行, 组装只读 cost_ms)

写: 段当量表.npz  按**段角色** 6 片 (段当量 = 按键时间 + cost × P_err(该角色的段类))
      —— 2026-09-21 起 6 片 (version 5; 09-19 的 5 片把 L3i1 与 L4i1 合成 seg_p0n1s1,
         时间侧段位落地后两者按键时间不同, 必须分开; 错误侧两者的错误率仍同属 c1, 已验可合并)
    总当量-2-4键.txt  2/3/4 键总当量 = 段当量求和 (ms, 期望耗时原值不归一化)

角色 → 段类与总当量映射:
  段类 (错误侧)     : c0 角点 / c1 首段 / c2 短尾 / c3 中段 / c4 长尾
  角色 → 段类       : L2i1→c0  L3i1→c1  L3i2→c2  L4i1→c1  L4i2→c3  L4i3→c4
  T₂(xy)   = seg_L2i1[x,y]
  T₃(xyz)  = seg_L3i1[x,y,z] + seg_L3i2[x,y,z]
  T₄(wxyz) = seg_L4i1[w,x,y] + seg_L4i2[w,x,y,z] + seg_L4i3[x,y,z]

运行顺序: 分析-按键时间.py --full (出按键时间表) → 分析-错误率.py (出错误率表)
       → 分析-错误修正时间.py (出错误修正时间实测值) → 本脚本。
"""
from pathlib import Path
import sys
import numpy as np

PROJ = Path(__file__).resolve().parent / "产物"   # 2026-09-05 目录重组: 产物入 产物/
# 错误修正时间 = 实测标量 (09-06 起动态来源; 09-14 曾部署分手版, 09-15 用户决策回退——
#   手效应是退格位依赖的布局结构 (BS=CapsLock→左目标同手串行惩罚), 入表牺牲通用性;
#   txt 为 label 行式, 手别行保留为诊断, 组装只读 cost_ms 行)
try:
    _cost = {}
    for l in (PROJ / "错误修正时间-实测值.txt").read_text(encoding="utf-8").splitlines():
        if not l or l.startswith("#") or l.startswith("label"):
            continue
        p_ = l.split("\t")
        _cost[p_[0]] = p_[1:]
    ERR_MS, ERR_N = float(_cost["cost_ms"][0]), int(_cost["cost_ms"][1])
except Exception:
    sys.exit("缺 产物/错误修正时间-实测值.txt — 先跑 分析-错误修正时间.py (错误修正时间实测值来源)")
L = "abcdefghijklmnopqrstuvwxyz;,./"   # 30 键 (3 行 10 列完整 QWERTY)

# 段角色片 → 错误侧段类 (索引与字母序与 分析-按键时间.py ROLE_CELL 一致)
ROLE_CLS = {"L2i1": 0, "L3i1": 1, "L3i2": 2, "L4i1": 1, "L4i2": 3, "L4i3": 4}
ROLE_NAMES = ["L2i1", "L3i1", "L3i2", "L4i1", "L4i2", "L4i3"]

z = np.load(PROJ / "按键时间表.npz", allow_pickle=False)
assert tuple(z["letters"]) == tuple(L) and int(z["version"]) == 3, \
    "按键时间表版本不符 (期望 version 3 = 段角色 6 片); 先跑 分析-按键时间.py --full"
R = {k: z[f"r_{k}"].astype(np.float64) for k in ROLE_NAMES}
assert R["L2i1"].shape == (30, 30)
assert R["L3i1"].shape == R["L3i2"].shape == R["L4i1"].shape == R["L4i3"].shape == (30, 30, 30)
assert R["L4i2"].shape == (30, 30, 30, 30)
print(f"按键时间表: 6 角色片, {sum(v.size for v in R.values()):,} 条 (version {int(z['version'])})")

# ── 键对错误率表: 900 键对 × 5 段类 ──
COLS = ["err_p0n0s1", "err_p0n1s1", "err_p1n0s2", "err_p1n1s2", "err_p1n0s3"]
CLS_LAB = ["c0 角点(∅,∅,1)", "c1 首段(∅,有,1)", "c2 短尾(有,∅,2)",
           "c3 中段(有,有,2)", "c4 长尾(有,∅,3)"]
perr = {}
with open(PROJ / "键对错误率表.txt", encoding="utf-8") as f:
    head = f.readline().rstrip("\n").split("\t")
    ci = [head.index(c) for c in COLS]
    for line in f:
        parts = line.rstrip("\n").split("\t")
        perr[parts[0]] = [float(parts[i]) for i in ci]
P5 = np.zeros((5, len(L), len(L)))               # [段类, a, b]
for ab, v in perr.items():
    i, j = L.index(ab[0]), L.index(ab[1])
    for c in range(5):
        P5[c, i, j] = v[c]
assert len(perr) == len(L)**2, "错误率表应覆盖 900 键对"
print(f"键对错误率: {len(perr)} 对 × 5 段类 (错误修正时间实测标量 {ERR_MS:.1f}ms, n={ERR_N})  类均值: "
      + " / ".join(f"c{c} {P5[c].mean()*100:.2f}%" for c in range(5)))

# ── 段当量 = 按键时间(角色) + 错误修正时间 × P_err(该角色的段类) ──
# 每个角色的 P_err 广播到该角色的索引维: 首段 (a,b,·) / 末段 (·,a,b) / 中段 (·,a,b,·) / 角点 (a,b)
BR = {"L2i1": P5[0],                      # (a, b) 角点
      "L3i1": P5[1][:, :, None],          # (a, b, n) 首段 → P 在 (a,b) 上广播
      "L3i2": P5[2][None, :, :],          # (p, a, b) 末段 → P 在 (a,b) 上广播
      "L4i1": P5[1][:, :, None],
      "L4i2": P5[3][None, :, :, None],
      "L4i3": P5[4][None, :, :]}
seg = {f"seg_{k}": R[k] + ERR_MS * BR[k] for k in ROLE_NAMES}
np.savez_compressed(PROJ / "段当量表.npz", **seg,
                    letters=np.array(list(L)),
                    version=np.int64(5), cost_ms=np.float64(ERR_MS),
                    segments=np.array([f"seg_{k}" for k in ROLE_NAMES]),
                    roles=np.array(ROLE_NAMES),
                    errcls=np.array([f"c{ROLE_CLS[k]}" for k in ROLE_NAMES]),
                    note=np.array(f"段当量 = 按键时间 S(段角色,…) + {ERR_MS:.1f}×P_err(a,b|该角色段类) ms; "
                                  "段角色 = (码长,段位): L2i1 角点/L3i1·L4i1 首段(同段类 c1)/"
                                  "L3i2 短尾/L4i2 中段/L4i3 长尾; "
                                  "T2=seg_L2i1 T3=seg_L3i1+seg_L3i2 "
                                  "T4=seg_L4i1+seg_L4i2+seg_L4i3 (2026-09-21 段位特征落地, "
                                  "version 5; 旧 5 片版把 L3i1 与 L4i1 合成一片 — 时间侧角色不同必须分)"))
print(f"输出: 段当量表.npz  (6 个段角色片; 段当量 = 按键时间 + {ERR_MS:.1f}×段类错误率, version 5)")

# ── 2-4 键总当量表 = 段当量求和 ──
T2c = np.maximum(seg["seg_L2i1"], 0.0)
T3c = seg["seg_L3i1"] + seg["seg_L3i2"]
T4c = seg["seg_L4i1"][:, :, :, None] + seg["seg_L4i2"] + seg["seg_L4i3"][None, :, :, :]

total = len(L)**2 + len(L)**3 + len(L)**4
with open(PROJ / "总当量-2-4键.txt", "w", encoding="utf-8") as f:
    f.write(f"# 2/3/4 键总当量 (ms, 期望耗时原值含错误成本: 段当量 = 按键时间 + {ERR_MS:.1f}ms×P_err(a,b|段类), 实测错误修正时间标量)\n")
    f.write("# 段角色 = (码长,段位); 段类 = (有前键,有后键,段位): c0 角点/c1 首段/c2 短尾/c3 中段/c4 长尾\n")
    f.write("# T2=seg_L2i1(xy) T3=seg_L3i1(xyz)+seg_L3i2(xyz) "
            "T4=seg_L4i1(wxy)+seg_L4i2(wxyz)+seg_L4i3(xyz)  (2026-09-21 段位特征落地)\n")
    f.write("code\t当量\n")
    buf = []
    for (i, j), v in np.ndenumerate(T2c):
        buf.append(f"{L[i]}{L[j]}\t{v:.2f}\n")
    for (i, j, k), v in np.ndenumerate(T3c):
        buf.append(f"{L[i]}{L[j]}{L[k]}\t{v:.2f}\n")
    f.writelines(buf); buf = []
    for (i, j, k, m), v in np.ndenumerate(T4c):
        buf.append(f"{L[i]}{L[j]}{L[k]}{L[m]}\t{v:.2f}\n")
        if len(buf) >= 100_000:
            f.writelines(buf); buf = []
    f.writelines(buf)
print(f"输出: 总当量-2-4键.txt ({total:,} 条, ms, 段当量求和)")

# ── 自检: 角色映射与公式还原 ──
a, b, c, d = "t", "h", "e", "s"
ki = {ch: L.index(ch) for ch in (a, b, c, d)}
t2 = seg["seg_L2i1"][ki[a], ki[b]]
t3 = seg["seg_L3i1"][ki[a], ki[b], ki[c]] + seg["seg_L3i2"][ki[a], ki[b], ki[c]]
t4 = (seg["seg_L4i1"][ki[a], ki[b], ki[c]] + seg["seg_L4i2"][ki[a], ki[b], ki[c], ki[d]]
      + seg["seg_L4i3"][ki[b], ki[c], ki[d]])
print(f"自检 thes: T₂={t2:.1f} T₃={t3:.1f} T₄={t4:.1f} (公式 = 表列值, 差应为 0)")
# 段位落地的结构影响 (示意, 非旧模型实测差): 3 键码 (a,b,c) 的段现在各取自己的角色片,
# 旧单表版两段都被迫借 4 键角色片 (L4i1 首段 / L4i3 长尾)
d_first = seg["seg_L3i1"][ki[a], ki[b], ki[c]] - seg["seg_L4i1"][ki[a], ki[b], ki[c]]
d_tail = seg["seg_L3i2"][ki[a], ki[b], ki[c]] - seg["seg_L4i3"][ki[a], ki[b], ki[c]]
print(f"自检 thes 角色片差 (仅示结构, 旧单表版 3 键段借 4 键片): "
      f"首段 L3i1−L4i1 = {d_first:+.1f}ms; 短尾 L3i2−长尾 L4i3 = {d_tail:+.1f}ms")
