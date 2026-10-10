#!/usr/bin/env python3
"""
用按键时间表 + 错误率表 + 错误修正时间实测值组装标准产物
（2026-08-31 定稿; 2026-09-19 术语统一 + 段类重参数化; 2026-09-21 段位特征落地;
  2026-10-10 错误侧改 (p,a,b,n) 条件化、按段角色 6 片 —— 与时间表同构）

读: 按键时间表.npz     按**段角色** 6 片 (version 3) 按键时间 S(角色,…) ms
        角色 = (码长, 段位): L2i1=2键第1段 / L3i1=3键第1段 / L3i2=3键第2段 /
        L4i1=4键第1段 / L4i2=4键第2段 / L4i3=4键第3段 (2026-10-07 展示名; 旧称 角点/首段/短尾/中段/长尾)
        (2026-09-21 段位特征落地: 同一 (p,a,b,n) 签名在 3 键与 4 键码中角色不同, 4-D 单表
         无法表达 → 拆片。旧版单表 F(31,30,30,31) version 2)
    错误率表.npz        **同构 6 段角色片** P_err(角色, 该角色的键) (version 1, 2026-10-10 起)
        r_L2i1[a,b] / r_L3i1·r_L4i1[a,b,n] / r_L3i2·r_L4i3[p,a,b] / r_L4i2[p,a,b,n]
        —— 索引键序与时间表相同 ⇒ 两表逐片直接相加, 无需广播。
        取代旧 键对错误率表.txt（900 键对 × 5 段类, 2026-09-19~10-10）：那个视图把 p/n 身份
        抹成"有前键/有后键"两列, 无法与时间侧 S(p,a,b,n) 对齐; 现按真实上下文逐码取用。
        (模型形式与依据见 分析-错误率.py / README §4.6)
    错误修正时间-实测值.txt  标量 cost_ms (实测净成本; label 行式另含手别诊断行, 组装只读 cost_ms)

写: 段当量表.npz  按**段角色** 6 片 (段当量 = 按键时间 + cost × P_err(该角色, 该上下文))
    总当量-2-4键.txt  2/3/4 键总当量 = 段当量求和 (ms, 期望耗时原值不归一化)

角色 → 键序（与 分析-按键时间.py ROLE_CELL / 分析-错误率.py ROLE6 一致）:
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

# 段角色片名（索引与字母序与 分析-按键时间.py ROLE_CELL / 分析-错误率.py ROLE6 一致）
ROLE_NAMES = ["L2i1", "L3i1", "L3i2", "L4i1", "L4i2", "L4i3"]

z = np.load(PROJ / "按键时间表.npz", allow_pickle=False)
assert tuple(z["letters"]) == tuple(L) and int(z["version"]) == 3, \
    "按键时间表版本不符 (期望 version 3 = 段角色 6 片); 先跑 分析-按键时间.py --full"
R = {k: z[f"r_{k}"].astype(np.float64) for k in ROLE_NAMES}
assert R["L2i1"].shape == (30, 30)
assert R["L3i1"].shape == R["L3i2"].shape == R["L4i1"].shape == R["L4i3"].shape == (30, 30, 30)
assert R["L4i2"].shape == (30, 30, 30, 30)
print(f"按键时间表: 6 角色片, {sum(v.size for v in R.values()):,} 条 (version {int(z['version'])})")

# ── 错误率表: 同构 6 段角色片 (2026-10-10 起; 取代 900×5 的 键对错误率表.txt) ──
ZE = np.load(PROJ / "错误率表.npz", allow_pickle=False)
assert tuple(ZE["letters"]) == tuple(L) and int(ZE["version"]) == 1, \
    "错误率表版本不符 (期望 version 1 = 段角色 6 片 (p,a,b,n) 条件化); 先跑 分析-错误率.py"
assert list(ZE["roles"]) == [f"r_{k}" for k in ROLE_NAMES]
E = {k: ZE[f"r_{k}"].astype(np.float64) for k in ROLE_NAMES}
for k in ROLE_NAMES:
    assert E[k].shape == R[k].shape, f"错误率片 {k} 形状与时间片不一致: {E[k].shape} vs {R[k].shape}"
print(f"错误率表: 6 角色片, {sum(v.size for v in E.values()):,} 条 (version {int(ZE['version'])}, "
      f"(p,a,b,n) 条件化; 错误修正时间实测标量 {ERR_MS:.1f}ms, n={ERR_N})  片均值: "
      + " / ".join(f"{k} {E[k].mean()*100:.2f}%" for k in ROLE_NAMES))

# ── 段当量 = 按键时间(角色,上下文) + 错误修正时间 × P_err(同角色同上下文) ──
# 两表同构 ⇒ 逐片直接相加（旧版需把 (a,b) 的 5 段类错误率广播到角色索引维, 现已不需要）
seg = {f"seg_{k}": R[k] + ERR_MS * E[k] for k in ROLE_NAMES}
np.savez_compressed(PROJ / "段当量表.npz", **seg,
                    letters=np.array(list(L)),
                    version=np.int64(6), cost_ms=np.float64(ERR_MS),
                    segments=np.array([f"seg_{k}" for k in ROLE_NAMES]),
                    roles=np.array(ROLE_NAMES),
                    errsrc=np.array("错误率表.npz (p,a,b,n) 条件化 6 角色片"),
                    note=np.array(f"段当量 = 按键时间 S(段角色,p,a,b,n) + {ERR_MS:.1f}×P_err(同角色同上下文) ms; "
                                  "段角色 = (码长,段位): L2i1=2键第1段/L3i1=3键第1段/L3i2=3键第2段/"
                                  "L4i1=4键第1段/L4i2=4键第2段/L4i3=4键第3段 (旧称 角点/首段/短尾/中段/长尾); "
                                  "T2=seg_L2i1 T3=seg_L3i1+seg_L3i2 "
                                  "T4=seg_L4i1+seg_L4i2+seg_L4i3 (2026-10-10 错误侧改 (p,a,b,n) 条件化, "
                                  "version 6; 与时间表同构; 旧 5 片版把 L3i1/L4i1 合成一片 — 两者时间与错误两侧都不同)"))
print(f"输出: 段当量表.npz  (6 个段角色片; 段当量 = 按键时间 + {ERR_MS:.1f}×P_err(角色,上下文), version 6)")

# ── 2-4 键总当量表 = 段当量求和 ──
T2c = np.maximum(seg["seg_L2i1"], 0.0)
T3c = seg["seg_L3i1"] + seg["seg_L3i2"]
T4c = seg["seg_L4i1"][:, :, :, None] + seg["seg_L4i2"] + seg["seg_L4i3"][None, :, :, :]

total = len(L)**2 + len(L)**3 + len(L)**4
with open(PROJ / "总当量-2-4键.txt", "w", encoding="utf-8") as f:
    f.write(f"# 2/3/4 键总当量 (ms, 期望耗时原值含错误成本: 段当量 = 按键时间 + {ERR_MS:.1f}ms×P_err(段角色, p,a,b,n), 实测错误修正时间标量)\n")
    f.write("# 段角色 = (码长,段位): L2i1 2键第1段/L3i1 3键第1段/L3i2 3键第2段/L4i1 4键第1段/L4i2 4键第2段/L4i3 4键第3段 (旧称 角点/首段/短尾/中段/长尾)\n")
    f.write("# 错误率自 2026-10-10 起按 (p,a,b,n) 条件化 (错误率表.npz, 与按键时间表同构 6 角色片)\n")
    f.write("# T2=seg_L2i1(xy) T3=seg_L3i1(xyz)+seg_L3i2(xyz) "
            "T4=seg_L4i1(wxy)+seg_L4i2(wxyz)+seg_L4i3(xyz)  (时间侧 2026-09-21 段位落地; 错误侧 10-10 上下文化)\n")
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
# 旧单表版两段都被迫借 4 键角色片 (L4i1=4键第1段 / L4i3=4键第3段)
d_first = seg["seg_L3i1"][ki[a], ki[b], ki[c]] - seg["seg_L4i1"][ki[a], ki[b], ki[c]]
d_tail = seg["seg_L3i2"][ki[a], ki[b], ki[c]] - seg["seg_L4i3"][ki[a], ki[b], ki[c]]
print(f"自检 thes 角色片差 (仅示结构, 旧单表版 3 键段借 4 键片): "
      f"3键第1段−4键第1段 = {d_first:+.1f}ms; 3键第2段−4键第3段 = {d_tail:+.1f}ms")
