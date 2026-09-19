#!/usr/bin/env python3
"""
用按键时间表 + 键对错误率表 + 错误修正时间实测值组装标准产物
（2026-08-31 定稿; 2026-09-19 术语统一: 段当量 = 按键时间 + 错误率 × 错误修正时间,
  总当量（2/3/4 键当量）= 段当量求和）

读: 按键时间表.npz      F[p,a,b,n] (31,30,30,31) 按键时间 S(p,a,b,n) ms (前键p/键对a,b/后键n 条件)
    键对错误率表.txt    900 键对 × 4 签名类 P_err(a,b | 有无前键, 有无后键)
       (2026-09-02 签名版: 错误率随码内位置爬升, 角点 ~0.6% ↔ 尾段 ~3.2%)
    错误修正时间-实测值.txt  错误修正时间标量 cost_ms (2026-09-06 起替代固定 500ms 常数, 用户决策;
       由 分析-错误修正时间.py 从 数据/错误修正时间-错误键.tsv 自动导出, 随采集动态更新;
       label 行式另含手别诊断行 cost_left/cost_right/hand_beta——09-14 分手部署 09-15 回退
       (手效应=退格位依赖的布局结构, 入表牺牲通用性), 组装只读 cost_ms;
       忽略连续文本 vs 4键限长文本的 regime 差异——错误修正时间难测, 连续文本更简单)

写: 段当量表.npz  C[p,a,b,n] = S(p,a,b,n) + cost_ms × P_err(a,b | sig(p), sig(n))
    （错误项按段 (p,n) 签名取列: p≠∅=有前键, n≠∅=有后键, 与 S 同语义;
      cost_ms = 错误修正时间实测标量 (净成本: 首错按下→首错位打对), 见 README §4.6/§4.7）
    总当量-2-4键.txt  2/3/4 键总当量 = 段当量求和 (ms, 期望耗时原值不归一化):
      T₂(ab)   = C[∅,a,b,∅]                                  (角点, 2 键试次验证中)
      T₃(abc)  = C[∅,a,b,c] + C[a,b,c,∅]
      T₄(abcd) = C[∅,a,b,c] + C[a,b,c,d] + C[b,c,d,∅]

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
E = 30                                  # ∅ 索引

z = np.load(PROJ / "按键时间表.npz", allow_pickle=False)
F = z["F"].astype(np.float64)
assert tuple(z["letters"]) == tuple(L) and int(z["empty"]) == E and int(z["version"]) == 2
print(f"按键时间表: F{z['F'].shape} ({z['F'].size:,} 条, npz)")

perr = {}
with open(PROJ / "键对错误率表.txt", encoding="utf-8") as f:
    head = f.readline().rstrip("\n").split("\t")
    ci = [head.index(c) for c in ("err_p0n0", "err_p0n1", "err_p1n0", "err_p1n1")]
    for line in f:
        parts = line.rstrip("\n").split("\t")
        perr[parts[0]] = [float(parts[i]) for i in ci]
Pmat = np.zeros((2, 2, len(L), len(L)))          # [有前键, 有后键, a, b]
for ab, v in perr.items():
    i, j = L.index(ab[0]), L.index(ab[1])
    Pmat[0, 0, i, j], Pmat[0, 1, i, j], Pmat[1, 0, i, j], Pmat[1, 1, i, j] = v
assert len(perr) == len(L)**2, "错误率表应覆盖 900 键对"
print(f"键对错误率: {len(perr)} 对 × 4 签名类 (错误修正时间实测标量 {ERR_MS:.1f}ms, n={ERR_N})  类均值: "
      f"角点 {Pmat[0,0].mean()*100:.2f}% / 首段 {Pmat[0,1].mean()*100:.2f}% / "
      f"尾段 {Pmat[1,0].mean()*100:.2f}% / 中段 {Pmat[1,1].mean()*100:.2f}%")

# ── 段当量表: 按键时间 + 错误修正时间×P_err(a,b|签名), 按段 (p,n) 签名取列 ──
# Pexp[p,a,b,n]: p<30=有前键, p=30=∅; n<30=有后键, n=30=∅ → 值 = Pmat[有无前键, 有无后键, a, b]
Pexp = np.empty((31, len(L), len(L), 31), dtype=Pmat.dtype)
Pexp[:30, :, :, :30] = Pmat[1, 1][None, :, :, None]   # 有前+有后 (中段)
Pexp[:30, :, :, 30]  = Pmat[1, 0][None, :, :]         # 有前+无后 (尾段)
Pexp[30,  :, :, :30] = Pmat[0, 1][:, :, None]         # 无前+有后 (首段)
Pexp[30,  :, :, 30]  = Pmat[0, 0]                     # 无前+无后 (角点)
for p_, a_, b_, n_ in [(0,1,2,3), (30,1,2,3), (0,1,2,30), (30,1,2,30), (15,7,8,29), (29,0,5,0)]:
    assert abs(Pexp[p_, a_, b_, n_] - Pmat[int(p_ < 30), int(n_ < 30), a_, b_]) < 1e-15
# 段当量表 (2026-09-15 回退标量——09-14 分手部署经用户决策撤销: 手效应 (+69ms, t=4.15) 是
# 退格位依赖的布局结构, 保当量表对任意退格位的通用性; 手别系数保留在 错误修正时间-实测值.txt
# 作诊断, 不参与组装)
C = F + ERR_MS * Pexp
np.savez_compressed(PROJ / "段当量表.npz", F=C,
                    letters=np.array(list(L)), empty=np.int64(E),
                    version=np.int64(3), cost_ms=np.float64(ERR_MS),
                    note=np.array(f"段当量 C(p,a,b,n)=按键时间 S(p,a,b,n)+{ERR_MS:.1f}×P_err(a,b|有无前键p,有无后键n) ms "
                                  "(错误修正时间=实测标量, 来源 错误修正时间-实测值.txt; 09-14 分手版次日回退——"
                                  "手效应为退格位依赖结构, 保通用性); "
                                  "T2=C[30,a,b,30] T3=C[30,a,b,c]+C[a,b,c,30] "
                                  "T4=C[30,a,b,c]+C[a,b,c,d]+C[b,c,d,30]"))
print(f"输出: 段当量表.npz  (C{C.shape}, 段当量 = 按键时间 + {ERR_MS:.1f}×签名错误率 (实测标量 n={ERR_N}), version 3)")

# ── 2-4 键总当量表 = 段当量求和 ──
T2c = np.maximum(C[E, :, :, E], 0.0)
T3c = C[E, :, :, :E] + C[:E, :, :, E]                       # [a,b,c]
T4c = C[E, :, :, :E][:, :, :, None] + C[:E, :, :, :E] + C[:E, :, :, E][None, :, :, :]

total = len(L)**2 + len(L)**3 + len(L)**4
with open(PROJ / "总当量-2-4键.txt", "w", encoding="utf-8") as f:
    f.write(f"# 2/3/4 键总当量 (ms, 期望耗时原值含错误成本: 段当量 = 按键时间 + {ERR_MS:.1f}ms×P_err(a,b|有无前键,有无后键), 实测错误修正时间标量; 09-14 分手版 09-15 回退保通用性)\n")
    f.write("# T2=C[∅,a,b,∅](角点,2键验证中) T3=C[∅,a,b,c]+C[a,b,c,∅] T4=C[∅,a,b,c]+C[a,b,c,d]+C[b,c,d,∅]\n")
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

# ── 自检: 抽查公式还原 ──
a, b, c, d = "t", "h", "e", "s"
ki = {ch: L.index(ch) for ch in (a, b, c, d)}
t2 = C[E, ki[a], ki[b], E]
t3 = C[E, ki[a], ki[b], ki[c]] + C[ki[a], ki[b], ki[c], E]
t4 = C[E, ki[a], ki[b], ki[c]] + C[ki[a], ki[b], ki[c], ki[d]] + C[ki[b], ki[c], ki[d], E]
print(f"自检 thes: T₂={t2:.1f} T₃={t3:.1f} T₄={t4:.1f} (公式 = 表列值, 差应为 0)")
