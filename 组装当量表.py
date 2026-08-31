#!/usr/bin/env python3
"""
用段表 + 错误率表组装标准产物（2026-08-31 定稿，无可选旗标）

读: 当量-段表.npz      F[p,a,b,n] (31,30,30,31) 原始段当量 S(p,a,b,n) ms
    当量-键对错误率.txt 900 键对 P_err(a,b)

写: 当量-修正段表.npz  C[p,a,b,n] = S(p,a,b,n) + 500ms × P_err(a,b)
    （修正只依赖段自身键对 (a,b), 对 p/n 维广播; 错误损失 500ms = 退格+注意力+
      重输, 数据推导 ~455ms, 与键对难度解耦——见 README §5.3）
    当量-2-4键.txt     2/3/4 键总当量 = 修正段求和 (ms, 期望耗时原值不归一化):
      T₂(ab)   = C[∅,a,b,∅]                                  (角点, 2 键试次验证中)
      T₃(abc)  = C[∅,a,b,c] + C[a,b,c,∅]
      T₄(abcd) = C[∅,a,b,c] + C[a,b,c,d] + C[b,c,d,∅]

运行顺序: 击键模型.py --full (出段表) → 分析-错误率规律.py (出错误率表) → 本脚本。
"""
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parent
ERR_MS = 500.0
L = "abcdefghijklmnopqrstuvwxyz;,./"   # 30 键 (3 行 10 列完整 QWERTY)
E = 30                                  # ∅ 索引

z = np.load(PROJ / "当量-段表.npz", allow_pickle=False)
F = z["F"].astype(np.float64)
assert tuple(z["letters"]) == tuple(L) and int(z["empty"]) == E and int(z["version"]) == 2
print(f"段表: F{z['F'].shape} ({z['F'].size:,} 条, npz)")

perr = {}
with open(PROJ / "当量-键对错误率.txt", encoding="utf-8") as f:
    f.readline()
    for line in f:
        ab, p = line.rstrip("\n").split("\t")
        perr[ab] = float(p)
P = np.array([[perr.get(a + b, 0.019) for b in L] for a in L])   # 缺省 1.9%
print(f"键对错误率: {len(perr)} 条 (错误损失 {ERR_MS:.0f}ms, 缺省 1.9%)")

# ── 修正段表: +500×P_err(a,b), 对 p/n 维广播 ──
C = F + ERR_MS * P[None, :, :, None]
np.savez_compressed(PROJ / "当量-修正段表.npz", F=C,
                    letters=np.array(list(L)), empty=np.int64(E),
                    version=np.int64(2),
                    note=np.array("修正段当量 C(p,a,b,n)=S(p,a,b,n)+500×P_err(a,b) ms; "
                                  "T2=C[30,a,b,30] T3=C[30,a,b,c]+C[a,b,c,30] "
                                  "T4=C[30,a,b,c]+C[a,b,c,d]+C[b,c,d,30]"))
print(f"输出: 当量-修正段表.npz  (C{C.shape}, 修正 = 段 + 500×P_err)")

# ── 2-4 键总当量表 = 修正段求和 ──
T2c = np.maximum(C[E, :, :, E], 0.0)
T3c = C[E, :, :, :E] + C[:E, :, :, E]                       # [a,b,c]
T4c = C[E, :, :, :E][:, :, :, None] + C[:E, :, :, :E] + C[:E, :, :, E][None, :, :, :]

total = len(L)**2 + len(L)**3 + len(L)**4
with open(PROJ / "当量-2-4键.txt", "w", encoding="utf-8") as f:
    f.write("# 2-4 键位当量 (ms, 期望耗时原值含错误成本: 修正段 = 段 + 500ms×P_err)\n")
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
print(f"输出: 当量-2-4键.txt ({total:,} 条, ms, 修正段求和)")

# ── 自检: 抽查公式还原 ──
a, b, c, d = "t", "h", "e", "s"
ki = {ch: L.index(ch) for ch in (a, b, c, d)}
t2 = C[E, ki[a], ki[b], E]
t3 = C[E, ki[a], ki[b], ki[c]] + C[ki[a], ki[b], ki[c], E]
t4 = C[E, ki[a], ki[b], ki[c]] + C[ki[a], ki[b], ki[c], ki[d]] + C[ki[b], ki[c], ki[d], E]
print(f"自检 thes: T₂={t2:.1f} T₃={t3:.1f} T₄={t4:.1f} (公式 = 表列值, 差应为 0)")
