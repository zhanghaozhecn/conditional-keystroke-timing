#!/usr/bin/env python3
"""
用段表 + 错误率表组装标准产物（2026-08-31 定稿，无可选旗标）

读: 当量-段表.npz      F[p,a,b,n] (31,30,30,31) 原始段当量 S(p,a,b,n) ms
    当量-键对错误率.txt 900 键对 × 4 签名类 P_err(a,b | 有无前键, 有无后键)
       (2026-09-02 签名版: 错误率随码内位置爬升, 角点 ~0.6% ↔ 尾段 ~3.2%)

写: 当量-修正段表.npz  C[p,a,b,n] = S(p,a,b,n) + 500ms × P_err(a,b | sig(p), sig(n))
    （错误项按段 (p,n) 签名取列: p≠∅=有前键, n≠∅=有后键, 与 S 同语义;
      错误损失 500ms = 退格+注意力+重输, 数据推导 ~455ms, 与键对难度解耦——见 README §4.6）
    当量-2-4键.txt     2/3/4 键总当量 = 修正段求和 (ms, 期望耗时原值不归一化):
      T₂(ab)   = C[∅,a,b,∅]                                  (角点, 2 键试次验证中)
      T₃(abc)  = C[∅,a,b,c] + C[a,b,c,∅]
      T₄(abcd) = C[∅,a,b,c] + C[a,b,c,d] + C[b,c,d,∅]

运行顺序: 击键模型.py --full (出段表) → 分析-错误率规律.py (出错误率表) → 本脚本。
"""
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parent / "产物"   # 2026-09-05 目录重组: 四件产物入 产物/
ERR_MS = 500.0
L = "abcdefghijklmnopqrstuvwxyz;,./"   # 30 键 (3 行 10 列完整 QWERTY)
E = 30                                  # ∅ 索引

z = np.load(PROJ / "当量-段表.npz", allow_pickle=False)
F = z["F"].astype(np.float64)
assert tuple(z["letters"]) == tuple(L) and int(z["empty"]) == E and int(z["version"]) == 2
print(f"段表: F{z['F'].shape} ({z['F'].size:,} 条, npz)")

perr = {}
with open(PROJ / "当量-键对错误率.txt", encoding="utf-8") as f:
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
print(f"键对错误率: {len(perr)} 对 × 4 签名类 (错误损失 {ERR_MS:.0f}ms)  类均值: "
      f"角点 {Pmat[0,0].mean()*100:.2f}% / 首段 {Pmat[0,1].mean()*100:.2f}% / "
      f"尾段 {Pmat[1,0].mean()*100:.2f}% / 中段 {Pmat[1,1].mean()*100:.2f}%")

# ── 修正段表: +500×P_err(a,b|签名), 按段 (p,n) 签名取列 ──
# Pexp[p,a,b,n]: p<30=有前键, p=30=∅; n<30=有后键, n=30=∅ → 值 = Pmat[有无前键, 有无后键, a, b]
Pexp = np.empty((31, len(L), len(L), 31), dtype=Pmat.dtype)
Pexp[:30, :, :, :30] = Pmat[1, 1][None, :, :, None]   # 有前+有后 (中段)
Pexp[:30, :, :, 30]  = Pmat[1, 0][None, :, :]         # 有前+无后 (尾段)
Pexp[30,  :, :, :30] = Pmat[0, 1][:, :, None]         # 无前+有后 (首段)
Pexp[30,  :, :, 30]  = Pmat[0, 0]                     # 无前+无后 (角点)
for p_, a_, b_, n_ in [(0,1,2,3), (30,1,2,3), (0,1,2,30), (30,1,2,30), (15,7,8,29), (29,0,5,0)]:
    assert abs(Pexp[p_, a_, b_, n_] - Pmat[int(p_ < 30), int(n_ < 30), a_, b_]) < 1e-15
C = F + ERR_MS * Pexp
np.savez_compressed(PROJ / "当量-修正段表.npz", F=C,
                    letters=np.array(list(L)), empty=np.int64(E),
                    version=np.int64(3),
                    note=np.array("修正段当量 C(p,a,b,n)=S(p,a,b,n)+500×P_err(a,b|有无前键p,有无后键n) ms; "
                                  "T2=C[30,a,b,30] T3=C[30,a,b,c]+C[a,b,c,30] "
                                  "T4=C[30,a,b,c]+C[a,b,c,d]+C[b,c,d,30]"))
print(f"输出: 当量-修正段表.npz  (C{C.shape}, 修正 = 段 + 500×签名错误率, version 3)")

# ── 2-4 键总当量表 = 修正段求和 ──
T2c = np.maximum(C[E, :, :, E], 0.0)
T3c = C[E, :, :, :E] + C[:E, :, :, E]                       # [a,b,c]
T4c = C[E, :, :, :E][:, :, :, None] + C[:E, :, :, :E] + C[:E, :, :, E][None, :, :, :]

total = len(L)**2 + len(L)**3 + len(L)**4
with open(PROJ / "当量-2-4键.txt", "w", encoding="utf-8") as f:
    f.write("# 2-4 键位当量 (ms, 期望耗时原值含错误成本: 修正段 = 段 + 500ms×P_err(a,b|有无前键,有无后键))\n")
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
