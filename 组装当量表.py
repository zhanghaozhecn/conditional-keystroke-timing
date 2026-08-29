#!/usr/bin/env python3
"""
用段表组装 2-4 键位当量表
读: 当量-段表.npz (F[p,a,b,n] 形状 (31,30,30,31), p/n 维 0-29=键、30=∅; 2026-08-29
    对称段模型 v2, 自 当量-段表.txt 文本格式迁移)
写: 当量-2-4键.txt (code\t当量, 单位 ms 期望键对耗时原值, 2026-08-19 起不再相对归一化),
   顺序 aa-zz, aaa-zzz, aaaa-zzzz

公式 (对称段模型, 词内语义):
  T₂(ab)   = F[∅,a,b,∅]                                  (角点, 2 键试次验证中)
  T₃(abc)  = F[∅,a,b,c] + F[a,b,c,∅]
  T₄(abcd) = F[∅,a,b,c] + F[a,b,c,d] + F[b,c,d,∅]
当量 = 期望键对/整串耗时 (模型预测的原始 ms 值)

--expected: 同时输出期望当量表 (当量-期望-2-4键.txt)
  期望段当量 = 段当量 + 500ms × P_err(a,b)   (错误损失固定 500ms:
  退格+注意力+重输, 与键对难度解耦; 500 与 分析-错误当量.py 推导 455ms 吻合)
  P_err 读自 当量-键对错误率.txt (分析-错误率规律.py 9 特征逻辑回归导出)
--baseline: 同时输出传统两键累加对照表 (当量-两键累加-2-4键.txt)
  T = Σ F[∅,ki,ki+1,∅] 逐对累加 (T₂ 角点切片), 不含前键条件与后键条件 —
  系统性低估整串 (对比见 README §5.2 / 对比-条件vs两键.py)
"""
import argparse, itertools
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--expected", action="store_true", help="同时导出期望当量表")
parser.add_argument("--baseline", action="store_true",
                    help="同时导出传统两键累加对照表 (当量-两键累加-2-4键.txt, 仅 T₂ 角点切片逐对累加, 同为 ms)")
args = parser.parse_args()

PROJ = Path(__file__).resolve().parent
SEG = PROJ / "当量-段表.npz"
OUT = PROJ / "当量-2-4键.txt"
OUT_EXP = PROJ / "当量-期望-2-4键.txt"
OUT_BASE = PROJ / "当量-两键累加-2-4键.txt"   # 传统基线: T = Σ F[∅,ki,ki+1,∅]
ERR_MS = 500.0   # 固定错误损失 (ms): 退格 + 注意力 + 重输, 数据推导 455 近似
L = "abcdefghijklmnopqrstuvwxyz;,./"  # 30 键 (3 行 10 列完整 QWERTY)
L26 = "abcdefghijklmnopqrstuvwxyz"
KI = {c: i for i, c in enumerate(L)}   # 0-29; ∅ = 30
E = 30

z = np.load(SEG, allow_pickle=False)
F = z["F"]                                # (31,30,30,31) float32
assert tuple(z["letters"]) == tuple(L) and int(z["empty"]) == 30 and int(z["version"]) == 2

# ── 向量化组装 (切片索引均按 F[p,a,b,n] 维序核对) ──
seg1 = F[E, :, :, :E].astype(np.float64)      # 首段 S(∅,a,b,c) → [a,b,c]
seg2 = F[:E, :, :, :E].astype(np.float64)     # 中段 S(a,b,c,d) → [a,b,c,d]
seg2end = F[:E, :, :, E].astype(np.float64)   # 尾段 S(a,b,c,∅) → [a,b,c]
seg3 = F[:E, :, :, E].astype(np.float64)      # 尾段 S(b,c,d,∅) → [b,c,d]
B = np.maximum(F[E, :, :, E].astype(np.float64), 0.0)   # T₂ 角点 (30,30)

T3 = seg1 + seg2end                                             # [a,b,c]
T4 = seg1[:, :, :, None] + seg2 + seg3[None, :, :, :]           # [a,b,c,d]

def main():
    print(f"段表: F{F.shape} ({F.size:,} 条, npz)")
    mat_min = min(B[KI[a], KI[b]] for a in L26 for b in L26)
    print(f"参考 (最快速字母键对): {mat_min:.0f}ms — 仅提示, 不用于归一化")

    perr = {}
    P = None
    if args.expected:
        with open(PROJ / "当量-键对错误率.txt", encoding="utf-8") as f:
            f.readline()
            for line in f:
                ab, p = line.rstrip("\n").split("\t")
                perr[ab] = float(p)
        P = np.array([[perr.get(a + b, 0.019) for b in L] for a in L])  # (30,30)
        print(f"键对错误率: {len(perr):,} 条 (错误损失固定 {ERR_MS}ms, 缺省 1.9%)")

    # ── 条件当量表 ──
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# 2-4 键位当量 (ms, 期望键对耗时原始值; 2026-08-19 起不再相对归一化)\n")
        f.write("# 对称段模型 v2: T2=F[∅,a,b,∅](角点,2键验证中) T3=F[∅,a,b,c]+F[a,b,c,∅] "
                "T4=F[∅,a,b,c]+F[a,b,c,d]+F[b,c,d,∅]\n")
        f.write("code\t当量\n")
        buf = []
        for i, a in enumerate(L):
            for j, b in enumerate(L):           # aa-zz
                buf.append(f"{a}{b}\t{B[i,j]:.2f}\n")
        for (i, j, k), v in np.ndenumerate(T3):  # aaa-zzz
            buf.append(f"{L[i]}{L[j]}{L[k]}\t{v:.2f}\n")
        f.writelines(buf); buf = []
        for (i, j, k, m), v in np.ndenumerate(T4):  # aaaa-zzzz
            buf.append(f"{L[i]}{L[j]}{L[k]}{L[m]}\t{v:.2f}\n")
            if len(buf) >= 100_000:
                f.writelines(buf); buf = []
        f.writelines(buf)

    total = len(L)**2 + len(L)**3 + len(L)**4
    print(f"输出: {OUT} ({total:,} 条, ms)")

    # ── 两键累加对照表 (--baseline): 传统方法 T = Σ B[ki,ki+1] (T₂ 角点切片) ──
    if args.baseline:
        with open(OUT_BASE, "w", encoding="utf-8") as f:
            f.write("# 两键累加对照 (传统基线, ms 期望耗时原值; B = T₂ 角点切片 F[∅,a,b,∅])\n")
            f.write("code\t当量\n")
            buf = []
            for i, a in enumerate(L):
                for j, b in enumerate(L):
                    buf.append(f"{a}{b}\t{B[i,j]:.2f}\n")
            for a, b, c in itertools.product(L, L, L):
                v = B[KI[a], KI[b]] + B[KI[b], KI[c]]
                buf.append(f"{a}{b}{c}\t{v:.2f}\n")
            for a, b, c, d in itertools.product(L, L, L, L):
                v = B[KI[a], KI[b]] + B[KI[b], KI[c]] + B[KI[c], KI[d]]
                buf.append(f"{a}{b}{c}{d}\t{v:.2f}\n")
                if len(buf) >= 100_000:
                    f.writelines(buf); buf = []
            f.writelines(buf)
        print(f"输出: {OUT_BASE} ({total:,} 条, 两键累加基线, ms)")

    # ── 期望当量表 (--expected): 段当量 + 500ms×P_err(该段键对) ──
    if args.expected:
        B_exp = B + ERR_MS * P                                        # P(a,b)
        T3_exp = T3 + ERR_MS * (P[:, :, None] + P[None, :, :])        # +P(a,b)+P(b,c)
        T4_exp = (T4 + ERR_MS * (P[:, :, None, None]                  # P(a,b)
                                 + P[None, :, :, None]                # P(b,c)
                                 + P[None, None, :, :]))              # P(c,d)
        with open(OUT_EXP, "w", encoding="utf-8") as f:
            f.write(f"# 期望当量 = 段当量 + {ERR_MS:.0f}ms×P_err (ms 期望耗时原值; 对称段模型 v2)\n")
            f.write("code\t当量\n")
            buf = []
            for i, a in enumerate(L):
                for j, b in enumerate(L):
                    buf.append(f"{a}{b}\t{B_exp[i,j]:.2f}\n")
            for (i, j, k), v in np.ndenumerate(T3_exp):
                buf.append(f"{L[i]}{L[j]}{L[k]}\t{v:.2f}\n")
            f.writelines(buf); buf = []
            for (i, j, k, m), v in np.ndenumerate(T4_exp):
                buf.append(f"{L[i]}{L[j]}{L[k]}{L[m]}\t{v:.2f}\n")
                if len(buf) >= 100_000:
                    f.writelines(buf); buf = []
            f.writelines(buf)
        print(f"输出: {OUT_EXP} ({total:,} 条, 期望当量, ms)")

if __name__ == "__main__":
    main()
