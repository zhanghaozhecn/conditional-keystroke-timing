#!/usr/bin/env python3
"""
用段表组装 2-4 键位当量表
读: 当量-段表.txt (27×26×26, S(p,a,b) ms, _=空前键)
写: 当量-2-4键.txt (code\\t当量, 归一化), 顺序 aa-zz, aaa-zzz, aaaa-zzzz

公式:
  T₂(ab)   = S[_,a,b]
  T₃(abc)  = S[_,a,b] + S[a,b,c]
  T₄(abcd) = S[_,a,b] + S[a,b,c] + S[b,c,d]
归一化: 除以最快速键对 (空前键最小段)

--expected: 同时输出期望当量表 (当量-期望-2-4键.txt)
  期望段当量 = 条件段当量 + 500ms × P_err(a,b)   (错误损失固定 500ms:
  退格+注意力+重输, 与键对难度解耦; 500 与 分析-错误当量.py 推导 455ms 吻合)
  P_err 读自 当量-键对错误率.txt (分析-错误率规律.py 9 特征逻辑回归导出)
"""
import argparse, itertools
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--expected", action="store_true", help="同时导出期望当量表")
args = parser.parse_args()

PROJ = Path(__file__).resolve().parent
SEG = PROJ / "当量-段表.txt"
OUT = PROJ / "当量-2-4键.txt"
OUT_EXP = PROJ / "当量-期望-2-4键.txt"
ERR_MS = 500.0   # 固定错误损失 (ms): 退格 + 注意力 + 重输, 数据推导 455 近似
L = "abcdefghijklmnopqrstuvwxyz;,./"  # 30 键 (3 行 10 列完整 QWERTY)

def main():
    # ── 读段表 ──
    seg = {}
    with open(SEG, encoding="utf-8") as f:
        f.readline(); f.readline(); f.readline()
        for line in f:
            p, a, b, ms = line.rstrip("\n").split("\t")
            seg[(p, a, b)] = float(ms)
    print(f"段表: {len(seg):,} 条")

    # ── 读键对错误率表 (--expected 时) ──
    perr = {}
    if args.expected:
        with open(PROJ / "当量-键对错误率.txt", encoding="utf-8") as f:
            f.readline()
            for line in f:
                ab, p = line.rstrip("\n").split("\t")
                perr[ab] = float(p)
        print(f"键对错误率: {len(perr):,} 条 (错误损失固定 {ERR_MS}ms)")

    # ── 空前键切片 = 2 键当量 (ms) ──
    B = {}
    for a in L:
        for b in L:
            B[(a, b)] = max(0.0, seg[("_", a, b)])
    # 归一化基准: 限定字母键对 (符号键无样本, 嵌入中性, 会污染最小值)
    LETTERS26 = "abcdefghijklmnopqrstuvwxyz"
    mat_min = min(v for (a, b), v in B.items()
                  if a in LETTERS26 and b in LETTERS26)
    print(f"基准 (最快速字母键对): {mat_min:.0f}ms")

    # ── 组装 + 写入 ──
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# 2-4 键位当量 (归一化, 基准=最快速键对 {mat_min:.0f}ms)\n")
        f.write("code\t当量\n")
        buf = []
        for a, b in itertools.product(L, L):           # aa-zz
            buf.append(f"{a}{b}\t{B[(a,b)] / mat_min:.2f}\n")
        for a, b, c in itertools.product(L, L, L):     # aaa-zzz
            v = (B[(a,b)] + seg[(a,b,c)]) / mat_min
            buf.append(f"{a}{b}{c}\t{v:.2f}\n")
        for a, b, c, d in itertools.product(L, L, L, L):  # aaaa-zzzz
            v = (B[(a,b)] + seg[(a,b,c)] + seg[(b,c,d)]) / mat_min
            buf.append(f"{a}{b}{c}{d}\t{v:.2f}\n")
        f.writelines(buf)

    total = len(L)**2 + len(L)**3 + len(L)**4
    print(f"输出: {OUT} ({total:,} 条)")

    # ── 期望当量表 (--expected): 段当量 + 500ms×P_err, 同基准归一化 ──
    if args.expected:
        def exp_seg(prev, a, b):
            """期望段当量 = S(prev,a,b) + 500×P_err(a,b)"""
            return seg[(prev, a, b)] + ERR_MS * perr.get(a + b, 0.019)  # 缺省=全局 1.9%
        with open(OUT_EXP, "w", encoding="utf-8") as f:
            f.write(f"# 期望当量 = 条件段当量 + {ERR_MS:.0f}ms×P_err (基准=最快速键对 {mat_min:.0f}ms)\n")
            f.write("code\t当量\n")
            buf = []
            for a, b in itertools.product(L, L):       # aa-zz
                buf.append(f"{a}{b}\t{exp_seg('_', a, b) / mat_min:.2f}\n")
            for a, b, c in itertools.product(L, L, L):     # aaa-zzz
                v = (exp_seg('_', a, b) + exp_seg(a, b, c)) / mat_min
                buf.append(f"{a}{b}{c}\t{v:.2f}\n")
            for a, b, c, d in itertools.product(L, L, L, L):  # aaaa-zzzz
                v = (exp_seg('_', a, b) + exp_seg(a, b, c) + exp_seg(b, c, d)) / mat_min
                buf.append(f"{a}{b}{c}{d}\t{v:.2f}\n")
            f.writelines(buf)
        print(f"输出: {OUT_EXP} ({total:,} 条, 期望当量)")

if __name__ == "__main__":
    main()
