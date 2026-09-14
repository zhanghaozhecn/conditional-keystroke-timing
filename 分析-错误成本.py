#!/usr/bin/env python3
"""
错误成本分析 — 聚合 数据/错误成本-错误键.tsv (2026-09-05)

episode 判定唯一源 = 采集-错误成本.py (实时判定, 只落盘错误键事件 + 每串汇总行;
连锁错(含同位二次错)在采集端吸收为单事件——错误只计首错、时长只记首错间隔, 用户决策 2026-09-05);
本脚本只做聚合: 全体/立即发现/延迟发现均值、盲打长度分布、连续模式按错率
(对标停止范式 P_err 的 regime 差异)、时长异常剔除 (09-13: 过程分桶 imm×nbs档, 桶内单侧
med+3×1.4826×MAD 围栏——注意力中断只拖慢不加快, 语义同 S 模型; 退格/盲打是时长簿记结构
必须分桶否则误删合法长纠正; 桶 n<8 不设栏; 剔除只作用于时长统计, 率仍用全部闭合 episode)、
手别系数 (09-14 用户决策采用分手错误时间: 立即基元 imm&nbs=1 控序号 OLS 手效应, |t|≥1 启用,
cost_left/right = 标量×因子, 延迟=立即+过程增量使基元无构成混杂)、
时间趋势 (全局序号: 线性趋势+四分位分段+子组前后半;
坐姿稳定性判据段 (09-07 用户决策: 现值认定稳定, 续采至练习效应可忽略即定案——
三判据 = 簇稳健SE≤30ms 且 近10坐姿斜率不显著 (09-12 改近端窗口, 见段内注释)
且 近5坐姿均值与全体差<40ms);
session=进程生命周期, 长短不一且开关与坐姿不对应, 不参与分组——2026-09-06 用户决策)、
按目标键/键对分组
(等成本假设的检验入口——现行假设任意键纠正时间相同, 实际可能与目标键或
前键+目标键键对有关, 样本攒够后由此检验)。
文件格式 (HEADER): kind=ep 行 = 错误事件 (pos/t_ms/dur_ms/got/want/prev/n_wrong/
blind/n_bs/immediate); kind=sum 行 = 串汇总 (pos=总按键, t_ms=串时长, dur_ms=错误按下数,
n_wrong=假警报退格, blind=截断 episode 数)。
样本量判据: 目标 ~150-200 episode (SD≈500ms 时 SE≤50ms)。

用法: python 分析-错误成本.py [记录.tsv] [--discard N]   # 记录默认 数据/错误成本-错误键.tsv;
      N=丢弃前 N 次按键 (熟悉期, 默认 684=首坐姿全部按键, 09-06 序号口径标定; 0=不弃)
      弃置与分组均按累计按键序号——session 字段仅采集端存档, 不参与分析 (09-06 用户决策)。
"""
import sys
import math
from pathlib import Path
from collections import defaultdict

_DIR = Path(__file__).resolve().parent
RAW = _DIR / "数据" / "错误成本-错误键.tsv"
DISCARD = 684          # 丢弃前 N 次按键 (熟悉期; 684=首坐姿全部按键, 09-06 序号口径标定)
args = sys.argv[1:]
if "--discard" in args:
    DISCARD = int(args[args.index("--discard") + 1])
    i = args.index("--discard")
    args = args[:i] + args[i + 2:]
if args and not args[0].startswith("--"):
    RAW = Path(args[0])

# ── 1. 读记录 (序号口径: 弃置按累计按键定位; 串归属规则 = 累计起点 ≥ 弃置线的串整体
#     保留、跨界串整体弃置——ep 行无按键计数无法串内再切, 默认 684 恰为坐姿边界无跨界) ──
episodes = []                       # dict(session, stream, pos, dur, want, prev, got, n_wrong, blind, nbs, imm)
ranges = []                         # 串区间: sum 行按文件序累计按键 [start, end)
cum = 0
with open(RAW, encoding="utf-8") as f:
    f.readline()
    for line in f:
        row = line.rstrip("\n").split("\t")
        if len(row) < 13 or row[2] != "sum":
            continue
        n = int(row[3])
        ranges.append(dict(key=(row[0], int(row[1])), start=cum, end=cum + n,
                           press=n, wrong=int(row[5]),
                           false_bs=int(row[9]), trunc=int(row[10])))
        cum += n
kept = {r["key"] for r in ranges if r["start"] >= DISCARD}
kept_ranges = [r for r in ranges if r["key"] in kept]
if DISCARD:
    drop = [r for r in ranges if r["key"] not in kept]
    print(f"丢弃前 {DISCARD} 按键 (熟悉期): 实弃 {sum(r['press'] for r in drop)} 按键 / {len(drop)} 串")
with open(RAW, encoding="utf-8") as f:
    f.readline()
    for line in f:
        row = line.rstrip("\n").split("\t")
        if len(row) < 13 or row[2] != "ep" or (row[0], int(row[1])) not in kept:
            continue
        episodes.append(dict(session=row[0], stream=int(row[1]), pos=int(row[3]),
                             dur=float(row[5]), got=row[6], want=row[7], prev=row[8],
                             n_wrong=int(row[9]), blind=int(row[10]),
                             nbs=int(row[11]), imm=row[12] == "1"))
p = sum(r["press"] for r in kept_ranges)
w = sum(r["wrong"] for r in kept_ranges)
fb = sum(r["false_bs"] for r in kept_ranges)
tr = sum(r["trunc"] for r in kept_ranges)

# ── 1.5 时长异常剔除: 过程分桶 + 桶内单侧 3σ-MAD 围栏 (2026-09-13) ──
# 语义同 S 模型剔除 (README §4.5): 注意力中断只拖慢不加快 → 只删过大侧, 偏快是真实行为不删。
# 桶 = (发现型 imm × 退格数档: nbs≤3 精确 / ≥4 合并)——退格与盲打是时长的过程簿记结构
# (+~400ms/退格、+~350ms/盲打字母, 09-07 特点分析), 不分桶会把合法的连锁/盲打长纠正误删;
# 桶 n<8 不设栏 (保守保留, 小桶 MAD 不稳); 围栏 = med + 3×1.4826×MAD, MAD 对待剔点稳健, 单遍不迭代。
# 剔除只作用于时长统计 (均值/坐姿序列/趋势/导出 cost), 事件计数与率仍用全部闭合 episode——错误确实发生过。
def med_(v):
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
buckets = defaultdict(list)
for i, e in enumerate(episodes, 1):
    buckets[(e["imm"], min(e["nbs"], 4))].append((i, e))
outliers = []
for (im_, nb_), grp in sorted(buckets.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
    if len(grp) < 8:
        continue
    dv = [e["dur"] for _, e in grp]
    m_ = med_(dv)
    mad_ = med_([abs(d - m_) for d in dv])
    if mad_ <= 0:
        continue
    fence_ = m_ + 3 * 1.4826 * mad_
    for i, e in grp:
        if e["dur"] >= fence_:
            e["out"] = True
            outliers.append((i, e, m_, mad_, fence_))
print("时长异常剔除 (过程分桶 imm×nbs档, 桶内单侧 med+3×1.4826×MAD, 桶 n<8 不设栏):")
for i, e, m_, mad_, fence_ in outliers:
    print(f"  剔 #{i} {e['dur']:.0f}ms {e['session']} #{e['stream']} pos{e['pos']} "
          f"{e['got']}→{e['want']} (imm={int(e['imm'])} nbs={e['nbs']} blind={e['blind']}: "
          f"med {m_:.0f} MAD {mad_:.0f} 栏 {fence_:.0f})")
if not outliers:
    print("  无超栏 episode")
episodes_ok = [e for e in episodes if not e.get("out")]
print(f"  保留 {len(episodes_ok)}/{len(episodes)} 进入时长统计 (率统计仍用全部闭合 episode)")

# ── 2. 统计 ──
def stats(v):
    n = len(v)
    if not n:
        return "n=0"
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0
    sv = sorted(v)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    return (f"n={n}  mean={m:.0f}ms  SE={sd / math.sqrt(n):.0f}  SD={sd:.0f}  "
            f"median={med:.0f}  min={sv[0]:.0f}  max={sv[-1]:.0f}")

durs = [e["dur"] for e in episodes_ok]
imm = [e["dur"] for e in episodes_ok if e["imm"]]
delayed = [e["dur"] for e in episodes_ok if not e["imm"]]

print("=" * 78)
print(f"按键总数 {p} (错误按下 {w}, {w / max(p, 1) * 100:.2f}%)"
      f" | episode 闭合 {len(episodes)} / 截断弃置 {tr} | 假警报退格 {fb}")
print(f"episode 率: {len(episodes) / max(p, 1) * 100:.2f}%/按键"
      f"  (停止范式键对 P_err 参考: 角 0.55/首 1.0/中 2.2/尾 3.2%)")
print(f"\n【全体 episode】 {stats(durs)}")
print(f"【立即发现】(错后首事件=退格) {stats(imm)}")
print(f"【延迟发现】(盲打≥1) {stats(delayed)}")
bl = defaultdict(int)
for e in episodes_ok:
    bl[min(e["blind"], 4)] += 1
print("盲打长度分布 (错误按下→首次退格之间的字母数): "
      + "  ".join(f"{k if k < 4 else '4+'}={v}" for k, v in sorted(bl.items())))
nb = defaultdict(int)
for e in episodes_ok:
    nb[min(e["nbs"], 5)] += 1
print("退格数分布: " + "  ".join(f"{k if k < 5 else '5+'}={v}" for k, v in sorted(nb.items())))

# ── 时间趋势: 全局序号 (文件追加序 = 时间序; session 边界只是进程生命周期, 不参与分组) ──
print("\n时间趋势 (按全局序号):")
n_ep = len(durs)
if n_ep >= 8:
    mid = (n_ep + 1) / 2
    m0 = sum(durs) / n_ep
    xs = range(1, n_ep + 1)
    sxx = sum((i - mid) ** 2 for i in xs)
    sxy = sum((i - mid) * (d - m0) for i, d in zip(xs, durs))
    syy = sum((d - m0) ** 2 for d in durs)
    slope = sxy / sxx
    r = sxy / math.sqrt(sxx * syy) if syy else 0.0
    sse = sum((d - (m0 + slope * (i - mid))) ** 2 for i, d in zip(xs, durs))
    se_sl = math.sqrt(sse / (n_ep - 2) / sxx)
    print(f"  线性趋势 {slope:+.1f} ms/episode  (r={r:+.2f}, t={slope / se_sl if se_sl else 0:+.2f}, df={n_ep - 2})")
    # 四分位分段: 均值 / 立即占比 / 段内错误率 (分母 = 段所跨串的按键合计, 串区间已在读入时累计)
    ordered = [(r["key"], r["press"]) for r in kept_ranges]
    pos_of = {st: i for i, (st, _) in enumerate(ordered)}
    k = max(1, n_ep // 4)
    for b, (a, z) in enumerate([(0, k), (k, 2 * k), (2 * k, 3 * k), (3 * k, n_ep)]):
        seg = episodes_ok[a:z]
        if not seg:
            continue
        st0 = (seg[0]["session"], seg[0]["stream"])
        st1 = (seg[-1]["session"], seg[-1]["stream"])
        press_seg = sum(p for _, p in ordered[pos_of[st0]:pos_of[st1] + 1]) \
            if st0 in pos_of and st1 in pos_of else 0
        mi = sum(1 for e in seg if e["imm"])
        print(f"  第{b + 1}段 #{a + 1}-#{z}: n={len(seg)}  mean={sum(e['dur'] for e in seg) / len(seg):.0f}ms"
              f"  立即{mi}/{len(seg)}  按键{press_seg}  错误率{len(seg) / max(press_seg, 1) * 100:.2f}%")
    # 构成去混淆: 整体趋势若由立即/延迟占比变化驱动, 子组各自前后半会持平
    for name, sub in (("立即", imm), ("延迟", delayed)):
        if len(sub) >= 8:
            h = len(sub) // 2
            print(f"  {name}子组前后半: {sum(sub[:h]) / h:.0f}ms → {sum(sub[h:]) / (len(sub) - h):.0f}ms")
    # 坐姿稳定性 (2026-09-07 用户决策: 现值认定为稳定时间, 续采至练习效应可忽略即定案)
    # 定案判据 (三条全满足): ① 簇稳健 SE ≤ 30ms (均值位置钉住, T₄ 传导 <±2ms)
    #   ② 近端窗口 (近 10 坐姿, 不足 10 用全部) 斜率不显著 (|t| < t_{.975,df})。
    #     2026-09-12 改: 全序列版把真实早期下降史 (952→711) 永久计入, 平台期 MC
    #     (实验-错误数据收益对比.py) 示 +12 坐姿仅 ~40% 通过、80% 需 30-40 坐姿 = 实际不可达;
    #     近 10 窗口 = 对持续下降 (-22ms/坐姿) 检出功效 ~65% 与历史惯性的折中 (09-13 用户确认)
    #   ③ 最近 5 坐姿均值与全体均值差 < 40ms (近端平稳)
    sits = []
    for s in dict.fromkeys(e["session"] for e in episodes_ok):   # 文件序 = 坐姿时间序
        v = [e["dur"] for e in episodes_ok if e["session"] == s]
        if len(v) >= 2:
            sits.append((len(v), sum(v) / len(v)))
    if len(sits) >= 4:
        ns_ = len(sits)
        tot = sum(c for c, _ in sits)
        grand2 = sum(c * m for c, m in sits) / tot
        vc = sum(c * c * (m - grand2) ** 2 for c, m in sits) / tot ** 2 * ns_ / (ns_ - 1)
        se_cl = math.sqrt(vc)
        win = sits[-10:]                                   # ② 近端窗口: 只验最近 10 坐姿内无趋势
        mw = len(win)
        mid = (mw + 1) / 2
        mm = sum(m for _, m in win) / mw
        sxx = sum((i - mid) ** 2 for i in range(1, mw + 1))
        sxy = sum((i - mid) * (m - mm) for i, (_, m) in enumerate(win, 1))
        sl = sxy / sxx
        sse = sum((m - (mm + sl * (i - mid))) ** 2 for i, (_, m) in enumerate(win, 1))
        se_sl = math.sqrt(sse / (mw - 2) / sxx) if sse else 0.0
        tc = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
              8: 2.306, 9: 2.262, 10: 2.228}.get(mw - 2, 2.0)
        tval = sl / se_sl if se_sl else 0.0
        recent = sum(m for _, m in sits[-5:]) / min(5, ns_)
        c1, c2, c3 = se_cl <= 30, abs(tval) < tc, abs(recent - grand2) < 40
        print(f"  坐姿均值序列 ({ns_} 坐姿): " + " ".join(f"{m:.0f}" for _, m in sits))
        print(f"  定案判据: ① 簇SE {se_cl:.0f}ms {'✓' if c1 else '✗'}(≤30)  "
              f"② 近{mw}坐姿斜率 {sl:+.1f}ms/坐姿 t={tval:+.2f} {'✓' if c2 else '✗'}(|t|<{tc:.2f})  "
              f"③ 近5坐姿均值 {recent:.0f} vs 全体 {grand2:.0f} {'✓' if c3 else '✗'}(差<40)")
        print("  → " + ("✓✓ 三判据全满足, 可定案 (用户确认后冻结 cost)" if c1 and c2 and c3
                        else "未达定案判据, 继续采集监控"))

# 等成本假设检验入口: 按目标键 / 前键+目标键键对 (样本稀疏时仅积累观察)
by_want = defaultdict(list)
for e in episodes_ok:
    by_want[e["want"]].append(e["dur"])
sig = sorted(((k, len(v), sum(v) / len(v)) for k, v in by_want.items() if len(v) >= 5),
             key=lambda x: -x[1])
if len(sig) >= 5:
    print("\n按目标键 (n≥5): " + "  ".join(f"{k}:{n}个/{m:.0f}ms" for k, n, m in sig))
    ms = [m for _, _, m in sig]
    print(f"  目标键均值极差 {max(ms) - min(ms):.0f}ms (稀疏, 攒样本后再判)")
else:
    print("\n按目标键分组: 样本不足 (每组 n≥5 才显示)")
by_pair = defaultdict(list)
for e in episodes_ok:
    if e["prev"] != "-":
        by_pair[e["prev"] + e["want"]].append(e["dur"])
top = sorted(by_pair.items(), key=lambda kv: -len(kv[1]))[:8]
if top and len(top[0][1]) >= 3:
    print("按键对 top8 (n≥3): " + "  ".join(f"{k}:{len(v)}个/{sum(v) / len(v):.0f}ms" for k, v in top))
else:
    print("按键对分组: 样本不足")
n_multi = sum(1 for e in episodes_ok if e["n_wrong"] >= 2)
print(f"多次错按 episode (同位/连锁吸收): {n_multi}/{len(episodes_ok)}" + (f" ({n_multi / len(episodes_ok) * 100:.0f}%)" if episodes_ok else ""))

print(f"\n样本量: {len(episodes_ok)}/150 目标 (剔异常后; 闭合共 {len(episodes)})"
      + ("  ✓ 够用" if len(episodes_ok) >= 150 else "  → 继续采集"))

# ── 手别系数 (2026-09-14 计算; 09-15 用户决策: 为当量表通用性回退标量部署——手效应是
# 退格位依赖的布局结构 (BS=CapsLock→左目标 got→BS→want 同手串行惩罚), 入表牺牲通用性;
# 手别行保留为诊断导出, 组装当量表.py 只读 cost_ms) ──────────────
# 手因子取自立即基元 (imm & nbs=1 & 围栏内): 延迟 = 立即 + 过程增量 (09-14 验证截距 686≈
# 立即 683), 立即基元无 imm/delayed 构成混杂; 基元内控全局序号 (练习) 的 OLS:
# dur ~ 1 + 序号(居中) + 左手。差值信噪比 |t|≥1 时手别诊断有效; 练习漂移双手同步 →
# 差值对漂移免疫 (09-14 前后半 +69/+78 稳定)。
LEFT_KEYS = frozenset("qwertasdfgzxcvb")   # QWERTY 左手 15 键 (物理布局; 与 击键模型.py COL_TO_HAND 对拍一致)
bim = [e for e in episodes_ok if e["imm"] and e["nbs"] == 1]
nb = len(bim)
mu_bim = sum(e["dur"] for e in bim) / nb if nb else 0.0
hand_beta = hand_se = hand_t = 0.0
FACTOR_L = FACTOR_R = None
lv = [e["dur"] for e in bim if e["want"] in LEFT_KEYS]
rv = [e["dur"] for e in bim if e["want"] not in LEFT_KEYS]
if nb >= 60 and lv and rv:
    mx, mh = (nb - 1) / 2.0, len(lv) / nb
    rows_ = [(1.0, i - mx, (1.0 if e["want"] in LEFT_KEYS else 0.0) - mh) for i, e in enumerate(bim)]
    yv_ = [e["dur"] for e in bim]
    A = [[sum(r[u] * r[v] for r in rows_) for v in range(3)] for u in range(3)]
    bvec = [sum(r[u] * y for r, y in zip(rows_, yv_)) for u in range(3)]
    def _solve3(A, rhs):                    # 3x3 高斯消元 (含单位化回代)
        M = [row[:] + [rhs[u]] for u, row in enumerate(A)]
        for c in range(3):
            p_ = max(range(c, 3), key=lambda k: abs(M[k][c])); M[c], M[p_] = M[p_], M[c]
            pv = M[c][c]
            M[c] = [x / pv for x in M[c]]
            for r_ in range(3):
                if r_ != c and M[r_][c]:
                    f_ = M[r_][c]
                    M[r_] = [a_ - f_ * b_ for a_, b_ in zip(M[r_], M[c])]
        return [M[u][3] for u in range(3)]
    coef = _solve3(A, bvec)
    a22 = _solve3(A, [0.0, 0.0, 1.0])[2]    # (A^-1)[2][2]
    rss = sum((y - (coef[0] + r[1] * coef[1] + r[2] * coef[2])) ** 2 for r, y in zip(rows_, yv_))
    hand_beta = coef[2]
    hand_se = math.sqrt(max(rss / (nb - 3) * a22, 0.0)) if nb > 3 else 0.0
    hand_t = hand_beta / hand_se if hand_se else 0.0
    if abs(hand_t) >= 1.0:
        FACTOR_L = 1.0 + hand_beta / 2.0 / mu_bim
        FACTOR_R = 1.0 - hand_beta / 2.0 / mu_bim
print(f"\n手别系数 (诊断, 不入表——09-15 回退标量; 立即基元 imm&nbs=1, n={nb}): 左 {sum(lv)/max(len(lv),1):.0f}ms (n={len(lv)}) "
      f"vs 右 {sum(rv)/max(len(rv),1):.0f}ms (n={len(rv)}); 控序号 OLS 手效应 "
      f"{hand_beta:+.1f}±{hand_se:.1f}ms (t={hand_t:+.2f}) → "
      + (f"因子 {FACTOR_L:.4f}/{FACTOR_R:.4f} (仅记录)" if FACTOR_L
         else "|t|<1, 手别差异信噪比不足"))

# ── 导出实测值 (组装当量表.py 的错误成本来源; 2026-09-06 起替代固定 500ms, 随采集动态更新;
#    2026-09-14 起含手别行, 用户决策: 忽略连续文本 vs 4键限长文本的错误时间 regime 差异) ──
OUT_COST = _DIR / "产物" / "错误成本-实测值.txt"
OUT_COST.parent.mkdir(exist_ok=True)
if durs:
    m = sum(durs) / len(durs)
    sd = math.sqrt(sum((x - m) ** 2 for x in durs) / (len(durs) - 1)) if len(durs) > 1 else 0.0
    with open(OUT_COST, "w", encoding="utf-8") as f:
        f.write("# 错误成本实测值 (分析-错误成本.py 自动导出; 组装当量表.py / 分析-错误率规律.py 读取)\n")
        f.write(f"# 口径: 全体闭合 episode 均值 (净成本), 弃置前 {DISCARD} 按键, 连锁错吸收单事件\n")
        f.write(f"# 时长异常剔除: 过程分桶 (发现型×退格数档) 桶内单侧 med+3×1.4826×MAD 围栏, "
                f"本次剔 {len(outliers)} 条 (桶 n<8 不设栏); 率统计仍用全部闭合 episode\n")
        f.write("# 手别: 立即基元 (imm&nbs=1) 控序号 OLS 手效应, |t|≥1 启用; cost_left/right = 标量×因子\n")
        f.write("label\tvalue\tn\textra\n")
        f.write(f"cost_ms\t{m:.1f}\t{len(durs)}\t{sd / math.sqrt(len(durs)):.1f}\n")
        if FACTOR_L:
            f.write(f"cost_left\t{m * FACTOR_L:.1f}\t{len(lv)}\tfactor {FACTOR_L:.4f}\n")
            f.write(f"cost_right\t{m * FACTOR_R:.1f}\t{len(rv)}\tfactor {FACTOR_R:.4f}\n")
            f.write(f"hand_beta\t{hand_beta:.1f}\t{hand_se:.1f}\tt {hand_t:+.2f}\tn_base {nb}\n")
    print(f"导出实测值 → 产物/错误成本-实测值.txt (cost={m:.1f}ms, n={len(durs)}"
          + (f"; 手别 {m*FACTOR_L:.1f}/{m*FACTOR_R:.1f}ms" if FACTOR_L else "; 手别未启用") + ")")
