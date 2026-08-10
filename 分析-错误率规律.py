#!/usr/bin/env python3
"""
错误率规律探索 + 简单模型（可解释特征逻辑回归）。

动机: 键位嵌入模型 (6000 参数) 在 259 个错误事件下无判别力 (AUC≈0.5) —
      复杂模型的稀疏问题。简单模型用键位特征 (距离/同指/弱指等, ~10 维),
      每特征上万样本, 不依赖键对级数据, 可研究"什么因素影响正确率"并
      给出结构化估算: p(a,b) = σ(β·features)。

用法: python 分析-错误率规律.py [数据.tsv]
"""
import sys, csv
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, ".")
from 击键模型 import LETTERS, LETTER_TO_COL, LETTER_TO_ROW, COL_TO_FINGER, COL_TO_HAND

PATH = sys.argv[1] if len(sys.argv) > 1 else "击键测速数据.tsv"

# 键位特征 (9 特征, 2026-08-08 定稿, AUC 0.635):
#   演进: 8 特征 (删列差 CD, 生理学无依据+零损失) → 7 特征 (删行差 RD,
#     与 BT/BB 冗余: 控制目标行后行差无梯度, 去 RD 0.606→0.603 持平)
#     → 9 特征 (坏指对 4 个替代邻近 ADJ+弱指 W, +b左小指, 0.606→0.635)。
#   坏指对 (同手数据): 无名-中指 3.85% 最高, 同指同列 3.17%, 小指-无名 2.68%,
#     同指双列 2.33% —— 具体指对关系替代几何"邻近" (跨手邻近=食指特殊性,
#     归入跨手 SH=0 基线)。
#   弱指不对称 (2026-08-08): a 小指无效应 (1.83 vs 1.91); b 小指左右不对称
#     (左 2.61% vs 右 1.35% 互相抵消) → 只保留 b 左小指特征 BLP。
#   行难度 (2026-08-08): 目标行绝对难度主导 (BT/BB, 同手跨手同幅度),
#     起点行/行移动无独立贡献。
def col_dist(a, b):
    sh = int(COL_TO_HAND[LETTER_TO_COL[a]] == COL_TO_HAND[LETTER_TO_COL[b]])
    return abs(LETTER_TO_COL[a] - LETTER_TO_COL[b]) * sh   # 备用
def row_dist(a, b):
    sh = int(COL_TO_HAND[LETTER_TO_COL[a]] == COL_TO_HAND[LETTER_TO_COL[b]])
    return abs(LETTER_TO_ROW[a] - LETTER_TO_ROW[b]) * sh   # 备用
def same_hand(a, b): return int(COL_TO_HAND[LETTER_TO_COL[a]] == COL_TO_HAND[LETTER_TO_COL[b]])
def same_finger(a, b): return int(COL_TO_FINGER[LETTER_TO_COL[a]] == COL_TO_FINGER[LETTER_TO_COL[b]])
def _f(a): return COL_TO_FINGER[LETTER_TO_COL[a]]
def bad_samecol(a, b):   # B1 同指同列 (同列上下移动)
    return int(same_hand(a, b) and _f(a) == _f(b) and
               LETTER_TO_COL[a] == LETTER_TO_COL[b] and a != b)
def bad_ringmid(a, b):   # B2 无名-中指 (同手)
    return int(same_hand(a, b) and {_f(a), _f(b)} in ({1, 2}, {6, 5}))
def bad_pinkyring(a, b): # B3 小指-无名 (同手)
    return int(same_hand(a, b) and {_f(a), _f(b)} in ({0, 1}, {7, 6}))
def bad_index2c(a, b):   # B4 同指双列 (食指内外横向)
    return int(same_hand(a, b) and _f(a) == _f(b) and
               abs(LETTER_TO_COL[a] - LETTER_TO_COL[b]) > 0)
def b_leftpinky(a, b):   # BLP b 为左小指 (仅左小指, 右小指是保护项不混入)
    return int(_f(b) == 0)
def top_row(a): return int(LETTER_TO_ROW[a] == 0)
def bottom_row(a): return int(LETTER_TO_ROW[a] == 2)

# ── 键对事件 (修正统计) ─────────────────────────────
ok_ev = Counter(); err_ev = Counter()
def add_events(code, act, ts_ok):
    if ts_ok:
        for i in range(3):
            prev = code[i-1] if i > 0 else "_"
            ok_ev[(prev, code[i], code[i+1])] += 1
        return
    pos = None
    for i, (c, a) in enumerate(zip(code, act)):
        if c != a: pos = i; break
    if pos is None: return
    for i in range(pos):
        prev = code[i-1] if i > 0 else "_"
        ok_ev[(prev, code[i], code[i+1])] += 1
    if pos >= 1:
        prev = code[pos-2] if pos >= 2 else "_"
        err_ev[(prev, code[pos-1], code[pos])] += 1

rows = list(csv.DictReader(open(PATH, encoding="utf-8"), delimiter="\t"))
for r in rows:
    code = r["code"]
    if len(code) != 4: continue
    if r["error"] == "0": add_events(code, code, True)
    else: add_events(code, r["actual"], False)

# 事件: (a, b, ok)
events = []
for (_, a, b), n in ok_ev.items(): events.extend([(a, b, 1)] * n)
for (_, a, b), n in err_ev.items(): events.extend([(a, b, 0)] * n)
N = len(events)
POS = sum(1 for e in events if e[2] == 0)
print(f"键对事件 {N} (错误 {POS}, {POS/N*100:.2f}%)")

# ── 1. 探索: 错误率 × 特征分层 ─────────────────────
def layer_rate(feat_fn, name, groups):
    """feat_fn: (a,b)->group key; groups: 组名列表"""
    print(f"\n[{name}]")
    cnt = Counter(); err = Counter()
    for a, b, ok in events:
        g = feat_fn(a, b)
        cnt[g] += 1
        if not ok: err[g] += 1
    for g in groups:
        n, e = cnt[g], err[g]
        if n:
            print(f"  {g}: 错误率 {e/n*100:5.2f}%  (n={n})")
        else:
            print(f"  {g}: 无样本")

layer_rate(lambda a, b: "同手" if same_hand(a,b) else "跨手", "手",
           ["同手", "跨手"])
layer_rate(lambda a, b: "同指" if same_finger(a,b) else "异指", "指",
           ["同指", "异指"])
layer_rate(lambda a, b: "同指同列" if bad_samecol(a,b) else "非",
           "同指同列", ["同指同列", "非"])
layer_rate(lambda a, b: "无名-中指" if bad_ringmid(a,b) else "非",
           "无名-中指", ["无名-中指", "非"])
layer_rate(lambda a, b: "小指-无名" if bad_pinkyring(a,b) else "非",
           "小指-无名", ["小指-无名", "非"])
layer_rate(lambda a, b: "同指双列" if bad_index2c(a,b) else "非",
           "同指双列", ["同指双列", "非"])
layer_rate(lambda a, b: "b左小指" if b_leftpinky(a,b) else "非",
           "b左小指", ["b左小指", "非"])
layer_rate(lambda a, b: f"b行={LETTER_TO_ROW[b]}", "目标键行",
           [f"b行={r}" for r in range(3)])

# ── 2. 简单模型: 特征逻辑回归 ───────────────────────
print("\n=== 特征逻辑回归 (估算键对正确率) ===")
def feats(a, b):
    return [1.0,
            same_hand(a, b), same_finger(a, b),
            bad_samecol(a, b), bad_ringmid(a, b),
            bad_pinkyring(a, b), bad_index2c(a, b),
            b_leftpinky(a, b),
            top_row(b), bottom_row(b)]
X = np.array([feats(a, b) for a, b, _ in events], dtype=np.float32)
Y = np.array([1.0 if ok else 0.0 for _, _, ok in events], dtype=np.float32)  # 1=正确
# 标准化 (除第一列截距)
mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-6
Xs = np.concatenate([X[:, :1], (X[:, 1:] - mu) / sd], axis=1)

# 5-fold AUC + 系数 (sklearn, 带类权重)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

aucs = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for tr, te in skf.split(Xs, Y):
    m = LogisticRegression(max_iter=2000, C=10.0)
    m.fit(Xs[tr], Y[tr])
    aucs.append(roc_auc_score(Y[te], m.predict_proba(Xs[te])[:, 1]))
print(f"5-fold AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# 全量拟合系数 (标准化空间 → 还原到原始特征尺度)
m = LogisticRegression(max_iter=2000, C=10.0)
m.fit(Xs, Y)
names = ["截距", "同手", "同指", "同指同列", "无名-中指", "小指-无名",
         "同指双列", "b左小指", "b顶行", "b底行"]
w_raw = np.zeros(len(names))
# 还原: logit = intercept + c0·x0(常数列=1.0) + Σ c_i·(x_i-μ)/σ
w_raw[0] = m.intercept_[0] + m.coef_[0][0] - (m.coef_[0][1:] * mu / sd).sum()
w_raw[1:] = m.coef_[0][1:] / sd
print("\n系数 (β, 正值=更正确; 原始特征尺度, 截距含标准化偏置补偿):")
for n_, b_ in zip(names, w_raw):
    print(f"  {n_:6s}: {b_:+.3f}")
prob = m.predict_proba(Xs)[:, 1]

# 验证还原公式: 抽几个键对对比模型输出
print("\n公式验证 (还原系数 logit vs 模型输出):")
for ab_ in ["ab", "fg", "aa", "sz"]:
    a_, b_ = ab_[0], ab_[1]
    x = np.array(feats(a_, b_), dtype=np.float32)
    logit = w_raw[0] + (w_raw[1:] * x[1:]).sum()
    p_formula = 1 / (1 + np.exp(-logit))
    xs = np.concatenate([x[:1], (x[1:] - mu) / sd])
    p_model = m.predict_proba(xs.reshape(1, -1))[0, 1]
    print(f"  {ab_}: 公式 P_corr={p_formula:.4f}  模型={p_model:.4f}  "
          f"({'✓' if abs(p_formula-p_model)<1e-3 else '✗'})")

# ── 3. 键对级估算 vs 直接统计 (n≥10) ───────────────
print("\n=== 键对级: 模型估算 vs 直接统计 (n≥10) ===")
pair_model = defaultdict(list); pair_stat = defaultdict(lambda: [0, 0])
for (a, b, ok), p in zip(events, prob):
    pair_model[(a, b)].append(p)
    pair_stat[(a, b)][0] += 1
    pair_stat[(a, b)][1] += 1 if ok else 0
diffs = []
for k, n in pair_stat.items():
    if n[0] < 10: continue
    p_model = 1 - np.mean(pair_model[k])
    p_stat = 1 - n[1] / n[0]
    diffs.append(abs(p_model - p_stat))
d = np.array(diffs)
print(f"  n≥10 键对 {len(d)} 个: |模型错误率-统计错误率| MAE = {d.mean()*100:.2f}pp")
print(f"  (统计错误率本身: 0-25%, 模型输出受特征结构约束)")

# 输出估算表 top/bottom
print("\n模型估算错误率 top10:")
est = {}
for k, ps in pair_model.items():
    est[k] = 1 - np.mean(ps)
for k in sorted(est, key=lambda k: -est[k])[:10]:
    print(f"  {k}: {est[k]*100:.1f}%  (n={pair_stat[k][0]})")

# ── 4. 导出键对错误率表 (900 键对全量, 直接错误率, 无需正确率中间步骤) ──
print("\n=== 导出 当量-键对错误率.txt (900 键对, 期望当量 = 段当量 + 500ms×P_err) ===")
with open("当量-键对错误率.txt", "w", encoding="utf-8") as f:
    f.write("pair\terr_rate\n")
    for a in LETTERS:
        for b in LETTERS:
            x = np.array([feats(a, b)], dtype=np.float32)
            xs = np.concatenate([x[:, :1], (x[:, 1:] - mu) / sd], axis=1)
            p_corr = float(m.predict_proba(xs)[0, 1])
            f.write(f"{a}{b}\t{1 - p_corr:.4f}\n")
print("  已导出 当量-键对错误率.txt")
print("  期望当量 = 条件段当量 S(prev,a,b) + 500ms × P_err(a,b)")
