#!/usr/bin/env python3
"""
错误率规律探索 + 签名条件逻辑回归（2026-09-02 签名版）。

模型: P_err(a,b | 有无前键, 有无后键) = σ(w·feat9(a,b) + α·有前键 + β·有后键 + γ·前×后)
      签名与 S(p,a,b,n) 的 (p,n) 语义统一, 角点 (0,0) 为参考电平:
        4键 pos1→(0,1) 首段   pos2→(1,1) 中段   pos3→(1,0) 尾段   2键 pos1→(0,0) 角点
      采纳依据 (2026-09-02): 错误率随码内位置爬升 (角点 0.63 / 首段 0.98 / 中段 2.16 /
      尾段 3.17%), 无上下文模型对 T₂ 角点高估 ~7ms/条目; 交互项必需——纯加法形式角点
      仍高估 2× (O/E 0.51), LRT χ²₁=5.97。事件级 AUC 0.650→0.698。
事件口径 (2026-09-02 修正): 4 键 + 2 键行 (2 键行即角点 (0,0) 类, 首次参与);
      正确 trial 每键对 1 ok 事件; 错误 trial 在错键截断——错键对记 1 err、其前的
      键对记 ok (旧版把错键对同时记 ok+err, 双重计数 623 条, 已修); 首键即错不记
      事件 (键对未尝试)。仍用全部数据 (含练习期, 用户决策——错误事件稀疏)。

用法: python 分析-错误率规律.py [数据.tsv] [--dual]
"""
import sys, csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import norm

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from 击键模型 import LETTERS, LETTER_TO_COL, LETTER_TO_ROW, COL_TO_FINGER, COL_TO_HAND

PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else str(_DIR / "击键测速数据.tsv")
DUAL = "--dual" in sys.argv  # 双向计数: 错按键对 (目标首键,实际错键) 同记 (2026-08-10 实验)

# 键位特征 (9 特征, 2026-08-08 定稿, AUC 0.635):
#   演进: 8 特征 (删列差 CD, 生理学无依据+零损失) → 7 特征 (删行差 RD,
#     与 BT/BB 冗余: 控制目标行后行差无梯度, 去 RD 0.606→0.603 持平)
#     → 9 特征 (坏指对 4 个替代邻近 ADJ+弱指 W, +b左小指, 0.606→0.635)。
#   坏指对 (同手数据): 无名-中指 3.85% 最高, 同指同列 3.17%, 小指-无名 2.68%,
#     同指双列 2.33% —— 具体指对关系替代几何"邻近" (跨手邻近=食指特殊性,
#     归入跨手 SH=0 基线)。
#   弱指不对称 (2026-08-08): a 小指无效应 (1.83 vs 1.91); b 小指左右不对称
#     (左 2.61% vs 右 1.35% 互相抵消) → 只保留 b 左小指特征 BLP。
#   行难度 (2026-08-08): 目标行绝对难度主导 (BT/BB, 同手跨行同幅度),
#     起点行/行移动无独立贡献。
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

# ── 文献新特征 (2026-08-10 补充, 见下方注释) ─────────
def mirror_pair(a, b):    # MIR 异手对称手指对 (Grudin 1983: 29% 换位错误发生在异手对称手指)
    return int(not same_hand(a, b) and _f(a) + _f(b) == 7)
def cross_row_same_finger(a, b):  # CRS 同指跨行 (keygen: 同指顶↔底行最慢最易错; MacNeilage: 垂直错误主体)
    return int(same_hand(a, b) and _f(a) == _f(b) and
               LETTER_TO_ROW[a] != LETTER_TO_ROW[b])
def adjacent_finger(a, b):  # ADJ 同手相邻手指 (sEMG 共激活, 尤其中指-无名指)
    return int(same_hand(a, b) and abs(_f(a) - _f(b)) == 1)
def b_toward_thumb(a, b):   # BTT b 比 a 更靠拇指侧 (Lachnit 1990: 向拇指侧相邻手指误击率最高, 方向不对称)
    if not same_hand(a, b): return 0
    return int(_f(b) > _f(a)) if COL_TO_HAND[LETTER_TO_COL[a]] == 0 else int(_f(a) > _f(b))

# 特征集: 9 特征基线 (2026-08-08 定稿) vs 13 特征 (文献扩充)
FEATS_BASE = [same_hand, same_finger, bad_samecol, bad_ringmid,
              bad_pinkyring, bad_index2c, b_leftpinky,
              lambda a, b: top_row(b), lambda a, b: bottom_row(b)]
FEATS_NEW  = FEATS_BASE + [mirror_pair, cross_row_same_finger,
                           adjacent_finger, b_toward_thumb]
NAMES_BASE = ["同手", "同指", "同指同列", "无名-中指", "小指-无名",
              "同指双列", "b左小指", "b顶行", "b底行"]
NAMES_NEW  = NAMES_BASE + ["镜像手指", "跨行同指", "相邻手指", "b靠拇指侧"]

# ── 键对事件 (签名口径, 2026-09-02) ──────────────────
# 事件 = (a, b, ok, 有前键, 有后键); 签名映射与 S(p,a,b,n) 一致
SIG = {4: {1: (0, 1), 2: (1, 1), 3: (1, 0)},   # 4 键: pos1 首段 / pos2 中段 / pos3 尾段
       2: {1: (0, 0)}}                          # 2 键: 角点
events = []
def add_events(code, act, ts_ok):
    L_ = len(code)
    cls = SIG.get(L_)
    if cls is None: return
    if ts_ok:
        for i in range(1, L_):
            pv, nx = cls[i]
            events.append((code[i-1], code[i], 1, pv, nx))
        return
    pos = None
    for i, (c, a) in enumerate(zip(code, act)):
        if c != a: pos = i; break
    if pos is None or pos == 0: return   # 未定位 / 首键即错: 无键对事件
    for i in range(1, pos):              # 错键之前的键对 → ok (错键对只记 err, 修双重计数)
        pv, nx = cls[i]
        events.append((code[i-1], code[i], 1, pv, nx))
    pv, nx = cls[pos]
    events.append((code[pos-1], code[pos], 0, pv, nx))
    # 双向计数 (--dual): 错按键对 (目标首键,实际错键) 同记, 签名同错键位置
    if DUAL and act[pos] in LETTERS and act[pos] != code[pos]:
        events.append((code[pos-1], act[pos], 0, pv, nx))

rows = list(csv.DictReader(open(PATH, encoding="utf-8"), delimiter="\t"))
for r in rows:
    code = r["code"]
    if r["error"] == "0": add_events(code, code, True)
    else: add_events(code, r["actual"], False)

N = len(events)
POS = sum(1 for e in events if e[2] == 0)
n2 = sum(1 for e in events if e[3] == 0 and e[4] == 0)
e2 = sum(1 for e in events if e[3] == 0 and e[4] == 0 and e[2] == 0)
print(f"键对事件 {N} (错误 {POS}, {POS/N*100:.2f}%)  | 其中 2 键角点类 {n2} (错误 {e2})")

# ── 1. 探索: 错误率 × 特征分层 (签名混合口径) ─────────
def layer_rate(feat_fn, name, groups):
    """feat_fn: (a,b)->group key; groups: 组名列表"""
    print(f"\n[{name}]")
    cnt = Counter(); err = Counter()
    for a, b, ok, _, _ in events:
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
layer_rate(lambda a, b: "镜像手指" if mirror_pair(a,b) else "非",
           "镜像手指 (异手对称)", ["镜像手指", "非"])
layer_rate(lambda a, b: "跨行同指" if cross_row_same_finger(a,b) else "非",
           "跨行同指", ["跨行同指", "非"])
layer_rate(lambda a, b: "相邻手指" if adjacent_finger(a,b) else "非",
           "相邻手指 (同手)", ["相邻手指", "非"])
layer_rate(lambda a, b: "b靠拇指" if b_toward_thumb(a,b) else "b离拇指" if adjacent_finger(a,b) else "非",
           "相邻方向", ["b靠拇指", "b离拇指", "非"])

# ── 2. 键对特征集选择: 9 vs 13 (5-fold AUC, 签名混合口径) ──
print("\n=== 键对特征逻辑回归 (特征集选择) ===")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def feats(a, b, fs):
    return [1.0] + [f(a, b) for f in fs]

Y = np.array([1.0 if ok else 0.0 for _, _, ok, _, _ in events], dtype=np.float32)  # 1=正确

def fit_and_eval(fs, label):
    X = np.array([feats(a, b, fs) for a, b, _, _, _ in events], dtype=np.float32)
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-6
    Xs = np.concatenate([X[:, :1], (X[:, 1:] - mu) / sd], axis=1)
    aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(Xs, Y):
        m = LogisticRegression(max_iter=2000, C=10.0)
        m.fit(Xs[tr], Y[tr])
        aucs.append(roc_auc_score(Y[te], m.predict_proba(Xs[te])[:, 1]))
    auc, sd_auc = np.mean(aucs), np.std(aucs)
    print(f"  {label:24s}: 5-fold AUC = {auc:.3f} ± {sd_auc:.3f}  ({len(fs)} 特征, N={len(Y)})")
    return auc

auc_base = fit_and_eval(FEATS_BASE, "基线 9 特征")
auc_new = fit_and_eval(FEATS_NEW, "文献扩充 13 特征")
if auc_new >= auc_base:
    FEATS_F, NAMES_F = FEATS_NEW, NAMES_NEW
    print(f"  → 采用 13 特征扩充 (AUC {auc_new:.3f} ≥ 基线 {auc_base:.3f})")
else:
    FEATS_F, NAMES_F = FEATS_BASE, NAMES_BASE
    print(f"  → 保留 9 特征基线 (AUC {auc_base:.3f} ≥ 扩充 {auc_new:.3f})")

# ── 3. 最终模型: feat9 + 签名特征 (α/β/γ, IRLS + Wald) ──
print("\n=== 签名条件模型: P_err = σ(w·feat + α·有前键 + β·有后键 + γ·前×后) ===")
Yerr = 1.0 - Y                                   # 1=错误
PV = np.array([e[3] for e in events], dtype=float)
NX = np.array([e[4] for e in events], dtype=float)
A_ = [e[0] for e in events]; B_ = [e[1] for e in events]
F9 = np.array([[f(a, b) for f in FEATS_F] for a, b in zip(A_, B_)], dtype=float)
mu9, sd9 = F9.mean(0), F9.std(0) + 1e-12
F9s = (F9 - mu9) / sd9
one = np.ones((N, 1))
X = np.concatenate([one, PV[:, None], NX[:, None], (PV * NX)[:, None], F9s], axis=1)

def irls(X, y, iters=60):
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(y.mean(), 1e-6) / (1 - y.mean()))
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30); p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-12
        H = X.T @ (X * W[:, None]); g = X.T @ (y - p)
        step = np.linalg.solve(H, g); b += step
        if np.max(np.abs(step)) < 1e-10: break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    ll = float(np.sum(y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300)))
    return b, np.sqrt(np.diag(np.linalg.inv(H))), ll, p

b, se, ll, p_err_hat = irls(X, Yerr)
b0, alpha, beta, gamma = b[0], b[1], b[2], b[3]
print(f"  有前键 α = {alpha:+.3f} ± {se[1]:.3f} (z={alpha/se[1]:+.2f})   "
      f"有后键 β = {beta:+.3f} ± {se[2]:.3f} (z={beta/se[2]:+.2f})   "
      f"前×后 γ = {gamma:+.3f} ± {se[3]:.3f} (z={gamma/se[3]:+.2f}, "
      f"LRT p={2*norm.sf(abs(gamma/se[3])):.3f})")
w_raw = b[4:] / sd9                       # 键对特征还原到原始尺度
base0 = b0 - (b[4:] * mu9 / sd9).sum()    # 角点类 (0,0) 截距

# 各签名类校准 (饱和类哑变量 → 聚合 O/E 应为 1)
print("\n签名类校准 (观察错误 / 模型期望):")
for nm, (v0, v1) in {"(0,0) 角点": (0, 0), "(0,1) 首段": (0, 1),
                     "(1,1) 中段": (1, 1), "(1,0) 尾段": (1, 0)}.items():
    m = (PV == v0) & (NX == v1)
    o = float(Yerr[m].sum()); e = float(p_err_hat[m].sum())
    print(f"  {nm:10s} n={int(m.sum()):6d}  obs={int(o):4d}  exp={e:7.1f}  O/E = {o/e:.2f}")

# 公式验证 (还原系数 vs 模型输出, 中段类)
def perr_formula(a, b_, pv, nx):
    logit = base0 + alpha*pv + beta*nx + gamma*pv*nx + (w_raw * np.array([f(a, b_) for f in FEATS_F])).sum()
    return 1 / (1 + np.exp(-logit))
print("\n公式验证 (还原系数公式 vs 标准化空间模型, 中段类):")
for ab_ in ["ab", "fg", "aa", "sz"]:
    a_, b__ = ab_[0], ab_[1]
    xs = np.concatenate([[1.0, 1.0, 1.0, 1.0],
                         (np.array([f(a_, b__) for f in FEATS_F]) - mu9) / sd9])
    pm = 1 / (1 + np.exp(-(b @ xs)))
    pf = perr_formula(a_, b__, 1, 1)
    print(f"  {ab_}: 公式 P_err={pf:.4f}  模型={pm:.4f}  ({'✓' if abs(pf-pm)<1e-6 else '✗'})")

# ── 4. 导出: 900 键对 × 4 签名类 ─────────────────────
print("\n=== 导出 当量-键对错误率.txt (900 键对 × 4 签名类) ===")
pairs = [(a, b) for a in LETTERS for b in LETTERS]
Xp9 = np.array([[f(a, b) for f in FEATS_F] for a, b in pairs], dtype=float)
base900 = base0 + Xp9 @ w_raw
sig = lambda z: 1 / (1 + np.exp(-z))
P00, P01 = sig(base900), sig(base900 + beta)
P10, P11 = sig(base900 + alpha), sig(base900 + alpha + beta + gamma)
with open(_DIR / "当量-键对错误率.txt", "w", encoding="utf-8") as f:
    f.write("pair\terr_p0n0\terr_p0n1\terr_p1n0\terr_p1n1\n")
    for (a, b), v00, v01, v10, v11 in zip(pairs, P00, P01, P10, P11):
        f.write(f"{a}{b}\t{v00:.4f}\t{v01:.4f}\t{v10:.4f}\t{v11:.4f}\n")
print(f"  已导出 当量-键对错误率.txt (列: 角点/首段/尾段/中段, p=有前键 n=有后键)")
print(f"  类均值 P_err: 角点 {P00.mean()*100:.2f}% / 首段 {P01.mean()*100:.2f}% / "
      f"尾段 {P10.mean()*100:.2f}% / 中段 {P11.mean()*100:.2f}%")
print(f"  平均错误成本 (×500ms): T₂ 500×角点 = {500*P00.mean():.2f}ms | "
      f"T₃ 500×(首+尾) = {500*(P01+P10).mean():.2f}ms | T₄ 500×(首+中+尾) = {500*(P01+P11+P10).mean():.2f}ms")
k = max(range(900), key=lambda i: P11[i])
print(f"  极值键对: {pairs[k][0]}{pairs[k][1]} 角点 {P00[k]*100:.1f}% / 尾段 {P10[k]*100:.1f}% / 中段 {P11[k]*100:.1f}%")
