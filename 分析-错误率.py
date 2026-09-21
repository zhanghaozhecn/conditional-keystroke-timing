#!/usr/bin/env python3
"""
错误率规律探索 + 签名×段位条件逻辑回归（2026-09-19 重参数化；2026-09-21 去交互项）。

模型: P_err(a,b | 段类) = σ(w·feat(a,b) + β·有后键 + δ·(段位−1))
      段类 = (有前键, 有后键, 段位)，共 5 类（由 码长×段位 映射）:
        c0 角点 (∅,∅,1) 2键seg1 | c1 首段 (∅,有,1) 3/4键seg1 | c2 短尾 (有,∅,2) 3键seg2
        c3 中段 (有,有,2) 4键seg2 | c4 长尾 (有,∅,3) 4键seg3
      五类电平 (相对 c0 角点): c1=+β, c2=+δ, c3=+β+δ, c4=+2δ  ← **纯加法形式**
      **交互项 γ 已删 (2026-09-21 用户决策「λ 交互现在就可以简化掉」)**: γ(有后键×段位)
      在段类重参数化后的口径下历次都不显著 (−0.10±0.17, z=−0.58) 且判定它需 ~1276 场
      数据 (README §5.7)、点估计 ≈0 意味其对总当量影响在第四位 → 删掉换 2 个自由度。
      本脚本每次运行**并排报一次带 γ 的拟合** (deviance + LRT)，作为「删得对不对」的常设监控。
      重参数化依据 (2026-09-19, 六格拟合): 旧四列签名在仅 2/4 键数据上与 (码长,段位)
      一一对应 (无损重参数化)，加入 3 键后「尾段」类被劈为 c2/c4——实测 1.64% vs 3.15%
      (z=−3.08, p=0.0020) → 旧模型显著拟合不足 (D=10.17→8.87, p=0.006→0.012)。
      **「有前键」不再单列**: 观测数据里 有前键 ⟺ 段位≥2 完全共线，旧模型的
      α=+2.10 实为位置效应——引入段位项后其系数塌缩为 0 (z=0.02)；改由
      δ·(段位−1) 承载「码内累积负担」，β 承载「末位保护」。
事件口径 (2026-09-02 修正, 2026-09-19 起 3 键行入模): 2/3/4 键行全部参与;
      正确 trial 每键对 1 ok 事件; 错误 trial 在错键截断——错键对记 1 err、其前的
      键对记 ok (旧版把错键对同时记 ok+err, 双重计数 623 条, 已修); 首键即错不记
      事件 (键对未尝试)。仍用全部数据 (含练习期, 用户决策——错误事件稀疏)。

用法: python 分析-错误率.py [数据.tsv] [--dual]
"""
import sys, csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import norm

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
import importlib
_kj = importlib.import_module("分析-按键时间")
LETTERS = _kj.LETTERS
LETTER_TO_COL = _kj.LETTER_TO_COL
LETTER_TO_ROW = _kj.LETTER_TO_ROW
COL_TO_FINGER = _kj.COL_TO_FINGER
COL_TO_HAND = _kj.COL_TO_HAND

PATH = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else str(_DIR / "数据" / "击键测速数据.tsv")
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

# ── 键对事件 (签名×段位口径, 2026-09-19) ─────────────
# 事件 = (a, b, ok, 段类); 段类由 (码长, 段位) 映射, 属性见 CLS_ATTR
SEGCLS = {2: {1: 0},                 # 角点 (∅,∅,1)
          3: {1: 1, 2: 2},           # 首段 (∅,有,1) | 短尾 (有,∅,2)
          4: {1: 1, 2: 3, 3: 4}}     # 首段 | 中段 (有,有,2) | 长尾 (有,∅,3)
CLS_ATTR = {0: (0, 0, 1), 1: (0, 1, 1), 2: (1, 0, 2), 3: (1, 1, 2), 4: (1, 0, 3)}  # (有前键,有后键,段位)
CLS_NAME = {0: "角点(∅,∅,1)", 1: "首段(∅,有,1)", 2: "短尾(有,∅,2)",
            3: "中段(有,有,2)", 4: "长尾(有,∅,3)"}
events = []
def add_events(code, act, ts_ok):
    L_ = len(code)
    cls = SEGCLS.get(L_)
    if cls is None: return
    if ts_ok:
        for i in range(1, L_):
            events.append((code[i-1], code[i], 1, cls[i]))
        return
    pos = None
    for i, (c, a) in enumerate(zip(code, act)):
        if c != a: pos = i; break
    if pos is None or pos == 0: return   # 未定位 / 首键即错: 无键对事件
    for i in range(1, pos):              # 错键之前的键对 → ok (错键对只记 err, 修双重计数)
        events.append((code[i-1], code[i], 1, cls[i]))
    events.append((code[pos-1], code[pos], 0, cls[pos]))
    # 双向计数 (--dual): 错按键对 (目标首键,实际错键) 同记, 段类同错键位置
    if DUAL and act[pos] in LETTERS and act[pos] != code[pos]:
        events.append((code[pos-1], act[pos], 0, cls[pos]))

rows = list(csv.DictReader(open(PATH, encoding="utf-8"), delimiter="\t"))
for r in rows:
    code = r["code"]
    if r["error"] == "0": add_events(code, code, True)
    else: add_events(code, r["actual"], False)

N = len(events)
POS = sum(1 for e in events if e[2] == 0)
n_c = Counter(e[3] for e in events)
e_c = Counter(e[3] for e in events if e[2] == 0)
print(f"键对事件 {N} (错误 {POS}, {POS/N*100:.2f}%) — 段类分布:")
for c_ in range(5):
    print(f"  c{c_} {CLS_NAME[c_]:14s} n={n_c[c_]:6d}  错误 {e_c[c_]:4d}  "
          f"= {e_c[c_]/max(n_c[c_], 1)*100:.3f}%")

# ── 1. 探索: 错误率 × 特征分层 (签名混合口径) ─────────
def layer_rate(feat_fn, name, groups):
    """feat_fn: (a,b)->group key; groups: 组名列表"""
    print(f"\n[{name}]")
    cnt = Counter(); err = Counter()
    for a, b, ok, _ in events:
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

Y = np.array([1.0 if ok else 0.0 for _, _, ok, _ in events], dtype=np.float32)  # 1=正确

def fit_and_eval(fs, label):
    X = np.array([feats(a, b, fs) for a, b, _, _ in events], dtype=np.float32)
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

# ── 3. 最终模型: feat + 签名×段位 加法形式 (β/δ, IRLS + Wald) ──
print("\n=== 签名×段位条件模型（加法）: P_err = σ(w·feat + β·有后键 + δ·(段位−1)) ===")
Yerr = 1.0 - Y                                   # 1=错误
CL = np.array([e[3] for e in events], dtype=int)
PV = np.array([CLS_ATTR[c_][0] for c_ in CL], dtype=float)        # 有前键 (与 段位≥2 完全共线, 不入模)
NX = np.array([CLS_ATTR[c_][1] for c_ in CL], dtype=float)        # 有后键
PS = np.array([CLS_ATTR[c_][2] - 1 for c_ in CL], dtype=float)    # 段位−1
A_ = [e[0] for e in events]; B_ = [e[1] for e in events]
F9 = np.array([[f(a, b) for f in FEATS_F] for a, b in zip(A_, B_)], dtype=float)
mu9, sd9 = F9.mean(0), F9.std(0) + 1e-12
F9s = (F9 - mu9) / sd9
one = np.ones((N, 1))
X = np.concatenate([one, NX[:, None], PS[:, None], F9s], axis=1)

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
b0, beta, delta = b[0], b[1], b[2]
print(f"  有后键 β = {beta:+.3f} ± {se[1]:.3f} (z={beta/se[1]:+.2f}, LRT p={2*norm.sf(abs(beta/se[1])):.4f})   "
      f"段位 δ = {delta:+.3f} ± {se[2]:.3f} (z={delta/se[2]:+.2f}, LRT p={2*norm.sf(abs(delta/se[2])):.4f})")
# 常设监控: 并排拟合一次带 γ 的版本, 报 deviance 差与 LRT (γ 仍不显著则可继续用加法形式)
Xg = np.concatenate([one, NX[:, None], PS[:, None], (NX * PS)[:, None], F9s], axis=1)
bg, seg_, llg, _ = irls(Xg, Yerr)
gam, gam_se = bg[3], seg_[3]
D = 2 * (llg - ll)
print(f"  [监控] 带交互 γ = {gam:+.3f} ± {gam_se:.3f} (z={gam/gam_se:+.2f})  "
      f"deviance 差 D = {D:.3f} (df=1, p={2*norm.sf(abs(gam/gam_se)):.3f})  "
      f"→ {'加法形式维持 (γ 不显著)' if abs(gam/gam_se) < 1.96 else '⚠ γ 转显著, 需复审模型形式'}")
w_raw = b[3:] / sd9                       # 键对特征还原到原始尺度
base0 = b0 - (b[3:] * mu9 / sd9).sum()    # 角点类 c0 截距

# 各段类校准 (饱和类哑变量 → 聚合 O/E 应为 1)
print("\n段类校准 (观察错误 / 模型期望):")
for c_ in range(5):
    m = (CL == c_)
    o = float(Yerr[m].sum()); e = float(p_err_hat[m].sum())
    print(f"  c{c_} {CLS_NAME[c_]:14s} n={int(m.sum()):6d}  obs={int(o):4d}  exp={e:7.1f}  O/E = {o/e:.2f}")

# 公式验证 (还原系数 vs 模型输出, 每类各取一对)
def perr_formula(a, b_, c_):
    n_, s_ = CLS_ATTR[c_][1], CLS_ATTR[c_][2] - 1
    logit = base0 + beta*n_ + delta*s_ \
        + (w_raw * np.array([f(a, b_) for f in FEATS_F])).sum()
    return 1 / (1 + np.exp(-logit))
print("\n公式验证 (还原系数公式 vs 标准化空间模型):")
for ab_, c_ in [("ab", 0), ("fg", 1), ("aa", 2), ("sz", 3), ("sz", 4)]:
    a_, b__ = ab_[0], ab_[1]
    n_, s_ = CLS_ATTR[c_][1], CLS_ATTR[c_][2] - 1
    xs = np.concatenate([[1.0, n_, s_],
                         (np.array([f(a_, b__) for f in FEATS_F]) - mu9) / sd9])
    pm = 1 / (1 + np.exp(-(b @ xs)))
    pf = perr_formula(a_, b__, c_)
    print(f"  {ab_} c{c_}: 公式 P_err={pf:.4f}  模型={pm:.4f}  ({'✓' if abs(pf-pm)<1e-6 else '✗'})")

# ── 4. 导出: 900 键对 × 5 段类 ───────────────────────
print("\n=== 导出 键对错误率表.txt (900 键对 × 5 段类) ===")
pairs = [(a, b) for a in LETTERS for b in LETTERS]
Xp9 = np.array([[f(a, b) for f in FEATS_F] for a, b in pairs], dtype=float)
base900 = base0 + Xp9 @ w_raw
sig = lambda z: 1 / (1 + np.exp(-z))
# 五类电平 (相对 c0 角点): c1=+β, c2=+δ, c3=+β+δ, c4=+2δ   (加法形式, 2026-09-21 去 γ)
LV = np.array([0.0, beta, delta, beta + delta, 2*delta])
P5 = np.stack([sig(base900 + lv) for lv in LV])            # (5, 900)
COLS = ["err_p0n0s1", "err_p0n1s1", "err_p1n0s2", "err_p1n1s2", "err_p1n0s3"]
(_DIR / "产物").mkdir(exist_ok=True)
with open(_DIR / "产物" / "键对错误率表.txt", "w", encoding="utf-8") as f:
    f.write("pair\t" + "\t".join(COLS) + "\n")
    for k_, (a, b) in enumerate(pairs):
        f.write(f"{a}{b}\t" + "\t".join(f"{P5[c_][k_]:.4f}" for c_ in range(5)) + "\n")
print("  已导出 键对错误率表.txt (列 = 段类: p=有前键, n=有后键, s=段位)")
print("  类均值 P_err: " + "  ".join(f"c{c_} {P5[c_].mean()*100:.2f}%" for c_ in range(5)))
try:
    _ec = None
    for l in (_DIR / "产物" / "错误修正时间-实测值.txt").read_text(encoding="utf-8").splitlines():
        if l.startswith("cost_ms\t"):
            _ec = float(l.split("\t")[1]); break
    print(f"  平均错误成本 (×实测 {_ec:.0f}ms, 来源 产物/错误修正时间-实测值.txt): "
          f"T₂ = {_ec*P5[0].mean():.2f}ms | T₃ = {_ec*(P5[1]+P5[2]).mean():.2f}ms | "
          f"T₄ = {_ec*(P5[1]+P5[3]+P5[4]).mean():.2f}ms")
except Exception:
    print("  平均错误成本: 未找到 产物/错误修正时间-实测值.txt (先跑 分析-错误修正时间.py)")
k = max(range(900), key=lambda i: P5[4][i])
print(f"  极值键对: {pairs[k][0]}{pairs[k][1]} 角点 {P5[0][k]*100:.1f}% / 短尾 {P5[2][k]*100:.1f}% / "
      f"中段 {P5[3][k]*100:.1f}% / 长尾 {P5[4][k]*100:.1f}%")

# ── 5. README §5.5 实证: 签名 vs 类盲 (2026-09-18 自 实验-README数字刷新.py 并入) ──
# 类盲 = 同特征集但无签名项的逻辑回归; 差异 = 位置信息缺失对 T₂ 角点条目的系统性高估
_ec2 = None
try:
    for l in (_DIR / "产物" / "错误修正时间-实测值.txt").read_text(encoding="utf-8").splitlines():
        if l.startswith("cost_ms\t"):
            _ec2 = float(l.split("\t")[1]); break
except OSError:
    pass
if _ec2:
    b_bl, _, _, _ = irls(np.concatenate([one, F9s], axis=1), Yerr)
    Xp9s = (Xp9 - mu9) / sd9
    P_bl = sig(b_bl[0] + Xp9s @ b_bl[1:])
    print("\n=== README §5.5: 签名 vs 类盲 (类盲=无签名对照; 偏差与 P_err 成正比, 扭曲 2 键简码排序) ===")
    print(f"  角点类均值: 签名 {P5[0].mean()*100:.2f}% vs 类盲 {P_bl.mean()*100:.2f}%  "
          f"→ 高估 {_ec2*(P_bl.mean()-P5[0].mean()):.1f}ms/条目")
    hi = np.argsort(P5[0])[-90:]
    print(f"  高错误对 (角点前 10%): 签名 {P5[0][hi].mean()*100:.2f}% vs 类盲 {P_bl[hi].mean()*100:.2f}%  "
          f"→ 高估 {_ec2*(P_bl[hi].mean()-P5[0][hi].mean()):.1f}ms/条目")
    print(f"  平均错误成本: 类盲 T₂ {_ec2*P_bl.mean():.1f} / T₃ {2*_ec2*P_bl.mean():.1f} / T₄ {3*_ec2*P_bl.mean():.1f}"
          f"   签名 T₂ {_ec2*P5[0].mean():.1f} / T₃ {_ec2*(P5[1]+P5[2]).mean():.1f} / "
          f"T₄ {_ec2*(P5[1]+P5[3]+P5[4]).mean():.1f}")
else:
    print("\n(未找到 产物/错误修正时间-实测值.txt, 跳过 §5.5 类盲对比)")
