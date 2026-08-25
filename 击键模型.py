#!/usr/bin/env python3
"""
条件击键当量模型（条件段模型）
  研究阶段（默认）: 训练段模型 + 报告总时间 MAE/R²
  完成阶段 --full: 追加导出段当量母表 ((len(LETTERS)+1)×len(LETTERS)², 其他当量由后续模块查表组合)

架构: 每段 t = 双线性交互(e_p,e_a,e_b) + MLP([e_p;e_a;e_b;φ(a,b)])  (前键为空的段 = 两键当量)
  前键空 = 索引 30 (可学习的"静止起始"嵌入)
  两键当量  T₂(ab)       = S(∅, a, b)
  三键当量  T₃(abc)      = S(∅,a,b) + S(a,b,c)
  四键当量  T₄(abcd)     = S(∅,a,b) + S(a,b,c) + S(b,c,d)
  共享 MLP, 全部 3 段样本训练 (每 trial 3 段)

键位嵌入: e_k = E[k] ∈ ℝ²⁰  (30 键各独立, 索引 30 = 空前键)
交互特征: φ(a,b) 8 维 = [同手, 同指, 同键, 列距, 行距, Fitts, 镜像手指, 跨行同指] (2026-08-11 文献扩展)

依据 (实验-特征与目标.py / 实验-分段模型.py):
  - φ 已饱和: 加列差/方向特征均无增益 (实验-方向特征.py)
  - 段间强交互: 相加模型 MAE 120ms vs 条件段模型 90.5ms
  - 前键条件决定性: 无前键 R² = -0.94
"""
import argparse, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

# 运行环境固定 (可复现性, 2026-08-18 实测):
# ① 单线程 — 模型仅 6.4k 参数/批 256, 多线程开销 > 收益 (单线程 12s vs 多线程 15s/次);
# ② 导入时播种 torch 默认 RNG — 新进程默认 RNG 是熵播种的, 模型构造 (嵌入/线性层
#   初始化) 依赖它, 而 train(seed) 只重播轨迹不重播构造 → 历史上同命令重跑结果漂移
#   ("±1.4ms 种子波动"的主要来源)。播种后同代码+同数据跨进程逐位可复现;
#   best-of-N 的多次构造沿确定性流推进, 初始化仍互不相同, 多样性不受影响。
torch.set_num_threads(1)
torch.manual_seed(0)

# ═══════════════════ 键盘布局 ═══════════════════
# 3 行 10 列完整 QWERTY: 30 键 (26 字母 + ; , . /)
LETTER_TO_COL = {
    'q':0,'w':1,'e':2,'r':3,'t':4,'y':5,'u':6,'i':7,'o':8,'p':9,
    'a':0,'s':1,'d':2,'f':3,'g':4,'h':5,'j':6,'k':7,'l':8,';':9,
    'z':0,'x':1,'c':2,'v':3,'b':4,'n':5,'m':6,',':7,'.':8,'/':9,
}
LETTER_TO_ROW = {
    'q':0,'w':0,'e':0,'r':0,'t':0,'y':0,'u':0,'i':0,'o':0,'p':0,
    'a':1,'s':1,'d':1,'f':1,'g':1,'h':1,'j':1,'k':1,'l':1,';':1,
    'z':2,'x':2,'c':2,'v':2,'b':2,'n':2,'m':2,',':2,'.':2,'/':2,
}
COL_TO_FINGER = [0,1,2,3,3,4,4,5,6,7]
COL_TO_HAND   = [0,0,0,0,0,1,1,1,1,1]
LETTERS = 'abcdefghijklmnopqrstuvwxyz;,./'
KEY_TO_IDX = {c:i for i,c in enumerate(LETTERS)}
EMPTY = 30  # 空前键索引 (静止起始)

def _precompute_phi():
    """φ(a,b) 8 维 — 2026-08-11 文献特征扩展 (实验-特征扩展.py):
    [同手, 同指, 同键, 列距, 行距, Fitts, 镜像手指, 跨行同指]
    文献依据: İşeri & Ekşioğlu 2015 (列>行>手 重要性排序, 显式几何有独立信息);
    Fitts (MT = a + b·log2(1+D/W), W=键宽, 行距≈2 键宽); keygen (同指跨行最慢);
    Grudin 1983 (镜像手指换位). 消融两轮:
    - 逐特征 LOO 全部持平 (实验-特征消融.py, 2026-08-11, Δ<0.2ms) — φ 特征间互为冗余
    - 整体删除 +1.7ms (实验-结构消融.py, 2026-08-14, 无泄漏 trial 测试口径) — φ 作为整体
      提供嵌入补不上的几何先验, 价值在稀疏三元组外推 (2/3 段表条目零样本); LOO 在
      训练分布内做所以测不出。保留 φ8 有实证支撑 (见 README §6.4)"""
    feats = {}
    for a in LETTERS:
        ca,ra = LETTER_TO_COL[a], LETTER_TO_ROW[a]
        for b in LETTERS:
            cb,rb = LETTER_TO_COL[b], LETTER_TO_ROW[b]
            fa,fb = COL_TO_FINGER[ca], COL_TO_FINGER[cb]
            same_hand = 1.0 if COL_TO_HAND[ca]==COL_TO_HAND[cb] else 0.0
            same_fing = 1.0 if fa==fb else 0.0
            same_key = 1.0 if a==b else 0.0
            col_d = float(abs(ca-cb))
            row_d = float(abs(ra-rb))
            fitts = float(np.log2(1 + np.sqrt(col_d**2 + (2*row_d)**2)))
            mirror = 1.0 if (not same_hand) and fa+fb==7 else 0.0
            crsf = 1.0 if same_hand and same_fing and ra!=rb else 0.0
            feats[(a,b)] = (same_hand, same_fing, same_key, col_d, row_d,
                            fitts, mirror, crsf)
    return feats
PHI = _precompute_phi()

# ═══════════════════ 模型 ═══════════════════

class RMSNorm(nn.Module):
    """RMSNorm: x / RMS(x) × g (LLaMA 风格, 无均值中心化)"""
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.g * x.shape[-1] ** 0.5

class _SegModel(nn.Module):
    """段模型公共接口: 单段/整串预测 + 模型 IO, 子类只需实现 _batch(ids, ph)。
    (KeystrokeModel / CPModel / 结构消融变体共用; 基类不含参数, 不影响 state_dict 键名)"""
    @property
    def _dev(self): return next(self.mlp.parameters()).device
    def seg(self, prev, a, b):
        """单段预测 (字符串输入, prev=None 表空前键)"""
        ids = torch.tensor([[EMPTY if prev is None else KEY_TO_IDX[prev],
                             KEY_TO_IDX[a], KEY_TO_IDX[b]]])
        ph = torch.tensor([PHI[(a,b)]])
        return self._batch(ids, ph).item()
    def tri_total(self, abc):
        a,b,c = abc
        return self.seg(None,a,b) + self.seg(a,b,c)
    def total(self, code):
        a,b,c,d = code
        return self.seg(None,a,b) + self.seg(a,b,c) + self.seg(b,c,d)
    def nparam(self): return sum(p.numel() for p in self.parameters())
    def save(self, path): torch.save(self.state_dict(), path)
    def load(self, path): self.load_state_dict(torch.load(path, weights_only=True))

class KeystrokeModel(_SegModel):
    """条件段模型: 键位嵌入 (20 维) + φ(8 特征) + 双线性交互 + MLP 头。
    算法对比 (实验-双线性扩容.py, 5 seed 验证集): de20/dh64 单层
    验证MAE 61.7/R² 0.395 反超 CP rank=32 (64.3/0.343) → 成为默认;
    继续加大 (de24/96 单层 65.3 过拟合, de32/128 双层 62.8, de48/192 三层 60.8 方差大)
    无稳定收益 → 默认 de20/dh64。
    激活/归一化 (实验-激活函数.py, 5 seed): relu 最优 (leaky/sigmoid/relu²/silu/swiglu
    全灭); 隐藏层后 RMSNorm 微增益 59.4/0.401 vs 61.7/0.395 (噪声边缘, 不恶化) → 落地。
    特征扩展 (实验-特征扩展.py/实验-特征消融.py, 2026-08-11): φ 2→8 维
    (文献几何/手指特征), 粗对比 38.4→37.4 似有增益, 但逐特征 LOO 全部持平
    (Δ<0.2ms)。结构消融 (实验-结构消融.py, 2026-08-14): 整体删 φ +1.7ms —
    φ 作为整体的价值在稀疏三元组外推, 保留有实证支撑 (README §6.4)。
    结构消融同轮: W3 跨键双线性 +2.1ms (最值钱组件), MLP 头仅 +0.5ms,
    huber/l1 无收益 — 现有元素全部保留, 结构已收敛 (残差/固有噪声 = 0.91)。"""
    def __init__(self, d_embed=20, d_hidden=64):
        super().__init__()
        self.E_key = nn.Embedding(len(LETTERS) + 1, d_embed)  # 30 键 + 空前键
        self.W = nn.Parameter(torch.randn(3, d_embed, d_embed) * 0.05)  # 3 对双线性交互
        self.mlp = nn.Sequential(nn.Linear(d_embed*3+8, d_hidden), nn.ReLU(),
                                 RMSNorm(d_hidden), nn.Linear(d_hidden, 1))
    @property
    def _dev(self): return next(self.mlp.parameters()).device
    def _batch(self, ids, ph):
        """ids: (B,3) [prev,a,b], ph: (B,3)"""
        B = ids.shape[0]
        ids = ids.to(self._dev)
        e = self.E_key(ids)
        # 显式二阶交互: (前键,首键), (首键,次键), (前键,次键)
        bil = (torch.einsum('bi,ij,bj->b', e[:,0], self.W[0], e[:,1]) +
               torch.einsum('bi,ij,bj->b', e[:,1], self.W[1], e[:,2]) +
               torch.einsum('bi,ij,bj->b', e[:,0], self.W[2], e[:,2]))
        mlp = self.mlp(torch.cat([e.reshape(B, -1), ph.to(self._dev).reshape(B, -1)], dim=1)).squeeze(-1)
        return bil + mlp

class CPModel(_SegModel):
    """CP 张量分解: S(p,a,b) ≈ Σ_r U(p)·V(a)·W(b) + MLP 头 (三向联合交互)。
    备选架构 (--model cp): 双线性扩容后验证集 R² 0.395 vs CP 0.343 被反超
    (实验-双线性扩容.py, 5 seed); 保留作对照。rank 扫描 8→32 提升、48 回落。"""
    def __init__(self, rank=32, d_embed=12, d_hidden=24):  # rank 扫描: 32 最优 (66.1/0.402), 48 回落
        super().__init__()
        self.U = nn.Embedding(len(LETTERS)+1, rank)   # 前键因子
        self.V = nn.Embedding(len(LETTERS)+1, rank)   # 首键因子
        self.W = nn.Embedding(len(LETTERS)+1, rank)   # 次键因子
        self.E_key = nn.Embedding(len(LETTERS)+1, d_embed)
        self.mlp = nn.Sequential(nn.Linear(d_embed*3+8, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))
    @property
    def _dev(self): return next(self.mlp.parameters()).device
    def _batch(self, ids, ph):
        """ids: (B,3) [prev,a,b], ph: (B,3)"""
        B = ids.shape[0]
        ids = ids.to(self._dev)
        cp = (self.U(ids[:,0]) * self.V(ids[:,1]) * self.W(ids[:,2])).sum(1)
        e = self.E_key(ids)
        mlp = self.mlp(torch.cat([e.reshape(B, -1), ph.to(self._dev).reshape(B, -1)], dim=1)).squeeze(-1)
        return cp + mlp

# ═══════════════════ 数据 ═══════════════════

def load_data(path):
    """[(code, [b_d,c_d,d_d]), ...]  error=0, 仅格式/有效性校验 (异常剔除在段级)"""
    data = []
    with open(path, encoding="utf-8") as f:
        hdr = {h:i for i,h in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            row = line.strip().split("\t")
            if len(row)<11 or row[hdr.get("error",8)]!="0": continue
            code = row[hdr["code"]]
            if len(code)!=4 or not all(c in LETTERS for c in code): continue
            try:
                bd,cd,dd = float(row[hdr["b_d"]]),float(row[hdr["c_d"]]),float(row[hdr["d_d"]])
            except ValueError: continue
            if bd<=0 or cd<=bd or dd<=cd: continue
            data.append((code,[bd,cd,dd]))
    return data

def mad_filter_segments(data, tgt, k=3.0):
    """段级异常剔除 (MAD 自适应, 按段位置分桶; 单侧上围栏 — 异常只可能偏大):
      剔除 seg ≥ median + k·1.4826·MAD 的段 (注意力中断; 中位数稳健, 不被长尾污染;
      2026-08-20 起单侧化: 右偏分布下 MAD 下界恒为负, 实测下侧剔除恒 0,
      双侧公式与单侧数值等价, 单侧与"异常只偏大"语义对齐)。
      不做硬性上下限:
        - 快段受单侧语义保护 (实测最快常规击键 15-40ms, 硬下限会误删基准键对)
        - MAD 上界 (实测 265-412ms) 已严于任何固定上限, 500ms 硬顶从不额外触发
      返回 (keep 掩码, 三段齐全的 trial 子集)。实验对比: MAD k=3 总 MAE 89.6→73.4ms,
      R² 0.275→0.314 (实验-剔除对比.py), 优于固定阈值 (600/2000) 与 k=2/4/5。"""
    n = len(data)
    pos = np.tile([0,1,2], n)
    keep = np.ones(len(tgt), dtype=bool)
    for p in range(3):
        idx = np.where(pos==p)[0]
        vals = tgt[idx]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        sigma = 1.4826 * mad if mad > 0 else 1.0
        keep[idx] = vals < med + k * sigma
    tri_ok = keep.reshape(-1, 3).all(axis=1)
    data_full = [d for d, ok in zip(data, tri_ok) if ok]
    return keep, data_full

def residual_filter_segments(data, prev, a, b, ph, tgt, k=3.0, m0_seeds=5):
    """B4b 两阶段残差剔除 (2026-08-12 落地, 实验-剔除优化/诊断/修复/终局):
      1. B0 段位 MAD 剔除 → 训 M0 (m0_seeds 验证集选优) — 干净模型提供期望行为
      2. M0 推全数据 → 带符号残差 r = y-ŷ (正 = 比预期慢)
      3. r 按键对 (a,b) 分桶, 单侧上围栏剔除 r ≥ med + k·1.4826·MAD —
         键对系统性偏差中心化后只删"比该键对合理预期慢"的段 (真注意力中断);
         异常只可能偏大: rollover/预备连击 (比预期快) 是真实打字行为不删
         (2026-08-20 单侧化, 原 |残差| 双侧判据曾把 16%/35% 的删除错放快侧)。
         模型学不会的键对 (pq 等跨手小指难键对) 不被误删 — 全局残差法会把
         "比模型预测慢"的真实慢段全删, pq 当量 162→95ms 失真 (循环论证,
         实验-剔除过删检查.py E2); 键对分桶修复后当量恢复 (实验-剔除修复.py)。
      依据: 固定测试集终局对比各方案 44-47ms 在噪声内 (剔除收益在当量
      稳健性而非预测精度); 剔除率 ~5% 在文献范围 (5-10%)。
      注意: 最终 keep 仅由第 2 阶段在全数据上的键对分桶残差判定 (替换而非
      交集第 1 阶段 — keep0 只用于定义 M0 训练集; 温和离群段被键对内残差
      判定"复活"属设计行为, 属保守方向)。
      返回 (keep 掩码, 三段齐全 trial 子集)。"""
    keep0 = mad_filter_segments(data, tgt)[0]
    # M0: 段位 MAD 干净数据训练, 验证集选优 (与主训练同协议)
    best_va, M0 = float("inf"), None
    for s in range(m0_seeds):
        m = KeystrokeModel()
        _, va = train(m, prev[keep0], a[keep0], b[keep0], ph[keep0], tgt[keep0], seed=s)
        if va < best_va:
            best_va, M0 = va, m
    ids = torch.tensor(np.stack([prev, a, b], axis=1))
    with torch.no_grad():
        pred = M0._batch(ids, torch.tensor(ph)).numpy()
    resid = tgt - pred    # 带符号残差: 正 = 比模型预期慢 (中断), 负 = 比预期快 (rollover/预备)
    # 单侧上围栏剔除 (2026-08-20): 异常只可能偏大 — 注意力中断只拖慢不拖快,
    # 打得比预期快是 rollover/预备连击, 属真实打字行为不删 (双侧 |残差| 判据曾把
    # 16% (全量) / 35% (测试集) 的删除错放在快侧, 且小桶下围栏会误删"预测特别准"段)。
    # 按键对分桶中心化后只删 r > med + k·σ (保守方向: 边界段保留; n=1 桶恒保留)。
    key = a.astype(np.int64) * len(LETTERS) + b
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order]) != 0) + 1
    keep = np.ones(len(tgt), dtype=bool)
    for idx in np.split(order, bounds):
        v = resid[idx]
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        sigma = 1.4826 * mad if mad > 0 else 1.0
        keep[idx] = v < med + k * sigma
    tri_ok = keep.reshape(-1, 3).all(axis=1)
    data_full = [d for d, ok in zip(data, tri_ok) if ok]
    return keep, data_full

def build_tensors(data):
    """全部段样本 (每 trial 3 段): (prev, a, b, phi, target)"""
    segs = []
    for code, ts in data:
        a,b,c,d = code
        segs.append((EMPTY, a, b, PHI[(a,b)], ts[0]))
        segs.append((a, b, c, PHI[(b,c)], ts[1]-ts[0]))
        segs.append((b, c, d, PHI[(c,d)], ts[2]-ts[1]))
    prev = np.array([s[0] if isinstance(s[0], int) else KEY_TO_IDX[s[0]] for s in segs], dtype=np.int64)
    a = np.array([KEY_TO_IDX[s[1]] for s in segs], dtype=np.int64)
    b = np.array([KEY_TO_IDX[s[2]] for s in segs], dtype=np.int64)
    ph = np.array([s[3] for s in segs], dtype=np.float32)
    tgt = np.array([s[4] for s in segs], dtype=np.float32)
    return prev, a, b, ph, tgt

# ═══════════════════ 训练 ═══════════════════

def train(model, prev, a, b, ph, tgt, epochs=200, seed=0, bs=256, lr=0.001,
          patience=60, loss_fn="mse"):
    """批量训练 + 早停 (80/20 随机划分), 模型原地更新, 返回 (全数据段MAE, 验证集段MAE)。
    验证集 MAE 用于 best-of-N 选优 (选最强模型必须用验证集, 全数据含训练集会偏向过拟合)。
    loss_fn: mse (主流程) | huber50 | l1 (实验-结构消融.py 目标函数变体) — 规范实现,
    实验-结构消融.py 的 train_v 仅转发至此 (2026-08-18 合并双胞胎实现, 防口径漂移)。
    判据口径 (勿单侧更改): 早停/存档用验证集 MSE, best-of-N 选优用返回的验证集
    MAE — 两者错配为已知现状, 历史结论均在此口径下取得。"""
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(tgt); nv = int(n*.2)
    idx = np.random.permutation(n)
    tr_i, va_i = idx[nv:], idx[:nv]
    yt = torch.tensor(tgt, dtype=torch.float32)
    X = torch.stack([torch.tensor(prev), torch.tensor(a), torch.tensor(b)], dim=1)
    PH = torch.tensor(ph)
    X_tr, X_va = X[tr_i], X[va_i]
    PH_tr, PH_va = PH[tr_i], PH[va_i]
    yt_tr, yt_va = yt[tr_i], yt[va_i]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_va, best_st, pat = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_i))
        for s in range(0, len(perm), bs):
            sel = perm[s:s+bs]
            pred = model._batch(X_tr[sel], PH_tr[sel])
            if loss_fn == "mse":
                loss = ((pred - yt_tr[sel])**2).mean()
            elif loss_fn == "huber50":
                loss = F.huber_loss(pred, yt_tr[sel], delta=50.0)
            else:  # l1
                loss = (pred - yt_tr[sel]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            va = ((model._batch(X_va, PH_va) - yt_va)**2).mean().item()
        if va < best_va:
            best_va = va; best_st = {k: v.clone() for k, v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
        if pat >= patience: break

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        pred = model._batch(X, PH).numpy()
        va_pred = model._batch(X_va, PH_va).numpy()
    return (float(np.mean(np.abs(pred - tgt))),
            float(np.mean(np.abs(va_pred - yt_va.numpy()))))

def eval_seg(model, prev, a, b, ph, tgt):
    """段 MAE (任意数据, 一次批量前向)"""
    model.eval()
    with torch.no_grad():
        pred = model._batch(torch.tensor(np.stack([prev, a, b], axis=1)),
                            torch.tensor(ph)).numpy()
    return float(np.mean(np.abs(pred - tgt)))

def total_preds(model, data):
    """四键总时间预测 (ms, float64) — 三段一次批量前向, eval_total/对比脚本共用"""
    ids, phs = [], []
    for (x, y, z, w), _ in data:
        ids += [[EMPTY, KEY_TO_IDX[x], KEY_TO_IDX[y]],
                [KEY_TO_IDX[x], KEY_TO_IDX[y], KEY_TO_IDX[z]],
                [KEY_TO_IDX[y], KEY_TO_IDX[z], KEY_TO_IDX[w]]]
        phs += [PHI[(x, y)], PHI[(y, z)], PHI[(z, w)]]
    model.eval()
    with torch.no_grad():
        pred = model._batch(torch.tensor(ids), torch.tensor(phs)).numpy()
    return pred.reshape(-1, 3).astype(np.float64).sum(axis=1)

def eval_total(model, data):
    """四键总时间 MAE/R² (全数据)。数值与逐 trial 逐段求和等价
    (2026-08-18 批量化: 4539 trial 8.4s → ~0.01s)"""
    if not data:
        return float("nan"), float("nan")
    ys = np.array([ts[2] for _, ts in data])
    errs = np.abs(total_preds(model, data) - ys)
    mae = float(errs.mean())
    r2 = 1 - float(np.sum(errs**2)) / max(float(np.sum((ys-ys.mean())**2)), 1e-9)
    return mae, r2

# ═══════════════════ 导出 ═══════════════════

def build_seg_table(model):
    """段表 F[p,a,b] = S(p,a,b) ms, 形状 (n,n,n)。n³ 一次批量前向 (n=键数)。"""
    n = len(LETTERS)
    tri = [(p,a,b) for p in LETTERS for a in LETTERS for b in LETTERS]
    ids = torch.tensor([[KEY_TO_IDX[p], KEY_TO_IDX[a], KEY_TO_IDX[b]] for p,a,b in tri])
    ph = torch.tensor([PHI[(a,b)] for _,a,b in tri])
    with torch.no_grad():
        return model._batch(ids, ph).numpy().reshape(n, n, n)

def _bigram_ms(model):
    """空前键 B[a,b] = S(∅,a,b) ms (未归一化), 形状 (30,30)"""
    n = len(LETTERS)
    ids = torch.tensor([[EMPTY, KEY_TO_IDX[a], KEY_TO_IDX[b]]
                        for a in LETTERS for b in LETTERS])
    ph = torch.tensor([PHI[(a,b)] for a in LETTERS for b in LETTERS])
    with torch.no_grad():
        return model._batch(ids, ph).numpy().reshape(n, n)

def export_seg_table(model, out_path):
    """(len(LETTERS)+1)×len(LETTERS)×len(LETTERS) 段当量母表: S(p,a,b) ms 原始值 (不归一化)。
    p='_' 表空前键 (静止起始)。其他当量由后续模块查表组合, 避免冗余存储:
      T₂(ab)   = S[_,a,b]
      T₃(abc)  = S[_,a,b] + S[a,b,c]
      T₄(abcd) = S[_,a,b] + S[a,b,c] + S[b,c,d]
      任意前缀续打: 已打 p, 续打 xyz = S[p,x,y] + S[x,y,z]
    """
    n = len(LETTERS)
    B = _bigram_ms(model)          # ∅ 切片 30×30
    F = build_seg_table(model)     # 字母前键 26³
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 段当量母表 S(p,a,b): 段时间 ms 原始值 (未归一化, 可除以最小值归一化)\n")
        f.write("# p=前键, _=空前键(静止起始); 组合: T_n(k1..kn) = S[_,k1,k2] + Σ S[ki-1,ki,ki+1]\n")
        f.write("p\ta\tb\tms\n")
        buf = []
        for i, a in enumerate(LETTERS):
            for j, b in enumerate(LETTERS):
                buf.append(f"_\t{a}\t{b}\t{max(0,B[i,j]):.1f}\n")
        for pi, p in enumerate(LETTERS):
            for i, a in enumerate(LETTERS):
                for j, b in enumerate(LETTERS):
                    buf.append(f"{p}\t{a}\t{b}\t{max(0,F[pi,i,j]):.1f}\n")
        f.writelines(buf)
    print(f"  当量-段表: {out_path}  ({(len(LETTERS)+1)*n*n:,} 条)")

# ═══════════════════ 主流程 ═══════════════════

def main():
    ap = argparse.ArgumentParser(description="条件击键当量模型: 训练 + 测试集四宫格评估 + 可选导出")
    ap.add_argument("--full", action="store_true",
                    help="追加部署模型训练 (全数据) + 导出段当量表与模型")
    ap.add_argument("--model", choices=["bi", "cp"], default="bi",
                    help="bi=双线性 de20/dh64 验证集误差最小 (默认) | cp=CP 张量分解 rank=32")
    ap.add_argument("--trials", type=int, default=10, help="best-of-N 训练次数")
    args = ap.parse_args()
    full, model_name, trials = args.full, args.model, args.trials
    PROJ = Path(__file__).resolve().parent
    DATA = PROJ / "击键测速数据.tsv"
    if not DATA.exists(): print(f"无数据: {DATA}"); sys.exit(1)

    print(f"加载: {DATA}")
    data = load_data(str(DATA))
    print(f"4 键样本: {len(data)}, 段样本: {len(data)*3}")

    # ── 主口径划分: 固定 trial 测试集 (2026-08-19 起统一口径; 与 4 个实验脚本同协议) ──
    rng = np.random.RandomState(2024)
    idx = rng.permutation(len(data))
    nt = int(len(data) * .2)
    test_idx = set(idx[:nt])
    train_data = [d for i, d in enumerate(data) if i not in test_idx]
    test_data = [d for i, d in enumerate(data) if i in test_idx]
    print(f"划分版本: N={len(data)} trial, 测试集 {len(test_data)} ({len(test_data)*100/len(data):.0f}%), 种子 2024")
    print(f"训练 trial {len(train_data)} / 测试 trial {len(test_data)}")

    # ── 评估模型: 训练集 B4b + best-of-N (测试 trial 的段不进训练/验证, 无泄漏) ──
    prev, a, b, ph, tgt = build_tensors(train_data)
    keep, _ = residual_filter_segments(train_data, prev, a, b, ph, tgt)
    print(f"训练集 B4b 剔除 (键对分桶带符号残差, 单侧上围栏 k=3): 保留 {int(keep.sum())}/{len(tgt)} 段")
    prev, a, b, ph, tgt = prev[keep], a[keep], b[keep], ph[keep], tgt[keep]

    model_name_full = {"bi": "双线性 de20/dh64", "cp": "CP rank=32"}.get(model_name, model_name)
    print(f"\n=== 评估模型训练 ({model_name_full}, best-of-{trials}, 验证集选优) ===")
    best_va, best_m = float("inf"), None
    for t in range(trials):
        m = CPModel() if model_name == "cp" else KeystrokeModel()
        seg_mae, va_mae = train(m, prev, a, b, ph, tgt, seed=t)
        star = ""
        if va_mae < best_va:
            best_va, best_m = va_mae, m
            star = "  ← 新最优"
        print(f"  trial {t+1:2d}/{trials}: 段MAE={seg_mae:5.1f}ms  验证段MAE={va_mae:5.1f}ms{star}")

    # 测试集独立 B4b (保留口径; 内部自训 M0, 与训练集剔除无信息交换)
    tp, ta, tb, tph, tt = build_tensors(test_data)
    tkeep, test_full = residual_filter_segments(test_data, tp, ta, tb, tph, tt)
    print(f"测试集 B4b: 保留 {int(tkeep.sum())}/{len(tt)} 段, 完整 trial {len(test_full)}/{len(test_data)}")

    # ── 主指标: 保留口径 (2026-08-26 起默认只展示保留口径, 全口径需时另行说明) ──
    seg_keep = eval_seg(best_m, tp[tkeep], ta[tkeep], tb[tkeep], tph[tkeep], tt[tkeep])
    tot_keep, r2_keep = eval_total(best_m, test_full)
    print(f"\n=== 主指标 (固定 trial 测试集 {len(test_data)}, 保留口径, 参数 {best_m.nparam()}, 验证段MAE={best_va:.1f}) ===")
    print(f"保留口径: 段MAE={seg_keep:5.1f}  总MAE={tot_keep:5.1f}  R²={r2_keep:+.3f}")

    if not full:
        print("\n研究模式: 仅训练评估。加 --full 追加部署模型训练 (全数据) + 导出当量表。")
        return

    # ── 部署模型: 全数据 B4b + best-of-N (当量表条目质量优先; 评估指标以上方四宫格为准) ──
    print(f"\n=== 部署模型 (全数据训练, {trials} trials — 导出用, 指标见上方评估模型) ===")
    dprev, da, db, dph, dtgt = build_tensors(data)
    dkeep, _ = residual_filter_segments(data, dprev, da, db, dph, dtgt)
    print(f"全数据 B4b: 保留 {int(dkeep.sum())}/{len(dtgt)} 段")
    dprev, da, db, dph, dtgt = dprev[dkeep], da[dkeep], db[dkeep], dph[dkeep], dtgt[dkeep]
    dbest_va, deploy_m = float("inf"), None
    for t in range(trials):
        m = CPModel() if model_name == "cp" else KeystrokeModel()
        seg_mae, va_mae = train(m, dprev, da, db, dph, dtgt, seed=t)
        star = ""
        if va_mae < dbest_va:
            dbest_va, deploy_m = va_mae, m
            star = "  ← 新最优"
        print(f"  trial {t+1:2d}/{trials}: 段MAE={seg_mae:5.1f}ms  验证段MAE={va_mae:5.1f}ms{star}")

    print("\n=== 导出 ===")
    deploy_m.save(str(PROJ/"keystroke_model.pt"))
    print(f"  模型 → {PROJ/'keystroke_model.pt'}")
    export_seg_table(deploy_m, str(PROJ/"当量-段表.txt"))


if __name__ == "__main__":
    main()
