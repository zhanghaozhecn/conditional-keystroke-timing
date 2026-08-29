#!/usr/bin/env python3
"""
条件击键当量模型（对称段模型 v2, sym_phi）
  研究阶段（默认）: 训练段模型 + 报告总时间 MAE/R²
  完成阶段 --full: 追加导出部署模型 (全数据) 与 4-D 段当量母表 (npz)

段定义: t = S(p,a,b,n) — 前两键 p,a + 当前键对 a,b + **后继键 n** (2026-08-29 落地,
实验-后键条件系列 → README §6.6)。n=∅(索引 30) 表无后继 (词末"甩出")。
架构: 双线性交互(e_p,e_a,e_b) 3 项 + MLP([e_p;e_a;e_b;φ15])
  φ15 = φ8(a,b) 几何 + φsuc7 后键特征 [存在, 同手(b,n), 同指(b,n), 同键(b,n),
  Fitts(b,n), 同手(a,n), Fitts(a,n)]; n=∅ 时 φsuc 全 0 (存在位=0)。
  sym_phi 形式: n 仅经手工特征进入 (实验证明几何身份≈嵌入身份, 5 双线性无增益),
  无未训练嵌入角点问题; 模型结构相对 v1 仅 MLP 输入 68→75 (+448 参数)。

查询公式 (词内语义, 全部查询点有训练分布覆盖, T₂ 角点 (p=∅,n=∅) 除外—2 键试次验证中):
  两键当量  T₂(ab)   = S(∅,a,b,∅)
  三键当量  T₃(abc)  = S(∅,a,b,c) + S(a,b,c,∅)
  四键当量  T₄(abcd) = S(∅,a,b,c) + S(a,b,c,d) + S(b,c,d,∅)

键位嵌入: e_k = E[k] ∈ ℝ²⁰ (30 键各独立, 索引 30 = ∅ 前键/后键)
导出: 当量-段表.npz — F[p,a,b,n] 形状 (31,30,30,31), p/n 维 0-29=键、30=∅,
  附 letters/empty/version 元数据; 组装当量表.py / 组装-chai当量表.py 查表组合。
"""
import argparse, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

# 运行环境固定 (可复现性, 2026-08-18 实测):
# ① 单线程 — 模型仅 ~6.8k 参数/批 256, 多线程开销 > 收益;
# ② 导入时播种 torch 默认 RNG — 新进程默认 RNG 是熵播种的, 模型构造 (嵌入/线性层
#   初始化) 依赖它, 而 train(seed) 只重播轨迹不重播构造 → 历史上同命令重跑结果漂移
#   ("±1.4ms 种子波动"的主要来源)。播种后同代码+同数据跨进程逐位可复现;
#   best-of-N 的多次构造沿确定性流推进, 初始化仍互不相同, 多样性不受影响。
# 注意: 模型构造消耗全局 RNG — 跨脚本绝对数字不可比 (构造时机不同 → 初始化不同),
# 变体对比须同脚本同流内进行 (08-27 实验教训)。
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
EMPTY = 30  # ∅ 索引 (空前键 / 无后继键, 可学习嵌入)

def _precompute_phi():
    """φ(a,b) 8 维 — 2026-08-11 文献特征扩展 (实验-特征扩展.py):
    [同手, 同指, 同键, 列距, 行距, Fitts, 镜像手指, 跨行同指]
    文献依据: İşeri & Ekşioğlu 2015 (列>行>手 重要性排序, 显式几何有独立信息);
    Fitts (MT = a + b·log2(1+D/W), W=键宽, 行距≈2 键宽); keygen (同指跨行最慢);
    Grudin 1983 (镜像手指换位). 消融两轮:
    - 逐特征 LOO 全部持平 (实验-特征消融.py, 2026-08-11, Δ<0.2ms) — φ 特征间互为冗余
    - 整体删除 +1.7ms (实验-结构消融.py, 2026-08-14, 无泄漏 trial 测试口径) — φ 作为整体
      提供嵌入补不上的几何先验, 价值在稀疏三元组外推 (2/3 段表条目零样本); LOO 在
      训练分布内做所以测不出。保留 φ 有实证支撑 (见 README §6.4)"""
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

# ── 后键特征 LUT (整数索引, 对称矩阵 30×30) ──
_COL = np.array([LETTER_TO_COL[c] for c in LETTERS])
_ROW = np.array([LETTER_TO_ROW[c] for c in LETTERS])
_FING = np.array([COL_TO_FINGER[c] for c in _COL])
_HAND = np.array([COL_TO_HAND[c] for c in _COL])
_SUC_FITTS = np.log2(1 + np.sqrt((_COL[:,None]-_COL[None,:])**2 +
                                 (2*np.abs(_ROW[:,None]-_ROW[None,:]))**2))  # [i,j]

def phi_suc(a, b, n):
    """φsuc 7 维: [存在, 同手(b,n), 同指(b,n), 同键(b,n), Fitts(b,n), 同手(a,n), Fitts(a,n)]
    a,b,n: 键索引数组 (b/a 恒为真实键 0-29; n 可为 EMPTY=30 → 全 0)"""
    a = np.asarray(a); b = np.asarray(b); n = np.asarray(n)
    real = n != EMPTY
    nc = np.where(real, n, 0)
    out = np.zeros((len(a), 7), dtype=np.float32)
    out[real, 0] = 1.0
    out[real, 1] = (_HAND[b] == _HAND[nc])[real]
    out[real, 2] = (_FING[b] == _FING[nc])[real]
    out[real, 3] = (b == nc)[real]
    out[real, 4] = _SUC_FITTS[b, nc][real]
    out[real, 5] = (_HAND[a] == _HAND[nc])[real]
    out[real, 6] = _SUC_FITTS[a, nc][real]
    return out

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
    ids 约定: (B,4) [prev, a, b, n]。基类不含参数, 不影响 state_dict 键名。"""
    @property
    def _dev(self): return next(self.mlp.parameters()).device
    def seg(self, prev, a, b, nxt=None):
        """单段预测 (字符串输入; prev=None 表空前键, nxt=None 表无后继=词末)"""
        ids = torch.tensor([[EMPTY if prev is None else KEY_TO_IDX[prev],
                             KEY_TO_IDX[a], KEY_TO_IDX[b],
                             EMPTY if nxt is None else KEY_TO_IDX[nxt]]])
        ph = torch.tensor([np.concatenate([PHI[(a,b)], phi_suc(
            [KEY_TO_IDX[a]], [KEY_TO_IDX[b]],
            [EMPTY if nxt is None else KEY_TO_IDX[nxt]])[0]])], dtype=torch.float32)
        return self._batch(ids, ph).item()
    def tri_total(self, abc):
        a,b,c = abc
        return self.seg(None,a,b,c) + self.seg(a,b,c)
    def total(self, code):
        a,b,c,d = code
        return self.seg(None,a,b,c) + self.seg(a,b,c,d) + self.seg(b,c,d)
    def nparam(self): return sum(p.numel() for p in self.parameters())
    def save(self, path): torch.save(self.state_dict(), path)
    def load(self, path): self.load_state_dict(torch.load(path, weights_only=True))

class KeystrokeModel(_SegModel):
    """对称段模型 (sym_phi): 键位嵌入 (20 维) + φ15 + 双线性交互 3 项 + MLP 头。
    v1 → v2 (2026-08-29): 增后键条件 — MLP 输入 68→75 (+448 参数), 结构其余不变;
    依据 实验-后键条件系列 (README §6.6): 对称化总MAE −1.3~−1.6 (两组独立 best-of-10
    方向一致), 分解 = 存在位(段位通道) −0.4~−1.0 + 后键几何身份 −1.0;
    n 仅经 φsuc 特征进入 (几何身份≈嵌入身份, 5 双线性无增益)。
    v1 依据 (实验-双线性扩容.py 等): de20/dh64 单层最优; relu 最优; RMSNorm 微增益;
    W3 跨键双线性 +2.1ms (最值钱组件); 结构已收敛 (残差/固有噪声 = 0.91)。"""
    def __init__(self, d_embed=20, d_hidden=64):
        super().__init__()
        self.E_key = nn.Embedding(len(LETTERS) + 1, d_embed)  # 30 键 + ∅
        self.W = nn.Parameter(torch.randn(3, d_embed, d_embed) * 0.05)  # (p,a),(a,b),(p,b)
        self.mlp = nn.Sequential(nn.Linear(d_embed*3+15, d_hidden), nn.ReLU(),
                                 RMSNorm(d_hidden), nn.Linear(d_hidden, 1))
    @property
    def _dev(self): return next(self.mlp.parameters()).device
    def _batch(self, ids, ph):
        """ids: (B,4) [prev,a,b,n] (n 仅经 ph 生效, 此处只嵌前 3 键), ph: (B,15)"""
        B = ids.shape[0]
        ids = ids.to(self._dev)
        e = self.E_key(ids[:, :3])
        # 显式二阶交互: (前键,首键), (首键,次键), (前键,次键)
        bil = (torch.einsum('bi,ij,bj->b', e[:,0], self.W[0], e[:,1]) +
               torch.einsum('bi,ij,bj->b', e[:,1], self.W[1], e[:,2]) +
               torch.einsum('bi,ij,bj->b', e[:,0], self.W[2], e[:,2]))
        mlp = self.mlp(torch.cat([e.reshape(B, -1), ph.to(self._dev).reshape(B, -1)], dim=1)).squeeze(-1)
        return bil + mlp

class CPModel(_SegModel):
    """CP 张量分解: S(p,a,b,n) ≈ Σ_r U(p)·V(a)·W(b) + MLP 头 (e_n+φ15 并入头)。
    备选架构 (--model cp): v1 对比验证集 R² 0.395 vs CP 0.343 被反超 (实验-双线性
    扩容.py, 5 seed); 保留作对照。v2 对称化: 后键经 e_n 并入 MLP 头 (对照口径)。"""
    def __init__(self, rank=32, d_embed=12, d_hidden=24):
        super().__init__()
        self.U = nn.Embedding(len(LETTERS)+1, rank)   # 前键因子
        self.V = nn.Embedding(len(LETTERS)+1, rank)   # 首键因子
        self.W = nn.Embedding(len(LETTERS)+1, rank)   # 次键因子
        self.E_key = nn.Embedding(len(LETTERS)+1, d_embed)
        self.mlp = nn.Sequential(nn.Linear(d_embed*4+15, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))
    @property
    def _dev(self): return next(self.mlp.parameters()).device
    def _batch(self, ids, ph):
        """ids: (B,4) [prev,a,b,n], ph: (B,15)"""
        B = ids.shape[0]
        ids = ids.to(self._dev)
        cp = (self.U(ids[:,0]) * self.V(ids[:,1]) * self.W(ids[:,2])).sum(1)
        e = self.E_key(ids)
        mlp = self.mlp(torch.cat([e.reshape(B, -1), ph.to(self._dev).reshape(B, -1)], dim=1)).squeeze(-1)
        return cp + mlp

# ═══════════════════ 数据 ═══════════════════

def load_data(path):
    """[(code, [b_d,c_d,d_d]), ...]  error=0, 仅格式/有效性校验 (异常剔除在段级)。
    2 键行 (code 长度 2, 08-29 起混采 T₂ 角点数据) 在此被 len==4 过滤 —
    角点验证分析另行读取, 不进段模型训练。"""
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

def residual_filter_segments(data, prev, a, b, nxt, ph, tgt, k=3.0, m0_seeds=5):
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
         键对分桶保持 (a,b) 二维 (v2 对称模型后键信息在残差中心化中自然吸收)。
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
        _, va = train(m, prev[keep0], a[keep0], b[keep0], nxt[keep0], ph[keep0], tgt[keep0], seed=s)
        if va < best_va:
            best_va, M0 = va, m
    ids = torch.tensor(np.stack([prev, a, b, nxt], axis=1))
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
    """全部段样本 (每 trial 3 段): (prev, a, b, nxt, phi15, target)
    段: S(∅,a,b,c) / S(a,b,c,d) / S(b,c,d,∅) — 与 v1 同序 (slot 0/1/2 交错)"""
    segs = []
    for code, ts in data:
        a,b,c,d = code
        segs.append((EMPTY, a, b, c, ts[0]))
        segs.append((a, b, c, d, ts[1]-ts[0]))
        segs.append((b, c, d, EMPTY, ts[2]-ts[1]))
    prev = np.array([s[0] if isinstance(s[0], int) else KEY_TO_IDX[s[0]] for s in segs], dtype=np.int64)
    a = np.array([KEY_TO_IDX[s[1]] for s in segs], dtype=np.int64)
    b = np.array([KEY_TO_IDX[s[2]] for s in segs], dtype=np.int64)
    nxt = np.array([s[3] if isinstance(s[3], int) else KEY_TO_IDX[s[3]] for s in segs], dtype=np.int64)
    ph8 = np.array([PHI[(s[1], s[2])] for s in segs], dtype=np.float32)
    ph = np.concatenate([ph8, phi_suc(a, b, nxt)], axis=1)
    tgt = np.array([s[4] for s in segs], dtype=np.float32)
    return prev, a, b, nxt, ph, tgt

# ═══════════════════ 训练 ═══════════════════

def train(model, prev, a, b, nxt, ph, tgt, epochs=200, seed=0, bs=256, lr=0.001,
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
    X = torch.stack([torch.tensor(prev), torch.tensor(a), torch.tensor(b),
                     torch.tensor(nxt)], dim=1)
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

def eval_seg(model, prev, a, b, nxt, ph, tgt):
    """段 MAE (任意数据, 一次批量前向)"""
    model.eval()
    with torch.no_grad():
        pred = model._batch(torch.tensor(np.stack([prev, a, b, nxt], axis=1)),
                            torch.tensor(ph)).numpy()
    return float(np.mean(np.abs(pred - tgt)))

def total_preds(model, data):
    """四键总时间预测 (ms, float64) — 三段一次批量前向, eval_total/对比脚本共用"""
    ids, phs = [], []
    for (x, y, z, w), _ in data:
        ix, iy, iz, iw = (KEY_TO_IDX[x], KEY_TO_IDX[y], KEY_TO_IDX[z], KEY_TO_IDX[w])
        ids += [[EMPTY, ix, iy, iz], [ix, iy, iz, iw], [iy, iz, iw, EMPTY]]
        for (pa, pb, pn) in ((ix, iy, iz), (iy, iz, iw), (iz, iw, EMPTY)):
            phs.append(np.concatenate([PHI[(LETTERS[pa], LETTERS[pb])],
                                       phi_suc([pa], [pb], [pn])[0]]))
    model.eval()
    with torch.no_grad():
        pred = model._batch(torch.tensor(ids), torch.tensor(np.array(phs, dtype=np.float32))).numpy()
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
    """4-D 段表 F[p,a,b,n] = S(p,a,b,n) ms, 形状 (31,30,30,31)。
    p/n 维 0-29 = 键, 30 = ∅ (与模型索引一致); a/b 维 0-29 = 键。
    按 p 分 31 批 (每批 30×30×31 = 27,900 行) 一次前向。"""
    n = len(LETTERS)
    tri = [(a, b, x) for a in range(n) for b in range(n) for x in list(range(n)) + [EMPTY]]
    ph8 = np.array([PHI[(LETTERS[a], LETTERS[b])] for a, b, _ in tri], dtype=np.float32)
    ta = np.array([t[0] for t in tri]); tb = np.array([t[1] for t in tri])
    tn = np.array([t[2] for t in tri])
    phs = np.concatenate([ph8, phi_suc(ta, tb, tn)], axis=1)
    F = np.empty((n + 1, n, n, n + 1), dtype=np.float32)
    with torch.no_grad():
        for p in range(n + 1):
            ids = torch.tensor([[p, a, b, x] for a, b, x in tri])
            F[p] = model._batch(ids, torch.tensor(phs)).numpy().reshape(n, n, n + 1)
    return F

def _bigram_ms(model):
    """T₂ 角点 B[a,b] = S(∅,a,b,∅) ms (词末两键, 未归一化), 形状 (30,30)。
    角点 (p=∅,n=∅) 无训练样本 — 特征空间内插, 2 键试次验证中 (README §6.6)"""
    n = len(LETTERS)
    ids = torch.tensor([[EMPTY, a, b, EMPTY] for a in range(n) for b in range(n)])
    ph8 = np.array([PHI[(LETTERS[a], LETTERS[b])] for a in range(n) for b in range(n)], dtype=np.float32)
    phs = np.concatenate([ph8, phi_suc(np.arange(n).repeat(n), np.tile(np.arange(n), n),
                                       np.full(n*n, EMPTY))], axis=1)
    with torch.no_grad():
        return model._batch(ids, torch.tensor(phs)).numpy().reshape(n, n)

def export_seg_table(model, out_path):
    """4-D 段当量母表 (npz): F (31,30,30,31) float32 + 元数据。
    查询组合 (组装当量表.py / 组装-chai当量表.py):
      T₂(ab)   = F[∅,a,b,∅]           (角点, 2 键试次验证中)
      T₃(abc)  = F[∅,a,b,c] + F[a,b,c,∅]
      T₄(abcd) = F[∅,a,b,c] + F[a,b,c,d] + F[b,c,d,∅]
    v1 的 3-D 文本段表 (当量-段表.txt) 由本文件替代 (2026-08-29 npz 化)。"""
    F = build_seg_table(model)
    np.savez_compressed(out_path, F=F,
                        letters=np.array(list(LETTERS)), empty=np.int64(EMPTY),
                        version=np.int64(2),
                        note=np.array("S(p,a,b,n) ms 原始值; p/n 维 0-29=键 30=EMPTY; "
                                      "T2=F[30,a,b,30] T3=F[30,a,b,c]+F[a,b,c,30] "
                                      "T4=F[30,a,b,c]+F[a,b,c,d]+F[b,c,d,30]"))
    print(f"  当量-段表: {out_path}  (F {F.shape}, {F.size:,} 条, npz)")

# ═══════════════════ 主流程 ═══════════════════

def main():
    ap = argparse.ArgumentParser(description="对称击键当量模型: 训练 + 测试集评估 + 可选导出")
    ap.add_argument("--full", action="store_true",
                    help="追加部署模型训练 (全数据) + 导出 4-D 段当量表 (npz) 与模型")
    ap.add_argument("--model", choices=["bi", "cp"], default="bi",
                    help="bi=双线性 de20/dh64 sym_phi (默认) | cp=CP 张量分解 rank=32")
    ap.add_argument("--trials", type=int, default=10, help="best-of-N 训练次数")
    args = ap.parse_args()
    full, model_name, trials = args.full, args.model, args.trials
    PROJ = Path(__file__).resolve().parent
    DATA = PROJ / "击键测速数据.tsv"
    if not DATA.exists(): print(f"无数据: {DATA}"); sys.exit(1)

    print(f"加载: {DATA}")
    data = load_data(str(DATA))
    print(f"4 键样本: {len(data)}, 段样本: {len(data)*3}")

    # ── 主口径划分: 固定 trial 测试集 (2026-08-19 起统一口径; 与实验脚本同协议) ──
    rng = np.random.RandomState(2024)
    idx = rng.permutation(len(data))
    nt = int(len(data) * .2)
    test_idx = set(idx[:nt])
    train_data = [d for i, d in enumerate(data) if i not in test_idx]
    test_data = [d for i, d in enumerate(data) if i in test_idx]
    print(f"划分版本: N={len(data)} trial, 测试集 {len(test_data)} ({len(test_data)*100/len(data):.0f}%), 种子 2024")
    print(f"训练 trial {len(train_data)} / 测试 trial {len(test_data)}")

    # ── 评估模型: 训练集 B4b + best-of-N (测试 trial 的段不进训练/验证, 无泄漏) ──
    prev, a, b, nxt, ph, tgt = build_tensors(train_data)
    keep, _ = residual_filter_segments(train_data, prev, a, b, nxt, ph, tgt)
    print(f"训练集 B4b 剔除 (键对分桶带符号残差, 单侧上围栏 k=3): 保留 {int(keep.sum())}/{len(tgt)} 段")
    prev, a, b, nxt, ph, tgt = prev[keep], a[keep], b[keep], nxt[keep], ph[keep], tgt[keep]

    model_name_full = {"bi": "双线性 de20/dh64 sym_phi", "cp": "CP rank=32"}.get(model_name, model_name)
    print(f"\n=== 评估模型训练 ({model_name_full}, best-of-{trials}, 验证集选优) ===")
    best_va, best_m = float("inf"), None
    for t in range(trials):
        m = CPModel() if model_name == "cp" else KeystrokeModel()
        seg_mae, va_mae = train(m, prev, a, b, nxt, ph, tgt, seed=t)
        star = ""
        if va_mae < best_va:
            best_va, best_m = va_mae, m
            star = "  ← 新最优"
        print(f"  trial {t+1:2d}/{trials}: 段MAE={seg_mae:5.1f}ms  验证段MAE={va_mae:5.1f}ms{star}")

    # 测试集独立 B4b (保留口径; 内部自训 M0, 与训练集剔除无信息交换)
    tp, ta, tb, tn, tph, tt = build_tensors(test_data)
    tkeep, test_full = residual_filter_segments(test_data, tp, ta, tb, tn, tph, tt)
    print(f"测试集 B4b: 保留 {int(tkeep.sum())}/{len(tt)} 段, 完整 trial {len(test_full)}/{len(test_data)}")

    # ── 主指标: 保留口径 (2026-08-26 起默认只展示保留口径, 全口径需时另行说明) ──
    seg_keep = eval_seg(best_m, tp[tkeep], ta[tkeep], tb[tkeep], tn[tkeep], tph[tkeep], tt[tkeep])
    tot_keep, r2_keep = eval_total(best_m, test_full)
    print(f"\n=== 主指标 (固定 trial 测试集 {len(test_data)}, 保留口径, 参数 {best_m.nparam()}, 验证段MAE={best_va:.1f}) ===")
    print(f"保留口径: 段MAE={seg_keep:5.1f}  总MAE={tot_keep:5.1f}  R²={r2_keep:+.3f}")

    if not full:
        print("\n研究模式: 仅训练评估。加 --full 追加部署模型训练 (全数据) + 导出当量表。")
        return

    # ── 部署模型: 全数据 B4b + best-of-N (当量表条目质量优先; 评估指标以上方为准) ──
    print(f"\n=== 部署模型 (全数据训练, {trials} trials — 导出用, 指标见上方评估模型) ===")
    dprev, da, db, dn, dph, dtgt = build_tensors(data)
    dkeep, _ = residual_filter_segments(data, dprev, da, db, dn, dph, dtgt)
    print(f"全数据 B4b: 保留 {int(dkeep.sum())}/{len(dtgt)} 段")
    dprev, da, db, dn, dph, dtgt = dprev[dkeep], da[dkeep], db[dkeep], dn[dkeep], dph[dkeep], dtgt[dkeep]
    dbest_va, deploy_m = float("inf"), None
    for t in range(trials):
        m = CPModel() if model_name == "cp" else KeystrokeModel()
        seg_mae, va_mae = train(m, dprev, da, db, dn, dph, dtgt, seed=t)
        star = ""
        if va_mae < dbest_va:
            dbest_va, deploy_m = va_mae, m
            star = "  ← 新最优"
        print(f"  trial {t+1:2d}/{trials}: 段MAE={seg_mae:5.1f}ms  验证段MAE={va_mae:5.1f}ms{star}")

    print("\n=== 导出 ===")
    deploy_m.save(str(PROJ/"keystroke_model.pt"))
    print(f"  模型 → {PROJ/'keystroke_model.pt'}  (v2 对称段模型, 与 v1 权重不兼容)")
    export_seg_table(deploy_m, str(PROJ/"当量-段表.npz"))


if __name__ == "__main__":
    main()
