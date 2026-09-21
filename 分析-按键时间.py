#!/usr/bin/env python3
"""
条件击键当量模型（v3 混合部署模型）
  研究阶段（默认）: 训练段模型 + 报告总时间 MAE/R²
  完成阶段 --full: 追加导出部署模型 (全数据) 与 4-D 按键时间母表 (npz)

段定义: t = S(角色, p,a,b,n) — 前两键 p,a + 当前键对 a,b + **后继键 n** (v2, 2026-08-29,
实验-后键条件系列, 结论已存档)。n=∅(索引 30) 表无后继 (词末"甩出")。
**段角色 (v4, 2026-09-21 段位特征正式落地)**: S 的条件里加入 (码长 L, 段位 i) —
段身份 (p,a,b,n) 只编码「是否首段 / 是否末段」, 同一签名在 3 键与 4 键码里角色不同
(如 3键seg2 与 4键seg3 同签名 (有前键,无后键))。实现 = φ19 = φ15 + [i==2, i==3,
L==3, L==4]; 3 键进训练池 (train4+train3+train2)。依据 实验/实验-段位特征.py 三刷:
T₃ 总MAE −1.2~−1.6ms、T₄ −0.3~−0.5ms、T₂ 恒零 (同号显著, 两种测试划分)。
神经分量: 双线性交互(e_p,e_a,e_b) 3 项 + MLP([e_p;e_a;e_b;φ19]) 79→128→64→1 + Dropout0.2
  (v3.1, 2026-09-02 采纳 d20w128: deep2 加深(v3)再放宽+Dropout — 部署级 blend 总 −0.96,
  dropout 使 5 成员去相关、集成收益放大, 实验-MLP调参.py 第四轮)
  φ15 = φ8(a,b) 几何 + φsuc7 后键特征 [存在, 同手(b,n), 同指(b,n), 同键(b,n),
  Fitts(b,n), 同手(a,n), Fitts(a,n)]; n=∅ 时 φsuc 全 0 (存在位=0)。
部署形态 (v3, 2026-09-02, 用户确认采纳): BlendModel = 0.3×XGBoost(独热124+φ19)
  + 0.7×(deep2 × 固定种子 0-4 平均) — 无 best-of 选优 → 无选优抽签噪声 (决策带
  从 ±3 缩至仅测试重洗 ~±1ms); 依据 实验-架构对比3.py (09-14 当前数据重跑):
  原始口径 总 41.6 / R²+0.592 / 段 25.8 / 角点 16.0 (对单模 base 44.0 −2.4, 对 ens5 −0.6)。

查询公式 (词内语义, 全部查询点有训练分布覆盖, T₂ 角点由 2 键试次内插):
  两键当量  T₂(xy)   = R(L2i1)[x,y]
  三键当量  T₃(xyz)  = R(L3i1)[x,y,z] + R(L3i2)[x,y,z]
  四键当量  T₄(wxyz) = R(L4i1)[w,x,y] + R(L4i2)[w,x,y,z] + R(L4i3)[x,y,z]

键位嵌入: e_k = E[k] ∈ ℝ²⁰ (30 键各独立, 索引 30 = ∅ 前键/后键)
导出 (2026-09-05 起入 产物/): 按键时间表.npz — 2026-09-21 起按段角色 **6 片** (version 3;
  旧版单表 F(31,30,30,31) 无法表达同签名跨码长的角色分裂, 与错误侧 09-19 段类重参数化同因),
  附 letters/empty/roles/version 元数据; 组装当量表.py 查表组合 (段当量 = 按键时间 + 错误率×错误修正时间)。
  模型: 按键时间模型-神经.pt (deep2×5 权重) + 按键时间模型-xgb.json (XGB 分量)。
README 实证数字 (2026-09-18 起随 --full 输出, readme_evidence): §5.4 类型×前键 / 键对难度 /
  模型视角 / 排序扭曲 / §6.1 rollover — 原 实验-README数字刷新.py 并入删除 (同 §5.3 内嵌演进);
  §5.5 签名vs类盲随 分析-错误率.py 输出。
"""
import argparse, itertools, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

# 运行环境固定 (可复现性, 2026-08-18 实测):
# ① 单线程 — 模型仅 ~6.8k 参数/批 256, 多线程开销 > 收益;
# ② 导入时播种 torch 默认 RNG — 新进程默认 RNG 是熵播种的, 模型构造 (嵌入/线性层
#   初始化) 依赖它, 而 train(seed) 只重播轨迹不重播构造 → 历史上同命令重跑结果漂移
#   ("±1.4ms 种子波动"的主要来源)。播种后同代码+同数据跨进程逐位可复现;
#   best-of-N 的多次构造沿确定性流推进, 初始化仍互不相同, 多样性不受影响。
# 注意: 裸构造 KeystrokeModel() 仍消耗全局 RNG — 跨脚本绝对数字不可比, 变体对比须
#   同脚本同流内进行 (08-27 实验教训); train_members 路径 (2026-09-09) 的成员构造
#   显式种子化, 与调用流位置无关, 跨脚本可比。
torch.set_num_threads(1)
torch.manual_seed(0)

# 并行训练 (2026-09-09): 成员构造显式种子化 (fork_rng + manual_seed(seed)), train()
# 入口本就重播 torch+numpy RNG → 每个成员 = (seed, 数据) 的纯函数, 多进程并行与串行
# 逐位一致; 2026-08-18 方案里"构造沿全局流推进"的跨成员 RNG 纠缠就此消除 (成员多样
# 性不变: 各种子初始化仍互不相同)。历史指标因此在噪声带内挪动一次 (~±1ms, 等同一次
# 测试集重洗)。KJ_SEQ=1 强制串行 (调试/逐位对照)。
_POOL = None

def _train_member(seed, prev, a, b, nxt, ph, tgt, loss_fn="mse"):
    """单成员: 构造 (seed 显式种子化) + 训练 → (model, 全数据段MAE, 验证段MAE)"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        m = KeystrokeModel(d_phi=np.asarray(ph).shape[1])   # φ 宽度取自数据 (段位落地后 = D_PHI 19)
    seg_mae, va_mae = train(m, prev, a, b, nxt, ph, tgt, seed=seed, loss_fn=loss_fn)
    return m, seg_mae, va_mae

def _main_spawn_safe():
    """Windows spawn 子进程会重导入调用方主模块顶层代码 — 主脚本无 __main__ 保护时
    并行会让每个子进程重跑其训练逻辑; 检测不到保护时回退串行 (结果一致, 速度同旧版)"""
    mn = sys.modules.get("__main__")
    f = getattr(mn, "__file__", "")
    if not f:
        return False
    try:
        return '"__main__"' in Path(f).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

def train_members(jobs):
    """jobs: [(seed, prev, a, b, nxt, ph, tgt[, loss_fn])] → 按序 [(model, 段MAE, 验证段MAE)]。
    各 job 为纯函数 → 结果与执行方式 (并行/串行) 无关。调用方主脚本须有 __main__ 保护"""
    if os.environ.get("KJ_SEQ") or len(jobs) <= 1 or not _main_spawn_safe():
        return [_train_member(*j) for j in jobs]
    global _POOL
    if _POOL is None:
        import atexit, multiprocessing as mp
        _POOL = mp.get_context("spawn").Pool(min(len(jobs), os.cpu_count() or 1))
        atexit.register(_POOL.terminate)        # 避免 interpreter 退出时 Pool.__del__ 报错
    try:
        return _POOL.starmap(_train_member, jobs)
    except Exception as e:                      # 并行环境异常 → 串行兜底 (结果一致)
        print(f"  [并行不可用 ({type(e).__name__}: {e}), 回退串行]")
        return [_train_member(*j) for j in jobs]

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

# ═══════════════ 段角色 (2026-09-21 段位特征正式落地) ═══════════════
# 段身份 (p,a,b,n) 只编码「是否首段 / 是否末段」——同一签名在 3 键与 4 键码里可能是
# 不同角色 (3键seg2 与 4键seg1 签名 (1,0)/(0,1) 各自对应不同段位)。错误率侧 2026-09-19
# 已按「段类 = 签名 × 段位」重参数化; 时间侧同因落地: 模型输入加 φ19 = φ15 + [i==2,
# i==3, L==3, L==4] (i=段位, L=码长), 按键时间表由单一 4-D F 改为**按角色分 6 片**。
# 依据 实验/实验-段位特征.py 三刷(988→1322 trial × 两种划分): T₃ 总MAE −1.2~−1.6ms
# (同号显著)、T₄ −0.3~−0.5ms (小样本时不显著, 3 键数据到 1322 trial 后显著)、T₂ 恒零;
# T₃ 时间偏差 −3.7→−2.0 (收窄未消除)。3 键入训是 L 特征可学的前提 (否则 L==3 恒 0)。
ROLES = ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3))   # (码长, 段位), 索引即角色 id
ROLE_ID = {r: i for i, r in enumerate(ROLES)}
ROLE_SLICE = (("r_L2i1", (2, 1)), ("r_L3i1", (3, 1)), ("r_L3i2", (3, 2)),
              ("r_L4i1", (4, 1)), ("r_L4i2", (4, 2)), ("r_L4i3", (4, 3)))
# 角色 → (参与索引的键数, 该角色的段在 (p,a,b,n) 上如何取用这些键)
# 索引顺序即该角色实际用到的键序: 首段 (a,b,n) / 末段 (p,a,b) / 中段 (p,a,b,n) / 角点 (a,b)
ROLE_CELL = {
    (2, 1): (2, lambda t: (EMPTY, t[0], t[1], EMPTY)),      # T₂ 角点 S(∅,x,y,∅)
    (3, 1): (3, lambda t: (EMPTY, t[0], t[1], t[2])),       # 3 键首段 S(∅,x,y,z)
    (3, 2): (3, lambda t: (t[0], t[1], t[2], EMPTY)),       # 3 键短尾 S(x,y,z,∅)
    (4, 1): (3, lambda t: (EMPTY, t[0], t[1], t[2])),       # 4 键首段 S(∅,w,x,y)
    (4, 2): (4, lambda t: (t[0], t[1], t[2], t[3])),        # 4 键中段 S(w,x,y,z)
    (4, 3): (3, lambda t: (t[0], t[1], t[2], EMPTY)),       # 4 键长尾 S(x,y,z,∅)
}
ROLE_ERR = {0: "c0", 1: "c1", 2: "c2", 3: "c1", 4: "c3", 5: "c4"}  # 角色 → 错误侧段类
CORNER_ROLE = ROLE_ID[(2, 1)]      # 2 键角点槽位 (B4b 分桶 / §5.4 桶筛选用)
FIRST_SEG_ROLES = (ROLE_ID[(2, 1)], ROLE_ID[(3, 1)], ROLE_ID[(4, 1)])   # p=∅ 段

def phi_role(role):
    """φ19 的段位块 4 维: [i==2, i==3, L==3, L==4] (role 数组 = 角色 id)。
    6 个角色在该 4 维空间两两可分辨 (L2i1=[0,0,0,0] 为参考电平), 与实验变体逐位一致"""
    role = np.asarray(role)
    i = np.array([ROLES[r][1] for r in role.ravel()]).reshape(role.shape)
    L = np.array([ROLES[r][0] for r in role.ravel()]).reshape(role.shape)
    return np.stack([(i == 2), (i == 3), (L == 3), (L == 4)], axis=-1).astype(np.float32)

def role_of(code_idx):
    """(码长, 段位) → 角色 id; 段位 1-based, 段数 = 码长-1"""
    return ROLE_ID[(len(code_idx), code_idx)]

D_PHI = 19   # φ 宽度 = 15 (φ8 + φsuc7) + 4 (段位块)

def _precompute_phi():
    """φ(a,b) 8 维 — 2026-08-11 文献特征扩展 (实验-特征扩展.py):
    [同手, 同指, 同键, 列距, 行距, Fitts, 镜像手指, 跨行同指]
    文献依据: İşeri & Ekşioğlu 2015 (列>行>手 重要性排序, 显式几何有独立信息);
    Fitts (MT = a + b·log2(1+D/W), W=键宽, 行距≈2 键宽); keygen (同指跨行最慢);
    Grudin 1983 (镜像手指换位). 消融两轮:
    - 逐特征 LOO 全部持平 (实验-特征消融.py, 2026-08-11, Δ<0.2ms) — φ 特征间互为冗余
    - 整体删除 +1.7ms/角点 +3.2ms (09-14 当前数据复测, 实验-架构对比3.py no_phi;
      原 2026-08-14 结构消融同值) — φ 作为整体
      提供嵌入补不上的几何先验, 价值在稀疏三元组外推 (2/3 按键时间表条目零样本); LOO 在
      训练分布内做所以测不出。保留 φ 有实证支撑 (结构消融已删, 结论存档)"""
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
PHI_MAT = np.array([[PHI[(x, y)] for y in LETTERS] for x in LETTERS], dtype=np.float32)  # (30,30,8) 批量用

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
    """对称段模型 (sym_phi deep2): 键位嵌入 (20 维) + φ15 + 双线性交互 3 项 + 深化 MLP 头。
    v3 (2026-09-02): MLP 64→32→1 两隐藏层 (deep2), 依据 实验-架构对比.py 稳定期重审 —
    加一层对 base 10/10 配对 −2.2~−2.5ms, v1"单层最优"为混合数据伪收敛 (实验-架构对比)。
    部署形态 = BlendModel (本类×5 固定种子平均 + XGB 0.3 混合, 见下方混合段)。
    v2 (2026-08-29): 增后键条件 — n 仅经 φsuc 特征进入 (几何身份≈嵌入身份);
    依据 实验-后键条件系列(已删, 结论存档): 对称化总MAE −1.3~−1.6。
    v1 依据 (实验-双线性扩容.py 等): de20 单层最优; relu 最优; RMSNorm 微增益;
    W3 跨键双线性 +2.1ms。"""
    def __init__(self, d_embed=20, d_hidden=128, p_drop=0.2, d_phi=D_PHI):
        super().__init__()
        self.d_phi = d_phi   # φ 宽度自适应 (2026-09-21): 段位特征实验需要 φ19, 默认 15 行为不变
        self.E_key = nn.Embedding(len(LETTERS) + 1, d_embed)  # 30 键 + ∅
        self.W = nn.Parameter(torch.randn(3, d_embed, d_embed) * 0.05)  # (p,a),(a,b),(p,b)
        # v3.1 d20w128: d_hidden 64→128 + Dropout 0.2 (实验-MLP调参.py 部署级 blend
        # 总 −0.96; dropout 成员去相关 → 种子平均收益放大)
        self.mlp = nn.Sequential(nn.Linear(d_embed*3+d_phi, d_hidden), nn.ReLU(),
                                 RMSNorm(d_hidden), nn.Dropout(p_drop),
                                 nn.Linear(d_hidden, d_hidden//2), nn.ReLU(),
                                 nn.Dropout(p_drop), nn.Linear(d_hidden//2, 1))
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
    备选架构 (历史对照): v1 验证集 R² 0.395 vs CP 0.343; 稳定期重审 50.6 vs 47.5;
    当前数据重审 48.7 vs 44.0 (实验-架构对比.py, 09-14) 三时代皆差, 维持废弃。"""
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

# ═══════════════════ 混合部署模型 (v3, 2026-09-02) ═══════════════════

W_XGB = 0.3   # 混合权重: 0.3×XGB + 0.7×(deep2 五种子平均) — 实验-架构对比2.py 双批定标
XGB_PARAMS = dict(n_estimators=500, learning_rate=0.06, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  random_state=0, n_jobs=-1, verbosity=0)

def design_matrix(prev, a, b, nxt, ph):
    """XGB 设计矩阵: 独热(p,a,b,n 各31列) 124 + φ15 = 139 维"""
    n = len(prev)
    oh = np.zeros((n, 4 * 31), dtype=np.float32)
    for j, arr in enumerate((prev, a, b, nxt)):
        oh[np.arange(n), np.asarray(arr) + j * 31] = 1.0
    return np.concatenate([oh, np.asarray(ph).reshape(n, -1).astype(np.float32)], axis=1)

def train_xgb(prev, a, b, nxt, ph, tgt):
    """XGB 分量 (确定性, 种子稳定 ±0.18ms — 实验-架构对比2)"""
    from xgboost import XGBRegressor
    X = design_matrix(prev, a, b, nxt, ph)
    return XGBRegressor(**XGB_PARAMS).fit(X, tgt)

class BlendModel:
    """v3 混合部署模型: W×XGB + (1-W)×(KeystrokeModel deep2 × 固定 5 种子平均)。
    无 best-of 选优 (固定种子 0-4 全体平均) → 无选优抽签噪声, 同数据逐位确定。
    依据 实验-架构对比3.py (09-14 当前数据重跑): 总 41.6 / R²+0.592 / 段 25.8 / 角点 16.0
    (原始口径), 对 ens5 单独 −0.6ms, 对单模 base 44.0 −2.4ms。
    接口与 _SegModel 鸭子类型兼容 (seg/total/_batch/nparam)。"""
    SEEDS = (0, 1, 2, 3, 4)
    def __init__(self, xgb_fit, members):
        self.xgb, self.members = xgb_fit, members
    @classmethod
    def fit(cls, prev, a, b, nxt, ph, tgt):
        xgb_fit = train_xgb(prev, a, b, nxt, ph, tgt)
        members = [m for m, _, _ in train_members([(s, prev, a, b, nxt, ph, tgt)
                                                   for s in cls.SEEDS])]
        return cls(xgb_fit, members)
    def _batch(self, ids, ph):
        """ids: torch (B,4) [p,a,b,n]; 返回 torch 张量 (调用方 .numpy())"""
        ids_np = ids.numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
        ph_np = ph.numpy() if isinstance(ph, torch.Tensor) else np.asarray(ph)
        with torch.no_grad():
            d = torch.stack([m._batch(ids, ph) for m in self.members]).mean(0)
        x = torch.from_numpy(self.xgb.predict(design_matrix(
            ids_np[:, 0], ids_np[:, 1], ids_np[:, 2], ids_np[:, 3], ph_np)).astype(np.float32))
        return W_XGB * x + (1 - W_XGB) * d
    # ── 单查询接口 (与 _SegModel 同形) ──
    def seg(self, prev, a, b, nxt=None):
        ids = torch.tensor([[EMPTY if prev is None else KEY_TO_IDX[prev],
                             KEY_TO_IDX[a], KEY_TO_IDX[b],
                             EMPTY if nxt is None else KEY_TO_IDX[nxt]]])
        ph = torch.tensor([np.concatenate([PHI[(a, b)], phi_suc(
            [KEY_TO_IDX[a]], [KEY_TO_IDX[b]],
            [EMPTY if nxt is None else KEY_TO_IDX[nxt]])[0]])], dtype=torch.float32)
        return self._batch(ids, ph).item()
    def total(self, code):
        a, b, c, d = code
        return self.seg(None, a, b, c) + self.seg(a, b, c, d) + self.seg(b, c, d)
    def nparam(self): return sum(m.nparam() for m in self.members)
    def eval(self): return self   # 兼容 eval_seg 的 model.eval() (成员常驻 eval 态)
    def save(self, path_pt, path_xgb):
        torch.save({"seeds": list(self.SEEDS),
                    "members": [m.state_dict() for m in self.members]}, path_pt)
        self.xgb.get_booster().save_model(path_xgb)
    @classmethod
    def load(cls, path_pt, path_xgb):
        import xgboost as xgb
        ck = torch.load(path_pt, weights_only=True)
        members = []
        for sd in ck["members"]:
            m = KeystrokeModel(); m.load_state_dict(sd); m.eval(); members.append(m)
        bst = xgb.Booster(); bst.load_model(path_xgb)
        class _BstWrap:   # 与 sklearn 拟合对象同形 (predict(ndarray))
            def predict(self, X): return bst.inplace_predict(np.ascontiguousarray(X, dtype=np.float32))
        return cls(_BstWrap(), members)

# ═══════════════════ 数据 ═══════════════════

# 项目相对路径常量 (2026-09-05 目录重组: 数据/产物/实验 三分; 单一规范源, 下游一律引用)
_DIR      = Path(__file__).resolve().parent
DATA_TSV  = _DIR / "数据" / "击键测速数据.tsv"
ART_DIR   = _DIR / "产物"                 # 全部计算产物 (导出方负责 mkdir)
SEG_NPZ   = ART_DIR / "按键时间表.npz"      # 按键时间母表 (本文件 --full 导出)
ERR_TXT   = ART_DIR / "键对错误率表.txt" # 键对错误率表 (分析-错误率.py 导出)
CHEN_TXT  = _DIR / "数据" / "陈一凡当量表.txt"  # 陈表对比数据 (键对\t当量, 随仓库分发)
CORR_NPZ  = ART_DIR / "段当量表.npz"  # 段当量表 (组装当量表.py)
T24_TXT   = ART_DIR / "总当量-2-4键.txt"     # 2/3/4 键总当量 (组装当量表.py)
MODEL_PT  = ART_DIR / "按键时间模型-神经.pt"
MODEL_XGB = ART_DIR / "按键时间模型-xgb.json"

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

def load_trials(path):
    """按时间序 (文件追加序 = 采集先后) 读全部 ok trial, 不按 session 分组 (2026-09-06 起
    序号口径, 用户决策: session=进程生命周期, 行数 4~836 悬殊且开关与坐姿不对应, 仅作
    采集端存档字段; 依据 实验/实验-序号趋势对比.py: 两种口径边界互证一致, session 内无热身
    效应 → 边界不携带状态信息)。
    返回 rows: [(code, times)] 文件顺序; 4 键 times=[b_d,c_d,d_d] (均自首键按下累计,
    d_d=trial 总时), 3 键 [b_d,c_d] (09-19 补采透传, d_d=0), 2 键 [b_d] (T₂ 角点, 08-31 入训)。
    3 键稳定期 09-21 晚达 9 块全程并**正式入训** (train_all = train4 + train3 + train2, §5.6):
    φ19 段位块的 L==3 维只在 3 键样本上可辨识; 改前实测 T₃ 低估 -7.3ms (seg2 占 -5.7) 且
    单纯加 3 键训练只修 +0.3ms — 缺的是特征里的段位信息。
    过滤同 08-31 版。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        hdr = {h: i for i, h in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            row = line.strip().split("\t")
            if len(row) < 12 or row[hdr.get("error", 8)] != "0":
                continue
            code = row[hdr["code"]]
            if not all(c in LETTERS for c in code):
                continue
            try:
                bd, cd, dd = float(row[hdr["b_d"]]), float(row[hdr["c_d"]]), float(row[hdr["d_d"]])
            except ValueError:
                continue
            if len(code) == 4:
                if bd <= 0 or cd <= bd or dd <= cd:
                    continue
                rows.append((code, [bd, cd, dd]))
            elif len(code) == 3:
                if bd <= 0 or cd <= bd:
                    continue
                rows.append((code, [bd, cd]))
            elif len(code) == 2:
                if bd <= 0:
                    continue
                rows.append((code, [bd]))
    return rows

def _block_boundary(tots, block, band, ref_last, smooth):
    """单码长序号块边界: ok trial 总时时间序 → (起始块 k, R, 块数 nb)。
    k=None: 块数<2 (数据不足, R 不可估); k==nb: 末块即超带 (退化, 稳定期空)。
    方法: 块总时中位 → smooth 块滚动中位 → R=末 ref_last 平滑值中位 →
    自末向前最早连续 ≤R×band 的块起 (确定性无随机)。"""
    nb = -(-len(tots) // block)
    if nb < 2:
        return None, None, nb
    med = np.array([np.median(tots[b * block:(b + 1) * block]) for b in range(nb)])
    sm = np.array([np.median(med[max(0, i - smooth + 1):i + 1]) for i in range(nb)])
    R = float(np.median(sm[-ref_last:]))
    k = nb
    while k - 1 >= 0 and sm[k - 1] <= R * band:
        k -= 1
    return k, R, nb


def stable_pools(path, block=150, band=1.10, ref_last=5, smooth=3, min_blocks=8):
    """稳定期标准划分 — 分码长序号块口径 (2026-09-19 用户决策: 三种码长入队时间不同、
    练习轨迹各自独立 (实验-分码长练习趋势.py: 2键全程无趋势/3键单 session), 稳定期按码长
    分别计算各自截取; 旧"4键边界+同截归属"对 2键恰巧等价——2键入队晚于 4键边界且自身无
    趋势, 其边界自然回溯至块 0 → train4/train2/test4/test2 内容与旧版逐位一致, 已验证)。
    每码长 (4键 d_d / 2键 b_d / 3键 c_d): ok trial 按时间序切 block 等大块 → 块中位 →
    smooth 滚动中位 → R=末 ref_last 中位 → 自末向前最早连续 ≤R×band 的块起。
    稳定块数 < min_blocks 的码长照常截取并告警 (如积累中的 3 键: 边界随数据自动收紧)。
    划分: 各码长稳定期分别 RandomState(2024) 顺序 permutation 80/20 (先 4 后 2 后 3,
    固定顺序 → 4/2 键随机流与旧版一致)。返回 dict:
      train_all  训练池 = train4 + train3 + train2 (拼接顺序固定; 2026-09-21 段位特征落地后
                 3 键入训 — L==3 特征只在 3 键样本上可辨识, 且这正是 T₃ 偏差的修法)
      train4/test4/train2/test2/train3/test3  各码长 train/test (test3 现为 T₃ 留出指标)
      deploy4/deploy2/deploy3  各码长稳定期全量 (部署 = 当前状态语义)
      desc  划分说明 (含各码长边界/R/告警)
    依据: 角点 S(∅,a,b,∅) 原为零样本外推 (实测偏差 −6.0±0.8ms), 2 键参与训练使其
    内插化; 模型结构共享使稀疏覆盖 (n=1) 也被整体统计强度正则化。"""
    rows = load_trials(path)
    idx_n = {n: [i for i, (c, _) in enumerate(rows) if len(c) == n] for n in (4, 2, 3)}
    tcol = {4: 2, 2: 0, 3: 1}          # 4键 d_d / 2键 b_d / 3键 c_d
    bounds, Rs, nbs = {}, {}, {}
    for n in (4, 2, 3):
        bounds[n], Rs[n], nbs[n] = _block_boundary(
            [rows[i][1][tcol[n]] for i in idx_n[n]], block, band, ref_last, smooth)
    stable, warn = {}, []
    for n in (4, 2, 3):
        k, R, nb = bounds[n], Rs[n], nbs[n]
        if R is None:                                   # 块数<2: 全保留
            stable[n] = [rows[i] for i in idx_n[n]]
        elif k >= nb:                                   # 末块超带: 稳定期空 (同旧版退化)
            stable[n] = []
            warn.append(f"{n}键末块超带 稳定期空")
        else:
            stable[n] = [rows[i] for i in idx_n[n][k * block:]]
            if nb - k < min_blocks:
                warn.append(f"{n}键稳定期仅 {nb - k} 块 (<{min_blocks})")
    rng = np.random.RandomState(2024)
    out = {}
    for n in (4, 2, 3):
        i_n = rng.permutation(len(stable[n]))
        nt_n = int(len(stable[n]) * .2)
        t_set = set(i_n[:nt_n])
        out[f"train{n}"] = [stable[n][i] for i in range(len(stable[n])) if i not in t_set]
        out[f"test{n}"] = [stable[n][i] for i in range(len(stable[n])) if i in t_set]
        out[f"deploy{n}"] = list(stable[n])

    def bd_txt(n):
        k, R, nb = bounds[n], Rs[n], nbs[n]
        if R is None:
            return f"{n}键 块数<2 暂全保留"
        if k >= nb:
            return f"{n}键 末块超带 稳定期空"
        tag = f"({nb - k}/{nb} 块)" if nb - k < min_blocks else ("(全程)" if k == 0 else "")
        return f"{n}键 块[{k}] (ok行 #{idx_n[n][k * block] + 1}){tag} R={R:.0f}"

    desc = (f"稳定期(分码长): {' | '.join(bd_txt(n) for n in (4, 2, 3))}  "
            f"(4键 {len(stable[4])} + 3键 {len(stable[3])} + 2键 {len(stable[2])} trial"
            + (f"; ⚠ {'; '.join(warn)}" if warn else "") + ")")
    return {"train_all": out["train4"] + out["train3"] + out["train2"], "desc": desc, **out}

def _seg_layout(data):
    """段槽位与 trial 归属 (6 角色: 4 键 (4,1)/(4,2)/(4,3) · 3 键 (3,1)/(3,2) · 2 键角点 (2,1))。
    B4b 的段位分桶与 trial 完整性检查共用 — 2026-09-21 段位特征落地后分桶按 (码长,段位):
    4 键与 2 键段的分桶划分与旧 4 槽版逐位等价 (旧槽 0/1/2 = 4 键三段, 旧槽 3 = 角点),
    仅新增 3 键两段各自成桶。"""
    pos, tri = [], []
    for ti, (code, _) in enumerate(data):
        for i in range(len(code) - 1):
            pos.append(ROLE_ID[(len(code), i + 1)]); tri.append(ti)
    return np.array(pos), np.array(tri)

def mad_filter_segments(data, tgt, k=3.0):
    """段级异常剔除 (MAD 自适应, 按段位置分桶; 单侧上围栏 — 异常只可能偏大):
      剔除 seg ≥ median + k·1.4826·MAD 的段 (注意力中断; 中位数稳健, 不被长尾污染;
      2026-08-20 起单侧化: 右偏分布下 MAD 下界恒为负, 实测下侧剔除恒 0,
      双侧公式与单侧数值等价, 单侧与"异常只偏大"语义对齐)。
      不做硬性上下限:
        - 快段受单侧语义保护 (实测最快常规击键 15-40ms, 硬下限会误删基准键对)
        - MAD 上界 (实测 265-412ms) 已严于任何固定上限, 500ms 硬顶从不额外触发
      槽位分桶 (08-31 起): 4 键 trial 段 slot 0/1/2 + 2 键角点段 slot 3 (独立桶)。
      返回 (keep 掩码, 段齐全的 trial 子集)。"""
    pos, tri = _seg_layout(data)
    keep = np.ones(len(tgt), dtype=bool)
    for p in np.unique(pos):
        idx = np.where(pos == p)[0]
        vals = tgt[idx]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        sigma = 1.4826 * mad if mad > 0 else 1.0
        keep[idx] = vals < med + k * sigma
    tri_ok = np.array([keep[tri == t].all() for t in range(len(data))])
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
    # M0: 段位 MAD 干净数据训练, 验证集选优 (与主训练同协议; 5 种子并行 09-09)
    outs = train_members([(s, prev[keep0], a[keep0], b[keep0], nxt[keep0], ph[keep0], tgt[keep0])
                          for s in range(m0_seeds)])
    best_va, M0 = float("inf"), None
    for m, _, va in outs:
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
    _, tri = _seg_layout(data)
    tri_ok = np.array([keep[tri == t].all() for t in range(len(data))])
    data_full = [d for d, ok in zip(data, tri_ok) if ok]
    return keep, data_full

def model_dphi(model):
    """模型的 φ 期望宽度 (用于构造查询特征; BlendModel 取成员; 未知则 D_PHI)。
    部署模型恒为 D_PHI=19; 该助手让查询侧在 φ15 对照/旧权重下也能工作。"""
    m = getattr(model, "members", [model])[0]
    return int(getattr(m, "d_phi", D_PHI))

def build_tensors(data):
    """全部段样本: (prev, a, b, nxt, phi15, target)。
    4 键 trial: 段 S(∅,a,b,c) / S(a,b,c,d) / S(b,c,d,∅) — 角色 (4,1)/(4,2)/(4,3);
    3 键 trial: 段 S(∅,a,b,c) / S(a,b,c,∅)           — 角色 (3,1)/(3,2);
    2 键 trial: 单段 S(∅,a,b,∅) = T₂ 角点            — 角色 (2,1)。
    φ19 = φ15 + 段位块 [i==2, i==3, L==3, L==4] (2026-09-21 段位特征落地)"""
    segs = []
    for code, ts in data:
        n = len(code)
        if n == 2:
            segs.append((EMPTY, code[0], code[1], EMPTY, ts[0], ROLE_ID[(2, 1)]))
            continue
        for i in range(n - 1):        # 段 i (0-based, 段位 = i+1): code[i]→code[i+1]
            pv = EMPTY if i == 0 else code[i - 1]
            nx = code[i + 2] if i + 2 < n else EMPTY      # 后继键 = 段末键的下一个
            t0 = 0.0 if i == 0 else ts[i - 1]
            segs.append((pv, code[i], code[i + 1], nx, ts[i] - t0, ROLE_ID[(n, i + 1)]))
    prev = np.array([s[0] if isinstance(s[0], int) else KEY_TO_IDX[s[0]] for s in segs], dtype=np.int64)
    a = np.array([KEY_TO_IDX[s[1]] for s in segs], dtype=np.int64)
    b = np.array([KEY_TO_IDX[s[2]] for s in segs], dtype=np.int64)
    nxt = np.array([s[3] if isinstance(s[3], int) else KEY_TO_IDX[s[3]] for s in segs], dtype=np.int64)
    ph8 = np.array([PHI[(s[1], s[2])] for s in segs], dtype=np.float32)
    role = np.array([s[5] for s in segs], dtype=np.int64)
    ph = np.concatenate([ph8, phi_suc(a, b, nxt), phi_role(role)], axis=1)
    tgt = np.array([s[4] for s in segs], dtype=np.float32)
    return prev, a, b, nxt, ph, tgt

# ═══════════════════ 训练 ═══════════════════

def train(model, prev, a, b, nxt, ph, tgt, epochs=200, seed=0, bs=256, lr=0.001,
          patience=60, loss_fn="mse"):
    """批量训练 + 早停 (80/20 随机划分), 模型原地更新, 返回 (全数据段MAE, 验证集段MAE)。
    验证集 MAE 用于 best-of-N 选优 (选最强模型必须用验证集, 全数据含训练集会偏向过拟合)。
    loss_fn: mse (主流程) | huber50 | l1 — 目标函数变体 (2026-08-18 自实验-结构消融.py
    合并双胞胎实现, 防口径漂移; 该脚本已删, 结论已存档)。
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
    """总时间预测 (ms, float64) — 每 trial 逐段一次批量前向 (码长 2/3/4 通用), 末段累计时。
    段角色随 (码长, 段位) 进入 φ19 (2026-09-21 段位特征落地); 2 键 = 单段角点。"""
    if not data:
        return np.zeros(0)
    ids, phs, nseg = [], [], []
    for code, _ in data:
        n = len(code)
        nseg.append(n - 1)
        for i in range(n - 1):
            pv = EMPTY if i == 0 else KEY_TO_IDX[code[i - 1]]
            nx = KEY_TO_IDX[code[i + 2]] if i + 2 < n else EMPTY
            ix, iy = KEY_TO_IDX[code[i]], KEY_TO_IDX[code[i + 1]]
            ids.append([pv, ix, iy, nx])
            phs.append(np.concatenate([PHI[(code[i], code[i + 1])],
                                       phi_suc([ix], [iy], [nx])[0],
                                       phi_role([ROLE_ID[(n, i + 1)]])[0]]))
    model.eval()
    ph = np.array(phs, dtype=np.float32)[:, :model_dphi(model)]
    with torch.no_grad():
        pred = model._batch(torch.tensor(ids), torch.tensor(ph)).numpy()
    tri_of = np.repeat(np.arange(len(data)), nseg)          # 段 → trial 归属求和
    return np.bincount(tri_of, weights=pred, minlength=len(data))

def eval_total(model, data):
    """总时间 MAE/R² (全数据; 码长 2/3/4 通用)。数值与逐 trial 逐段求和等价
    (2026-08-18 批量化: 4539 trial 8.4s → ~0.01s)。实测值取 trial 末段累计时
    (4 键 d_d / 3 键 c_d / 2 键 b_d)"""
    if not data:
        return float("nan"), float("nan")
    ys = np.array([ts[len(c) - 2] for c, ts in data])
    errs = np.abs(total_preds(model, data) - ys)
    mae = float(errs.mean())
    r2 = 1 - float(np.sum(errs**2)) / max(float(np.sum((ys-ys.mean())**2)), 1e-9)
    return mae, r2

# ═══════════════════ 导出 ═══════════════════

def build_seg_table(model, chunk=100_000):
    """按键时间母表 — 2026-09-21 段位特征落地后按**段角色分 6 片** (单一 4-D F 已不可能:
    同一 (p,a,b,n) 格在 3 键与 4 键码中角色不同, 段位块 φ19 使 S 依赖角色)。返回 dict:
      r_L2i1 (30,30)               T₂ 角点 S(∅,x,y,∅)
      r_L3i1 / r_L4i1 (30,30,30)   首段 S(∅,x,y,z)  (3 键 / 4 键各自角色)
      r_L3i2 / r_L4i3 (30,30,30)   末段 S(x,y,z,∅)  (3 键短尾 / 4 键长尾)
      r_L4i2 (30,30,30,30)         中段 S(w,x,y,z)
    共 918,900 条; 每片按自身索引维的字母序排列 (索引即该角色实际用到的键)。
    旧单表 864,900 条 = 4 键三段 (p,a,b,n) 网格 + 角点, 其中 (1,0) 格被 3/4 键共用——
    这正是段位特征要拆开的东西。"""
    n = len(LETTERS)
    out = {}
    model.eval()
    with torch.no_grad():
        for role, (nk, cell) in ROLE_CELL.items():
            L, i = role
            tup = np.array(list(itertools.product(range(n), repeat=nk)), dtype=np.int64)
            pv = np.empty(len(tup), np.int64); av = np.empty(len(tup), np.int64)
            bv = np.empty(len(tup), np.int64); nv = np.empty(len(tup), np.int64)
            for r, t in enumerate(tup):
                p_, a_, b_, n_ = cell(t)
                pv[r], av[r], bv[r], nv[r] = p_, a_, b_, n_
            rid = np.full(len(tup), ROLE_ID[role], np.int64)
            ph = np.concatenate([PHI_MAT[av, bv], phi_suc(av, bv, nv), phi_role(rid)], axis=1)
            pred = np.empty(len(tup), np.float32)
            phw = ph[:, :model_dphi(model)]        # 按模型宽度取 (部署恒 19)
            for s in range(0, len(tup), chunk):
                ids = torch.tensor(np.stack([pv[s:s+chunk], av[s:s+chunk],
                                             bv[s:s+chunk], nv[s:s+chunk]], axis=1))
                pred[s:s+chunk] = model._batch(ids, torch.tensor(phw[s:s+chunk])).numpy()
            out[f"r_L{L}i{i}"] = pred.reshape((n,) * nk)
    return out

def _bigram_ms(model):
    """T₂ 角点 B[a,b] = S(∅,a,b,∅) ms (词末两键, 未归一化), 形状 (30,30)。
    角点角色 (2,1) 有 2 键实测训练样本 (08-31 起), 属内插而非外推"""
    n = len(LETTERS)
    ids = torch.tensor([[EMPTY, a, b, EMPTY] for a in range(n) for b in range(n)])
    ar = np.arange(n).repeat(n); br = np.tile(np.arange(n), n)
    rid = np.full(n * n, ROLE_ID[(2, 1)], np.int64)
    phs = np.concatenate([PHI_MAT[ar, br],
                          phi_suc(ar, br, np.full(n * n, EMPTY)),
                          phi_role(rid)], axis=1)
    with torch.no_grad():
        return model._batch(ids, torch.tensor(phs[:, :model_dphi(model)])).numpy().reshape(n, n)

def load_chen():
    """陈一凡 1986 两键当量表 txt (键对\t当量, 900 行定向) → {(a,b): 当量}。
    来源 2026-09-09 自极速赛码器 击键当量.xls 一次性导出 (本表随仓库分发,
    对比自此内嵌主流程——原 对比-条件vs陈表.py 与 评估-段表.npz 中间产物均废)"""
    d = {}
    for line in CHEN_TXT.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            ab, v = line.split("\t")
            d[(ab[0], ab[1])] = float(v)
    return d

def chen_comparison(m, pools, test_full):
    """陈一凡表对比 (README §5.3): [1] 4 码累加 (陈表 ×k₄) + [2] 2 码角点 (×k₂),
    陈表两项各自独立比例缩放。2026-09-09 内嵌主流程 (用户决策): 直接用评估模型
    in-memory (T₄ = total_preds / 角点 = B 切片) — 无重训、无中间产物, 与 §5.2
    主指标严格同一模型。"""
    from collections import defaultdict
    from scipy.stats import spearmanr
    CYv = np.array([[load_chen()[(x, y)] for y in LETTERS] for x in LETTERS])
    KI = KEY_TO_IDX

    ys = np.array([ts[2] for _, ts in test_full])
    p_cond = total_preds(m, test_full)
    p_chen = np.array([CYv[KI[c[0]], KI[c[1]]] + CYv[KI[c[1]], KI[c[2]]] + CYv[KI[c[2]], KI[c[3]]]
                       for c, _ in test_full], dtype=np.float64)
    print(f"\n=== 陈一凡表对比 [1] 4 码累加 (保留口径, n={len(test_full)} trial) ===")
    print(f"  {'方法':12s} {'MAE':>7s}ms {'×k':>8s} {'scale后MAE':>10s} {'scale后R²':>9s} {'排名':>7s}")
    for name, p in (("条件段模型", p_cond), ("陈一凡累加", p_chen)):
        k = float((ys * p).sum() / (p ** 2).sum())
        errs = np.abs(p * k - ys)
        r2s = 1 - float(np.sum(errs ** 2)) / max(float(np.sum((ys - ys.mean()) ** 2)), 1e-9)
        sp, _ = spearmanr(p, ys)
        print(f"  {name:12s} {np.abs(p - ys).mean():7.1f} {k:8.3f} {errs.mean():10.1f} {r2s:+9.3f} {sp:7.4f}")

    pv = defaultdict(list)
    for code, ts in pools["test2"]:
        pv[(KI[code[0]], KI[code[1]])].append(ts[0])
    B = _bigram_ms(m)                     # 角点切片 S(∅,a,b,∅)
    samples = []                          # (实测, 模型预测, 陈表值)
    for (i, j), v in pv.items():
        v = np.array(v)
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        sig = 1.4826 * mad if mad > 0 else 1.0
        c = v[v < med + 3 * sig]
        samples += [(t, B[i, j], CYv[i, j]) for t in c]
    samples = np.array(samples)
    y2, pc, pcy = samples[:, 0], samples[:, 1], samples[:, 2]
    k2 = float((y2 * pcy).sum() / (pcy ** 2).sum())
    print(f"=== 陈一凡表对比 [2] 2 码角点 (留出 trial 清洗后, n={len(y2)}) ===")
    for name, p, k in (("条件段模型(角点)", pc, None), ("陈一凡表", pcy, k2)):
        pp = p if k is None else p * k
        err = np.abs(pp - y2)
        sp, _ = spearmanr(p, y2)
        tag = "本征 ms" if k is None else f"×k₂={k:.1f}"
        print(f"  {name:14s}: MAE {err.mean():6.1f}ms (中位 {np.median(err):5.1f})  [{tag}]  排名 {sp:.4f}")

def export_seg_table(model, out_path, T=None):
    """按键时间母表 (npz) — 2026-09-21 起按段角色 6 片 (version 3; 旧版单表 F(31,30,30,31) version 2)。
    段角色 = (码长, 段位); 同一 (p,a,b,n) 格在 3 键与 4 键码中角色不同 (段位块 φ19 参与建模),
    故 4-D 单表无法表达 (与错误侧 2026-09-19 段类重参数化同因)。查询组合:
      T₂(xy)    = r_L2i1[x,y]
      T₃(xyz)   = r_L3i1[x,y,z] + r_L3i2[x,y,z]
      T₄(wxyz)  = r_L4i1[w,x,y] + r_L4i2[w,x,y,z] + r_L4i3[x,y,z]
    各片索引即该角色实际用到的键 (见 ROLE_CELL), 键序与 LETTERS 一致。
    T 可由调用方传入 (README 实证块复用, 免二次前向)。"""
    if T is None:
        T = build_seg_table(model)
    np.savez_compressed(out_path, **T,
                        letters=np.array(list(LETTERS)), empty=np.int64(EMPTY),
                        version=np.int64(3),
                        roles=np.array([f"L{L}i{i}" for L, i in ROLES]),
                        note=np.array("按键时间 S(段角色,p,a,b,n) ms; 段角色 = (码长,段位); "
                                      "T2=r_L2i1[x,y] T3=r_L3i1[x,y,z]+r_L3i2[x,y,z] "
                                      "T4=r_L4i1[w,x,y]+r_L4i2[w,x,y,z]+r_L4i3[x,y,z]; "
                                      "片索引即该角色用到的键序 (首段 a,b,n / 末段 p,a,b / 中段 p,a,b,n)"))
    tot = sum(v.size for v in T.values())
    print(f"  按键时间表: {out_path}  (6 角色片 {[v.shape for v in T.values()]}, {tot:,} 条, npz)")

def readme_evidence(deploy_all, dprev, da, db, dn, dph, dtgt, dkeep, T):
    """README 实证数字 (2026-09-18 起随 --full 输出; 原 实验/实验-README数字刷新.py 并入删除,
    同 §5.3 陈表对比的"独立脚本→内嵌"演进)。复用内存中的部署池 B4b 掩码与按键时间表 F, 零额外训练;
    每块首行即数字意义注记, 与 README 节号对应。§5.5 签名vs类盲在 分析-错误率.py。"""
    from collections import defaultdict
    from scipy.stats import spearmanr

    def ptype(a, b):
        if a == b: return "同键重复"
        if COL_TO_HAND[LETTER_TO_COL[a]] != COL_TO_HAND[LETTER_TO_COL[b]]: return "跨手"
        return "同指异键" if COL_TO_FINGER[LETTER_TO_COL[a]] == COL_TO_FINGER[LETTER_TO_COL[b]] else "同手异指"

    pos, _ = _seg_layout(deploy_all)
    four = np.isin(pos, [ROLE_ID[(4, 1)], ROLE_ID[(4, 2)], ROLE_ID[(4, 3)]])  # §5.4 语义 = 4 键段
    sel = dkeep & four
    print("\n=== README 实证数字 (复用部署 B4b 掩码与按键时间表, 零额外训练) ===")
    rows54 = defaultdict(lambda: defaultdict(list))
    for i in np.where(sel)[0]:
        rows54[ptype(LETTERS[da[i]], LETTERS[db[i]])]["空前" if pos[i] == ROLE_ID[(4, 1)] else "有前键"].append(dtgt[i])
    print("[§5.4 类型×前键] 段间隔中位 ms — 空前=词首段(静止启动) vs 有前键=词中段(手不回位), 差=前键条件效应")
    print(f"  {'类型':8s} {'空前(首段)':>10s} {'有前键':>8s} {'差':>7s}")
    for t in ("跨手", "同手异指", "同键重复", "同指异键"):
        m0 = np.median(rows54[t]["空前"]); m1 = np.median(rows54[t]["有前键"])
        print(f"  {t:8s} {m0:10.0f} {m1:8.0f} {m1-m0:+7.0f}   (n={len(rows54[t]['空前'])}/{len(rows54[t]['有前键'])})")
    v0 = [v for t in rows54 for v in rows54[t]["空前"]]
    v1 = [v for t in rows54 for v in rows54[t]["有前键"]]
    print(f"  {'混合平均':8s} {np.median(v0):10.0f} {np.median(v1):8.0f} {np.median(v1)-np.median(v0):+7.0f}")

    by_t = defaultdict(list)
    for i in np.where(sel)[0]:
        by_t[ptype(LETTERS[da[i]], LETTERS[db[i]])].append(dtgt[i])
    print("[键对难度参考] 同口径全段中位 ms (§5.5 末行): " + " < ".join(
        f"{t} {np.median(v):.0f}" for t, v in sorted(
            ((t, by_t[t]) for t in ("跨手", "同键重复", "同手异指", "同指异键")),
            key=lambda kv: np.median(kv[1]))))

    print("[模型视角] 部署表 S(∅,x,y) vs mean_p S(p,x,y) ms — 差=同键对『角点→有前键』的模型内抬升 (§5.4 末段)")
    for x, y in (("f", "i"), ("w", "v"), ("a", "z"), ("a", "a")):
        i, j = KEY_TO_IDX[x], KEY_TO_IDX[y]
        corner = float(T["r_L2i1"][i, j])                  # 角点角色 (2,1)
        withprev = float(np.mean(T["r_L4i2"][:, i, j, :]))   # 中段角色 (4,2), 对 p 与 n 求均值
        print(f"  {x}{y}: 角点 {corner:.0f} vs 有前键均值 {withprev:.0f}  差 {withprev-corner:+.0f}")

    print("[排序扭曲] 两键累加 vs 条件 T₄ (随机 2 万码) — 两键表不可修的结构性错位实证 (§5.4 末段)")
    rng = np.random.RandomState(7)
    codes = rng.randint(0, 30, size=(20000, 4))
    B = T["r_L2i1"]
    t4c = np.array([T["r_L4i1"][c, d, e] + T["r_L4i2"][c, d, e, g_] + T["r_L4i3"][d, e, g_]
                    for c, d, e, g_ in codes])
    t4t = np.array([B[c, d] + B[d, e] + B[e, g_] for c, d, e, g_ in codes])
    sp, _ = spearmanr(t4c, t4t)
    i1, i2 = rng.randint(0, 20000, 200000), rng.randint(0, 20000, 200000)
    m = i1 != i2
    dc, dt = t4c[i1[m]] - t4c[i2[m]], t4t[i1[m]] - t4t[i2[m]]
    nz = (dc != 0) & (dt != 0)
    print(f"  Spearman {sp:.4f}  不一致码对 {np.mean(np.sign(dc[nz]) != np.sign(dt[nz]))*100:.1f}%  "
          f"两键累加低估占比 {np.mean((t4t - t4c) < 0)*100:.1f}%")

    print("[§6.1 rollover] 释时全>0 无偏口径 — 重叠%=释时晚于后键按下 (串行假设的实证边界)")
    n_rows = 0; cnt = [0, 0, 0]; ov = [[], [], []]
    with open(DATA_TSV, encoding="utf-8") as fh:
        hdr = {h: i for i, h in enumerate(fh.readline().rstrip("\n").split("\t"))}
        for line in fh:
            row = line.rstrip("\n").split("\t")
            if len(row) < 12 or row[8] != "0" or len(row[hdr["code"]]) != 4:
                continue
            try:
                bd, cd, dd = float(row[hdr["b_d"]]), float(row[hdr["c_d"]]), float(row[hdr["d_d"]])
                au, bu, cu, du = (float(row[hdr[c_]]) for c_ in ("a_u", "b_u", "c_u", "d_u"))
            except ValueError:
                continue
            if min(au, bu, cu, du) <= 0:
                continue
            n_rows += 1
            for s, (rel, press) in enumerate(((au, bd), (bu, cd), (cu, dd))):
                if rel > press:
                    cnt[s] += 1; ov[s].append(rel - press)
    print(f"  样本 {n_rows} 行: " + "  ".join(
        f"{nm} 重叠 {cnt[s]/n_rows*100:.1f}% (幅值中位 {np.median(ov[s]) if ov[s] else 0:.0f}ms)"
        for s, nm in ((0, "首段"), (1, "中段"), (2, "尾段"))))

# ═══════════════════ 主流程 ═══════════════════

def main():
    ap = argparse.ArgumentParser(description="对称击键当量模型 v3 混合: 训练 + 测试集评估 + 可选导出")
    ap.add_argument("--full", action="store_true",
                    help="追加部署模型训练 (全数据) + 导出 4-D 按键时间表 (npz) 与模型")
    args = ap.parse_args()
    full = args.full
    DATA = DATA_TSV
    if not DATA.exists(): print(f"无数据: {DATA}"); sys.exit(1)

    print(f"加载: {DATA}")
    pools = stable_pools(str(DATA))
    data = pools["train_all"]            # 训练池 = 4 键 train + 3 键 train + 2 键角点 train (拼接序固定)
    train_data, test_data = pools["train4"], pools["test4"]
    print(f"4 键样本: {pools['desc']}")
    print(f"段样本: 训练池 {len(data)} trial → {3*len(pools['train4'])+2*len(pools['train3'])+len(pools['train2'])} 段")

    # ── 主口径划分: 稳定期内固定 trial 测试集 (4 键与 2 键各自 80/20, 同 RandomState(2024)
    #    顺序划分; 2 键测试 trial 不进训练 → 角点指标非循环) ──
    print(f"划分版本: 稳定期 4键 N={len(train_data)+len(test_data)}, 测试集 {len(test_data)} (20%), "
          f"+ 2键角点 训练 {len(pools['train2'])} / 留出 {len(pools['test2'])}, 种子 2024")
    print(f"训练 trial {len(train_data)}+{len(pools['train2'])}+{len(pools['train3'])} / "
          f"测试 trial {len(test_data)}+{len(pools['test3'])}+{len(pools['test2'])}")

    # ── 评估模型: 训练池 B4b (4键+2键角点) + v3 混合 (XGB 0.3 + deep2×5 固定种子平均;
    #    无 best-of 选优 → 无选优抽签噪声; 测试 trial 的段不进训练, 无泄漏) ──
    prev, a, b, nxt, ph, tgt = build_tensors(data)
    keep, _ = residual_filter_segments(data, prev, a, b, nxt, ph, tgt)
    print(f"训练池 B4b 剔除 (键对分桶带符号残差, 单侧上围栏 k=3): 保留 {int(keep.sum())}/{len(tgt)} 段 (4键+3键+2键角点)")
    prev, a, b, nxt, ph, tgt = prev[keep], a[keep], b[keep], nxt[keep], ph[keep], tgt[keep]

    print(f"\n=== 评估模型训练 (v3 混合: XGB {W_XGB} + deep2×{len(BlendModel.SEEDS)} 固定种子平均) ===")
    outs = train_members([(s, prev, a, b, nxt, ph, tgt) for s in BlendModel.SEEDS])
    for s, (_, seg_mae, va_mae) in zip(BlendModel.SEEDS, outs):
        print(f"  seed {s}: 段MAE={seg_mae:5.1f}ms  验证段MAE={va_mae:5.1f}ms")
    blend_m = BlendModel(train_xgb(prev, a, b, nxt, ph, tgt), [m for m, _, _ in outs])

    # ── 主指标: 保留口径 (2026-08-26 起默认只展示保留口径, 全口径需时另行说明) ──
    tp, ta, tb, tn, tph, tt = build_tensors(test_data)
    tkeep, test_full = residual_filter_segments(test_data, tp, ta, tb, tn, tph, tt)
    print(f"测试集 B4b: 保留 {int(tkeep.sum())}/{len(tt)} 段, 完整 trial {len(test_full)}/{len(test_data)}")
    seg_keep = eval_seg(blend_m, tp[tkeep], ta[tkeep], tb[tkeep], tn[tkeep], tph[tkeep], tt[tkeep])
    tot_keep, r2_keep = eval_total(blend_m, test_full)
    print(f"\n=== 主指标 (固定 trial 测试集 {len(test_data)}, 保留口径, v3 混合 {blend_m.nparam()} 参数×5+XGB) ===")
    print(f"保留口径: 段MAE={seg_keep:5.1f}  总MAE={tot_keep:5.1f}  R²={r2_keep:+.3f}")

    # ── T₂ 角点指标 (2 键留出 trial, 逐键对单侧 MAD 清洗后对比角点预测; 非循环) ──
    if pools["test2"]:
        from collections import defaultdict
        pv = defaultdict(list)
        for code, ts in pools["test2"]:
            pv[(KEY_TO_IDX[code[0]], KEY_TO_IDX[code[1]])].append(ts[0])
        B = _bigram_ms(blend_m)
        errs, diffs = [], []
        for (i, j), v in pv.items():
            v = np.array(v)
            med = np.median(v); mad = np.median(np.abs(v - med))
            sig = 1.4826 * mad if mad > 0 else 1.0
            c = v[v < med + 3 * sig]
            errs += list(np.abs(c - B[i, j])); diffs += list(c - B[i, j])
        errs, diffs = np.array(errs), np.array(diffs)
        print(f"角点指标 (T₂ 留出 n={len(errs)}): 段MAE={errs.mean():5.1f}  偏差={diffs.mean():+5.1f} (中位 {np.median(diffs):+.1f})")

    # ── T₃ 留出指标 (3 键 test3 = 20% 稳定期 3 键 trial; 09-21 段位落地后 3 键入训,
    #    test3 成干净留出 — 补齐旧管线「3 键无留出」的评估空位) ──
    if pools["test3"]:
        k3p, k3a, k3b, k3n, k3ph, k3t = build_tensors(pools["test3"])
        k3keep, test3_full = residual_filter_segments(pools["test3"], k3p, k3a, k3b, k3n, k3ph, k3t)
        if test3_full:
            mae3, r23 = eval_total(blend_m, test3_full)
            pred3 = total_preds(blend_m, test3_full)
            y3 = np.array([ts[1] for _, ts in test3_full])
            pos3, _ = _seg_layout(pools["test3"])
            ids3 = torch.tensor(np.stack([k3p[k3keep], k3a[k3keep], k3b[k3keep], k3n[k3keep]], axis=1))
            with torch.no_grad():
                seg3 = blend_m._batch(ids3, torch.tensor(k3ph[k3keep])).numpy()
            b3 = seg3 - k3t[k3keep]
            r3 = pos3[k3keep]
            print(f"T₃ 指标 (3 键留出 n={len(test3_full)}): 总MAE={mae3:5.1f}  "
                  f"偏差={float((pred3-y3).mean()):+5.1f} (中位 {np.median(pred3-y3):+.1f})  R²={r23:+.3f}")
            print(f"  段位偏差分解: 首段 L3i1 {b3[r3 == ROLE_ID[(3, 1)]].mean():+5.1f}ms "
                  f"(n={int((r3 == ROLE_ID[(3, 1)]).sum())})  |  "
                  f"短尾 L3i2 {b3[r3 == ROLE_ID[(3, 2)]].mean():+5.1f}ms "
                  f"(n={int((r3 == ROLE_ID[(3, 2)]).sum())})")

    # ── 陈一凡表对比 (内嵌 2026-09-09, README §5.3; 评估模型 in-memory, 无重训无中间产物) ──
    if CHEN_TXT.exists() and test_full:
        chen_comparison(blend_m, pools, test_full)
    elif not CHEN_TXT.exists():
        print("\n(陈一凡当量表.txt 缺失, 跳过陈表对比)")

    if not full:
        print("\n研究模式: 仅训练评估。加 --full 追加部署模型训练 (全数据) + 导出当量表。")
        return

    # ── 部署模型: 稳定期全量 B4b + v3 混合 (当量表条目质量优先; 评估指标以上方为准;
    #    2026-08-30 起部署池=稳定期, 08-31 起含 2 键角点全量 — 角点条目内插化;
    #    2026-09-02 起部署形态=XGB+deep2×5 混合, 固定种子无选优) ──
    print(f"\n=== 部署模型 (稳定期全量训练, v3 混合 — 导出用, 指标见上方评估模型) ===")
    deploy_all = pools["deploy4"] + pools["deploy3"] + pools["deploy2"]
    dprev, da, db, dn, dph, dtgt = build_tensors(deploy_all)
    dkeep, _ = residual_filter_segments(deploy_all, dprev, da, db, dn, dph, dtgt)
    print(f"全数据 B4b: 保留 {int(dkeep.sum())}/{len(dtgt)} 段  (稳定期全量, 4键+3键+2键角点)")
    ev_arrays = (dprev, da, db, dn, dph, dtgt, dkeep)   # README 实证块复用 (未过滤布局 + 掩码)
    dprev, da, db, dn, dph, dtgt = dprev[dkeep], da[dkeep], db[dkeep], dn[dkeep], dph[dkeep], dtgt[dkeep]
    douts = train_members([(s, dprev, da, db, dn, dph, dtgt) for s in BlendModel.SEEDS])
    for s, (_, seg_mae, va_mae) in zip(BlendModel.SEEDS, douts):
        print(f"  seed {s}: 段MAE={seg_mae:5.1f}ms  验证段MAE={va_mae:5.1f}ms")
    deploy_m = BlendModel(train_xgb(dprev, da, db, dn, dph, dtgt), [m for m, _, _ in douts])

    print("\n=== 导出 ===")
    ART_DIR.mkdir(exist_ok=True)
    deploy_m.save(str(MODEL_PT), str(MODEL_XGB))
    print(f"  模型 → 产物/按键时间模型-神经.pt (deep2×5 权重) + 产物/按键时间模型-xgb.json (XGB 分量; φ19 段位落地后与旧权重不兼容)")
    T = build_seg_table(deploy_m)
    export_seg_table(deploy_m, str(SEG_NPZ), T)
    readme_evidence(deploy_all, *ev_arrays, T)


if __name__ == "__main__":
    main()
