#!/usr/bin/env python3
"""
错误率规律探索 + **(p,a,b,n) 条件逻辑回归**（2026-09-19 重参数化；2026-09-21 去交互项；
2026-10-07 同指块改「同指 + 同键」；**2026-10-10 改 (p,a,b,n) 条件化 + 按段角色 6 片导出**）。

**当前模型（2026-10-10 用户决策「把错误侧也改成 p,a,b,n 格式」；同日晚些落地单键块, 再按用户
「键特征 / 跨键特征 / 角色」三类重组并删 3 列冗余）**:
    P_err(p,a,b,n | 段角色) = σ( 键特征·b 块(4) + 跨键·pa(2) + 跨键·ab(5) + 跨键·bn(2)
                                 + 跨键·an(2) + 角色电平(3) )            ← 共 18 列 + 截距 = 19
    **19 列的分类清单（列序即此; 运行时打印同一张表）**:
      键特征（单键函数）: 键·p 0 列 | 键·a 0 列 | **键·b 4 列** | 键·n 0 列
          = b左小指 / b顶行 / b底行 / **食指(b)**  ← 4 列都只是 b 的函数（与 a 无关）
      跨键特征（键对函数）: 跨键·pa 2 | 跨键·ab 5 | 跨键·bn 2 | 跨键·pb 0 | 跨键·an 2
          pa = 同键(p,a)（保护项） / 同手(p,a)  ← **无「同指」**: (p,a) 平面上同指异键无抬升（见下）
          ab = 同手 / 同指 / 同键 / 小指-无名 / 小指-食指（2026-10-10 删 无名-中指）
          bn = 同键(b,n) / Fitts(b,n)（2026-10-10 删 同手(b,n)、同指(b,n) —— 块内共线, 由 Fitts 承载）
          an = 同手(a,n) / Fitts(a,n)
      角色 3 列: 有后键 β / 段位−1 δ / λ(3键首段)
    ⚠ **归类纠正**: `b左小指/b顶行/b底行` 旧名 `ab·b…` 是误标（三者只是 b 的函数, 用 ID(b) 回归
      R²=1.000000）; 归入「键特征·b」后**导出侧必须把这一组合成一个沿 b 轴的加性项**（原先寄生在
      ab 项里）—— 漏掉会被「导出片 4 点公式验证」当场抓住（2026-10-10 本轮就是这样抓到的）。
    段角色 = (码长, 段位) 6 个（与时间侧 S(p,a,b,n) 同格式）:
      L2i1 2键第1段 | L3i1 3键第1段 | L3i2 3键第2段 | L4i1 4键第1段 | L4i2 4键第2段 | L4i3 4键第3段
    电平由 3 参数给出（旧 5 段类的差别只在把 L3i1/L4i1 分开为 λ）:
      L2i1=0 | L3i1=β+λ | L4i1=β | L3i2=δ | L4i2=β+δ | L4i3=2δ
      λ = 3键首段 vs 4键首段（历史 z=−0.42/p=0.67 可合并；此处作角色格式的一部分显式入模并监控）
    **角色口径为何不用 [P空, N空, 是3键]**（2026-10-10 用户提议, 实验/实验-错误率角色口径对比.py）:
      该 3 指示与 6 角色**双射**（标签层成立）, 但作为线性主效应只有 4 个自由参数（角色均值空间
      5 维）⇒ 它不含**码内位置**信息, 强制 (L3i2−L4i3)=(L3i1−L4i1), 而数据是 −0.81 vs −0.17 logit
      ⇒ D=23.15（df=2, p=9.4e-06）显著不足、配对 ΔAUC −0.00384、角点 O/E 0.731→0.403 ⇒ 维持 β/δ/λ
      （对饱和 D=2.59, p=0.274 够用）。最省结构版 [有后键, 段位−1]（去 λ）也够用且 AIC 最低, 见 README。
    ∅ 约定: p/n 缺失（段位 1 / 码末）时其上下文特征取 0，且上下文特征只在**其有定义的子集内
      中心化**（全样本中心化会让零填充把条目均值塞进电平项 ⇒ β 与块共线、解释崩塌）。
    **落地依据（2026-10-10，实验/实验-错误率前后键上下文.py）**: 后键身份块 LRT D=63.85/df=6/
      p<1e-4、配对 ΔAUC +0.0118(t=5.42)、时间两批都显著(+0.0071/+0.0120)、session 分层
      [同手(b,n)] z=+3.34(44/58 场)、类灵活性混淆排除(−0.00028)；同键(p,a) 子集 z=−2.55/p=0.011。
      本轮同数据同事件 LRT: 旧形式 loglik −4120.6 → 新 −4085.4（Δ+35.1, df≈7, p≈1.3e-12）。
    **同手(p,a) 落地（2026-10-10，用户问「pab3键同手会导致错误率上升这一特征需要单独加吗」）**：
      2×2 分层 [同手(p,a)]×[同手(a,b)] = 3.54% / 1.84% / 2.93% / 1.77%（三键同手格最高）；
      session 分层 z=+5.43（48/62 场同向, 符号检验 p<1e-4）、时间两批同号（+0.89/+1.88pp）、
      入模 LRT D=5.17（df=1, p=0.023）、配对 ΔAUC +0.00074（t=+3.03，达 小指-食指 落地时同一口径）
      ⇒ **加主效应**。**「三键同手」交互项本身不加**：主效应已入模时它 Δloglik 增益恰为 0
      （两者都 +2.58）、单独用 ΔAUC 只有 +0.00016(t=0.65) ⇒ 冗余（线性模型里主效应已表达该现象）。
      依据: 实验/实验-错误率三键同手.py。
      ⚠ **落地曾受阻于"表结构"**：旧 900×5 键对×段类表没有 p/n 轴（09-02 测到 +0.008 却因
      "产品级摆动真实"未采纳）；本版改为与时间表同构的 6 角色片 ⇒ 组装侧可逐码取真实上下文。
    导出: 产物/错误率表.npz（6 角色片, r_L2i1[a,b] / r_L3i1·r_L4i1[a,b,n] / r_L3i2·r_L4i3[p,a,b] /
      r_L4i2[p,a,b,n], version 1）—— **取代** 旧的 产物/键对错误率表.txt（900×5）。
**历史模型（2026-09-19~10-09）**: P_err(a,b|段类) = σ(w·feat(a,b) + β·有后键 + δ·(段位−1))，
      段类 = (有前键, 有后键, 段位) 共 5 类:
        c0 2键第1段 (∅,∅,1) | c1 3·4键第1段 (∅,有,1) | c2 3键第2段 (有,∅,2)
        c3 4键第2段 (有,有,2) | c4 4键第3段 (有,∅,3)
      五类电平 (相对 c0): c1=+β, c2=+δ, c3=+β+δ, c4=+2δ  ← 纯加法形式
      **交互项 γ 已删 (2026-09-21 用户决策「λ 交互现在就可以简化掉」)**: γ(有后键×段位)
      在段类重参数化后的口径下历次都不显著 (−0.10±0.17, z=−0.58) → 删掉换 2 个自由度。
      本脚本每次运行**并排报一次带 γ 的拟合** (deviance + LRT)，作为「删得对不对」的常设监控
      （2026-10-10 起该监控保留, 另加「6 角色电平饱和 vs 3 参数」的监控②）。
      重参数化依据 (2026-09-19, 六格拟合): 旧四列签名在仅 2/4 键数据上与 (码长,段位)
      一一对应 (无损重参数化)，加入 3 键后「尾段」类被劈为 c2/c4——实测 1.64% vs 3.15%
      (z=−3.08, p=0.0020) → 旧模型显著拟合不足 (D=10.17→8.87, p=0.006→0.012)。
      **「有前键」不再单列**: 观测数据里 有前键 ⟺ 段位≥2 完全共线，旧模型的
      α=+2.10 实为位置效应——引入段位项后其系数塌缩为 0 (z=0.02)；改由
      δ·(段位−1) 承载「码内累积负担」，β 承载「末位保护」。
**同指块重参数化 (2026-10-07 用户决策「同指块改成同指加同键」)**: 键对表上恒有
      同指 = 同键 + 同指同列 + 同指双列 (逐键对成立: 900 键对里 126 = 30 + 60 + 36,
      三者互斥)，故给定 同指 后「同指同列」与「同键」是同一残差的两种写法——实测
      把 同指同列 换成 同键 时拟合概率最大差 8e-17、同指块外系数逐位相同；再删
      同指双列 只剩 {同指, 同键} 两列，拟合概率最大差 1.6e-4、loglik 差 2.3e-4，
      因为「同指同列」与「同指双列」两态的实测错误率本就分不开 (3.305% vs 3.242%,
      n=2905/1758)，而同键 (码内重字 0.199%, n=1507) 与它们差得远——同指块实际
      只需 2 自由度。依据: 实验/实验-同指块等价性.py。**反例警告**: 删 同指双列
      而不补 同键 不是等价改动 (那时 同键 与 同指双列 别名化, loglik 掉 26.6/df=1)。
**特征集冻结 + 小指-食指 落地 (2026-10-07 用户决策「只保留 8 个特征」→「生产改成 9 特征」)**:
      先删掉 8 vs 12 的自动选型、固定基线 (依据 实验/实验-错误率特征精简评估.py: 两集配对折
      AUC 差 t=+0.11 属折间噪声, 但旧 `>=` 规则会让选型随数据翻转、把产物推着走——实测段当量
      |Δ| 均值 0.3~1.8ms、总当量 T₄ 尾上 11ms; 且逐特征只有 同键/小指-无名·b左小指/b顶行/
      b底行 承载, 4 个文献特征逐个加进基线全不显著; 聚合错误成本对特征选择完全不敏感,
      任何单项删除对 T₂/T₃/T₄ ≤0.08ms)。同日再按消融结果把 **小指-食指** 加入部署 (8→9):
      [同手, 同指, 同键, 无名-中指, 小指-无名, 小指-食指, b左小指, b顶行, b底行]。
      **小指-食指依据** (实验/实验-错误率新特征消融.py, 10 个未测候选里唯一命中; 该候选 =
      同手且两指相隔 ≥3, 即中间隔着无名/中指): 原始率 **1.79%** (72 键对/3522 事件/63 错)
      vs 同手 2.46% = **−0.67pp** (z=−2.73); 入模 z=−2.18, **LRT 4.97** (df=1, p=0.026);
      配对折 **ΔAUC +0.00152** (t=+3.35, 5 CV 种子×5 折, 效应量是文献扩充 4 特征块整体的
      5 倍); **两批独立数据同号且效应量几乎一致** (按采集时间序对半切: 前半 −0.65pp/ΔAUC
      +0.00139, 后半 −0.69pp/+0.00159); **session 分层 z=−3.06** (每场内以同手率作期望,
      观察 63 错 vs 期望 91.9), ≥20 事件的 48 场里 38 场为保护方向 (符号检验 p<0.001)。
      它是**保护项**: 小指与食指无共享 extrinsic 肌腱行程 (机械耦合最弱), 而错误率最高的
      无名-中指 3.06% / 小指-无名 3.61% 恰是耦合最强的; 对照 小指-中指 2.54% 与
      中指-食指 2.44% 都不具备该保护 ⇒ 不是"任何跨指"也不是"任何含食指", 是 {小指,食指}
      本身。其余 9 项候选 (坏指对方向性/目标键身份/手别/起点键位置) 全不显著。
      **监控项**: 下次数据刷新复核 小指-食指 的符号是否保持。
事件口径 (2026-09-02 修正, 2026-09-19 起 3 键行入模): 2/3/4 键行全部参与;
      正确 trial 每键对 1 ok 事件; 错误 trial 在错键截断——错键对记 1 err、其前的
      键对记 ok (旧版把错键对同时记 ok+err, 双重计数 623 条, 已修); 首键即错不记
      事件 (键对未尝试)。仍用全部数据 (含练习期, 用户决策——错误事件稀疏)。
**全特征消融已做过 (2026-10-10) —— 特征空间饱和, 勿重复做**: 把时间侧 φ36 + 独热 122 +
      嵌入/双线性的每条对应物建成 17 族，一次测齐 (加一/替换、逐项删 21 列、全模型 LOO、
      纯几何梯度、身份块 + Rao 得分检验、时间两批 + 前向/反向 session 留出、产物影响)。
      19 个候选里**只有 1 个过全部门槛**: 「b 由食指敲击」1 列 (LRT D=18.50, df=1,
      p=1.7e-05; 配对 ΔAUC +0.00348/t=+5.20; ΔAIC −16.5; 四项复核全同号; 原始率 1.55%
      295/19040 vs 非食指 2.07%, session 分层 z=−3.13; 中指单项 z=−0.14、「食指或中指」
      反而反号 ⇒ 就是「食指」12 键本身) —— **2026-10-10 已落地**（用户指令「加入 b 食指」,
      见上方单键块注释与 README §4.6 落地读数）。
      两条**必须记住的读数**:
        ① **键身份的大效应是假象** —— ID(b) 相对「去掉 b 行/列信息的参照」ΔAUC +0.0203(t=13.8)
           四项复核全同号, 但相对**部署基线**只剩 +0.0018(t=2.18) 且时间两批(−0.0044/−0.0041)
           与前向/反向留出(−0.0040/−0.0001) **全部反号**。根因: 部署的 3 个 b 单向特征
           (b顶行/b底行/b左小指) **完全落在 ID(b) 张成内 (R²=1.000000)**，已捕获其 91%
           (0.0184/0.0203) ⇒ **凡含 ID(b) 的臂必须同步删掉这 3 列并以此作参照**，否则是
           「加 29 列 + 删 3 列」的混淆对比。同理任何**单键独热块与其单向属性特征严格共线**。
        ② **参照必须含完整的行信息** —— 若参照已删掉 b 的行特征, 「加 b 的行+列」会把
           「恢复行信息」误读成新增益 (实测 +0.016 里绝大部分是行信息, 真实新增益 ≈0)。
      其余判定: (a,b) 换成时间侧 φ8 原样 8 项 ΔAUC −0.0214(t=−12.4) ⇒ 错误侧的定向特征
      (b顶行/b底行/b左小指 + 3 个指对) 才是增益来源; 后键块补齐 4 项 +0.0002、an 补齐 6 项
      −0.0013、pa 补齐 6 项 −0.0007、**pb 全 8 项 +0.00006 (时间侧 8/8 LOO 全变差, 错误侧无效)**、
      pn −0.0025 (与时间侧 T8 同结论) 均不加; 角色电平 3 参数够用 (饱和 D=2.56 p=0.28)、
      λ 可删 (D=0.58 p=0.45)、存在(p) 无用 (D=0.12); 后键块里 **同手(b,n)/同指(b,n) 确认是
      死重** (D=0.03/0.01, 块内共线由 Fitts(b,n) 承载; 保留作 φsuc 格式对称)。
      Rao 得分检验 (本轮新增工具, 900 列满交互不必拟合): (a,b) S=1133.8/df=890、(b,n)
      1322.3/df=895 等 p 虽小, 但 **df 与错误事件量 (880) 同阶 ⇒ 要 ~9000 错误事件才有
      10 EPV** ⇒ 结构存在但当前不可落地。
      依据: 实验/实验-错误率全特征消融.py (输出 实验/_错误率全特征消融-输出.txt)。
**特征必要性审计 (2026-10-10 用户追问「现有特征都有必要吗」; 实验/实验-错误率特征必要性与pa同指.py)**:
      部署 21 列逐列 nested 删项 (似然 D/p 与排序 配对 ΔAUC 两轴): **14 列双显著必须保留**;
      2 列单轴显著 (ab·小指-无名 p=0.035 / bn·同手(a,n) p=0.078 但排序 t=−3.03) ⇒ 保留;
      **3 列无证据 ⇒ 可删** (ab·无名-中指 p=0.148、bn·同手(b,n) p=0.68、bn·同指(b,n) p=0.53)
      —— 三列一起删 Δloglik −1.28、ΔAIC −3.4、**ΔBIC −29.7**、配对 ΔAUC +0.0005 (t=+1.28)、
      角色校准 O/E 逐位不动 (L3i1 1.000→1.000) ⇒ **删之无害且更省参**, 但**未落地** (改表属
      口径变更, 留待用户决策); **λ L3i1** 单看也无证据 (p=0.44) 但它正是两个「第1段」角色之间的
      唯一旋钮 —— 删它会把 L3i1 的 O/E 从 1.00 拖到 0.86 ⇒ **保留** (它就是"两首段可合并"这条
      监控结论的载体, 不是冗余)。
**pa 跳过「同指」是合理的 (2026-10-10 用户追问)**: pa 平面与 ab 平面的**状态结构不同** ——
      (a,b) 上 同指异键 3.35% vs 同手异指 2.38% (**Δ+0.97pp, z=+3.45**, 真风险态);
      (p,a) 上 同指异键 2.70% vs 同手异指 2.83% (**Δ−0.12pp, z=−0.36**, 完全无抬升)
      ⇒ 把 ab 的 {同手, 同指, 同键} 基照搬到 pa 是错的。入模复核: +同指(p,a) Δloglik +0.01
      (p=0.90)、配对 ΔAUC −0.00024 (t=−3.70, 反而变差); 而 {同手, 同键} 两列各自有据
      (删 同键 D=4.98/p=0.0016、删 同手 D=2.62/p=0.022), 换成 {同手, 同指} 掉 3.97 loglik。
      机制: (a,b) 是段内相邻击键, 同指连击机械上互相牵制; (p,a) 跨段边界 (段间有停顿/回位),
      该耦合不再显现; 跨边界真正有信号的是 **同键** (1.30% vs 2.83%, 保护; 与 同键(a,b) 的
      0.19% 同向) 与 **同手** (风险, +0.088)。

用法: python 分析-错误率.py [数据.tsv] [--dual]
"""
import os, sys, csv
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import chi2, norm

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

# 键位特征 (9 特征, 2026-10-07 小指-食指 落地后; 沿革: 原 9 特征 2026-08-08 定稿 → 同指块
#   重参数化 8 → 小指-食指 9):
#   演进: 8 特征 (删列差 CD, 生理学无依据+零损失) → 7 特征 (删行差 RD,
#     与 BT/BB 冗余: 控制目标行后行差无梯度, 去 RD 0.606→0.603 持平)
#     → 9 特征 (坏指对 4 个替代邻近 ADJ+弱指 W, +b左小指, 0.606→0.635)
#     → 8 特征 (2026-10-07: 同指块由 [同指, 同指同列, 同指双列] 改为 [同指, 同键]。
#       四态实测: 同键 0.199% / 同指同列 3.305% / 同指双列 3.242% / 异指 1.764% ——
#       「同指异键」两态分不开、与 同键 差得远, 故同指块只需 2 自由度;
#       等价性证据见模块 docstring 与 实验/实验-同指块等价性.py)
#     → 9 特征 (2026-10-07: 加 小指-食指。同手 6 个指对里最后补齐的一块 —— 且是**保护项**:
#       原始 1.79% vs 同手 2.46%, 两独立批同号, session 分层 z=−3.06, 见 docstring)
#   坏指对 (同手数据): 无名-中指 3.85% 最高, 小指-无名 2.68% (原表另有 同指同列 3.17% /
#     同指双列 2.33%, 2026-10-07 并入同指块, 由 同指 + 同键 表达) —— 具体指对关系
#     替代几何"邻近" (跨手邻近=食指特殊性, 归入跨手 SH=0 基线)。
#   弱指不对称 (2026-08-08): a 小指无效应 (1.83 vs 1.91); b 小指左右不对称
#     (左 2.61% vs 右 1.35% 互相抵消) → 只保留 b 左小指特征 BLP。
#   行难度 (2026-08-08): 目标行绝对难度主导 (BT/BB, 同手跨行同幅度),
#     起点行/行移动无独立贡献。
def same_hand(a, b): return int(COL_TO_HAND[LETTER_TO_COL[a]] == COL_TO_HAND[LETTER_TO_COL[b]])
def same_finger(a, b): return int(COL_TO_FINGER[LETTER_TO_COL[a]] == COL_TO_FINGER[LETTER_TO_COL[b]])
def same_key(a, b): return int(a == b)   # 码内重字 (2026-10-07 入基线: 同指块第 2 自由度)
def _f(a): return COL_TO_FINGER[LETTER_TO_COL[a]]
def bad_samecol(a, b):   # B1 同指同列 (同列上下移动; 2026-10-07 起不入特征集, 仅探索分层用)
    return int(same_hand(a, b) and _f(a) == _f(b) and
               LETTER_TO_COL[a] == LETTER_TO_COL[b] and a != b)
def bad_ringmid(a, b):   # B2 无名-中指 (同手)
    return int(same_hand(a, b) and {_f(a), _f(b)} in ({1, 2}, {6, 5}))
def bad_pinkyring(a, b): # B3 小指-无名 (同手)
    return int(same_hand(a, b) and {_f(a), _f(b)} in ({0, 1}, {7, 6}))
def pinky_index(a, b):   # PI 小指-食指 (同手, 两指相隔 ≥3; 2026-10-07 落地, **保护项**)
    return int(same_hand(a, b) and {_f(a), _f(b)} in ({0, 3}, {4, 7}))
def bad_index2c(a, b):   # B4 同指双列 (食指内外横向; 2026-10-07 起不入特征集, 仅探索分层用)
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

# ── 特征清单（**2026-10-10 按用户口径分类**: 键特征 / 跨键特征 / 角色）──
#   **键特征**（单键函数）: 键·p 0 列 | 键·a 0 列 | **键·b 4 列** | 键·n 0 列
#   **跨键特征**（键对函数, 顺序 = pa ab bn pb an）:
#       跨键·pa 2 列（同键/同手(p,a)）| 跨键·ab 5 列（同手/同指/同键/小指-无名/小指-食指）
#       | 跨键·bn 2 列（同键/Fitts(b,n)）| 跨键·pb 0 列 | 跨键·an 2 列（同手/Fitts(a,n)）
#   **角色** 3 列（有后键 / 段位−1 / λ）
# 归类的意义: `b左小指/b顶行/b底行` 三者**只是 b 的函数**（与 a 无关）, 旧命名 `ab·b左小指`
#   把它们挂在 ab 名下是误标 —— 2026-10-10 起按「键特征·b」呈现（数学上不变, 只改归属表述）。
# **2026-10-10 删 3 列冗余**（依据 实验/实验-错误率特征必要性与pa同指.py）: `无名-中指`（跨键·ab,
#   p=0.148）+ `同手(b,n)`·`同指(b,n)`（跨键·bn, p=0.68/0.53）三列在似然与排序上**均无证据**
#   （一起删 Δloglik −1.28、ΔAIC −3.4、ΔBIC −29.7、配对 ΔAUC +0.0005、角色校准 O/E 逐位不动）;
#   后者两列是「块内共线、由 Fitts(b,n) 承载」的列（09-19 已文档化）。
# 跨键·ab 保留 5 列; 跨键·bn 由 4 列降为 2 列 ⇒ 后键块 6 → 4（bn 2 + an 2）。
FEATS_AB = [same_hand, same_finger, same_key, bad_pinkyring, pinky_index]      # 跨键·ab (5)
NAMES_AB = ["同手", "同指", "同键", "小指-无名", "小指-食指"]
FEATS_BB = [b_leftpinky, lambda a, b: top_row(b), lambda a, b: bottom_row(b)]  # 键特征·b 的粗粒度 3 项
NAMES_BB = ["b左小指", "b顶行", "b底行"]
# 监控用特征集（不参与选型）: 基线 8 特征（ab 5 + 键·b 3）vs 12 特征（+4 个文献特征）
FEATS_BASE = FEATS_AB + FEATS_BB
FEATS_NEW  = FEATS_BASE + [mirror_pair, cross_row_same_finger,
                           adjacent_finger, b_toward_thumb]
NAMES_BASE = NAMES_AB + NAMES_BB
NAMES_NEW  = NAMES_BASE + ["镜像手指", "跨行同指", "相邻手指", "b靠拇指侧"]

# ── 键对事件 (签名×段位口径, 2026-09-19) ─────────────
# 事件 = (a, b, ok, 段类); 段类由 (码长, 段位) 映射, 属性见 CLS_ATTR
SEGCLS = {2: {1: 0},                 # 2键第1段 (∅,∅,1)
          3: {1: 1, 2: 2},           # 3·4键第1段 (∅,有,1) | 3键第2段 (有,∅,2)
          4: {1: 1, 2: 3, 3: 4}}     # 首段 | 4键第2段 (有,有,2) | 4键第3段 (有,∅,3)
CLS_ATTR = {0: (0, 0, 1), 1: (0, 1, 1), 2: (1, 0, 2), 3: (1, 1, 2), 4: (1, 0, 3)}  # (有前键,有后键,段位)
CLS_NAME = {0: "2键第1段(∅,∅,1)", 1: "3·4键第1段(∅,有,1)", 2: "3键第2段(有,∅,2)",
            3: "4键第2段(有,有,2)", 4: "4键第3段(有,∅,3)"}   # 旧称: 角点/首段/短尾/中段/长尾
events = []
# 段角色 (码长, 段位) → 6 角色索引（顺序与 分析-按键时间.py ROLES 一致）。
# 2026-10-10 起错误侧**按段角色建模与导出**（与时间侧同格式 (p,a,b,n)）：
# 旧的 5 段类把 L3i1/L4i1 合成 c1（历史检验 p=0.67 可合并），现按角色分开、并把
# 3键首段 vs 4键首段 的差显式入模为 λ（监控项，不显著时两片接近但不强制相等）。
ROLE6 = {(2, 1): 0, (3, 1): 1, (3, 2): 2, (4, 1): 3, (4, 2): 4, (4, 3): 5}
ROLE6_NAME = ["L2i1 2键第1段", "L3i1 3键第1段", "L3i2 3键第2段",
              "L4i1 4键第1段", "L4i2 4键第2段", "L4i3 4键第3段"]


def add_events(code, act, ts_ok, sess):
    """事件 = (a, b, ok, 段类, session, p, n, 段角色)。
    p/n/角色 三字段 2026-10-10 追加 —— p = 该键对**前一键**身份（段位 1 时为 ""）、
    n = **后一键**身份（码末为 ""）、段角色 = (码长, 段位) → 6 角色索引。
    前 4 个字段与旧版完全一致（a,b,ok,段类），故既有实验脚本的解包只需补位；
    这也是**唯一的事件口径来源**（勿在别处重造事件）。"""
    L_ = len(code)
    cls = SEGCLS.get(L_)
    if cls is None: return
    def ev(i, ok_):
        return (code[i-1], code[i], ok_, cls[i], sess,
                code[i-2] if i >= 2 else "", code[i+1] if i + 1 < L_ else "", ROLE6[(L_, i)])
    if ts_ok:
        for i in range(1, L_):
            events.append(ev(i, 1))
        return
    pos = None
    for i, (c, a) in enumerate(zip(code, act)):
        if c != a: pos = i; break
    if pos is None or pos == 0: return   # 未定位 / 首键即错: 无键对事件
    for i in range(1, pos):              # 错键之前的键对 → ok (错键对只记 err, 修双重计数)
        events.append(ev(i, 1))
    events.append(ev(pos, 0))
    # 双向计数 (--dual): 错按键对 (目标首键,实际错键) 同记, 段类同错键位置;
    # 上下文仍取**目标码**的 p/n (错误只改 b, 前后键身份按该段在码中的位置定义)
    if DUAL and act[pos] in LETTERS and act[pos] != code[pos]:
        events.append((code[pos-1], act[pos], 0, cls[pos], sess,
                       code[pos-2] if pos >= 2 else "", code[pos+1] if pos + 1 < L_ else "",
                       ROLE6[(L_, pos)]))

rows = list(csv.DictReader(open(PATH, encoding="utf-8"), delimiter="\t"))
for r in rows:
    code = r["code"]
    if r["error"] == "0": add_events(code, code, True, r.get("session", ""))
    else: add_events(code, r["actual"], False, r.get("session", ""))

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
    for a, b, ok, _, _, _, _, _ in events:
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

# ── 2. 键对特征集: 固定 8 特征（2026-10-07 用户决策「只保留 8 个特征」）+ 12 特征监控 ──
print("\n=== 键对特征逻辑回归 (特征集选择) ===")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def feats(a, b, fs):
    return [1.0] + [f(a, b) for f in fs]

Y = np.array([1.0 if ok else 0.0 for _, _, ok, _, _, _, _, _ in events], dtype=np.float32)  # 1=正确

def fit_and_eval(fs, label):
    X = np.array([feats(a, b, fs) for a, b, _, _, _, _, _, _ in events], dtype=np.float32)
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

auc_base = fit_and_eval(FEATS_BASE, "基线 8 特征（跨键·ab 5 + 键·b 3）")
auc_new = fit_and_eval(FEATS_NEW, "文献扩充 12 特征（仅监控）")
# 2026-10-07 用户决策「只保留 8 个特征」: 删掉自动选型, 固定为基线。
# 依据: 8 vs 12 的 AUC 差历次 ≤0.002 且配对复核判为折间噪声 (同划分逐折配对,
# 5 个 CV 种子 t=+0.11), 但按旧 `>=` 规则会让选型随数据翻转 → 产物被掷硬币推动
# (实测段当量 |Δ| 均值 0.3~1.8ms、T₄ 尾上 11ms)。冻结后产物跨刷新不再摆动。
# 同日再把 小指-食指 加入基线 (8→9, 同样用户决策, 依据见 docstring) —— 基线此后
# 只按「消融 + 两独立批复核」逐项增补, 不再走 AUC 自动选型。
# **2026-10-10 删 无名-中指（9→8）, 同批删 同手(b,n)/同指(b,n)**: 三列在似然与排序上
# 均无证据（见文件头「特征清单」注释与 实验/实验-错误率特征必要性与pa同指.py）。
# 见 实验/实验-错误率特征精简评估.py、实验/实验-错误率新特征消融.py;
# 12 特征 AUC 保留为只报不选的常设监控。
FEATS_F, NAMES_F = FEATS_BASE, NAMES_BASE     # 兼容别名（= 监控基线集, 已非模型的 ab 块）
print(f"  → 固定 8 特征基线（2026-10-07 定、2026-10-10 去 无名-中指；扩充 12 特征 AUC {auc_new:.3f} "
      f"{'≥' if auc_new >= auc_base else '<'} 基线 {auc_base:.3f}，仅作监控不参与选型）")

# ── 3. 最终模型: **(p,a,b,n) 条件化** + 角色电平 (β, δ, λ)；按 键特征/跨键特征/角色 三类组织 ──
# 2026-10-10 用户决策「把错误侧也改成 p,a,b,n 格式」——与时间侧 S(p,a,b,n) 同格式:
#   电平: 6 段角色由 3 参数给出（与旧 5 段类的差别只在把 L3i1/L4i1 分开为 λ）:
#     L2i1 = 0 | L3i1 = β+λ | L4i1 = β | L3i2 = δ | L4i2 = β+δ | L4i3 = 2δ
#     λ = 3键首段 vs 4键首段（历史检验 z=−0.42 / p=0.67 可合并；此处作为角色格式的一部分
#     显式入模并作监控，不强制相等，预期很小）
#   特征: 9 个 (a,b) 键对特征（不变）+ **后键身份块 6**（φsuc 同源；2026-10-10 实测全过门槛：
#     LRT D=63.85/df=6/p<1e-4、配对 ΔAUC +0.0118(t=5.42)、时间两批都显著、session 分层 z=+3.34，
#     类灵活性混淆已排除 −0.00028；见 实验/实验-错误率前后键上下文.py）
#     + **同键(p,a)**（段首键与前一键重复；子集 z=−2.55/p=0.011，保护方向，与 同键(a,b) 同向）
#   ∅ 约定: p/n 缺失时其上下文特征取 0 —— 存在性由角色电平承载（有前键 ⟺ 段位≥2 完全共线）
print("\n=== (p,a,b,n) 条件模型: P_err = σ(w·feat(a,b) + w_n·后键身份 + w_p·同键(p,a) + 段角色电平) ===")
Yerr = 1.0 - Y                                   # 1=错误
CL = np.array([e[3] for e in events], dtype=int)
RL = np.array([e[7] for e in events], dtype=int)                  # 6 段角色
PV = np.array([CLS_ATTR[c_][0] for c_ in CL], dtype=float)        # 有前键 (与 段位≥2 完全共线, 不入模)
NX = np.array([CLS_ATTR[c_][1] for c_ in CL], dtype=float)        # 有后键
PS = np.array([CLS_ATTR[c_][2] - 1 for c_ in CL], dtype=float)    # 段位−1
L3I1 = (RL == 1).astype(float)                                    # λ 指示位 (3键首段)
A_ = [e[0] for e in events]; B_ = [e[1] for e in events]
P_ = [e[5] for e in events]; Nn_ = [e[6] for e in events]
# 跨键·ab (5): 全样本中心化（处处有定义）
FF = np.array([[f(a, b) for f in FEATS_AB] for a, b in zip(A_, B_)], dtype=float)
muF, sdF = FF.mean(0), FF.std(0) + 1e-12
FFs = (FF - muF) / sdF
# 键特征·b (3 个粗粒度): 只依赖 b, 处处有定义 ⇒ 全样本中心化
BB = np.array([[f(a, b) for f in FEATS_BB] for a, b in zip(A_, B_)], dtype=float)
muB, sdB = BB.mean(0), BB.std(0) + 1e-12
BBc = (BB - muB) / sdB


def _fitts(x, y):
    return float(np.log2(1 + np.sqrt((LETTER_TO_COL[x] - LETTER_TO_COL[y]) ** 2
                                     + (2 * (LETTER_TO_ROW[x] - LETTER_TO_ROW[y])) ** 2)))


# 跨键·bn (2) + 跨键·an (2)（旧称「后键身份块 6」，2026-10-10 删 bn 的 同手/同指 两冗余列）:
#   bn 族: 同键(b,n) Fitts(b,n)；an 族: 同手(a,n) Fitts(a,n)。n=∅ → 0
#   删掉的 同手(b,n)/同指(b,n): 似然 D=0.17/0.39（p=0.68/0.53）、块内共线由 Fitts(b,n) 承载。
NF_FUN = [("同键(b,n)", lambda a, b, n: 1.0 if (n and b == n) else 0.0),
          ("Fitts(b,n)", lambda a, b, n: _fitts(b, n) if n else 0.0),
          ("同手(a,n)", lambda a, b, n: same_hand(a, n) if n else 0),
          ("Fitts(a,n)", lambda a, b, n: _fitts(a, n) if n else 0.0)]
FN = np.array([[f(a, b, n) for _, f in NF_FUN] for a, b, n in zip(A_, B_, Nn_)], dtype=float)
# 跨键·pa (2): 同键(p,a)（段首键与前一键重复, 保护项） + 同手(p,a)（2026-10-10 落地:
# 依据 实验/实验-错误率三键同手.py —— 2×2 分层 同手(p,a)×同手(a,b) 显示三键同手格 3.54%、
# 其余 1.77~2.93%；session 分层 z=+5.43(48/62 场), LRT D=5.17/p=0.023, 配对 ΔAUC +0.00074
# (t=+3.03)、时间两批同号; **「三键同手」交互项本身不需要**（主效应已入模时它 Δloglik 增益
# 恰为 0、ΔAUC 更低）⇒ 只加主效应。∅ 与中心化约定同 bn/an 块）
# **为何 pa 没有「同指」**(2026-10-10 判定, 依据 实验/实验-错误率特征必要性与pa同指.py):
# (a,b) 平面 同指异键 3.35% vs 同手异指 2.38%（+0.97pp, z=+3.45, 真风险态）而 (p,a) 平面
# 2.70% vs 2.83%（−0.12pp, z=−0.36, 无抬升）⇒ 两平面状态结构不同, 照搬 ab 的基是错的;
# +同指(p,a) Δloglik +0.01（p=0.90）、配对 ΔAUC −0.00024（t=−3.70, 反而更差）。
PF_FUN = [("同键(p,a)", lambda p, a: 1.0 if (p and p == a) else 0.0),
          ("同手(p,a)", lambda p, a: same_hand(p, a) if p else 0.0)]
FP = np.array([[f(p, a) for _, f in PF_FUN] for p, a in zip(P_, A_)], dtype=float)
# **子集中心化**（2026-10-10）: 后键块/同键(pa) 只在其有定义处（n/p 存在）中心化, 缺失处仍取 0。
# 若改用全样本中心化, 零填充会把条目均值的偏移塞进电平项 ⇒ β 与块共线 (β 由 +0.397 跳 +0.890、
# SE 0.111→0.369, 解释崩塌)。子集中心化后电平 = 「该角色在平均上下文下的水平」, 与块正交 ⇒
# β/δ 保持原解释（这与 小指-食指 等 (a,b) 特征的全样本中心化不同 —— 那些特征处处有定义）。
mask_n = np.array([n != "" for n in Nn_])
mask_p = np.array([p != "" for p in P_])
muN, sdN = FN[mask_n].mean(0), FN[mask_n].std(0) + 1e-12
muP, sdP = FP[mask_p].mean(0), FP[mask_p].std(0) + 1e-12
FNc = np.where(mask_n[:, None], (FN - muN) / sdN, 0.0)
FPc = np.where(mask_p[:, None], (FP - muP) / sdP, 0.0)
# 单键块 (1): **目标键 b 由食指敲**（2026-10-10 落地; 依据 实验/实验-错误率全特征消融.py）
# 这是错误侧「单键属性」里唯一过全部门槛的项，也是对照时间侧键嵌入/独热后剩下的唯一增量：
#   入模 z=−4.27、LRT D=18.50（df=1, p=1.7e-05）、配对 ΔAUC +0.00348（t=+5.20, 5 种子×5 折）、
#   ΔAIC −16.5（该轮全部候选最佳）; 稳健性四项全同号 —— 时间前半 +0.0016 / 后半 +0.0030、
#   前向 session 留出 +0.0040 / 反向 +0.0036; 原始率 1.55%（295/19040）vs 非食指 2.07%,
#   session 分层 z=−3.13。**它是「食指」（左 r/t/f/g/v/b + 右 y/u/h/j/n/m 共 12 键）本身**:
#   中指单独 z=−0.14、「食指或中指」1 列反而四项反号; 也不是频次代理（加 log 频次(b) 后
#   本项 z 仍 −3.67）。与已部署的 小指-食指（z=−2.48, 同为保护方向）同向 —— 生理读法:
#   食指无共享 extrinsic 肌腱行程且灵活性最高。
#   **归入「键特征·b」而非 (a,b) 特征**(2026-10-10 用户口径): 它是 b 的单向属性 —— 与
#   b左小指/b顶行/b底行 同类（那三列旧名 `ab·b左小指` 是误标, 数学上只是 b 的函数）;
#   若要补 b 的键身份分辨率, 应改用 b 的 29 列独热, 但那相对部署基线只剩 +0.0018（t=2.18）
#   且两批/留出全部反号 ⇒ 键身份整体不成立, 但「食指」这 1 列成立（详见 README §4.6）。
#   全样本中心化（b 恒为真实键, 该特征处处有定义 —— 与 bn/an/pa 块的子集中心化不同）。
FING_B = np.array([COL_TO_FINGER[LETTER_TO_COL[b]] for b in B_], dtype=float)
FI = np.isin(FING_B, [3, 4]).astype(float)[:, None]      # 食指: 左 3/4 列(r t f g v b) + 右 5/6 列(y u h j n m)
muI, sdI = FI.mean(0), FI.std(0) + 1e-12
FIc = (FI - muI) / sdI
# ── 设计矩阵: 按「键特征 / 跨键特征 / 角色」三类组织（2026-10-10 用户口径）──
#   列序 = [截距] + 键·b(4) + 跨键·pa(2) + 跨键·ab(5) + 跨键·bn(2) + 跨键·an(2) + 角色(3) = 19 列
#   （bn 与 an 相邻且 bn 在前 ⇒ 导出侧 w_n 仍是一段连续切片, N_B2N/N_A2N 照旧索引）
one = np.ones((N, 1))
_N_B2N = [i for i, (nm, _) in enumerate(NF_FUN) if "(b,n)" in nm]
_N_A2N = [i for i, (nm, _) in enumerate(NF_FUN) if "(a,n)" in nm]
assert len(_N_B2N) == 2 and len(_N_A2N) == 2, (_N_B2N, _N_A2N)
GROUPS = [("键特征·b", [f"键·b/{nm}" for nm in NAMES_BB] + ["键·b/食指(b)"],
           np.concatenate([BBc, FIc], 1)),
          ("跨键·pa", [f"跨键·pa/{nm}" for nm, _ in PF_FUN], FPc),
          ("跨键·ab", [f"跨键·ab/{nm}" for nm in NAMES_AB], FFs),
          ("跨键·bn", [f"跨键·bn/{NF_FUN[i][0]}" for i in _N_B2N], FNc[:, _N_B2N]),
          ("跨键·an", [f"跨键·an/{NF_FUN[i][0]}" for i in _N_A2N], FNc[:, _N_A2N]),
          ("角色", ["角色/β 有后键(NX)", "角色/δ 段位−1(PS)", "角色/λ 3键首段(L3i1)"],
           np.concatenate([NX[:, None], PS[:, None], L3I1[:, None]], 1))]
COLNAMES = ["截距"] + [nm for _, nms, _ in GROUPS for nm in nms]
X = np.concatenate([one] + [m for _, _, m in GROUPS], axis=1)
assert X.shape[1] == len(COLNAMES), (X.shape, len(COLNAMES))
IX = {nm: i for i, nm in enumerate(COLNAMES)}
I_NX, I_PS, I_LAM = IX["角色/β 有后键(NX)"], IX["角色/δ 段位−1(PS)"], IX["角色/λ 3键首段(L3i1)"]
I_FI = IX["键·b/食指(b)"]

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
b0, beta, delta, lam = b[0], b[I_NX], b[I_PS], b[I_LAM]
print(f"\n=== 特征清单（{X.shape[1]} 列 = 截距 + 键特征 {sum(1 for n in COLNAMES if n.startswith('键·'))} "
      f"+ 跨键特征 {sum(1 for n in COLNAMES if n.startswith('跨键·'))} "
      f"+ 角色 {sum(1 for n in COLNAMES if n.startswith('角色/'))}）===")
for gname, gnames, gmat in GROUPS:
    print(f"  {gname:10s} {len(gnames)} 列: " + ", ".join(n.split("/", 1)[1] for n in gnames))
print("  键特征按 p/a/b/n 分: p 0 / a 0 / **b %d** / n 0   "
      "跨键特征按 pa/ab/bn/pb/an 分: pa 2 / ab %d / bn 2 / pb 0 / an 2"
      % (len(NAMES_BB) + 1, len(NAMES_AB)))
print("\n=== 系数读数（按 键特征 / 跨键特征 / 角色 分组）===")
for gname, gnames, _ in GROUPS:
    if gname == "角色":
        continue
    print(f"  [{gname}] " + "  ".join(
        f"{n.split('/',1)[1]} {b[IX[n]]:+.3f}(z={b[IX[n]]/se[IX[n]]:+.2f})" for n in gnames))
print("  [角色] " + f"有后键 β {beta:+.3f}(z={beta/se[I_NX]:+.2f})  "
      f"段位 δ {delta:+.3f}(z={delta/se[I_PS]:+.2f})  "
      f"λ(L3i1−L4i1) {lam:+.3f}(z={lam/se[I_LAM]:+.2f}, p={2*norm.sf(abs(lam/se[I_LAM])):.3f})")
print(f"  键·b/食指(b) 的方向: {b[I_FI]:+.3f} = 保护（负 = 目标键由食指敲更不易错）")
# 常设监控①（旧监控项延续）: γ = 有后键×段位 —— 在 (p,a,b,n) 参数化下继续盯加法形式
Xg = np.concatenate([X, (NX * PS)[:, None]], axis=1)
bg, seg_, llg, _ = irls(Xg, Yerr)
gam, gam_se = bg[-1], seg_[-1]
print(f"  [监控①] 带交互 γ = {gam:+.3f} ± {gam_se:.3f} (z={gam/gam_se:+.2f})  D = {2*(llg-ll):.3f} "
      f"(df=1, p={2*norm.sf(abs(gam/gam_se)):.3f})  → "
      f"{'加法形式维持 (γ 不显著)' if abs(gam/gam_se) < 1.96 else '⚠ γ 转显著, 需复审模型形式'}")
# 常设监控②（新）: 6 角色电平饱和 (5 参数) vs 本模型 (3 参数 β/δ/λ) —— 电平参数化是否够用
DUM6 = np.zeros((N, 5))
for k in range(5):
    DUM6[:, k] = (RL == k + 1)
X_sat = np.concatenate([one, DUM6, FFs, FNc, FPc, BBc, FIc], axis=1)
b_sat, _, ll_sat, _ = irls(X_sat, Yerr)
D_sat = 2 * (ll_sat - ll)
print(f"  [监控②] 6 角色电平饱和 (5 参数) vs 本模型 (3 参数): D = {D_sat:.3f} (df=2, "
      f"p={chi2.sf(D_sat, 2):.4f})  → {'电平参数化够用' if chi2.sf(D_sat, 2) > 0.05 else '⚠ 需更自由的角色电平'}")

# 角色校准 (观察错误 / 模型期望)
print("\n角色校准 (观察错误 / 模型期望):")
for r in range(6):
    m = RL == r
    o, e_ = float(Yerr[m].sum()), float(p_err_hat[m].sum())
    print(f"  {ROLE6_NAME[r]:16s} n={int(m.sum()):6d}  obs={int(o):4d}  exp={e_:7.1f}  O/E = {o/e_:.2f}")

# ── 4. 导出: 错误率表.npz — **按段角色 6 片, (p,a,b,n) 条件化** ──
# 与 按键时间表.npz 完全同构（同样的 6 角色、同样的索引键序），故组装侧两表可直接相加:
#   r_L2i1[a,b] | r_L3i1[a,b,n] | r_L3i2[p,a,b] | r_L4i1[a,b,n] | r_L4i2[p,a,b,n] | r_L4i3[p,a,b]
# 每片的索引即该角色实际用到的键（全部为真实键, 无 ∅ —— 与时间侧 ROLE_CELL 一致）。
# 2026-10-10 起**取代** 键对错误率表.txt（900×5 的 (a,b)×段类 边缘视图）：那个视图对
# 3/4 键的段会把 p/n 身份抹掉, 无法与时间侧 S(p,a,b,n) 对齐; 组装改读本表后可逐码用真实上下文。
print("\n=== 导出 错误率表.npz (6 段角色片, (p,a,b,n) 条件化) ===")
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
LV6 = np.array([0.0, beta + lam, delta, beta, beta + delta, 2 * delta])   # 角色序 L2i1..L4i3
pairs = [(a, b) for a in LETTERS for b in LETTERS]
KI = {ch: i for i, ch in enumerate(LETTERS)}


def _ab_mat():                       # (5, 30, 30) 跨键·ab 特征查表
    M = np.zeros((len(FEATS_AB), 30, 30))
    for i, f in enumerate(FEATS_AB):
        for x in LETTERS:
            for y in LETTERS:
                M[i, LETTERS.index(x), LETTERS.index(y)] = f(x, y)
    return M


def _ctx_mats():
    """后键块分两族（索引轴不同, 勿混）: (b,n) 族 4 个（Mbn[i][b,n]）+ (a,n) 族 2 个（Man[j][a,n]）"""
    Mbn = np.zeros((len(N_B2N), 30, 30))
    for i, k in enumerate(N_B2N):
        f = NF_FUN[k][1]
        for x in LETTERS:
            for y in LETTERS:
                Mbn[i, LETTERS.index(x), LETTERS.index(y)] = f("a", x, y)   # b=x, n=y
    Man = np.zeros((len(N_A2N), 30, 30))
    for j, k in enumerate(N_A2N):
        f = NF_FUN[k][1]
        for x in LETTERS:
            for y in LETTERS:
                Man[j, LETTERS.index(x), LETTERS.index(y)] = f(x, "a", y)   # a=x, n=y
    Mpa_ = np.zeros((len(PF_FUN), 30, 30))
    for i, (_, f) in enumerate(PF_FUN):
        for x in LETTERS:
            for y in LETTERS:
                Mpa_[i, LETTERS.index(x), LETTERS.index(y)] = f(x, y)
    return Mbn, Man, Mpa_


# 跨键分两族（索引轴不同, 勿混）: bn 族 2 个 + an 族 2 个
N_B2N = [i for i, (nm, _) in enumerate(NF_FUN) if "(b,n)" in nm]
N_A2N = [i for i, (nm, _) in enumerate(NF_FUN) if "(a,n)" in nm]
assert len(N_B2N) == 2 and len(N_A2N) == 2, (N_B2N, N_A2N)
assert _N_B2N == N_B2N and _N_A2N == N_A2N, (_N_B2N, N_B2N, _N_A2N, N_A2N)
pairs = [(a, b) for a in LETTERS for b in LETTERS]
Mab = _ab_mat(); Mbn, Man, Mpa = _ctx_mats()
Zf = (Mab - muF.reshape(-1, 1, 1)) / sdF.reshape(-1, 1, 1)        # (5,30,30) 全样本标准化
Zn_b = (Mbn - muN[N_B2N].reshape(-1, 1, 1)) / sdN[N_B2N].reshape(-1, 1, 1)   # 子集中心化
Zn_a = (Man - muN[N_A2N].reshape(-1, 1, 1)) / sdN[N_A2N].reshape(-1, 1, 1)
Zp = (Mpa - muP.reshape(-1, 1, 1)) / sdP.reshape(-1, 1, 1)
I_AB, I_BN = IX["跨键·ab/" + NAMES_AB[0]], IX["跨键·bn/" + NF_FUN[_N_B2N[0]][0]]
I_AN, I_PA = IX["跨键·an/" + NF_FUN[_N_A2N[0]][0]], IX["跨键·pa/" + PF_FUN[0][0]]
w_ff = b[I_AB:I_AB + len(NAMES_AB)]               # 跨键·ab (5)
w_n = b[I_BN:I_BN + len(NF_FUN)]                  # 跨键·bn(2) + 跨键·an(2) —— 连续, 顺序同 NF_FUN
w_p = b[I_PA:I_PA + len(PF_FUN)]                  # 跨键·pa (2)
AB = np.einsum("i,ijk->jk", w_ff, Zf)             # (a,b) 键对项
BN = np.einsum("i,ijk->jk", w_n[N_B2N], Zn_b)     # (b,n) 项, 轴序 [b,n]
AN = np.einsum("i,ijk->jk", w_n[N_A2N], Zn_a)     # (a,n) 项, 轴序 [a,n]
PA = np.einsum("i,ijk->jk", w_p, Zp)              # (p,a) 项 = 同键(p,a) + 同手(p,a)
# 键特征·b (4 列) 全组: 只依赖 b ⇒ 合起来是一个沿 b 轴的加性项（30 维向量, 按字母序）。
# ⚠ 三者 (b左小指/b顶行/b底行) 原先寄生在 ab 项里（旧 FF 9 列含它们）, 2026-10-10 归到
#   「键特征·b」组后必须在此显式相加 —— 漏掉会被导出片的 4 点公式验证当场抓住。
FI_GRID = np.isin(np.array([COL_TO_FINGER[LETTER_TO_COL[c]] for c in LETTERS]), [3, 4]).astype(float)
BB_GRID = np.array([[f(LETTERS[0], y) for f in FEATS_BB] for y in LETTERS])   # (30,3) 与 a 无关
w_bb = b[IX["键·b/" + NAMES_BB[0]]:IX["键·b/" + NAMES_BB[0]] + len(NAMES_BB)]
BF = ((BB_GRID - muB) / sdB) @ w_bb + b[I_FI] * (FI_GRID - muI[0]) / sdI[0]   # (30,) 轴 = b


def role_err(role):
    """某角色的 P_err 片 —— 索引键序与时间侧 ROLE_CELL 一致:
    L2i1[a,b] / L3i1·L4i1[a,b,n] / L3i2·L4i3[p,a,b] / L4i2[p,a,b,n]（每片索引都是真实键, 无 ∅）。
    BF（食指(b) 项）沿 b 轴相加: [a,b]→轴1 / [a,b,n]→轴1 / [p,a,b]→轴2 / [p,a,b,n]→轴2"""
    lv = LV6[role]
    if role == 0:                                     # L2i1: (∅,a,b,∅)
        return sig(b0 + lv + AB + BF[None, :])
    if role in (1, 3):                                # L3i1 / L4i1: (∅,a,b,n)
        return sig(b0 + lv + AB[:, :, None]
                   + np.broadcast_to(BN[None, :, :], (30, 30, 30))       # (b,n) → 轴 1,2
                   + np.broadcast_to(AN[:, None, :], (30, 30, 30))       # (a,n) → 轴 0,2
                   + BF[None, :, None])
    if role in (2, 5):                                # L3i2 / L4i3: (p,a,b)
        return sig(b0 + lv + np.broadcast_to(AB[None, :, :], (30, 30, 30))
                   + np.broadcast_to(PA[:, :, None], (30, 30, 30))       # 同键(p,a)
                   + BF[None, None, :])
    return sig(b0 + lv + AB[None, :, :, None]                            # L4i2: (p,a,b,n)
               + np.broadcast_to(BN[None, None, :, :], (30, 30, 30, 30))   # (b,n) → 轴 2,3
               + np.broadcast_to(AN[None, :, None, :], (30, 30, 30, 30))   # (a,n) → 轴 1,3
               + np.broadcast_to(PA[:, :, None, None], (30, 30, 30, 30))   # (p,a) → 轴 0,1
               + BF[None, None, :, None])


ROLE6_SLICE = ["r_L2i1", "r_L3i1", "r_L3i2", "r_L4i1", "r_L4i2", "r_L4i3"]
SHAPE6 = {0: (30, 30), 1: (30, 30, 30), 2: (30, 30, 30), 3: (30, 30, 30), 4: (30, 30, 30, 30), 5: (30, 30, 30)}
E6 = {}
for r in range(6):
    E6[ROLE6_SLICE[r]] = role_err(r).astype(np.float32)
    assert E6[ROLE6_SLICE[r]].shape == SHAPE6[r], (r, E6[ROLE6_SLICE[r]].shape)
# 公式验证: 用还原尺度重算两个代表点（L2i1 与 L4i2）比对导出片
# 公式验证: 按**与事件同一套设计矩阵**逐项重算（含上下文项与角色电平指示）→ 与导出片比对
ROLE_IND = {0: (0.0, 0.0, 0.0), 1: (1.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0),
            3: (1.0, 0.0, 0.0), 4: (1.0, 1.0, 0.0), 5: (0.0, 2.0, 0.0)}   # (NX, 段位−1, L3i1)


def perr_ctx(p_, a_, b_, n_, role):
    """按**列名**组装一行（2026-10-10 起, 免于列序变动）：截距 + 键·b + 跨键 pa/ab/bn/an + 角色"""
    nx, ps_, l3 = ROLE_IND[role]
    v = {"截距": 1.0}
    for i, nm in enumerate(NAMES_BB):
        v[f"键·b/{nm}"] = (FEATS_BB[i](a_, b_) - muB[i]) / sdB[i]
    v["键·b/食指(b)"] = (float(COL_TO_FINGER[LETTER_TO_COL[b_]] in (3, 4)) - muI[0]) / sdI[0]
    for i, nm in enumerate(NAMES_AB):
        v[f"跨键·ab/{nm}"] = (FEATS_AB[i](a_, b_) - muF[i]) / sdF[i]
    for i, (nm, f) in enumerate(NF_FUN):
        v[f"跨键·{'bn' if '(b,n)' in nm else 'an'}/{nm}"] = \
            (f(a_, b_, n_) - muN[i]) / sdN[i] if n_ else 0.0
    for i, (nm, f) in enumerate(PF_FUN):
        v[f"跨键·pa/{nm}"] = (f(p_, a_) - muP[i]) / sdP[i] if p_ else 0.0
    v["角色/β 有后键(NX)"], v["角色/δ 段位−1(PS)"], v["角色/λ 3键首段(L3i1)"] = nx, ps_, l3
    return float(sig(np.array([v[nm] for nm in COLNAMES]) @ b))


print("\n公式验证 (同一设计矩阵重算 vs 导出片, 4 个代表上下文):")
for p_, a_, b_, n_, role in (("", "a", "b", "", 0), ("", "a", "b", "z", 1), ("w", "a", "b", "", 2),
                             ("w", "a", "b", "z", 4)):
    idx = tuple(KI[x] for x in (p_, a_, b_, n_) if x)
    pm = float(E6[ROLE6_SLICE[role]][idx])
    pf = perr_ctx(p_, a_, b_, n_, role)
    print(f"  (p={p_ or '∅'},a={a_},b={b_},n={n_ or '∅'}) {ROLE6_NAME[role]}: 重算 {pf:.6f}  "
          f"片值 {pm:.6f}  ({'✓' if abs(pf - pm) < 1e-6 else '✗'})")

# KJ_ERR_NOEXPORT=1 → 跳过产物写入（实验脚本 importlib 载入本模块时不再顺带重写产物）
OUT_NPZ = _DIR / "产物" / "错误率表.npz"
if os.environ.get("KJ_ERR_NOEXPORT", "0") == "1":
    print("\n  [KJ_ERR_NOEXPORT=1] 跳过 产物/错误率表.npz 写入")
else:
    (_DIR / "产物").mkdir(exist_ok=True)
    np.savez_compressed(OUT_NPZ, **E6, letters=np.array(list(LETTERS)), empty=np.int64(30),
                        version=np.int64(1), roles=np.array(ROLE6_SLICE),
                        note=np.array("P_err 按段角色 6 片, (p,a,b,n) 条件化; 索引键序 = 该角色实际用到的键 "
                                      "(L2i1[a,b] L3i1[a,b,n] L3i2[p,a,b] L4i1[a,b,n] L4i2[p,a,b,n] "
                                      "L4i3[p,a,b]); 与 按键时间表.npz 同构可逐片相加 (2026-10-10 起"
                                      "取代 键对错误率表.txt 的 900×5 边缘视图); 模型含 单键块"
                                      "「食指(b)」1 列 (2026-10-10 落地)"))
    print(f"\n  已导出 {OUT_NPZ.name}  ({sum(v.size for v in E6.values()):,} 条, version 1)")
print("  角色片均值 P_err: " + "  ".join(f"{ROLE6_NAME[r].split()[0]} {E6[ROLE6_SLICE[r]].mean()*100:.2f}%"
                                        for r in range(6)))
try:
    _ec = None
    for l in (_DIR / "产物" / "错误修正时间-实测值.txt").read_text(encoding="utf-8").splitlines():
        if l.startswith("cost_ms\t"):
            _ec = float(l.split("\t")[1]); break
    _m = lambda r: float(E6[ROLE6_SLICE[r]].mean())
    print(f"  平均错误成本 (×实测 {_ec:.0f}ms): T₂ = {_ec*_m(0):.2f}ms | "
          f"T₃ = {_ec*(_m(1)+_m(2)):.2f}ms | T₄ = {_ec*(_m(3)+_m(4)+_m(5)):.2f}ms")
except Exception:
    print("  平均错误成本: 未找到 产物/错误修正时间-实测值.txt (先跑 分析-错误修正时间.py)")
_flat = E6["r_L4i2"].reshape(-1)
_k = int(np.argmax(_flat))
_i = np.unravel_index(_k, (30, 30, 30, 30))
print(f"  极值: L4i2 最大 P_err 键对 {LETTERS[_i[1]]}{LETTERS[_i[2]]} (p={LETTERS[_i[0]]}, n={LETTERS[_i[3]]}) "
      f"= {_flat[_k]*100:.1f}%")

# ── 5. README §5.5 实证: 段角色 vs 类盲（同特征集但无角色电平）──
# 类盲 = 只有键对特征 + 后键身份 + 同键(p,a) 的逻辑回归; 差异 = 位置信息缺失对 T₂ 角点条目的系统性高估
_ec2 = None
try:
    for l in (_DIR / "产物" / "错误修正时间-实测值.txt").read_text(encoding="utf-8").splitlines():
        if l.startswith("cost_ms\t"):
            _ec2 = float(l.split("\t")[1]); break
except OSError:
    pass
if _ec2:
    # 类盲 = 无角色电平、无 p/n 上下文（= 跨键·ab + 键·b）; 与旧版 [one, FFs(9)] 同为 10 列
    b_bl, _, _, _ = irls(np.concatenate([one, FFs, BBc, FIc], axis=1), Yerr)
    _FI_PAIR = np.tile(np.array([[float(COL_TO_FINGER[LETTER_TO_COL[c]] in (3, 4))]
                                 for c in LETTERS]), (30, 1))     # 沿 a 广播（pairs 以 a 为外循环）
    XpF = np.concatenate([np.array([[f(x, y) for f in FEATS_AB + FEATS_BB]
                                    for x in LETTERS for y in LETTERS], dtype=float), _FI_PAIR], 1)
    XpFs = (XpF - np.concatenate([muF, muB, muI])) / np.concatenate([sdF, sdB, sdI])
    P_bl = sig(b_bl[0] + XpFs @ b_bl[1:])
    E0 = E6["r_L2i1"].reshape(-1)                    # 角点角色片 (a,b) 展平, 与 pairs 同序
    print("\n=== README §5.5: 段角色 vs 类盲 (类盲=无角色电平对照; 偏差与 P_err 成正比, 扭曲 2 键简码排序) ===")
    print(f"  角点类均值: 段角色 {E0.mean()*100:.2f}% vs 类盲 {P_bl.mean()*100:.2f}%  "
          f"→ 高估 {_ec2*(P_bl.mean()-E0.mean()):.1f}ms/条目")
    hi = np.argsort(E0)[-90:]
    print(f"  高错误对 (角点前 10%): 段角色 {E0[hi].mean()*100:.2f}% vs 类盲 {P_bl[hi].mean()*100:.2f}%  "
          f"→ 高估 {_ec2*(P_bl[hi].mean()-E0[hi].mean()):.1f}ms/条目")
    _m = lambda r: float(E6[ROLE6_SLICE[r]].mean())
    print(f"  平均错误成本: 类盲 T₂ {_ec2*P_bl.mean():.1f} / T₃ {2*_ec2*P_bl.mean():.1f} / "
          f"T₄ {3*_ec2*P_bl.mean():.1f}   段角色 T₂ {_ec2*_m(0):.1f} / T₃ {_ec2*(_m(1)+_m(2)):.1f} / "
          f"T₄ {_ec2*(_m(3)+_m(4)+_m(5)):.1f}")
else:
    print("\n(未找到 产物/错误修正时间-实测值.txt, 跳过 §5.5 类盲对比)")
