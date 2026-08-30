#!/usr/bin/env python3
"""
击键测速工具（通用）
- 2/4 键混采: 每试次以 P2 概率出 2 键码（其余 4 键），混入方式加权随机
- 多样性加权随机采样: 首键均匀随机, 后续键 w(b|a) ∝ 1/(n(a,b)+1)
  (欠样本键对优先——符号键 ;,./ 零样本时被自动补足; 样本量趋均衡后回归均匀随机;
   4 键与 2 键各用独立键对计数池: 4 键池喂 4 键段覆盖, 2 键池喂 T₂ 角点 900 键对覆盖)
- 2 键试次 (2026-08-29 增补, 服务对称段模型 T₂ 角点 S(∅,a,b,∅) 验证):
  记录行 code 长度 2, b_d=第 2 键时间, c_d/d_d=0 填充——下游 load_data 按 len(code)==4
  过滤, 旧流程不受影响
- 第一键按下开始计时，打完自动进入下一组
- 正确→记录数据，等末键释放后进入随机延迟
- 输错→记录错误（含实际输入），进入随机延迟
- 空格→丢弃当前组（不记录），进入随机延迟
- 试次结束统一延迟 300-800ms 随机（打断节律，防止预期性击键压缩当量差异）
- Esc→退出
- trial 序号在 session 内自增（疲劳建模用），退出重置
"""
import sys, time, random
from collections import defaultdict
from pathlib import Path
import tkinter as tk

OUT = Path("D:/OneDrive/typing/击键测速/击键测速数据.tsv")
LETTERS = 'abcdefghijklmnopqrstuvwxyz;,./'  # 30 键 (3 行 10 列完整 QWERTY)
P2 = 0.30  # 2 键试次概率 (加权混入)。~215 trial/session 下 30% ≈ 65 键对/session,
           # ~14 session 后 900 键对 ≈1 样本/对; 初期可临时调高 (如 0.5) 加速角点覆盖

HEADER = [
    "code",           # 显示的编码 (4 键 trial; 08-29 起混入 2 键)
    "b_d", "c_d", "d_d",           # 第 2/3/4 键按下时间 (ms from a_d); 2 键行 c_d/d_d=0
    "a_u", "b_u", "c_u", "d_u",    # 第 1/2/3/4 键放开时间 (ms from a_d); 2 键行 c_u/d_u=0
    "error",          # 0=正确, 1=错误
    "actual",         # 错误时记录实际按键序列
    "trial",          # 试次序号 (1-based, session 内自增)
    "session",        # 启动时间戳，区分不同次运行
]

def _load_pair_pool(path, code_len):
    """统计已有数据相邻键对样本数 → {键对: 计数}。
    code_len=4: t1/t2/t3 三段全算 (4 键池); code_len=2: 单段 (2 键池)"""
    counts = defaultdict(int)
    if not Path(path).exists():
        return counts
    with open(path, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        code_i = hdr.index("code") if "code" in hdr else 0
        for line in f:
            row = line.strip().split("\t")
            if len(row) <= code_i: continue
            code = row[code_i]
            if len(code) != code_len: continue
            for i in range(len(code) - 1):
                counts[code[i] + code[i+1]] += 1
    return counts

class SpeedTest:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("击键测速")
        self.root.geometry("550x420")
        self.root.configure(bg="#1e1e1e")

        self.total_trials = 0   # 总试次（含跳过，用于 block 计数）
        self.recorded = 0       # 已记录数（正确+错误）
        self.code = ""
        self.nkeys = 4          # 当前试次键数 (2 | 4)
        self.events = []        # [(type, key, ts_from_t0)]
        self.t0 = 0
        self.phase = "waiting"  # waiting | typing | blocked
        self._header_written = OUT.exists()
        self.pair_counts = _load_pair_pool(OUT, 4)   # 4 键欠样本加权池
        self.pair2_counts = _load_pair_pool(OUT, 2)  # 2 键欠样本加权池 (T₂ 角点覆盖)

        self.session_id = time.strftime("%Y%m%d-%H%M%S")

        self.root.bind("<KeyPress>", self.on_press)
        self.root.bind("<KeyRelease>", self.on_release)

        # --- UI ---
        self.label_title = tk.Label(self.root, text="击键测速（2/4 字母编码）",
                                     font=("Consolas", 14), fg="#fff", bg="#1e1e1e")
        self.label_title.pack(pady=10)

        self.label_code = tk.Label(self.root, text="",
                                    font=("Consolas", 48, "bold"), fg="#0f0", bg="#1e1e1e")
        self.label_code.pack(pady=15)

        self.label_typed = tk.Label(self.root, text="",
                                    font=("Consolas", 24), fg="#888", bg="#1e1e1e")
        self.label_typed.pack(pady=5)

        self.label_info = tk.Label(self.root, text="",
                                    font=("Consolas", 14), fg="#ccc", bg="#1e1e1e")
        self.label_info.pack(pady=5)

        self.label_count = tk.Label(self.root, text=f"已记录: {self.recorded}",
                                     font=("Consolas", 12), fg="#666", bg="#1e1e1e")
        self.label_count.pack(pady=5)

        self.label_block = tk.Label(self.root, text="",
                                    font=("Consolas", 10), fg="#555", bg="#1e1e1e")
        self.label_block.pack(pady=2)

        self.label_hint = tk.Label(self.root, text="输入显示的编码 | 空格跳过 | Esc 退出",
                                    font=("Consolas", 10), fg="#555", bg="#1e1e1e")
        self.label_hint.pack(pady=5)

        self._write_header()
        self.new_code()
        self.root.focus_set()

    def _write_header(self):
        if not self._header_written:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write("\t".join(HEADER) + "\n")
            self._header_written = True

    def _trial_num(self):
        """当前试次序号 (1-based)"""
        return self.total_trials + 1

    def new_code(self):
        # 混采: P2 概率 2 键 (T₂ 角点), 其余 4 键
        # 多样性加权采样: 首键均匀随机, 后续键 w(b|a) ∝ 1/(n(a,b)+1)
        # (4 键用 4 键池计数, 2 键用 2 键池计数——两池目标不同, 各自均衡)
        self.nkeys = 2 if random.random() < P2 else 4
        pool = self.pair2_counts if self.nkeys == 2 else self.pair_counts
        code = random.choice(LETTERS)
        for _ in range(self.nkeys - 1):
            weights = [1.0 / (pool.get(code[-1] + b, 0) + 1) for b in LETTERS]
            code += random.choices(LETTERS, weights=weights, k=1)[0]
        self.code = code
        self.phase = "waiting"
        self.events = []
        self.label_code.config(text=self.code, fg="#0f0")
        self.label_typed.config(text="")
        self.label_info.config(text="")
        self.label_block.config(text=f"Trial {self._trial_num()}")

    def on_press(self, event):
        # Esc 始终可用
        if event.keysym == 'Escape':
            self.root.quit()
            return

        # blocked / releasing 阶段：拦截所有输入（Esc 除外）
        if self.phase in ("blocked", "releasing"):
            return

        ch = event.char.lower()

        # 空格：丢弃当前组，不记录
        if event.keysym == 'space':
            if self.phase in ("waiting", "typing"):
                self.total_trials += 1
                self.label_info.config(text="␣ 已跳过")
                self.label_code.config(fg="#f44")
                self._schedule_next()
            return

        if not ch or ch not in LETTERS:
            return

        t = time.perf_counter()

        if self.phase == "waiting":
            self.phase = "typing"
            self.t0 = t

        self.events.append(('d', ch, t - self.t0))
        self.label_typed.config(text=self.label_typed.cget("text") + ch.upper())

        # 逐键校验：错第一键即停止
        downs = [e[1] for e in self.events if e[0] == 'd']
        idx = len(downs) - 1  # 当前键序号 (0-based)

        if downs[idx] != self.code[idx]:
            # 按错：立即拦截，不等全部键
            self._record(ok=False)
            self.label_info.config(text=f"✗ 期望 {self.code[idx].upper()}")
            self.label_code.config(fg="#f44")
            self._schedule_next()
        elif idx == self.nkeys - 1:
            # 末键正确：全部正确
            self.phase = "releasing"
            self.label_code.config(fg="#0f0")
            self.label_info.config(text="✓")
            self._check_all_released()
            self._release_timeout = self.root.after(500, self._force_advance)

    def on_release(self, event):
        ch = event.char.lower()
        if not ch or ch not in LETTERS:
            return
        t = time.perf_counter()
        if self.phase in ("typing", "blocked", "releasing"):
            self.events.append(('u', ch, t - self.t0))
            if self.phase == "releasing":
                self._check_all_released()

    def _record(self, ok):
        """ok=True: 正确试次; ok=False: 错误试次"""
        downs = [(e[1], e[2]) for e in self.events if e[0] == 'd']
        # 释放时间按 (键, 按下次序) FIFO 配对 — 同键连击的各次按下对应各自的释放
        # (v1 起的缺陷, 08-29 发现: 按键字符只留首次释放, 同字母位置共用首释时间,
        #  1342/1418 重复字母行受影响且早于自身按下时间; 游离释放=无对应按下, 忽略)
        pending, matched = {}, {}
        for e in self.events:
            if e[0] == 'd':
                pending.setdefault(e[1], []).append(e[2])
            else:  # 'u'
                q = pending.get(e[1])
                if q:
                    q.pop(0)
                    matched.setdefault(e[1], []).append(e[2])
        n = self.nkeys

        # 按下时间戳 (ms): 按先后顺序, 不足 4 位填 0 (文件格式固定 4 列)
        ts_d = [d[1] * 1000 for d in downs[:n]]
        while len(ts_d) < 4:
            ts_d.append(0.0)

        # 释放时间戳 (ms): 按 code 键位对应 (不足 4 键补空, 文件格式固定 4 列);
        # 同字母多次出现按出现次序取该键第 i 次释放
        if ok:
            keys = list(self.code)
        else:
            keys = [d[0] for d in downs[:n]]
        while len(keys) < 4:
            keys.append('')
        seen, ts_u = {}, []
        for ch in keys:
            o = seen.get(ch, 0); seen[ch] = o + 1
            rl = matched.get(ch, [])
            ts_u.append(rl[o] * 1000 if o < len(rl) else 0.0)

        actual = "".join(d[0] for d in downs) if not ok else ""
        tid = self._trial_num()

        self.recorded += 1

        # 显示
        if ok:
            parts = [f"t{i}={ts_d[i]-ts_d[i-1]:.0f}ms" for i in range(1, n)]
            self.label_info.config(text="  ".join(parts) + f"  T={ts_d[n-1]:.0f}ms  ✓")
        else:
            self.label_info.config(text=f"✗ 输入: {actual}")
        self.label_count.config(text=f"已记录: {self.recorded}")

        # 写入
        # code, b_d, c_d, d_d, a_u, b_u, c_u, d_u, error, actual, trial, session
        row = [
            self.code,
            f"{ts_d[1]:.1f}", f"{ts_d[2]:.1f}", f"{ts_d[3]:.1f}",
            f"{ts_u[0]:.1f}", f"{ts_u[1]:.1f}", f"{ts_u[2]:.1f}", f"{ts_u[3]:.1f}",
            "0" if ok else "1",
            actual,
            str(tid),
            self.session_id,
        ]
        with open(OUT, "a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")
        # 更新键对计数 (下次生成时欠样本优先; 2 键/4 键各自池)
        pool = self.pair2_counts if self.nkeys == 2 else self.pair_counts
        for i in range(self.nkeys - 1):
            pool[self.code[i] + self.code[i+1]] += 1
        self.total_trials += 1

    def _check_all_released(self):
        """检查是否全部键已释放，若是则记录并推进"""
        ups = [e[1] for e in self.events if e[0] == 'u']
        if len(ups) >= self.nkeys:
            if hasattr(self, '_release_timeout'):
                self.root.after_cancel(self._release_timeout)
                del self._release_timeout
            self._record(ok=True)
            self._schedule_next()

    def _force_advance(self):
        """超时兜底：释放不全也强制推进"""
        if self.phase == "releasing":
            self._record(ok=True)
            self._schedule_next()

    def _schedule_next(self, delay_ms=None):
        """统一试次结束→延迟→新码。
        延迟随机 300-800ms (打断节律, 防止预期性击键压缩当量差异)。
        期间 phase=blocked 拦截输入。"""
        if delay_ms is None:
            delay_ms = random.randint(300, 800)
        self.phase = "blocked"
        # 传方法本身而非调用 (带括号 = 立即同步执行 new_code, 试次间延迟失效:
        # 08-29 实测该回归使 时间显示被同帧清除 + 无节律间隔导致按键连打成卡壳)
        self.root.after(delay_ms, self.new_code)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    SpeedTest().run()
