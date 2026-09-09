#!/usr/bin/env python3
"""
错误成本采集 — 连续长随机串 + 自然回退纠错 (2026-09-05)
目的: 实测当量修正项的错误成本 (现行 500ms 为经验取整; 停止范式分解 ~455ms 只覆盖立即发现)。

测量定义 (净成本语义, README §4.7/§5.5): 从首错按下 → 首错位置的正确字母被按下
  (位置正确——盲打字母中出现的同字母不算); 盲打字母/多次回退/过位回退均含在内。
  反事实世界里尾段以同节奏平移 → 原始区间本身就是净成本, 无需扣正常打字时间。
  连锁错规则 (用户决策 2026-09-05): episode 开启期间的所有错按 (同位二次错、
  连锁打到后续位置的错) 全部吸收为同一错误事件——错误只计首错、时长只记首错
  间隔; 假警报退格 (无开敞 episode) 只计数。

范式: 单一显示框逐字符着色——白=未打, 绿=打对, 红=打错, 黄底=当前位置 (键对键转录
  范式; 着色使错误发现基本为立即型 → 测短码语境口径)。退格修正; 全部打对才完成
  (末位打错不跳串, 悬置至退格改对——2026-09-07 修正: 此前末位打错直接完成, 该错误
  计入事件数但永远无时长=截断), 完成后随机暂停 (800-1500ms) 进入下一串。

记录 (2026-09-05 定稿, 用户决策"只记录错误键"): episode 判定在本工具实时完成,
  仅落盘错误事件 + 每串一行汇总 → 数据/错误成本-错误键.tsv
  列: session/stream/kind/pos/t_ms/dur_ms/got/want/prev/n_wrong/blind/n_bs/immediate
    kind=ep : pos=首错位置, t_ms=episode 起始(串内相对), dur_ms=首错间隔时长, got=首错
              实按键, want=目标键, prev=前键(pos=0 记"-"), n_wrong=episode 内错按次数
              (同位/连锁吸收), blind=首退格前盲打字母数, n_bs=期间退格数,
              immediate=1 若首错后首事件是退格
    kind=sum: 串汇总, pos=总按键, t_ms=串时长, dur_ms=错误事件数(只计首错),
              n_wrong=假警报退格数, blind=串尾未闭合(截断)episode 数
  (初版无反馈显示下的试采数据 2026-09-05 已删除——5 错 4 截断, 该口径不可用)
分析: 分析-错误成本.py 只做聚合统计 (episode 判定唯一源 = 本工具)。
反馈 (2026-09-06): 状态栏实时显示 本次打开已采集错误 episode 数 (episode 闭合写入即 +1,
  中途退出的截断 episode 不计) + 当前串速度 = 正确字母按键数 / 自本串首键累计时长
  (错按/退格不计字母但耗时计入 → 有效吞吐口径; ≥5 个正确字母起显示)。
"""
import sys, time, random
from pathlib import Path
import tkinter as tk

OUT = Path(__file__).resolve().parent / "数据" / "错误成本-错误键.tsv"
LETTERS = 'abcdefghijklmnopqrstuvwxyz;,./'  # 30 键 (与主采集一致)
STREAM_LEN = (120, 200)   # 每串字母数随机区间
PAUSE = (800, 1500)       # 串间暂停 ms
COLS = 48                 # 显示每行字符数 (等宽字体对齐)
HEADER = ["session", "stream", "kind", "pos", "t_ms", "dur_ms",
          "got", "want", "prev", "n_wrong", "blind", "n_bs", "immediate"]


def _wrap(s):
    return "\n".join(s[i:i + COLS] for i in range(0, len(s), COLS))


class CostCollector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("错误成本采集（连续输入 + 回退纠错）")
        self.root.geometry("980x600")
        self.root.configure(bg="#1e1e1e")
        self.root.protocol("WM_DELETE_WINDOW", self._finish)

        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self.stream_no = 0
        self.done_count = 0
        self.sess_ep_count = 0   # 本次打开累计已采集错误 episode 数 (闭合写入即 +1)
        self.stream = ""
        self.pos = 0
        self.typed = ""
        self.t0 = None
        self.phase = "idle"     # idle=串已显示未开打 | typing | done=串间暂停
        self.ep = None          # 当前活跃 episode (dict) 或 None; 连锁错吸收, 全程至多一个
        self.n_press = 0        # 本串按键计数 (汇总用)
        self.n_wrong = 0        # 本串错误事件数 (只计首错——episode 期间的连锁/同位错不计)
        self.n_false_bs = 0     # 本串假警报退格计数

        self.root.bind("<KeyPress>", self.on_key)

        # --- UI: 单一显示框 (目标串, 按状态着色) ---
        tk.Label(self.root, text="连续输入显示的字母串 · 打错自行退格修正 · Esc 退出",
                 font=("Consolas", 13), fg="#fff", bg="#1e1e1e").pack(pady=8)
        self.label_info = tk.Label(self.root, text="", font=("Consolas", 13),
                                   fg="#ccc", bg="#1e1e1e")
        self.label_info.pack(pady=2)

        self.text_disp = tk.Text(self.root, font=("Consolas", 19), bg="#111", fg="#fff",
                                 bd=0, height=5, wrap="none", state="disabled")
        self.text_disp.pack(padx=14, pady=12, fill="both", expand=True)
        self.text_disp.tag_config("ok", foreground="#0f0")     # 打对 → 绿
        self.text_disp.tag_config("bad", foreground="#f44")    # 打错 → 红
        self.text_disp.tag_config("cur", background="#e8c34a", foreground="#111")  # 当前位置

        self.label_hint = tk.Label(self.root,
                                   text="白=未打 · 绿=对 · 红=错 · 黄底=当前位置 · 尽可能快",
                                   font=("Consolas", 11), fg="#555", bg="#1e1e1e")
        self.label_hint.pack(pady=4)

        self._write_header()
        self._new_stream()
        self.root.focus_set()

    def _write_header(self):
        if not OUT.exists():
            with open(OUT, "a", encoding="utf-8") as f:
                f.write("\t".join(HEADER) + "\n")

    def _new_stream(self):
        L = random.randint(*STREAM_LEN)
        self.stream = "".join(random.choice(LETTERS) for _ in range(L))
        self.stream_no += 1
        self.pos = 0
        self.typed = ""
        self.t0 = None
        self.phase = "idle"
        self.ep = None
        self.n_press = self.n_wrong = self.n_false_bs = 0
        self.n_correct = 0        # 本串正确字母按键数 (速度显示分子)
        self.text_disp.config(height=-(-L // COLS))
        self._refresh()

    def _t(self):
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
            return 0.0
        return (now - self.t0) * 1000

    def on_key(self, event):
        if event.keysym == "Escape":
            self._finish()
            return
        if self.phase == "done":
            return

        if event.keysym == "BackSpace":
            if self.pos == 0:
                return
            t = self._t()
            if self.ep is not None:
                self.ep["nbs"] += 1
                self.ep["bs_seen"] = True
            else:
                self.n_false_bs += 1
            self.pos -= 1
            self.typed = self.typed[:-1]
            self.phase = "typing"
            self._bookkeep("b")
            self._refresh()
            return

        ch = event.char.lower()
        if not ch or ch not in LETTERS:
            return
        if self.pos >= len(self.stream):
            return                      # 末位打错悬置: 仅退格可继续 (打对则上一按已完成本串)
        t = self._t()
        self.n_press += 1
        if ch == self.stream[self.pos]:
            self.n_correct += 1                              # 速度分子: 正确字母 (错按/退格不计)
        opened = False
        if ch != self.stream[self.pos]:
            if self.ep is None:                          # 首错 → 开 episode; 连锁/同位错吸收
                self.ep = dict(pos=self.pos, start_t=t, got=ch, n_wrong=1,
                               blind=0, nbs=0, imm=None, bs_seen=False)
                opened = True
                self.n_wrong += 1                        # 错误只计首错
            else:
                self.ep["n_wrong"] += 1                  # 连锁/同位: 吸收, 不计错误不计时
        elif self.ep is not None and self.pos == self.ep["pos"]:
            ep = self.ep                                 # 锚位(首错位置)的正确字母 → 闭合
            self.ep = None
            ep["dur"] = t - ep["start_t"]
            ep["want"] = self.stream[self.pos]
            ep["prev"] = self.stream[self.pos - 1] if self.pos else "-"
            self._write_ep(ep)
        self.pos += 1
        self.typed += ch
        self.phase = "typing"
        self._bookkeep("p", opened)
        self._refresh()
        if self.pos >= len(self.stream) and self.typed == self.stream:
            self._complete_stream()     # 全部打对才完成; 末位(或吸收残留)打错 → 退格修正后再完成

    def _bookkeep(self, kind, opened=False):
        """episode 簿记: imm 由起始后第一个事件定 (跳过开场事件本身); 首 p 即首盲字母"""
        ep = self.ep
        if ep is None or opened:
            return
        if ep["imm"] is None:
            ep["imm"] = (kind == "b")
            if kind == "p":
                ep["blind"] = 1
        elif kind == "p" and not ep["bs_seen"]:
            ep["blind"] += 1

    def _write_ep(self, ep):
        self.sess_ep_count += 1
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(f"{self.session_id}\t{self.stream_no}\tep\t{ep['pos']}\t{ep['start_t']:.1f}\t"
                    f"{ep['dur']:.1f}\t{ep['got']}\t{ep['want']}\t{ep['prev']}\t"
                    f"{ep['n_wrong']}\t{ep['blind']}\t{ep['nbs']}\t{1 if ep['imm'] else 0}\n")

    def _write_sum(self):
        dur = (time.perf_counter() - self.t0) * 1000 if (self.t0 is not None and self.n_press) else 0.0
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(f"{self.session_id}\t{self.stream_no}\tsum\t{self.n_press}\t{dur:.1f}\t"
                    f"{self.n_wrong}\t-\t-\t-\t{self.n_false_bs}\t{1 if self.ep else 0}\t0\t0\n")

    def _complete_stream(self):
        self._write_sum()
        self.done_count += 1
        self.phase = "done"
        el = (time.perf_counter() - self.t0) if self.t0 is not None else 0.0
        spd = f"{self.n_correct / el:.1f}" if el > 0 else "—"
        self.label_info.config(
            text=f"✓ 完成第 {self.stream_no} 串（{len(self.stream)} 字母，{spd} 字母/s）· "
                 f"本次已采集错误 {self.sess_ep_count}")
        self.root.after(random.randint(*PAUSE), self._new_stream)

    def _finish(self):
        # 中途退出: 已开打的串写汇总行 (开敞 episode 计入截断)
        if self.phase == "typing" and self.n_press:
            self._write_sum()
        self.root.destroy()

    def _refresh(self):
        """单一显示框: 已打前缀逐字符着色 (绿=对/红=错), 白=未打, 黄底=当前位置"""
        self.text_disp.config(state="normal")
        self.text_disp.delete("1.0", "end")
        self.text_disp.insert("1.0", _wrap(self.stream))
        for tag in ("ok", "bad", "cur"):
            self.text_disp.tag_remove(tag, "1.0", "end")
        for i, ch in enumerate(self.typed):        # 不变量 len(typed)==pos
            idx = f"{i // COLS + 1}.{i % COLS}"
            self.text_disp.tag_add("ok" if ch == self.stream[i] else "bad", idx, idx + "+1c")
        if self.pos < len(self.stream):
            idx = f"{self.pos // COLS + 1}.{self.pos % COLS}"
            self.text_disp.tag_add("cur", idx, idx + "+1c")
        self.text_disp.config(state="disabled")

        if self.phase != "done":
            if self.t0 is not None and self.n_correct >= 5:   # 前 5 个正确字母内样本太短, 不显示
                spd = f"{self.n_correct / (time.perf_counter() - self.t0):.1f}"
            else:
                spd = "—"
            self.label_info.config(
                text=f"第 {self.stream_no} 串 · {len(self.stream)} 字母 · 进度 {self.pos}/{len(self.stream)} · "
                     f"已完成 {self.done_count} · 本次错误 {self.sess_ep_count} · 速度 {spd} 字母/s")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    CostCollector().run()
