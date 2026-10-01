#!/usr/bin/env python3
"""采集输入层.py — 两个采集脚本共用的**禁用输入法**层 (唯一来源; 2026-10-01)

问题: 输入法处于**中文态**时, 字母先被 IME 拿去组字 ⇒ tkinter 只收到 `keysym='??'` / `char=''`
      的 <KeyPress>, 采集脚本 (读 `event.char`) 判定"没有按键"; 而且 IME 会输出汉字、弹候选框。
      (同源项目的复现与三层修法见 `../并击击键当量/README.md` §4.1 与
       `../并击击键当量/实验/实验-IME按键捕获.py` —— 真实注入按键的实测。)

本模块 = 通用做法 (SDL / Chromium / Wine 都用 `ImmAssociateContext`; `ImmDisableIME` 是 MSDN 的
进程级方案):
  ③ `ime_disable_process()`: `ImmDisableIME(0)` —— **进程级**禁用 IME。
     MSDN 硬约束: 必须在**第一个顶层窗口收到 WM_CREATE 之前**调用 ⇒ 建 `tk.Tk()` 之前调;
     失败 (返回 False) 无害 —— ② 仍在。成功 ⇒ 本进程不加载输入法, 键直接到窗口。
  ② `ImeGuard(root)`: `ImmAssociateContext(hwnd, NULL)` —— 摘掉**本窗口**的输入上下文;
     窗口每次激活与每 2 s 重摘一次 (系统会在激活/切换输入法后重新关联)。
     注: tkinter 自带 IME 支持 (建窗/获焦时自己挂上下文) ⇒ ② 与 ③ 互补, 两个都要。

用法 (采集脚本里各 3 行):
    from 采集输入层 import ime_disable_process, ImeGuard
    ime_disable_process()          # 建窗前
    ...  self.root = tk.Tk()  ...
    self.ime_guard = ImeGuard(self.root)     # 建窗后; 退出时 .stop()

不做的事: 本模块**不装低层键盘钩子** (并击项目那层是为"IME 连键都看不到"的额外保证);
  实测 ②+③ 已足够让中文态下的字母以干净 `char` 到达 tkinter ⇒ 串击保持脚本简单、无全局钩子。
"""
import sys

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _imm32 = ctypes.WinDLL("imm32", use_last_error=True)

    _GA_ROOT = 2
    _user32.GetAncestor.restype = ctypes.c_void_p
    _user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    _imm32.ImmAssociateContext.restype = ctypes.c_void_p
    _imm32.ImmAssociateContext.argtypes = [wintypes.HWND, ctypes.c_void_p]
    _imm32.ImmDisableIME.restype = wintypes.BOOL
    _imm32.ImmDisableIME.argtypes = [wintypes.DWORD]

    def ime_disable_process():
        """进程级禁用 IME (`ImmDisableIME(0)`); 必须在建第一个顶层窗口**之前**调用 (MSDN)。
        返回 True/False (失败无害 —— ImeGuard 仍在)。"""
        return bool(_imm32.ImmDisableIME(0))

    def tk_toplevel(root):
        """tkinter 顶层窗口句柄 (winfo_id() 在 Windows 上返回内部子窗口; 顶层要 GA_ROOT)"""
        return _user32.GetAncestor(wintypes.HWND(root.winfo_id()), _GA_ROOT) or root.winfo_id()

    def ime_detach(root):
        """摘掉本窗口 (顶层 + 子窗口 + 本线程焦点窗口) 的输入上下文; 返回摘掉旧上下文的窗口数"""
        n = 0
        for h in {tk_toplevel(root), root.winfo_id(), _user32.GetFocus()}:
            if h and _imm32.ImmAssociateContext(wintypes.HWND(h), None):
                n += 1
        return n

    class ImeGuard:
        """窗口级 IME 摘除 (②): 建窗后调一次; 激活时与每 interval_ms 重摘一次"""

        def __init__(self, root, interval_ms=2000):
            self.root, self.interval_ms = root, interval_ms
            self.n_detached = ime_detach(root)
            self._alive = True
            root.bind("<FocusIn>", self._on_focus, add="+")   # add="+" 不动脚本原有绑定
            self._tick()

        def _on_focus(self, event=None):
            if self._alive:
                ime_detach(self.root)

        def _tick(self):
            if not self._alive:
                return
            ime_detach(self.root)
            self.root.after(self.interval_ms, self._tick)

        def stop(self):
            self._alive = False

    def status():
        """给采集脚本打印的一行状态"""
        return "输入法: 进程级 ImmDisableIME(0) 已调用 (见返回值) + 窗口级摘除已启用"

else:                                   # 非 Windows: 空实现 (IME 是 Windows 特性)
    def ime_disable_process():
        return False

    def tk_toplevel(root):
        return 0

    def ime_detach(root):
        return 0

    class ImeGuard:
        def __init__(self, root, interval_ms=2000):
            self.n_detached = 0
            self._alive = True

        def stop(self):
            self._alive = False

    def status():
        return "输入法: 非 Windows 平台, 无需处理"


if __name__ == "__main__":
    print(f"IS_WINDOWS = {IS_WINDOWS}")
    if IS_WINDOWS:
        print("ime_disable_process() →", ime_disable_process(), "(需在建窗前调用才为 True)")
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        g = ImeGuard(r)
        print("ImeGuard: 摘除窗口数 =", g.n_detached, "| tk_toplevel =", hex(tk_toplevel(r) or 0))
        g.stop(); r.destroy()
    print(status())
