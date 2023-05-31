"""

https://qiita.com/tortuepin/items/9ede6ca603ddc74f91ba

https://stackoverflow.com/questions/64035952/how-to-key-press-detection-on-a-linux-terminal-low-level-style-in-python
https://stackoverflow.com/questions/2520893/how-to-flush-the-input-stream
https://atsuoishimoto.hatenablog.com/entry/20110329/1301324988

pipのkeyboard:
ImportError: You must be root to use this library on linux.
    hmm...
"""

import atexit
import builtins
import dataclasses
import fcntl
import os
import pprint
import sys
import termios
import threading
import time
import tty
import unicodedata
from datetime import datetime
from pathlib import Path
from posix import times_result
from pprint import pprint
from typing import Dict, List, Optional

from mypython.pyutil import singleton_class
from mypython.terminal import Prompt


class Key:
    Ctrl_A = "\x01"
    Ctrl_B = "\x02"
    # Ctrl_C = "\x03" # just Ctrl + C
    Ctrl_D = "\x04"
    Ctrl_E = "\x05"
    Ctrl_F = "\x06"
    Ctrl_G = "\x07"
    Ctrl_H = "\x08"
    # Ctrl_I = "\t"  # == tab key
    # Ctrl_J = "\n"  # == new line
    Ctrl_K = "\x0b"
    Ctrl_L = "\x0c"
    # Ctrl_M = "\n"  # == new line
    Ctrl_N = "\x0e"
    Ctrl_O = "\x0f"
    Ctrl_P = "\x10"
    Ctrl_Q = "\x11"
    Ctrl_R = "\x12"
    Ctrl_S = "\x13"
    Ctrl_T = "\x14"
    Ctrl_U = "\x15"
    Ctrl_V = "\x16"
    Ctrl_W = "\x17"
    Ctrl_X = "\x18"
    Ctrl_Y = "\x19"
    Ctrl_Z = "\x1a"

    Ctrl_Alt_A = "\x1b\x01"
    Ctrl_Alt_B = "\x1b\x02"
    Ctrl_Alt_E = "\x1b\x05"

    # http://hp.vector.co.jp/authors/VA010562/xehelp/html/HID00000453.htm
    # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797

    ESC = "\x1b"

    Up = "\x1b[A"
    Down = "\x1b[B"
    Right = "\x1b[C"
    Left = "\x1b[D"

    F1 = "\x1bOP"
    F2 = "\x1bOQ"
    F3 = "\x1bOR"
    F4 = "\x1bOS"
    F5 = "\x1b[15~"
    F6 = "\x1b[17~"
    F7 = "\x1b[18~"
    F8 = "\x1b[19~"
    F9 = "\x1b[20~"
    F10 = "\x1b[21~"
    F12 = "\x1b[24~"

    Delete = "\x1b[3~"
    Backspace = "\x7f"


_stdin_fd = sys.stdin.fileno()
_init_attr = termios.tcgetattr(_stdin_fd)


@singleton_class
class KeyInput:
    def __init__(self) -> None:
        self._e = threading.Event()
        self._using_input = False
        self._ie = threading.Event()
        self._input_last_len = 0
        self._c = ""
        self._c_buf: List[str] = []

        self._pressed: Dict[str, bool] = {}

        ### last define ###
        self._t = threading.Thread(target=self._listener, daemon=True)
        self._t.start()

    def get(self, block=True) -> str:
        """
        Return:
            この関数が呼ばれる直前に押したキー if block is False
            この関数が呼ばれた直後に押したキー otherwise
        """

        if block:
            while True:
                self._e.wait()
                self._e.clear()
                c = self._get_str()
                if c != "":
                    break
        else:
            c = self._get_str()

        return c

    def is_pressed(self, key: str):
        return self._pressed.pop(key, False)

    def reset_cache(self):
        self._pressed = {}

    def input(self, prompt=""):
        """
        Blocking
        Do not use raw builtins.input
        """
        # _termios_restore()
        # print(prompt)
        # print(prompt, end="") # not working with end=""
        os.write(sys.stdout.fileno(), prompt.encode(errors="replace"))
        self._c_buf.clear()
        self._using_input = True
        # _attr(~termios.ICANON)
        self._ie.wait()
        self._ie.clear()
        text = self._get_str()
        # _attr(~termios.ICANON & ~termios.ECHO)
        self._using_input = False
        return text

    def wait_until(self, key: str):
        """
        c = keyinput.get(block=True)
        if c == "r":
            ...

        というように実装したなら、任意のキーが打ち込まれた時点でblockingが解除されるので注意
        """

        while True:
            self._e.wait()
            self._e.clear()
            c = self._get_str()
            if c == key:
                break
            elif c == "":
                continue
            else:
                # VSCode: "terminal.integrated.enableBell": true
                # Hmm...
                # sys.stdout.write("\a")
                print(c, "pressed")

    def _inner(self):
        """
        Blocking

        thread ok
        複数のスレッドで動かさないこと。

        this class handle:
        ✘ Ctrl + C
        ✔ Ctrl + D
        """

        try:
            # if self._using_input:
            #     _attr(~termios.ICANON)
            # else:
            #     _attr(~termios.ICANON & ~termios.ECHO)

            _attr(~termios.ICANON & ~termios.ECHO)
            c = os.read(_stdin_fd, 1024).decode(
                errors="replace"
            )  # 1024: support long Japanese until enter
            if self._using_input:
                _edit(self._c_buf, c)
            else:
                self._c_buf.append(c)

            if len(c) == 1:
                self._pressed[c] = True

        finally:
            _termios_restore()

    def _get_str(self):
        if len(self._c_buf) == 0:
            return ""
        else:
            if self._using_input:
                text = "".join(self._c_buf[:-1])  # remove last '\n'
                self._c_buf.clear()
                return text
            else:
                c = self._c_buf[-1]
                self._c_buf.clear()
                return c

    def _listener(self):
        while True:
            self._inner()
            if self._using_input:
                if self._c_buf[-1] == "\n":
                    self._ie.set()
            else:
                self._e.set()

    @staticmethod
    def reset():
        _termios_restore()


# # # new[3]はlflags
# # ICANON(カノニカルモードのフラグ)を外す
# new[3] &= ~termios.ICANON
# # ECHO(入力された文字を表示するか否かのフラグ)を外す
# new[3] &= ~termios.ECHO  # 文字を表示しない
# self._new_attr = new


def _attr(lflags: Optional[int] = None):
    _new = _init_attr.copy()
    if lflags is not None:
        _new[3] &= lflags
    termios.tcsetattr(_stdin_fd, termios.TCSANOW, _new)


@atexit.register
def _termios_restore():
    # thread を併用した場合、ctrl+cをすると次にターミナル使うことが不可能になる問題の回避
    # https://qiita.com/qualitia_cdev/items/f536002791671c6238e3
    termios.tcsetattr(_stdin_fd, termios.TCSANOW, _init_attr)


# fl = fcntl.fcntl(sys.stdout.fileno(), fcntl.F_GETFL)
# fl = fl | os.O_NONBLOCK
# fcntl.fcntl(_stdin_fd, fcntl.F_SETFL, fl)


def _edit(buf: List[str], key: str) -> None:
    # TODO

    if key == Key.Backspace:
        # print(Prompt.cursor_left(1), end=)
        # os.write(_stdin_fd, Prompt.cursor_left(1).encode())
        os.write(sys.stdout.fileno(), Prompt.erase_left_character().encode())
        # sys.stdout.write(Prompt.erase_left_character())
        # self._c_buf.
        # 日本語の場合はカーソル2つ移動
    else:
        os.write(sys.stdout.fileno(), key.encode(errors="replace"))
        buf.append(key)


############################################################################################################


def _test1():
    import threading
    import time

    keyinput = KeyInput()

    def main():
        i = 0
        while True:
            i += 1
            print(f"===== {i} ======")

            c = keyinput.get()

            print(c)
            print(c.encode())

            if c == "q" or c == b"q" or c == Key.ESC:
                break

            if c == "n":
                # keyinput.reset()
                some_entry = keyinput.input("name? ")
                print(some_entry)

            if c == "p":
                keyinput.wait_until("r")
                print("resume")

    def hogeloop():
        for i in range(10):
            print(i)
            print("q")
            time.sleep(0.2)

    t = threading.Thread(target=hogeloop)
    t.start()
    main()


def _test2():
    keyinput = KeyInput()

    i = 0
    while True:
        i += 1

        c = keyinput.get(block=False)
        print(f"== {i}", c)

        if c == "q" or c == b"q" or c == Key.ESC:
            break

        if c == "p":
            print("pause")
            print("If you want resume, press 'r'.")
            keyinput.wait_until("r")
            print("resume")

            # c = keyinput.get(block=True)
            # print(c)
            # if c == "r":
            #     print("resume")
            # else:
            #     print("end")
            #     break

        if c == "n":
            keyinput.reset()
            some_entry = keyinput.input("name? ")
            print(some_entry)

        time.sleep(0.1)


def _test3():
    keyinput = KeyInput()

    i = 0
    while True:
        i += 1
        print(f"=== {i} ===")
        c = keyinput.get()
        print(c)
        print(len(c))
        print(c.encode())


def _test4():
    # _termios_restore()

    while True:
        text = builtins.input("> ")
        print(text)
        if text == "q" or text == "quit" or text == "exit":
            break


def _test5():
    keyinput = KeyInput()

    print("Phase 0")
    keyinput.wait_until("s")

    print("Phase 1")
    for i in range(3):
        c = keyinput.get(block=True)
        print(i, c, len(c))

    print("Phase 2")
    text = keyinput.input("input :  ")
    print("return :", text)
    print("return :", text.encode())

    print("Phase 3")
    for i in range(3):
        c = keyinput.get(block=True)
        print(i, c, len(c))

    print("Phase 4")
    for i in range(30):
        c = keyinput.get(block=False)
        print(i, c, len(c))
        time.sleep(0.1)

    print("Phase 5")
    text = keyinput.input("input :  ")
    print("return :", text)
    print("return :", text.encode())

    print("Phase 6")
    for i in range(30):
        c = keyinput.get(block=False)
        print(i, c, len(c))
        time.sleep(0.1)

    print("Phase 7")
    keyinput.wait_until("r")

    print("Phase 8")
    for i in range(5):
        c = keyinput.get(block=True)
        print(i, c, len(c))
        time.sleep(0.1)


# def _test4():
#     from pynput.keyboard import Controller, Key

#     keyboard = Controller()
#     keyboard.type

#     # Press and release space
#     keyboard.press(Key.space)
#     keyboard.release(Key.space)

#     # Type a lower case A; this will work even if no key on the
#     # physical keyboard is labelled 'A'
#     keyboard.press("a")
#     keyboard.release("a")

#     # Type two upper case As
#     keyboard.press("A")
#     keyboard.release("A")
#     with keyboard.pressed(Key.shift):
#         keyboard.press("a")
#         keyboard.release("a")

#     # Type 'Hello World' using the shortcut type method
#     keyboard.type("Hello World")


if __name__ == "__main__":
    # _test1()
    # _test2()
    # _test3()
    # _test4()
    _test5()
