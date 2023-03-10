# only standard library

import builtins
import os
import pprint
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from posix import times_result
from pprint import pformat


class Color:
    def rgb(r: int, g: int, b: int):
        return f"\x1b[38;2;{r};{g};{b}m"

    def bg_rgb(r: int, g: int, b: int):
        return f"\x1b[48;2;{r};{g};{b}m"

    reset = "\x1b[m"

    # Bright
    black = "\x1b[90m"
    red = "\x1b[91m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    blue = "\x1b[94m"
    magenta = "\x1b[95m"
    cyan = "\x1b[96m"
    white = "\x1b[97m"

    def print(*values, c=green, sep=" ", **kwargs):
        text = ""
        for i, e in enumerate(values):
            text += str(e)
            if i < len(values) - 1:
                text += sep
        builtins.print(c + text + Color.reset, sep=sep, **kwargs)

    def pprint(obj, c=green):
        s = pprint.pformat(obj)
        builtins.print(c + s + Color.reset)

    boldblack = "\x1b[1;90m"
    boldred = "\x1b[1;91m"
    boldgreen = "\x1b[1;92m"
    boldyellow = "\x1b[1;93m"
    boldblue = "\x1b[1;94m"
    boldmagenta = "\x1b[1;95m"
    boldcyan = "\x1b[1;96m"
    boldwhite = "\x1b[1;97m"

    coral = rgb(255, 127, 80)
    hotpink = rgb(255, 105, 180)
    purple = rgb(128, 0, 128)
    orange = rgb(255, 165, 0)

    bg_gold = bg_rgb(255, 215, 0)


class Prompt:
    del_left_line = "\x1b[1K"
    del_right_line = "\x1b[0K"
    del_line = "\r\x1b[K"
    # del_line = "\x1b[2K" # ダメ

    def __init__(self) -> None:
        self._prev_n_new_line = 0

    # def print(self, *args, **kwargs):
    #     self._prev_n_new_line = 0
    #     builtins.print(*args, **kwargs)

    # def print_update(self, small_obj, clip=True):
    #     texts = str(small_obj).split("\n")

    #     for _ in range(self._prev_n_new_line):
    #         builtins.print(Prompt.del_line + self.cursor_up(1), end="")

    #     self._prev_n_new_line = len(texts)

    #     for i, s in enumerate(texts):
    #         if clip:
    #             max_len = os.get_terminal_size().columns
    #             cnt, first_, last_ = Prompt._clip(s, max_len, max_len - 5, max_len)
    #             if cnt > max_len:
    #                 s = first_ + " ..."

    #         builtins.print(Prompt.del_line + s)

    #     # self.scroll_up(len(texts))

    @staticmethod
    def cursor_up(n: int):
        return f"\x1b[{n}A"

    @staticmethod
    def cursor_down(n: int):
        return f"\x1b[{n}B"

    @staticmethod
    def scroll_up(n: int):
        builtins.print(f"\x1b[{n}T", end="")

    @staticmethod
    def print_one_line(small_obj, clip=True):
        text = str(small_obj)

        if clip:
            text = Prompt.clip(text)

        builtins.print(Prompt.del_line + text, end="", flush=True)

    @staticmethod
    def clip(text: str):
        max_len = os.get_terminal_size().columns
        cnt, first_, last_ = Prompt._clip(text, max_len, max_len - 5, max_len)
        if cnt > max_len:
            text = first_ + " ..."
        return text

    # @staticmethod
    # def print_multi_line(obj, n: int, clip=True):
    #     texts = str(obj).split("\n")

    #     # builtins.print(Prompt.del_line)
    #     for _ in range(n):
    #         builtins.print(Prompt.del_line + Prompt.cursor_up(1), end="")

    #     for i, s in enumerate(texts[:n]):
    #         if clip:
    #             max_len = os.get_terminal_size().columns
    #             cnt, first_, last_ = Prompt._clip(s, max_len, max_len - 5, max_len)
    #             if cnt > max_len:
    #                 s = first_ + " ..."

    #         builtins.print(Prompt.del_line + s)

    #         # if i < n - 1:
    #         #     builtins.print(Prompt.del_line + s)
    #         # else:
    #         #     builtins.print(Prompt.del_line + s, end="")

    @staticmethod
    def fit_terminal(text):
        max_len = os.get_terminal_size().columns
        cnt, first_, last_ = Prompt._clip(text, max_len, 15, max_len - 20)
        if cnt <= max_len:
            return text
        else:
            return first_ + " ... " + last_

    @staticmethod
    def _clip(text, len_max, len_first_max, len_last_max):
        cnt = 0
        first_size = 0
        for i, c in enumerate(text):  # i: 0 to N-1
            if unicodedata.east_asian_width(c) in "FWA":
                cnt += 2
            else:
                cnt += 1

            if first_size == 0:
                if cnt == len_first_max:
                    first_size = i + 1
                elif cnt > len_first_max:
                    first_size = i

        last_size = 0

        if len_max <= cnt:
            cnt_ = 0
            for i, c in enumerate(reversed(text)):
                if unicodedata.east_asian_width(c) in "FWA":
                    cnt_ += 2
                else:
                    cnt_ += 1

                if last_size == 0:
                    if cnt_ == len_last_max:
                        last_size = i + 1
                        break
                    elif cnt_ > len_last_max:
                        last_size = i
                        break

            return cnt, text[:first_size], text[-last_size:]
        else:
            return cnt, None, None
