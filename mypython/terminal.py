import builtins
import os
import pprint
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

    def print(*values, color=None, c=None, sep=" ", **kwargs):
        assert (color is None) or (c is None)

        text = ""
        for i, e in enumerate(values):
            text += str(e)
            if i < len(values) - 1:
                text += sep

        c_ = color if color is not None else c
        if c_ is None:
            c_ = Color.green
        print(c_ + text + Color.reset, sep=sep, **kwargs)

    def pprint(obj, color=green):
        s = pprint.pformat(obj)
        print(color + s + Color.reset)

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
    del_line = "\r\x1b[K"

    @staticmethod
    def print_one_line(small_obj):
        builtins.print(Prompt.del_line + str(small_obj), end="")

    @staticmethod
    def fit_terminal(text):
        max_len = os.get_terminal_size().columns
        cnt, first_, last_ = Prompt._clip(text, max_len, 15, max_len - 20)
        if cnt <= max_len:
            return text
        else:
            return first_ + " ... " + last_

    @staticmethod
    def _clip(text, len_max, first_max, last_max):
        cnt = 0
        first_size = 0
        for i, c in enumerate(text):  # i: 0 to N-1
            if unicodedata.east_asian_width(c) in "FWA":
                cnt += 2
            else:
                cnt += 1

            if first_size == 0:
                if cnt == first_max:
                    first_size = i + 1
                elif cnt > first_max:
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
                    if cnt_ == last_max:
                        last_size = i + 1
                        break
                    elif cnt_ > last_max:
                        last_size = i
                        break

            return cnt, text[:first_size], text[-last_size:]
        else:
            return cnt, None, None
