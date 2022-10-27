import builtins
import shutil
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mypython.terminal import Color


def select_date(model_date_path) -> Optional[Path]:
    if not Path(model_date_path).exists():
        Color.print(f'"{model_date_path}" doesn\'t exist.', c=Color.red)
        return None
    if not Path(model_date_path).is_dir():
        Color.print(f'"{model_date_path}" is not a directory.', c=Color.red)
        return None

    w_dirs = [d for d in Path(model_date_path).glob("*") if d.is_dir()]
    w_dirs.sort()

    # weight/* の無いディレクトリを除去
    for w in reversed(w_dirs):
        if len(list(w.glob("weight/*"))) == 0:
            w_dirs.remove(w)

    if len(w_dirs) == 0:
        Color.print(
            f'Date and time doesn\'t exist in "{model_date_path}" directory.', c=Color.orange
        )
        return None

    for i, e in enumerate(w_dirs, 1):
        l = len(list(e.glob("weight/*")))
        print(i, ":", e.name, f"({l})")

    idx = _get_idx("Select date and time (or exit): ", len(w_dirs))
    if idx is None:
        return None
    else:
        return w_dirs[idx]


def select_weight(path: Path) -> Optional[Path]:
    weight_p = list(path.glob("weight/*"))
    weight_p.sort(key=lambda e: int(e.stem))  # weight_
    if len(weight_p) == 0:
        Color.print("Weight doesn't exist.", c=Color.orange)
        return None

    for i, e in enumerate(weight_p, 1):
        print(i, ":", e.name)

    idx = _get_idx("Choose weight (or exit): ", len(weight_p))
    if idx is None:
        return None
    else:
        return weight_p[idx]


def delete_useless_saves(model_date_path):
    w_dirs = [d for d in Path(model_date_path).glob("*") if d.is_dir()]
    w_dirs.sort()

    # weight/* の無いディレクトリを消去
    for w in reversed(w_dirs):
        if len(list(w.glob("weight/*"))) == 0:
            # w_dirs.remove(w)
            shutil.rmtree(w)


def _get_idx(text, len_list):
    while True:
        idx = builtins.input(Color.green + text + Color.reset)

        if idx == "exit":
            print("Bye!")
            return None
        else:

            try:
                idx = int(idx) - 1
                if 0 <= idx and idx < len_list:
                    return idx
                else:
                    Color.print(f"Please 1 to {len_list}. again.", c=Color.red)
            except ValueError:
                Color.print("Input integer or exit. again.", c=Color.red)


def cmap_plt(N: int, name="rainbow", reverse=False):
    color_map = cm.get_cmap(name)
    if reverse:
        return [color_map(i / N) for i in range(N)]
    else:
        return [color_map(1 - i / N) for i in range(N)]


def cmap_qt(N: int, name="rainbow", reverse=False):
    """
    transparency の減衰の例:
        symbolBrushes[:, 3] = np.linspace(10, 255, N)
    """
    return np.array(cmap_plt(N, name, reverse)) * 255
    # return (np.array(cmap_plt(N)) * 255).astype(np.uint8)
