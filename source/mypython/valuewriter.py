import io
import os
import pickle
import shutil
import struct
import sys
import time

# from mypython.terminal import Color
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


class ValueWriter:
    """Write immediately

      number of info (4 byte)
    + info (number of info byte, picke of dict)
    + data
    """

    dtype2struct = {
        "int8": "b",
        "uint8": "B",
        "int16": "h",  # max: 32767
        "uint16": "H",  # max: 65535
        "int32": "i",  # max: 2147483647
        "uint32": "I",  # max: 4294967295
        "int64": "q",
        "uint64": "Q",
        "float32": "f",
        "float64": "d",
    }

    core = "core"
    order = "order"

    def __init__(self, root, name_type: Dict[str, str] = {}) -> None:
        # https://qiita.com/halhorn/items/178ed670f05e9bbe6d0a
        # 外部でも共通インスタンスのdictだと共有されてしまう

        self.root_core = Path(root, self.core)
        self.name_type = name_type.copy()

        self.root_core.mkdir(parents=True, exist_ok=True)
        self.f_order = open(Path(root, self.order), "wb")
        self.fs: Dict[str, io.BufferedWriter] = {}
        for name, typ in self.name_type.items():
            self._make_f(name, typ)

    def write(self, name: str, value):
        """
        If type is not specified in name_type, type is determined by the first value input type corresponding to name

        name: max byte is 255
        """

        # assert type(name) == str
        # assert len(name.encode()) <= 255

        if not name in self.name_type:

            type_v = type(value)
            if type_v == int:
                typ = "int32"
            elif type_v == float:
                typ = "float32"
            elif type_v.__name__ in self.dtype2struct:
                # numpy.float64, etc.
                typ = type_v.__name__
            else:
                raise TypeError(f'"{name}" type: {type_v}')

            f = self._make_f(name, typ)

        else:
            f = self.fs[name]

        # if scalar:
        f.write(struct.pack(self.name_type[name], value))  # Faster than numpy.ndarray.tobytes
        f.flush()

    @staticmethod
    def load(root) -> Dict[str, np.ndarray]:
        root = Path(root)

        with open(Path(root, ValueWriter.order), "rb") as f:
            names_ = f.read()
        names = []

        i = 0
        while i < len(names_):
            s = i + 1
            names.append(names_[s : s + names_[i]].decode())
            i += names_[i] + 1

        # print(names)

        data = {}
        for name in names:
            with open(Path(root, ValueWriter.core, name), "rb") as f:
                byte = f.read()
            back_idx = 4
            len_info = struct.unpack("I", byte[:back_idx])[0]
            info = pickle.loads(byte[back_idx : back_idx + len_info])
            back_idx += len_info
            typ = np.dtype(info["typ"]).newbyteorder(info["endian"])
            data[name] = np.frombuffer(byte[back_idx:], dtype=typ)
        return data

    def _make_f(self, name: str, typ):
        f = open(self.root_core / name, "wb")
        info = {"typ": typ, "endian": sys.byteorder}
        info = pickle.dumps(info)
        f.write(struct.pack("I", len(info)))
        f.write(info)
        f.flush()

        self.fs[name] = f
        typ_ = self.dtype2struct.get(typ, None)
        if typ_ is not None:
            self.name_type[name] = typ_

        name_ = name.encode()
        self.f_order.write(struct.pack("B", len(name_)) + name_)
        self.f_order.flush()  # これが無いと、ファイルの中身があるにも関わらず読み込めない

        return f


if __name__ == "__main__":
    # Example

    root = "_hoge"

    vals = {
        "foo": [-53.63, 5624.146, 1944.78, -3462.7020, 652.89],
        "baraaa": [154451, 4652, -54276524, 6254, 7864],
        # "bar": [1544.51, 4652, -5427.6524, 6254, 7864],
        "bazz": np.random.randint(-300, 500, size=6, dtype=np.int32),
        "piyo": np.random.randn(3),
    }

    vwriter = ValueWriter(root)
    for k, vs in vals.items():
        for v in vs:
            # print(k, v)
            vwriter.write(k, v)
            # time.sleep(1)
            # ctrl + c

    vwriter.write("bazz", 114)
    vwriter.write("bazz", 514)
    vwriter.write("fuga", 8888)

    data = ValueWriter.load(root)
    pprint(data)
    print(data.keys())

    if os.path.isdir(root):
        shutil.rmtree(root)
