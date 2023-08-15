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
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np

T = Union[int, float, np.ndarray]


class ValueWriter:
    """
    ・Write immediately
    ・Small volume
    ・Fixed length

    Data specification
        'info' byte length (4 byte)
        + info (dict -> pickle)
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

    struct2dtype = {
        "b": "int8",
        "B": "uint8",
        "h": "int16",
        "H": "uint16",
        "i": "int32",
        "I": "uint32",
        "q": "int64",
        "Q": "uint64",
        "f": "float32",
        "d": "float64",
    }

    core = "core"
    order = "order"

    def __init__(self, root) -> None:
        # TODO: Add schema like polars

        self.root_core = Path(root, self.core)
        self.name2type = {}
        self.name2info = {}

        self.root_core.mkdir(parents=True, exist_ok=True)
        self.f_order = open(Path(root, self.order), "wb")
        self.fs: Dict[str, io.BufferedWriter] = {}
        for name, typ in self.name2type.items():
            self._make_f(name, typ)

    def write(self, name: str, value: T):
        """
        If type is not specified in name_type, type is determined by the first value input type corresponding to name

        name: max byte is 255
        """

        # assert type(name) == str
        # assert len(name.encode()) <= 255

        type_v = type(value)
        if type_v == int:
            typ = "int32"
            info = {"typ": typ, "endian": sys.byteorder}
        elif type_v == float:
            typ = "float32"
            info = {"typ": typ, "endian": sys.byteorder}
        elif type_v.__name__ in self.dtype2struct:
            # numpy.float64, etc.
            typ = type_v.__name__
            info = {"typ": typ, "endian": sys.byteorder}
        elif type_v == np.ndarray:
            typ = "ndarray"
            # info = {"typ": typ, "endian": value.dtype.byteorder, "dtype": value.dtype, "shape": value.shape}
            info = {"typ": typ, "dtype": value.dtype, "shape": value.shape}
        else:
            raise TypeError(
                f'Unsupported type: (name: "{name}", value: {value}, type: {type_v.__name__})'
            )

        if not name in self.name2info:
            self.name2info[name] = info
            f = self._make_f(name, typ, info)
        else:
            registered_type = self.name2info[name]["typ"]
            # print(name, value, typ, registered_type)
            if typ != registered_type:
                raise TypeError(
                    f'Changed type: (name: "{name}", value: {value}, type: {typ}, expected type: {registered_type})'
                )

            if typ == "ndarray":
                if info != self.name2info[name]:
                    raise ValueError(f"\nfirst: {self.name2info[name]}\nnow:   {info}")

            f = self.fs[name]

        if typ == "ndarray":
            f.write(value.tobytes())
        else:
            f.write(struct.pack(self.name2type[name], value))  # Faster than numpy.ndarray.tobytes

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
        for name_ in names:
            with open(Path(root, ValueWriter.core, name_), "rb") as f:
                byte = f.read()
            back_idx = 4
            len_info = struct.unpack("I", byte[:back_idx])[0]
            info = pickle.loads(byte[back_idx : back_idx + len_info])
            back_idx += len_info
            typ = info["typ"]

            # print(f"name: {name_}, data len: {len(byte[back_idx:])}")

            if typ == "ndarray":
                dtype_ = info["dtype"]
                # for safe
                l = len(byte[back_idx:])
                e = np.prod(info["shape"])
                lim = e * int(l // e)
                data[name_] = np.frombuffer(byte[back_idx : back_idx + lim], dtype=dtype_).reshape(
                    -1, *info["shape"]
                )
            else:
                dtype_ = np.dtype(typ).newbyteorder(info["endian"])
                data[name_] = np.frombuffer(byte[back_idx:], dtype=dtype_)

        return data

    def _make_f(self, name: str, typ: str, info: dict):
        info = pickle.dumps(info)
        f = open(self.root_core / name, "wb")
        f.write(struct.pack("I", len(info)))
        f.write(info)
        f.flush()

        self.fs[name] = f

        # Only Real type
        typ_ = self.dtype2struct.get(typ, None)
        if typ_ is not None:
            self.name2type[name] = typ_

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
        # "array": [np.random.randn(3, 2) for _ in range(4)],
        # "array": [np.array([0.1, 0.2, 0.3]) for _ in range(4)],
        "array": [
            np.arange(0.1, 0.6 + 0.01, 0.1).reshape(2, 3),
            np.arange(0.7, 1.2 + 0.01, 0.1).reshape(2, 3),
            # np.array([[1.3, 1.4, 1.5]]), # raise
        ],
    }

    # print(vals.keys())

    vwriter = ValueWriter(root)
    for k, vs in vals.items():
        for v in vs:
            # print(k, v)
            vwriter.write(k, v)
            # time.sleep(1)
            # ctrl + c

    vwriter.write("bazz", 114)
    vwriter.write("bazz", 514)
    # vwriter.write("bazz", 514.352)  # raise error
    # vwriter.write("bazz", "kakaka")  # raise error

    vwriter.write("new", 8888)

    print("===== load =====")

    data = ValueWriter.load(root)
    # pprint(data) # not true order printing
    # pprint(data.keys())
    for k, v in data.items():
        print(k)
        print(v)

    print(data["array"].shape)

    if os.path.isdir(root):
        shutil.rmtree(root)
