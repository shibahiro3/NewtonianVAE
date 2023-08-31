#!/usr/bin/env python3

import pickle
import re
import signal
from datetime import datetime
from numbers import Number
from pprint import pprint
from typing import Dict, List

import numpy as np

import mypython.qt.util as qut
from mypython import rdict
from mypython.qt.imports import *
from view import qt_http_vis as qvis

pg.setConfigOptions(
    antialias=True,
    # useOpenGL=False,
    # background="white",
    # foreground="black",  # 軸、軸ラベルの色
)


class LossWidget(QtWidgets.QVBoxLayout):
    def __init__(self, title="Loss", label: str = "min max") -> None:
        super().__init__()

        self.data: List[List[Number]] = []

        # plt.setLabel("left", "Y axis")
        self.w = pg.PlotWidget()
        self.w.clear()
        self.w.setTitle(title)
        self.w.showGrid(x=True, y=True)
        self.g_pool: List[pg.PlotDataItem] = []

        w_label = QtWidgets.QLabel(label)
        w_lineedit = QtWidgets.QLineEdit()
        # btn = QtWidgets.QPushButton("Apply")

        def on_clicked():
            try:
                min, max = re.split(" *, *| +", w_lineedit.text().strip())
                data = np.asarray(self.data)
                if min == "min":
                    min = np.nanmin(data)
                else:
                    min = float(min)

                if max == "max":
                    max = np.nanmax(data)
                else:
                    max = float(max)
                self.w.setYRange(min=min, max=max)
            except:
                w_lineedit.setText(w_lineedit.text() + " (error)")

        # btn.clicked.connect(on_clicked)
        w_lineedit.returnPressed.connect(on_clicked)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(w_label)
        bar.addWidget(w_lineedit)

        self.addWidget(self.w)
        self.addLayout(bar)

    def catch_data(self, x: List[Number]):
        """listの数値ごとに色をつけて複数の折れ線グラフを表示"""
        dl = len(self.data)
        nl = len(x)  # まぁ、これは一定であるものを想定　一定で無くても多分動く
        if dl < nl:
            color = ["skyblue", "orange", "green", "purple"]
            for i in range(dl, dl + nl):
                self.data.append([])
                self.g_pool.append(
                    self.w.plot(
                        [],
                        [],
                        pen=pg.mkPen(color=color[i], width=2),
                        symbol="o",
                        symbolBrush=color[i],
                        symbolPen=None,
                        symbolSize=5,
                    )
                )

        for i in range(len(self.data)):
            self.data[i].append(x[i])
            self.g_pool[i].setData(self.data[i])


class Window(qvis.BaseWindow):
    def __init__(self, this_addr) -> None:
        super().__init__(this_addr)

        self.setWindowTitle("Dict[str, number] Timeline Vis")

        self.main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.main_layout)
        self.losses: Dict[str, LossWidget] = {}

        def append(k, v):
            if k not in self.losses.keys():
                l = LossWidget(k)
                self.losses[k] = l
                self.main_layout.addLayout(l)
            self.losses[k].catch_data(v)

        def update(data: bytes):
            d = pickle.loads(data)
            # pprint(d)
            loss_names = d["train"].keys()
            for k in loss_names:
                append(k, [d["train"][k], d["valid"][k]])

        self.httpd.set_callback(update)
        self.start_httpd()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication([])
    win = Window(this_addr=("localhost", 50000))
    geometry = QApplication.primaryScreen().availableGeometry()
    win.resize(int(geometry.width() // 3), int(geometry.height() // 3))
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
