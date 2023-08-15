#!/usr/bin/env python3

import pickle
import signal
from datetime import datetime
from typing import Dict

import numpy as np

import mypython.qt.thread as qth
import mypython.qt.util as qut
from mypython import rdict
from mypython.qt.imports import *
from view import qt_http_vis


class Images(QWidget):
    def __init__(self, data: bytes) -> None:
        super().__init__()

        d: Dict[str, np.ndarray] = pickle.loads(data)

        layout = QVBoxLayout()

        print("===", datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "===")
        rdict.show(d)
        for k, v in d.items():
            layout.addWidget(QLabel(k))
            qi = qut.QtImage()
            layout.addWidget(qi)
            qi.setImage(qut.numpy2qimage(v))
            qi.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setLayout(layout)


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication([])
    win = qt_http_vis.Window(this_addr=("localhost", 12345), WidgetClass=Images)
    win.setWindowTitle("Dict[str, ndarray] Vis")
    geometry = QApplication.primaryScreen().availableGeometry()
    win.resize(int(geometry.width() // 3), int(geometry.height() // 1.5))
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
