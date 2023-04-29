import pickle
import signal
import socket
import threading
from typing import Optional

import numpy as np


class Receiver:
    def __init__(self, port, typ) -> None:
        self._data = None

        self.port = port
        self.typ = typ

        self.sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", self.port))

        self._is_running = False

        # ==========
        self.thread1 = threading.Thread(target=self._run)

    def start(self):
        self._is_running = True
        self.thread1.start()

    def _run(self):
        while self._is_running:
            # print("=== received ===")
            self._data, from_addr = self.sock.recvfrom(4096)

    def get(self, convert=None):
        if self._data is not None:
            if self.typ == "text":
                return self._data.decode("utf-8")
            elif self.typ == "pickle":
                return pickle.loads(self._data)
            else:
                assert NotImplementedError()

    def invoke_end(self):
        self._is_running = False

        # # ???  OSError: [Errno 9] Bad file descriptor
        # self.sock.sendto("-1 -1 -1 -1".encode("utf-8"), ("127.0.0.1", self.port))

        # self.thread1.join()
        self.sock.close()

    # def close(self):
    #     self.sock.close()
