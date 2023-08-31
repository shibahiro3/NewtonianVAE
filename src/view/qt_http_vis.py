from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)
from typing import Callable, Optional

import mypython.qt.util as qut
from mypython.qt.imports import *


class NewHTTPServer(HTTPServer):
    """
    https://stackoverflow.com/questions/49070708/how-to-pass-queue-to-basehttpserver
    """

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        bind_and_activate=True,
        callback: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)
        self.sig = qut.SignalWrap()
        if callback is not None:
            self.sig.connect_(callback)

    def set_callback(self, callback: Optional[Callable[[bytes], None]] = None):
        self.sig.connect_(callback)


class NewHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server: NewHTTPServer
        # print(self.server)

        # print(self.client_address)
        # print(self.command)  # POST
        # print(self.request_version)  # HTTP/1.1
        # print(self.headers)
        # print(self.path)
        # print(self.requestline)
        content_length = int(self.headers["Content-Length"])
        data = self.rfile.read(content_length)
        self.server.sig.emit_(data)

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        # silent
        return


def create_server(addr, callback: Optional[Callable[[bytes], None]] = None):
    return NewHTTPServer(addr, NewHTTPRequestHandler, callback=callback)


class BaseWindow(QWidget):
    def __init__(self, this_addr, update: Optional[Callable[[bytes], None]] = None) -> None:
        super().__init__()

        self.httpd = create_server(this_addr, update)
        self.th_server = qut.OneThread(fn=self.httpd.serve_forever)
        self.threadpool = QtCore.QThreadPool()

    def start_httpd(self):
        self.threadpool.start(self.th_server)

    def closeEvent(self, a0) -> None:
        self.httpd.shutdown()
        return super().closeEvent(a0)


class SimpleWindow(BaseWindow):
    def __init__(self, this_addr, WidgetClass) -> None:
        super().__init__(this_addr)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.w = QWidget()
        self.main_layout.addWidget(self.w)

        def update(data: bytes):
            self.w.deleteLater()
            # self.main_layout.removeWidget(self.w)
            self.w = WidgetClass(data)
            self.main_layout.addWidget(self.w)

        self.httpd.set_callback(update)
        self.start_httpd()
