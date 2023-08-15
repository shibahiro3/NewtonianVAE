from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)

import mypython.qt.thread as qth
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
    ) -> None:
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)
        self.sig = qth.SignalWrap()


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


class Window(QWidget):
    def __init__(self, this_addr, WidgetClass) -> None:
        super().__init__()

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.w = QWidget()
        self.main_layout.addWidget(self.w)

        self.httpd = NewHTTPServer(this_addr, NewHTTPRequestHandler)

        def update(ret):
            self.w.deleteLater()
            # self.main_layout.removeWidget(self.w)
            self.w = WidgetClass(ret)
            self.main_layout.addWidget(self.w)

        self.httpd.sig.connect_(update)

        self.th1 = qth.OneThread(fn=self.httpd.serve_forever)
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.start(self.th1)

    def closeEvent(self, a0) -> None:
        self.httpd.shutdown()
        return super().closeEvent(a0)
