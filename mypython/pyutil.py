import argparse
import datetime
import errno
import fcntl
import inspect
import math
import os
import pickle
import pprint
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from posix import times_result
from pprint import pformat


def s2dhms(sec):
    sec = math.floor(sec)
    d = math.floor(sec / 86400)
    _rest_sec = sec - d * 86400
    h = math.floor(_rest_sec / 3600)
    _rest_sec = _rest_sec - h * 3600
    m = math.floor(_rest_sec / 60)
    s = _rest_sec - m * 60
    return d, h, m, s


def s2dhms_str(sec, always_day=False):
    d, h, m, s = s2dhms(sec)
    if d > 0:
        return f"{d} Days {h}:{m:0>2}:{s:0>2}"
    else:
        if always_day:
            return f"{d} Days {h}:{m:0>2}:{s:0>2}"
        else:
            return f"{h}:{m:0>2}:{s:0>2}"
