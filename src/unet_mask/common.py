import os
import sys
from pathlib import Path
from pprint import pprint


def set_path(_file_):
    """_file_: __file__"""

    _dirname = Path(_file_).absolute().parent
    workspaceFolder = _dirname.parent.parent
    src = _dirname.parent

    assert workspaceFolder.name.startswith("NewtonianVAE")  # NewtonianVAE-branch
    assert src.name == "src"

    os.chdir(workspaceFolder)
    sys.path.insert(0, str(src))
