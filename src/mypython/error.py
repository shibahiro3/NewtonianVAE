import sys
from pathlib import Path
from typing import Union

from mypython.terminal import Color


def check_file(p: Union[str, Path], extension: str = None):
    """ex. extension = ".txt" """
    p = Path(p)
    if not p.exists():
        Color.print(f'"{p}" does not exist.', c=Color.red)
        sys.exit()
    if not p.is_file():
        Color.print(f'"{p}" is not file.', c=Color.red)
        sys.exit()
    if (extension is not None) and p.suffix != extension:
        Color.print(f'"{p}" extension is not {extension}.', c=Color.red)
        sys.exit()


def check_dir(p: Union[str, Path]):
    p = Path(p)
    if not p.exists():
        Color.print(f'"{p}" does not exist.', c=Color.red)
        sys.exit()
    if not p.is_dir():
        Color.print(f'"{p}" is not directory.', c=Color.red)
        sys.exit()
