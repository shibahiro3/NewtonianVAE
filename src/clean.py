#!/usr/bin/env python3


"""
find [PATH] -type d -empty
find [PATH] -type d -empty -delete
"""


import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List

from mypython.terminal import Color


def main():
    clean(sys.argv[1])


def clean(root):
    if not Path(root).exists():
        print(f'"{root}" does not exist')
        return
    if not Path(root).is_dir():
        print(f'"{root}" is not directory')
        return

    deletable: List[Path] = []
    warning: List[Path] = []

    print("Directories with no weight files...")
    for f in Path(root).glob("**/weight/"):
        if len(os.listdir(f)) == 0:
            print(f.resolve())
            deletable.append(f.parent)

            try:
                datetime.strptime(f.parent.name, "%Y-%m-%d_%H-%M-%S")  # for check (ValueError)
            except:
                warning.append(f.resolve())
                # continue

    if len(warning) > 0:
        Color.print("Warnings", c=Color.orange)
        for f in warning:
            Color.print(f, c=Color.orange)

    delete(deletable)


def delete(deletable: List[Path]):
    if len(deletable) > 0:
        res = input("Delete? [y/n] ")
        if res == "y":
            for f in deletable:
                if f.is_dir():
                    shutil.rmtree(f)
                    print(f"Deleted (d): {f.resolve()}")
                elif f.is_file():
                    os.remove(f)
                    print(f"Deleted (f): {f.resolve()}")
        else:
            print("Abort.")
    else:
        print("No deletable files")


if __name__ == "__main__":
    main()
