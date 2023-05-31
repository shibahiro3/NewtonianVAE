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


config = dict(
    metavar="FILE",
    help="Configuration file",
)
format_file = dict(
    help=r'You can save it in the "path: {results_dir: ...}" (in config file) directory by pressing the s key on the matplotlib window with specified format.'
)
format_video = dict(
    help='Format of the video to be saved\nYou can save the video with "--save-anim".'
)
default_fig_formats = ["png"]
# default_fig_formats = ["png", "pdf", "svg"]

# format:
#   png: versatile, raster
#   pdf: versatile, vector, for LaTeX
#   svg: vector, for PowerPoint
