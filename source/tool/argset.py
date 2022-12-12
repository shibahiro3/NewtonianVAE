import argparse
import inspect
from pprint import pprint


def save_anim(
    parser: argparse.ArgumentParser,
    action="store_true",
    help="Save animation",
    **kwargs,
):
    parser.add_argument("--save-anim", **_kwargs(locals()), **kwargs)


def save_result(
    parser: argparse.ArgumentParser,
    action="store_true",
    help="Save animation",
    **kwargs,
):
    parser.add_argument("--save-result", **_kwargs(locals()), **kwargs)


def resume(
    parser: argparse.ArgumentParser,
    action="store_true",
    help="Load the model and resume learning",
    **kwargs,
):
    parser.add_argument("--resume", **_kwargs(locals()), **kwargs)


def anim_mode(
    parser: argparse.ArgumentParser,
    type=str,
    choices=["freeze", "anim", "save"],
    default="anim",
    **kwargs,
):
    parser.add_argument("-a", "--anim-mode", **_kwargs(locals()), **kwargs)


def start_iter(
    parser: argparse.ArgumentParser,
    metavar="NUM",
    type=int,
    default=1,
    **kwargs,
):
    parser.add_argument("--start-iter", **_kwargs(locals()), **kwargs)


def output(
    parser: argparse.ArgumentParser,
    metavar="PATH",
    type=str,
    **kwargs,
):
    parser.add_argument("-o", "--output", **_kwargs(locals()), **kwargs)


def format(
    parser: argparse.ArgumentParser,
    metavar="FORMAT",
    type=str,
    nargs="*",
    default=["svg", "pdf"],
    **kwargs,
):
    parser.add_argument("--format", **_kwargs(locals()), **kwargs)


def env_domain(
    parser: argparse.ArgumentParser,
    metavar="ENV",
    type=str,
    required=False,
    **kwargs,
):
    parser.add_argument("--env-domain", **_kwargs(locals()), **kwargs)


def cf(
    parser: argparse.ArgumentParser,
    required=True,
    type=str,
    metavar="FILE",
    help="Configuration file of common",
    **kwargs,
):
    parser.add_argument("--cf", **_kwargs(locals()), **kwargs)


# def cf_eval(
#     parser: argparse.ArgumentParser,
#     required=True,
#     type=str,
#     metavar="FILE",
#     help="Configuration file of evaluation",
#     **kwargs,
# ):
#     parser.add_argument("--cf-eval", **_kwargs(locals()), **kwargs)


# def cf_simenv(
#     parser: argparse.ArgumentParser,
#     required=True,
#     type=str,
#     metavar="FILE",
#     help="Configuration file of simulation environment",
#     **kwargs,
# ):
#     parser.add_argument("--cf-simenv", **_kwargs(locals()), **kwargs)


def episodes(
    parser: argparse.ArgumentParser,
    required=True,
    type=int,
    metavar="E",
    help="Total number of episodes",
    **kwargs,
):
    parser.add_argument("--episodes", **_kwargs(locals()), **kwargs)


def steps(
    parser: argparse.ArgumentParser,
    required=False,
    type=int,
    metavar="E",
    help="Time steps",
    **kwargs,
):
    parser.add_argument("--steps", **_kwargs(locals()), **kwargs)


def watch(
    parser: argparse.ArgumentParser,
    type=str,
    choices=["render", "plt"],
    help=(
        "Check data without saving data. For rendering, "
        "you can choose to use OpenCV (render) or Matplotlib (plt)."
    ),
    **kwargs,
):
    parser.add_argument("--watch", **_kwargs(locals()), **kwargs)


def path_model(
    parser: argparse.ArgumentParser,
    required=True,
    type=str,
    metavar="DIR_PATH",
    help="Directory path for models managed by date and time",
    **kwargs,
):
    parser.add_argument("--path-model", **_kwargs(locals()), **kwargs)


def path_data(
    parser: argparse.ArgumentParser,
    required=True,
    type=str,
    metavar="DIR_PATH",
    help="Directory path where episode data exists",
    **kwargs,
):
    parser.add_argument("--path-data", **_kwargs(locals()), **kwargs)


def path_save(
    parser: argparse.ArgumentParser,
    required=True,
    type=str,
    metavar="DIR_PATH",
    help="Destination directory path",
    **kwargs,
):
    parser.add_argument("--path-save", **_kwargs(locals()), **kwargs)


def path_result(
    parser: argparse.ArgumentParser,
    required=False,
    type=str,
    metavar="DIR_PATH",
    help="Directory path for result\nIf specified, videos, etc., will be stored in that directory.",
    **kwargs,
):
    parser.add_argument("--path-result", **_kwargs(locals()), **kwargs)


def goal_img(
    parser: argparse.ArgumentParser,
    required=True,
    type=str,
    metavar="PATH",
    help="Goal image path (*.npy)",
    **kwargs,
):
    parser.add_argument("--goal-img", **_kwargs(locals()), **kwargs)


def fix_xmap_size(
    parser: argparse.ArgumentParser,
    required=True,
    type=float,
    metavar="S",
    help="xmap size",
    **kwargs,
):
    parser.add_argument("--fix-xmap-size", **_kwargs(locals()), **kwargs)


def position_size(
    parser: argparse.ArgumentParser,
    required=True,
    type=float,
    metavar="S",
    help="position size",
    **kwargs,
):
    parser.add_argument("--position-size", **_kwargs(locals()), **kwargs)


def alpha(
    parser: argparse.ArgumentParser,
    required=True,
    type=float,
    metavar="α",
    help="P gain",
    **kwargs,
):
    parser.add_argument("--alpha", **_kwargs(locals()), **kwargs)


def _kwargs(a):
    a.pop("parser")
    a.pop("kwargs")
    return a
