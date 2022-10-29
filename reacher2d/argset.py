import argparse


def parse_save_anim(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-s",
        "--save-anim",
        action="store_true",
        help="Save animation",
    )


def parse_cf(parser: argparse.ArgumentParser, default="../params.json5"):
    parser.add_argument(
        "--cf",
        type=str,
        default=default,
        metavar="FILE",
        help="Configuration file of common",
    )


def parse_cf_eval(parser: argparse.ArgumentParser, default="../params_eval.json5"):
    parser.add_argument(
        "--cf-eval",
        type=str,
        default=default,
        metavar="FILE",
        help="Configuration file of evaluation",
    )


def parse_cf_reacher2d(parser: argparse.ArgumentParser, default="../params_reacher2d.json5"):
    parser.add_argument(
        "--cf-reacher2d",
        type=str,
        default=default,
        metavar="FILE",
        help="Configuration file or Reacher-2D",
    )


def parse_episodes(parser: argparse.ArgumentParser, default=320):
    parser.add_argument(
        "--episodes",
        type=int,
        default=default,
        metavar="E",
        help="Total number of episodes",
    )


def parse_watch(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--watch",
        type=str,
        choices=["render", "plt"],
        help="Check data without saving data. For rendering, you can choose to use OpenCV (render) or Matplotlib (plt).",
    )


def parse_path_model(parser: argparse.ArgumentParser, default="saves"):
    parser.add_argument(
        "-m",
        "--path-model",
        type=str,
        default=default,
        metavar="FILE",
        help="Directory paths for models managed by date and time",
    )
