config = dict(
    metavar="FILE",
    help="Configuration file",
)
path_model = dict(
    metavar="DIR_PATH",
    help="Directory path for models managed by date and time\nThis need not be specified since the path stored in the parameter on configuration file is used, but if specified in this argument, this path is forced to be used.",
)
path_data = dict(
    metavar="DIR_PATH",
    help="Directory path where episode data exists\nAs with --path-model, not required.",
)
path_result = dict(
    metavar="DIR_PATH",
    help="Directory path for result\nAs with --path-model, not required.",
)
fotmat_file = dict(
    help="You can save it in the path-result directory by pressing the s key on the matplotlib window."
)
