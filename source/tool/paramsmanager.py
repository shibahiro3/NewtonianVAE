import dataclasses
from collections import ChainMap
from pathlib import Path
from pprint import pprint
from typing import Optional, Union

from mypython.pyutil import check_args_type
from mypython.terminal import Color
from third import json5


class _Converter:
    @property
    def _contents(self):
        return self.__dict__

    @property
    def kwargs(self):
        return self._contents

    def __str__(self) -> str:
        return _dumps(self._contents)

    def to_json(self):
        return self._contents

    def _save(self, path, contents: dict, msg="saved params:", lock=True):
        path = Path(path)
        assert path.suffix == ".json5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w") as f:
            f.write(_dumps(contents))
        if lock:
            path.chmod(0o444)
        Color.print(msg, path)

    def save(self, path, msg="saved params:", lock=True):
        self._save(path, self._contents, msg, lock)


@dataclasses.dataclass
class Train(_Converter):
    device: str
    dtype: str
    data_start: int
    data_stop: int
    batch_size: int
    epochs: int
    learning_rate: Union[int, float]
    save_per_epoch: int
    max_time_length: int
    grad_clip_norm: Union[None, int, float] = None
    seed: Optional[int] = None
    kl_annealing: bool = False
    check_value: bool = True

    def __post_init__(self):
        check_args_type(self, self.__dict__)
        assert self.dtype in ("float16", "float32")


@dataclasses.dataclass
class Eval(_Converter):
    device: str
    dtype: str
    data_start: Optional[int] = None
    data_stop: Optional[int] = None

    def __post_init__(self):
        check_args_type(self, self.__dict__)
        assert self.dtype in ("float16", "float32")


@dataclasses.dataclass
class Preprocess(_Converter):
    scale_u: Optional[list] = None
    scale_x: Optional[list] = None


@dataclasses.dataclass
class Paths(_Converter):
    data_dir: Optional[str] = None
    saves_dir: Optional[str] = None
    results_dir: Optional[str] = None
    resume_weight: Optional[str] = None
    used_nvae_weight: Optional[str] = None

    def __post_init__(self):
        check_args_type(self, self.__dict__)

    @property
    def _contents(self):
        return {k: str(v) for k, v in self.__dict__.items() if v is not None}


class Params(_Converter):
    def __init__(self, path) -> None:
        super().__init__()

        self.raw_: dict = json5.load(open(path))

        def instance_or_none(T, kwargs):
            return T(**kwargs) if kwargs is not None else None

        self.model: str = self.raw_.get("model", None)
        self.model_params: dict = self.raw_.get(self.model, None)
        self.train = instance_or_none(Train, self.raw_.get("train", None))
        self.eval = instance_or_none(Eval, self.raw_.get("eval", None))
        self.preprocess = instance_or_none(Preprocess, self.raw_.get("preprocess", None))
        self.path = instance_or_none(Paths, self.raw_.get("path", None))
        self.pid = None  # os.getpid()

    def _select(self, save_keys):
        model_params = {self.model: self.model_params}
        contents = ChainMap(model_params, self.__dict__)

        new_contents = {}
        for k in save_keys:
            new_contents[k] = contents[k]

        return new_contents

    def save_train(self, path):
        self._save(
            path=path,
            contents=self._select(["model", self.model, "train", "path", "pid"]),
        )

    def save_train_ctrl(self, path):
        self._save(
            path=path,
            contents=self._select(["model", self.model, "train", "preprocess", "path", "pid"]),
        )

    def save_simenv(self, path):
        self._save(
            path=path,
            contents={"ControlSuiteEnvWrap": self.raw_["ControlSuiteEnvWrap"]},
            msg="saved simenv params:",
        )


def default_to_json(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    else:
        raise TypeError(f"Object of type {obj.__class__} is not JSON serializable")


def _dumps(_contents: dict):
    ret = json5.dumps(_contents, default=default_to_json, indent=2)
    return ret
