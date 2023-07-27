import dataclasses
import os
import sys
from collections import ChainMap
from inspect import signature
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from typing_extensions import final

from mypython import rdict
from mypython.pyutil import check_args_type
from mypython.terminal import Color
from third import json5


def _remove_private(d: Dict[str, Any]):
    """
    Recursively deletes all keys in a dictionary that start with '_'.
    """
    d_ = d.copy()
    for k in list(d_.keys()):
        if k.startswith("_"):
            del d_[k]
        elif isinstance(d_[k], dict):
            _remove_private(d_[k])
    return d_


class _Converter:
    @property
    def _contents(self):
        return self.__dict__

    @property
    def kwargs(self):
        return self._contents

    def __str__(self) -> str:
        return _dumps(self._contents)

    @final
    def to_json(self):
        return _remove_private(self._contents)

    def _save(self, path, contents: Dict[str, Any], msg="saved params:", lock=True):
        contents = _remove_private(contents)

        path = Path(path)
        assert path.suffix == ".json5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w") as f:
            f.write(_dumps(_remove_private(contents)))
        if lock:
            path.chmod(0o444)
        Color.print(msg, path)

    def save(self, path, msg="saved params:", lock=True):
        self._save(path, _remove_private(self._contents), msg, lock)


@dataclasses.dataclass
class Train(_Converter):
    path: Union[List[str], str]
    device: str
    dtype: str
    epochs: int
    learning_rate: Union[int, float]
    save_per_epoch: int
    max_time_length: Union[int, str]
    batch_size: int
    grad_clip_norm: Union[None, int, float] = None
    seed: Optional[int] = None
    check_value: bool = True
    gradscaler_args: Optional[dict] = None
    use_autocast: bool = False
    load_all: bool = False


@dataclasses.dataclass
class Valid(_Converter):
    path: Union[List[str], str]
    batch_size: Optional[int] = None  # Deprecated (for backward compatibility)


@dataclasses.dataclass
class Test(_Converter):
    path: Union[List[str], str]
    device: str
    dtype: str  # "float16", "float32"


@dataclasses.dataclass
class Paths(_Converter):
    saves_dir: str
    results_dir: str
    resume_weight: Optional[str] = None
    used_nvae_weight: Optional[str] = None

    def __post_init__(self):
        check_args_type(self, self.__dict__)

    @property
    def _contents(self):
        contents = self.__dict__.copy()
        contents = {k: str(v) for k, v in contents.items() if v is not None}
        return contents


_T = TypeVar("_T")


def _i_n(T: Type[_T], kwargs) -> Optional[_T]:
    # instance or none -> i_n
    # return T.from_kwargs(**kwargs) if kwargs is not None else None
    return T(**kwargs) if kwargs is not None else None


class Params(_Converter):
    def __init__(self, path, exclusive_keys=None) -> None:
        super().__init__()

        with open(path) as f:
            self._raw: dict = json5.load(f)

        self.model: Optional[str] = self._raw.get("model", None)
        self.model_params: Optional[dict] = self._raw.get(self.model, None)
        self.train = _i_n(Train, self._raw.get("train", None))
        self.valid = _i_n(Valid, self._raw.get("valid", None))
        self.test = _i_n(Test, self._raw.get("test", None))
        self.path = _i_n(Paths, self._raw.get("path", None))
        self.pid = None  # os.getpid()
        self.others: dict = self._raw.get("others", {})  # (saved_)params.others.get(..., defalut)

    @property
    def raw(self):
        return self._raw

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
            contents=self._select(["model", self.model, "train", "valid", "path", "pid", "others"]),
        )

    def save_train_ctrl(self, path):
        self._save(
            path=path,
            contents=self._select(["model", self.model, "train", "path", "pid"]),
        )

    def save_simenv(self, path):
        self._save(
            path=path,
            contents={"ControlSuiteEnvWrap": self.raw["ControlSuiteEnvWrap"]},
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
