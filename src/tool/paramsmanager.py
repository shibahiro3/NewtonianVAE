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

    # @classmethod
    # def from_kwargs(cls, **kwargs):
    #     # fetch the constructor's signature
    #     cls_fields = {field for field in signature(cls).parameters}

    #     # split the kwargs into native ones and new ones
    #     native_args, new_args = {}, {}
    #     for name, val in kwargs.items():
    #         if name in cls_fields:
    #             native_args[name] = val
    #         else:
    #             if " " in name:
    #                 print(f'Can not set attribute "{name}"')
    #             else:
    #                 new_args[name] = val

    #     # use the native ones to create the class ...
    #     ret = cls(**native_args)

    #     # ... and add the new ones by hand
    #     for new_name, new_val in new_args.items():
    #         setattr(ret, new_name, new_val)
    #     return ret


def _add_root(
    root: Optional[str], path: Union[List[str], str, None]
) -> Union[List[str], str, None]:
    if (root is not None) and (path is not None):
        if type(path) == list:
            ret: List[str] = []
            for i in range(len(path)):
                ret.append(os.path.join(root, path[i]))
        else:
            ret: str = os.path.join(root, path)

        return ret
    else:
        return None


def _remove_root(root: str, path: Union[List[str], str, None]) -> Union[List[str], str, None]:
    if path is not None:
        if type(path) == list:
            ret: List[str] = []
            for i in range(len(path)):
                ret.append(str(Path(path[i]).relative_to(root)))
        else:
            ret: str = str(Path(path).relative_to(root))

        return ret
    else:
        return None


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

    _path_root: Union[List[str], str] = ""

    def __post_init__(self):
        # check_args_type(self, self.__dict__)
        pass

    @property
    def _contents(self):
        contents = self.__dict__.copy()
        contents["path"] = _remove_root(self._path_root, contents["path"])
        return contents


@dataclasses.dataclass
class Valid(_Converter):
    path: Union[List[str], str]
    batch_size: Optional[int] = None  # Deprecated (for backward compatibility)

    _path_root: Union[List[str], str] = ""

    def __post_init__(self):
        # check_args_type(self, self.__dict__)
        pass

    @property
    def _contents(self):
        contents = self.__dict__.copy()
        contents["path"] = _remove_root(self._path_root, contents["path"])
        return contents


@dataclasses.dataclass
class Test(_Converter):
    path: Union[List[str], str]
    device: str
    dtype: str

    _path_root: Union[List[str], str] = ""

    def __post_init__(self):
        # check_args_type(self, self.__dict__)
        assert self.dtype in ("float16", "float32")

    @property
    def _contents(self):
        contents = self.__dict__.copy()
        contents["path"] = _remove_root(self._path_root, contents["path"])
        return contents


@dataclasses.dataclass
class Preprocess(_Converter):
    scale_u: Optional[list] = None
    scale_x: Optional[list] = None


@dataclasses.dataclass
class Paths(_Converter):
    saves_dir: str
    results_dir: str
    resume_weight: Optional[str] = None
    used_nvae_weight: Optional[str] = None
    hidden_conf: Optional[str] = None

    _path_root: Union[List[str], str] = ""

    def __post_init__(self):
        check_args_type(self, self.__dict__)

    @property
    def _contents(self):
        contents = self.__dict__.copy()
        contents["saves_dir"] = _remove_root(self._path_root, contents["saves_dir"])
        contents["results_dir"] = _remove_root(self._path_root, contents["results_dir"])
        contents["resume_weight"] = _remove_root(self._path_root, contents["resume_weight"])
        contents = {k: str(v) for k, v in contents.items() if v is not None}
        return contents


@dataclasses.dataclass
class HiddenConf(_Converter):
    data_root: str = ""
    save_root: str = ""

    def __post_init__(self):
        check_args_type(self, self.__dict__)


_T = TypeVar("_T")


def _instance_or_none(T: Type[_T], kwargs) -> Optional[_T]:
    # return T.from_kwargs(**kwargs) if kwargs is not None else None
    return T(**kwargs) if kwargs is not None else None


def _get_hidden_conf(hidden_conf_path: Optional[str] = None):
    """
    json5 file:
    {
        data_root: "path",
        save_root: "path",
    }

    data_root/data...
    save_root/save... (weight, etc.)
    save_root/result...
    """

    hidden_conf = HiddenConf()
    if (hidden_conf_path is not None) and Path(hidden_conf_path).exists():
        with open(hidden_conf_path) as f:
            hidden_conf = HiddenConf(**json5.load(f))
    return hidden_conf


class Params(_Converter):
    def __init__(self, path, exclusive_keys=None) -> None:
        super().__init__()

        with open(path) as f:
            self._raw: dict = json5.load(f)

        def _wrap(x):
            return x
            # if exclusive_keys is None:
            #     return x

            # if "model" in exclusive_keys:
            #     return x
            # else:
            #     return None

        self.model: Optional[str] = _wrap(self._raw.get("model", None))
        self.model_params: Optional[dict] = _wrap(self._raw.get(self.model, None))
        self.train = _wrap(_instance_or_none(Train, self._raw.get("train", None)))
        self.valid = _wrap(_instance_or_none(Valid, self._raw.get("valid", None)))
        self.test = _wrap(_instance_or_none(Test, self._raw.get("test", None)))
        self.preprocess = _wrap(_instance_or_none(Preprocess, self._raw.get("preprocess", None)))
        self.path = _wrap(_instance_or_none(Paths, self._raw.get("path", None)))
        self.pid = None  # os.getpid()
        self.others: dict = self._raw.get("others", {})  # (saved_)params.others.get(..., defalut)
        self.hidden_conf = _wrap(_get_hidden_conf(self.path.hidden_conf))

        # DO NOT USE Path()  (glob require str)
        if self.train is not None:
            self.train.path = _add_root(self.hidden_conf.data_root, self.train.path)
            self.train._path_root = self.hidden_conf.data_root
        if self.valid is not None:
            self.valid.path = _add_root(self.hidden_conf.data_root, self.valid.path)
            self.valid._path_root = self.hidden_conf.data_root
        if self.test is not None:
            self.test.path = _add_root(self.hidden_conf.data_root, self.test.path)
            self.test._path_root = self.hidden_conf.data_root
        if self.path is not None:
            self.path.saves_dir = os.path.join(self.hidden_conf.save_root, self.path.saves_dir)
            self.path.results_dir = os.path.join(self.hidden_conf.save_root, self.path.results_dir)
            self.path._path_root = self.hidden_conf.save_root

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
            contents=self._select(["model", self.model, "train", "preprocess", "path", "pid"]),
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
