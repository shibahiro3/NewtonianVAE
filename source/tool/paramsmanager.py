from collections import ChainMap
from pathlib import Path
from pprint import pprint
from typing import Optional, Union

from third import json5

from mypython.pyutil import check_args_type
from mypython.terminal import Color


class _Converter:
    @property
    def _contents(self):
        return self.__dict__

    @property
    def kwargs(self):
        return self._contents

    def __str__(self) -> str:
        # Color.print(self.__class__, "__str__", c=Color.coral)
        return _dumps(self._contents)

    def to_json(self):
        # Color.print(self.__class__, "to_json")
        return self._contents

    def save(self, path, lock=True):
        path = Path(path)
        assert path.suffix == ".json5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w") as f:
            f.write(_dumps(self._contents))
        if lock:
            path.chmod(0o444)
        Color.print("saved params:", path)


class Train(_Converter):
    def __init__(
        self,
        device: str,
        dtype: str,
        data_start: int,
        data_stop: int,
        batch_size: int,
        epochs: int,
        learning_rate: Union[int, float],
        save_per_epoch: int,
        max_time_length: int,
        grad_clip_norm: Union[None, int, float] = None,
        seed: Optional[int] = None,
        kl_annealing: bool = False,
    ) -> None:
        check_args_type(self.__init__, locals())

        assert dtype in ("float16", "float32")

        self.data_start = data_start
        self.data_stop = data_stop
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.dtype = dtype
        self.max_time_length = max_time_length
        self.seed = seed
        self.grad_clip_norm = grad_clip_norm
        self.kl_annealing = kl_annealing

        # === Not related to learning ===
        self.save_per_epoch = save_per_epoch


class Eval(_Converter):
    def __init__(
        self,
        device: str,
        dtype: str,
        data_start: int,
        data_stop: int,
        result_path: str,
        training: bool = False,
    ) -> None:
        check_args_type(self.__init__, locals())

        assert dtype in ("float16", "float32")

        self.device = device
        self.dtype = dtype
        self.data_start = data_start
        self.data_stop = data_stop
        self.training = training
        self.result_path = result_path


class TrainExternal(_Converter):
    def __init__(
        self,
        data_path: str,
        save_path: str,
        data_id: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        check_args_type(self.__init__, locals())

        self.data_path = data_path
        self.save_path = save_path
        self.data_id = data_id
        self.resume_from = resume_from


class Params(_Converter):
    def __init__(self, path) -> None:
        self.raw_: dict = json5.load(open(path))

        def class_or_none(T, kwargs):
            return T(**kwargs) if kwargs is not None else None

        self.model: str = self.raw_.get("model", None)
        self.train = class_or_none(Train, self.raw_.get("train", None))
        self.external = class_or_none(TrainExternal, self.raw_.get("external", None))
        self.eval = class_or_none(Eval, self.raw_.get("eval", None))

    @property
    def _contents(self):
        tmp = self.__dict__.copy()
        tmp.pop("raw_")

        model_params = {self.model: self.raw_[self.model]}
        contents = ChainMap(model_params, tmp)
        # contents = dict(model_params, **tmp)

        save_keys = ["model", self.model, "train", "external"]

        new_contents = {}
        for k in save_keys:
            new_contents[k] = contents[k]

        # delete_keys = contents.keys() ^ save_keys
        # for k in delete_keys:
        #     contents.pop(k)

        return new_contents

    def save_simenv(self, path, lock=True):
        path = Path(path)
        assert path.suffix == ".json5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w") as f:
            f.write(_dumps({"ControlSuiteEnvWrap": self.raw_["ControlSuiteEnvWrap"]}))
        if lock:
            path.chmod(0o444)
        Color.print("saved simenv params:", path)


class ParamsEval(Eval):
    def __init__(self, path) -> None:
        self.raw_ = json5.load(open(path))
        super().__init__(**self.raw_["eval"])

    @property
    def _contents(self):
        ret = self.__dict__.copy()
        ret.pop("raw_")
        return ret


def default_to_json(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    else:
        raise TypeError(f"Object of type {obj.__class__} is not JSON serializable")


def _dumps(_contents: dict):
    ret = json5.dumps(_contents, default=default_to_json, indent=2)
    return ret
