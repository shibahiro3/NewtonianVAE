from collections import ChainMap
from pathlib import Path
from typing import Optional, Union

import json5
from torch import NumberType

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


class Train(_Converter):
    def __init__(
        self,
        device: str,
        dtype: str,
        data_start: int,
        data_stop: int,
        batch_size: int,
        epochs: int,
        learning_rate: NumberType,
        save_per_epoch: int,
        max_time_length: int,
        grad_clip_norm: Optional[NumberType] = None,
        seed: Optional[int] = None,
        kl_annealing=False,
    ) -> None:
        assert device in ("cpu", "cuda")
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
        self, device: str, dtype: str, data_start: int, data_stop: int, training: bool
    ) -> None:
        assert device in ("cpu", "cuda")
        assert dtype in ("float16", "float32")

        self.device = device
        self.dtype = dtype
        self.data_start = data_start
        self.data_stop = data_stop
        self.training = training


class TrainExternal(_Converter):
    def __init__(
        self,
        data_path: str,
        data_id: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> None:

        self.data_path = data_path
        self.data_id = data_id
        self.resume_from = resume_from


class Params(_Converter):
    def __init__(self, path) -> None:
        self.raw_: dict = json5.load(open(path))

        self.model: str = self.raw_["model"]
        self.train = Train(**self.raw_["train"])
        external = self.raw_.get("external")
        self.external = TrainExternal(**external) if external is not None else None

    @property
    def _contents(self):
        tmp = self.__dict__.copy()
        tmp.pop("raw_")

        model_params = {self.model: self.raw_[self.model]}
        # contents = ChainMap(model_params, ret)
        contents = dict(model_params, **tmp)
        return contents

    def save(self, path, lock=True):
        path = Path(path)
        assert path.suffix == ".json5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w") as f:
            f.write(_dumps(self._contents))
        if lock:
            path.chmod(0o444)


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
