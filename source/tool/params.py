from pprint import pformat
from typing import Union

import json5


class _Train:
    def __init__(
        self,
        device: str,
        dtype: str,
        data_start: int,
        data_stop: int,
        batch_size: int,
        epochs: int,
        grad_clip_norm: Union[int, float, None],
        learning_rate: Union[int, float],
        save_per_epoch: int,
        max_time_length: int,
        resume,
    ) -> None:
        assert device in ("cpu", "cuda")
        assert dtype in ("float16", "float32")

        self.data_start = data_start
        self.data_stop = data_stop
        self.batch_size = batch_size
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.learning_rate = learning_rate
        self.save_per_epoch = save_per_epoch
        self.device = device
        self.dtype = dtype
        self.resume = resume
        self.max_time_length = max_time_length

    @property
    def kwargs(self):
        return self.__dict__


class _Eval:
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

    @property
    def kwargs(self):
        return self.__dict__


class Params:
    def __init__(self, path) -> None:
        self.raw_ = json5.load(open(path))

        self.model: str = self.raw_["model"]
        self.train = _Train(**self.raw_["train"])

    def __str__(self):
        return pformat(self.raw_)


class ParamsEval(_Eval):
    def __init__(self, path) -> None:
        self.raw_ = json5.load(open(path))
        super().__init__(**self.raw_["eval"])

    @property
    def kwargs(self):
        ret = self.__dict__.copy()
        ret.pop("_raw")
        return ret

    def __str__(self):
        return pformat(self.raw_)


class _SimEnv:
    def __init__(
        self,
        env: str,
        seed: int,
        max_episode_length: int,
        action_repeat: int,
        bit_depth: int,
        action_type: str,
    ) -> None:
        domain, task = env.split("-")
        if domain == "pointmass":
            assert action_type in ("random", "circle")
        elif domain == "reacher":
            assert action_type in ("random", "paper", "equal_paper", "handmade")

        self.env = env
        self.seed = seed
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.action_type = action_type

    @property
    def kwargs(self):
        return self.__dict__


class ParamsSimEnv(_SimEnv):
    def __init__(self, path) -> None:
        self._raw = json5.load(open(path))
        super().__init__(**self._raw["simenv"])

    @property
    def kwargs(self):
        ret = self.__dict__.copy()
        ret.pop("_raw")
        return ret

    def __str__(self):
        return pformat(self._raw)
