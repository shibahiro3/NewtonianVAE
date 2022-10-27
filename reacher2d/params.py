from pprint import pprint
from typing import List, Optional, Tuple, Union

import json5


class _NewtonianVAE:
    def __init__(self, dim_x: int, transition_std: float, regularization: bool) -> None:
        assert type(dim_x) == int and dim_x > 0
        assert type(transition_std) == float
        assert type(regularization) == bool

        self.dim_x = dim_x
        self.transition_std = transition_std
        self.regularization = regularization

    @property
    def kwargs(self):
        return self.__dict__


class _NewtonianVAEDerivation:
    def __init__(self, dim_xhat: int, dim_pxhat_middle: int) -> None:
        assert type(dim_xhat) == int and dim_xhat > 0
        assert type(dim_pxhat_middle) == int and dim_pxhat_middle > 0

        self.dim_xhat = dim_xhat
        self.dim_pxhat_middle = dim_pxhat_middle

    @property
    def kwargs(self):
        return self.__dict__


class _General:
    def __init__(self, steps: int, derivation: bool) -> None:
        assert type(steps) == int and steps > 0
        assert type(derivation) == bool

        self.steps = steps
        self.derivation = derivation

    @property
    def kwargs(self):
        return self.__dict__


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
        resume,
    ) -> None:
        assert device in ("cpu", "cuda")
        assert dtype in ("float16", "float32")
        assert type(data_start) == int and data_start >= 0
        assert type(data_stop) == int and data_stop > data_start
        assert type(batch_size) == int and batch_size < data_stop - data_start
        assert type(epochs) == int and epochs > 0
        assert (
            type(grad_clip_norm) == float or type(grad_clip_norm) == int or grad_clip_norm is None
        )
        assert type(learning_rate) == float and learning_rate > 0
        assert type(save_per_epoch) == int and save_per_epoch > 0

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

    @property
    def kwargs(self):
        return self.__dict__


class _Eval:
    def __init__(
        self, device: str, dtype: str, data_start: int, data_stop: int, training: bool
    ) -> None:
        assert device in ("cpu", "cuda")
        assert dtype in ("float16", "float32")
        assert type(data_start) == int and data_start >= 0
        assert type(data_stop) == int and data_stop >= data_start
        assert type(training) == bool

        self.device = device
        self.dtype = dtype
        self.data_start = data_start
        self.data_stop = data_stop
        self.training = training

    @property
    def kwargs(self):
        return self.__dict__


class _Path:
    def __init__(self, data: str, model: str, result: str) -> None:
        assert type(data) == str
        assert type(model) == str
        assert type(result) == str

        self.data = data
        self.model = model
        self.result = result

    @property
    def kwargs(self):
        return self.__dict__


class _Reacher2D:
    def __init__(
        self,
        seed: int,
        max_episode_length: int,
        action_repeat: int,
        bit_depth: int,
        action_type: str,
    ) -> None:
        assert type(seed) == int
        assert type(max_episode_length) == int and max_episode_length > 0
        assert type(action_repeat) == int and action_repeat > 0
        assert type(bit_depth) == int
        assert action_type in ("random", "paper", "equal_paper", "handmade")

        self.seed = seed
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.action_type = action_type

    @property
    def kwargs(self):
        return self.__dict__


class Params:
    def __init__(self, path) -> None:
        self._raw = json5.load(open(path))

        self.newtonian_vae = _NewtonianVAE(**self._raw["newtonian_vae"])
        self.newtonian_vae_derivation = _NewtonianVAEDerivation(
            **self._raw["newtonian_vae_derivation"]
        )
        self.general = _General(**self._raw["general"])
        self.train = _Train(**self._raw["train"])
        self.path = _Path(**self._raw["path"])


class ParamsReacher2D(_Reacher2D):
    def __init__(self, path) -> None:
        self._raw = json5.load(open(path))
        super().__init__(**self._raw["reacher2d"])

    @property
    def kwargs(self):
        ret = self.__dict__.copy()
        ret.pop("_raw")
        return ret


class ParamsEval(_Eval):
    def __init__(self, path) -> None:
        self._raw = json5.load(open(path))
        super().__init__(**self._raw["eval"])

    @property
    def kwargs(self):
        ret = self.__dict__.copy()
        ret.pop("_raw")
        return ret


# if __name__ == "__main__":
#     params = Params("../params.json5")
#     params_reacher2d = ParamsReacher2D("../params_reacher2d.json5")
#     params_eval = ParamsEval("../params_eval.json5")
