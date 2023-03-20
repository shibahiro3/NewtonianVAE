"""
References:
    https://github.com/Kaixhin/PlaNet/blob/master/env.py
    https://github.com/deepmind/dm_control
"""


from collections import ChainMap
from numbers import Real
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

import mypython.vision as mv
from dm_control.suite.wrappers import pixels
from mypython.terminal import Color

# from dm_control import suite
from third.dm_control import suite

GYM_ENVS = [
    "Pendulum-v0",
    "MountainCarContinuous-v0",
    "Ant-v2",
    "HalfCheetah-v2",
    "Hopper-v2",
    "Humanoid-v2",
    "HumanoidStandup-v2",
    "InvertedDoublePendulum-v2",
    "InvertedPendulum-v2",
    "Reacher-v2",
    "Swimmer-v2",
    "Walker2d-v2",
]
CONTROL_SUITE_ENVS = [
    "cartpole-balance",
    "cartpole-swingup",
    "reacher-easy",
    "finger-spin",
    "cheetah-run",
    "ball_in_cup-catch",
    "walker-walk",
]
# CONTROL_SUITE_ACTION_REPEATS = {
#     "cartpole": 8,
#     "reacher": 4,
#     "finger": 2,
#     "cheetah": 4,
#     "ball_in_cup": 6,
#     "walker": 2,
# }


def img2obs(x):
    """range [0, 1] -> [-0.5, 0.5]"""
    return x - 0.5


def obs2img(x):
    """range [-0.5, 0.5] -> [0, 1]"""
    x = x + 0.5
    x = mv.clip(x, 0, 1)

    # TODO:
    # postprocess_observation

    x = mv.cnn2plt(x)
    return x


def preprocess_observation_(observation: Tensor, bit_depth: int) -> None:
    """Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])"""
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2**bit_depth).sub_(0.5)

    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2**bit_depth))


def postprocess_observation(observation, bit_depth):
    """Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])"""
    return np.clip(
        np.floor((observation + 0.5) * 2**bit_depth) * 2 ** (8 - bit_depth),
        0,
        2**8 - 1,
    ).astype(np.uint8)


def _images_to_observation(image: np.ndarray, bit_depth: int, size: tuple) -> Tensor:
    """
    Args:
        image: (240, 240, 3)
        size: (H, W)

    Returns:
        image: cnn input
            shape (3, H, W), range [-0.5, 0.5], color order = RGB
    """

    # Resize and put channel first
    image: Tensor = torch.tensor(
        cv2.resize(image, tuple(reversed(size)), interpolation=cv2.INTER_LINEAR).transpose(
            2, 0, 1
        ),  # change color order
        dtype=torch.float32,
    )

    # Quantise, centre and dequantise inplace
    preprocess_observation_(image, bit_depth)

    # assert image.shape[-2:] == size

    return image  # .unsqueeze(dim=0)  # Add batch dimension


class ControlSuiteEnv:
    def __init__(
        self,
        env: str,
        symbolic: bool,
        seed: int,
        max_episode_length: int,
        action_repeat: int,
        bit_depth: int,
        imgsize: Union[list, tuple],  # (H, W)
        camera_id=0,
        task_settings: Optional[dict] = None,
    ):
        assert max_episode_length <= 1000

        self.random = np.random.RandomState(seed)

        domain, task = env.split("-")

        if task_settings is None:
            task_kwargs = {"random": seed}
        else:
            task_kwargs = {"random": seed, "task_settings": task_settings}

        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs=task_kwargs)
        # if not symbolic:
        # self._env = pixels.Wrapper(self._env, pixels_only=False)

        self.symbolic = symbolic
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.imgsize = tuple(imgsize)
        self.camera_id = camera_id

        # if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
        #     print(
        #         "Using action repeat %d; recommended action repeat for domain is %d"
        #         % (action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain])
        #     )

    def reset(self) -> Tensor:
        self.t = 0  # Reset internal timer
        state = self._env.reset()

        if self.symbolic:
            return self._symbolic_state2observation(state)
        else:
            return _images_to_observation(self.adjust_camera(), self.bit_depth, self.imgsize)

    def step(self, action: Tensor) -> Tuple[Tensor, float, bool]:
        action = action.detach().cpu().numpy()

        reward = 0
        for k in range(self.action_repeat):
            state = self._env.step(action)
            self.t += 1  # Increment internal timer
            done = state.last() or self.t == self.max_episode_length
            if done:
                break
            else:
                reward += state.reward

        if self.symbolic:
            observation = self._symbolic_state2observation(state)
        else:
            observation = _images_to_observation(self.adjust_camera(), self.bit_depth, self.imgsize)

        return observation, reward, done

    def render(self):
        """綺麗なまま表示する"""
        cv2.imshow("screen", self.adjust_camera()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return (
            sum(
                [
                    (1 if len(obs.shape) == 0 else obs.shape[0])
                    for obs in self._env.observation_spec().values()
                ]
            )
            if self.symbolic
            else (3,) + self.imgsize
        )

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    @property
    def action_range(self):
        return float(self._env.action_spec().minimum[0]), float(self._env.action_spec().maximum[0])

    @staticmethod
    def _symbolic_state2observation(state):
        return torch.tensor(
            np.concatenate(
                [
                    np.asarray([obs]) if isinstance(obs, float) else obs
                    for obs in state.observation.values()
                ],
                axis=0,
            ),
            dtype=torch.float32,
        )  # .unsqueeze(dim=0)

    def adjust_camera(self):
        camimg = self._env.physics.render(camera_id=self.camera_id)
        return camimg

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self) -> Tensor:
        spec = self._env.action_spec()
        return torch.from_numpy(self.random.uniform(spec.minimum, spec.maximum, spec.shape))

    def zeros_action(self) -> Tensor:
        spec = self._env.action_spec()
        return torch.zeros(spec.shape)


class ControlSuiteEnvWrap(ControlSuiteEnv):
    def __init__(
        self,
        env: str,
        seed: int,
        max_episode_length: int,
        action_repeat: int,
        bit_depth: int,
        imgsize: Union[list, tuple],
        action_type: str = "default",
        position_wrap: str = "None",
        camera_id=0,
        task_settings: Optional[str] = None,
    ):
        super().__init__(
            env=env,
            symbolic=False,
            seed=seed,
            max_episode_length=max_episode_length,
            action_repeat=action_repeat,
            bit_depth=bit_depth,
            task_settings=task_settings,
            camera_id=camera_id,
            imgsize=imgsize,
        )

        self.action_type = action_type
        self.position_wrap = position_wrap

        domain, task = env.split("-")
        self.domain = domain
        self.task = task

        if domain == "reacher2d":
            if action_type == "default":
                pass
            elif action_type == "forward":
                self.sample_random_action = self.action_forward
            else:
                assert False

            if position_wrap == "None":
                pass
            elif position_wrap == "endeffector":
                self.position_wrapper = reacher_default2endeffectorpos
            else:
                assert False

        elif domain == "point_mass":
            if action_type == "default":
                pass
            elif action_type == "circle":
                self.sample_random_action = self.action_point_mass_circle
            elif action_type == "per_episode":
                self.sample_random_action = self.action_per_episode
            elif action_type == "random_walk":
                self.sample_random_action = lambda: self.action_random_walk(2, range=(-0.3, 0.3))
            else:
                assert False

        elif domain == "point_mass_3d":
            if action_type == "default":
                pass
            elif action_type == "random_walk":
                self.sample_random_action = lambda: self.action_random_walk(3, range=(-0.3, 0.3))
            else:
                assert False

    def _cameras(self) -> Dict[str, np.ndarray]:
        """raw (np.ndarray)
        (H, W, RGB) (plt)"""

        cameras = {}
        # https://github.com/deepmind/dm_control/issues/116
        for name in self._env.physics.named.model.cam_quat.axes.row.names:
            cameras[name] = self._env.physics.render(height=480, width=480, camera_id=name)
        return cameras

    def _observations(self, state):
        observations = state.observation
        observations["camera"] = {}
        for k, v in self._cameras().items():
            observations["camera"][k] = _images_to_observation(v, self.bit_depth, self.imgsize)
        return observations

    def render(self):
        """綺麗なまま表示する"""
        cameras = self._cameras()
        winname = "camera " + ", ".join(cameras.keys())
        mv.show_imgs_cv(list(cameras.values()), winname, cols=4, color_order="RGB", block=False)

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        return self._observations(state)

    def step(self, action: Tensor) -> Tuple[Dict[str, Union[np.ndarray, Tensor, Real]], bool]:
        """
        Returns: observation, reward, done, position
            * **observation**: shape = (3, H, W), range = [-0.5, 0.5], color order = RGB
            * **reward**: 0
            * **done**: True when time reaches max_episode_length // action_repeat
            * **position**: coordinate, or two angle, etc. depends on env
        """

        action = action.detach().cpu().numpy()

        done = False

        for k in range(self.action_repeat):
            state = self._env.step(action)
            self.t += 1  # Increment internal timer

            done = state.last() or self.t == self.max_episode_length
            if done:
                break

        return self._observations(state), done

    def adjust_camera(self):
        """Erase the cluster of stars on the left and right"""
        # https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L171

        # H, W = 240, 320 # default
        # H, W = 480, 640  # max
        # s = (W - H) // 2

        # camimg = self._env.physics.render(camera_id=0)[:, s : s + H, :]
        # camimg = self._env.physics.render(height=H, width=W, camera_id=0)[:, s : s + H, :]
        # print(camimg.shape)  # (H, W, 3) -> (H, H, 3)

        camimg = self._env.physics.render(height=480, width=480, camera_id=self.camera_id)

        return camimg

    @staticmethod
    def position_wrapper(position):
        return position

    def action_per_episode(self):
        if self.t == 0:
            self.action_mean = super().sample_random_action()

        return self.action_mean

    def action_random_walk(self, dim: int, range: tuple):
        if self.t == 0:
            self._action = torch.from_numpy(self.random.uniform(range[0], range[1], size=(dim,)))
        else:
            self._action += torch.from_numpy(self.random.uniform(range[0], range[1], size=(dim,)))

        # self._action[0] = 0
        # self._action[1] = 0
        # self._action[2] = 0
        return torch.clip(self._action, -1, 1)

    def action_point_mass_circle(self):
        theta = self.t / self.max_episode_length
        r = 0.6
        a = 0.5
        x = r * np.cos(2 * np.pi * theta) + self.random.uniform(-a, a)
        y = r * np.sin(2 * np.pi * theta) + self.random.uniform(-a, a)
        return torch.tensor([x, y])

    def action_forward(self):
        return torch.tensor([self.random.uniform(-0.3, 0.7), self.random.uniform(-0.2, 0.6)])


def reacher_default2endeffectorpos(position):
    """
    Deprecated
        Hard coding (0.12 from xml file)

    Args:
        position: shape (*, 2)

    Returns: target position (end effector)
    """
    arg = position[..., 0]
    wrist = 0.12 * roll_2d(arg)
    arg += position[..., 1]
    target = 0.12 * roll_2d(arg)
    return wrist + target


def roll_2d(arg):
    return np.stack([np.cos(arg), np.sin(arg)]).T


def reacher_fix_arg_range(arg):
    arg_ = np.where(
        (-np.pi < arg) & (arg < np.pi), arg, arg - 2 * np.pi * (arg // (2 * np.pi)) - np.pi
    )
    # assert (-np.pi < arg_).all().item() and (arg_ < np.pi).all().item()
    return arg_
