"""
Ref:
    https://github.com/Kaixhin/PlaNet/blob/master/env.py
    https://github.com/deepmind/dm_control
"""


from typing import Tuple, Union

import cv2
import numpy as np
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels
from torch import Tensor

from mypython.terminal import Color

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
    return x + 0.5


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


def _images_to_observation(image: np.ndarray, bit_depth: int, size=64) -> Tensor:
    """
    Returns:
        image: cnn input
            shape (3, size, size), range [-0.5, 0.5], color order = RGB
    """

    # Resize and put channel first
    image: Tensor = torch.tensor(
        cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
        dtype=torch.float32,
    )

    # Quantise, centre and dequantise inplace
    preprocess_observation_(image, bit_depth)

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
    ):
        assert max_episode_length <= 1000

        domain, task = env.split("-")
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed})
        if not symbolic:
            self._env = pixels.Wrapper(self._env, pixels_only=False)

        self.symbolic = symbolic
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

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
            return _images_to_observation(self.adjust_camera(), self.bit_depth)

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
            observation = _images_to_observation(self.adjust_camera(), self.bit_depth)

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
            else (3, 64, 64)
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
        camimg = self._env.physics.render(camera_id=0)  # id=1はfinger目線だったw
        return camimg

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self) -> Tensor:
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))

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
        action_type: str = "default",
        position_wrap: str = "None",
    ):
        super().__init__(
            env=env,
            symbolic=False,
            seed=seed,
            max_episode_length=max_episode_length,
            action_repeat=action_repeat,
            bit_depth=bit_depth,
        )

        self.action_type = action_type
        self.position_wrap = position_wrap

        domain, task = env.split("-")
        self.domain = domain
        self.task = task

        if domain == "reacher":
            if action_type == "default":
                pass

            elif action_type == "paper":
                # deprecated
                self.sample_random_action = lambda: torch.tensor(
                    [0.5 + (np.random.rand() - 0.5), -np.pi + 0.3 + np.random.rand() * 0.5]
                )

            elif action_type == "equal_paper":
                # deprecated
                self.sample_random_action = lambda: torch.tensor([np.random.rand(), -1])

            elif action_type == "handmade":
                self.sample_random_action = lambda: torch.tensor(
                    [np.random.uniform(-0.3, 0.7), np.random.uniform(-0.2, 0.6)]
                )

            elif action_type == "handmade2":
                self.sample_random_action = lambda: torch.tensor(
                    [np.random.uniform(-0.7, 1), np.random.uniform(-0.8, 1)]
                )

            elif action_type == "handmade3":
                self.sample_random_action = lambda: torch.tensor(
                    [np.random.uniform(-0.5, 0.7), np.random.uniform(-0.3, 0.5)]
                )
            elif action_type == "handmade4":

                def f():
                    a0min = np.random.uniform(-1, 1)
                    a0max = np.random.uniform(a0min, 1)
                    a1min = np.random.uniform(-1, 1)
                    a1max = np.random.uniform(a1min, 1)
                    # print("===========")
                    # print(a0min, a0max)
                    # print(a1min, a1max)
                    return torch.tensor(
                        [np.random.uniform(a0min, a0max), np.random.uniform(a1min, a1max)]
                    )

                self.sample_random_action = f

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

                def f():
                    theta = self.t / self.max_episode_length
                    r = 0.6
                    a = 0.5
                    x = r * np.cos(2 * np.pi * theta) + np.random.uniform(-a, a)
                    y = r * np.sin(2 * np.pi * theta) + np.random.uniform(-a, a)
                    return torch.tensor([x, y])

                self.sample_random_action = f

            else:
                assert False

    def step(self, action: Tensor) -> Tuple[Tensor, float, bool, np.ndarray]:
        """
        Returns: observation, reward, done, position
            * **observation**: shape = (3, 64, 64), range = [-0.5, 0.5], color order = RGB
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

        observation = _images_to_observation(self.adjust_camera(), self.bit_depth)
        position = self.position_wrapper(state.observation["position"])
        return observation, 0, done, position

    def adjust_camera(self):
        """Erase the cluster of stars on the left and right"""
        s = (320 - 240) // 2
        camimg = self._env.physics.render(camera_id=0)[:, s : s + 240, :]
        # print(camimg.shape)  # (240, 320, 3) -> (240, 240, 3)
        return camimg

    @staticmethod
    def position_wrapper(position):
        return position


def reacher_default2endeffectorpos(position):
    """
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
