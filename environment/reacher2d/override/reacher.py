# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Reacher domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05

# Changed/added by Sugar
# _SMALL_TARGET = .015
_SMALL_TARGET = .02


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('reacher.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Reacher(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns reacher with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Reacher(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Reacher domain."""

  def finger_to_target(self):
    """Returns the vector from target to finger in global coordinates."""

    # Changed/added by Sugar
    # Guarantee that the task will not be terminated
    # True is original implementation
    if False:
        return (self.named.data.geom_xpos['target', :2] -
                self.named.data.geom_xpos['finger', :2])
    else:
        return (np.array([0, 0.2]))

  def finger_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.finger_to_target())


class Reacher(base.Task):
  """A reacher `Task` to reach the target."""

  def __init__(self, target_size, random=None):
    """Initialize an instance of `Reacher`.

    Args:
      target_size: A `float`, tolerance to determine whether finger reached the
          target.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._target_size = target_size
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    physics.named.model.geom_size['target', 0] = self._target_size

    # Changed/added by Sugar
    # Initial arm position
    # True is original implementation
    if True:
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    else:
        # radian

        # zero
        physics.named.data.qpos["shoulder"] = np.array([0])
        physics.named.data.qpos["wrist"] = np.array([0])

        ### 160 [deg] == (8/9)π ≈ 2.79 [rad]

        # red (target)
        # physics.named.data.qpos["shoulder"] = np.array([(-7/9)*np.pi])
        # physics.named.data.qpos["wrist"] = np.array([(3/9)*np.pi])

        # green (target2)
        # physics.named.data.qpos["shoulder"] = np.array([(-4/9)*np.pi])
        # physics.named.data.qpos["wrist"] = np.array([(6/9)*np.pi])

        # yellow (target3)
        # physics.named.data.qpos["shoulder"] = np.array([(5/9)*np.pi])
        # physics.named.data.qpos["wrist"] = np.array([(3/9)*np.pi])

        # paper...?
        # Here is the initialize process. 
        # [0.0, 1.0) [rad] == [0.0, 57.29) [deg]
        # physics.named.data.qpos["shoulder"] = 0.5 + (np.random.rand() - 0.5)
        # [-2.84, -2.34) [rad] == [-162.72, -134.07) [deg]
        # physics.named.data.qpos["wrist"] = -np.pi + 0.3 + np.random.rand() * 0.5

        # print(physics.named.data.qpos)


    # Changed/added by Sugar
    # Randomize target position (red ball)
    # True is original implementation
    if False:
        angle = self.random.uniform(0, 2 * np.pi)
        radius = self.random.uniform(.05, .20)
        physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
        physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)
    else:
        physics.named.model.geom_pos["target"] = np.array([-0.07108755, -0.19531144, 0])

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state and the target position."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_target'] = physics.finger_to_target()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    radii = physics.named.model.geom_size[['target', 'finger'], 0].sum()
    return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))
