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

"""Point-mass domain."""

import collections

import numpy as np

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import io as resources
from dm_control.utils import rewards

# from mypython.receiver import Receiver
# receiver = Receiver(port=60333, typ="pickle")
# receiver.start()


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()

from pprint import pprint

# Changed/added by Sugar
from third.dm_control import read_model


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  # return common.read_model('point_mass.xml'), common.ASSETS
  return read_model("point_mass_3d.xml"), common.ASSETS  # Changed/added by Sugar


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, task_settings=None, environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_gains=False, random=random, task_settings=task_settings)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, task_settings=None, environment_kwargs=None):
  """Returns the hard point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_gains=True, random=random, task_settings=task_settings)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

  def mass_to_target(self):
    """Returns the vector from mass to target in global coordinate."""

    # Changed/added by Sugar
    # Guarantee that the task will not be terminated
    # True is original implementation
    if False:
        return (self.named.data.geom_xpos['target'] -
                self.named.data.geom_xpos['pointmass'])
    else:
        return (np.array([0, 0.2]))

  def mass_to_target_dist(self):
    """Returns the distance from mass to the target."""
    return np.linalg.norm(self.mass_to_target())


class PointMass(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, task_settings=None):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    super().__init__(random=random)

    # Changed/added by Sugar
    if task_settings is None:
      self.task_settings = {}
    else:
      self.task_settings = task_settings
    self._set_position()

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

       If _randomize_gains is True, the relationship between the controls and
       the joints is randomized, so that each control actuates a random linear
       combination of joints.

    Args:
      physics: An instance of `mujoco.Physics`.
    """

    ### Changed/added by Sugar

    # None or "defalut" are similar to original implementation
    target_position = self.task_settings.get("target_position", None)
    target_size = self.task_settings.get("target_size", None)

    self._set_position()
    if False:
      randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    else:
      physics.named.data.qpos["root_x"] = self.init_self_pos[0]
      physics.named.data.qpos["root_y"] = self.init_self_pos[1]
      physics.named.data.qpos["root_z"] = self.init_self_pos[2]

    if target_position is None or target_position == "default":
      pass
    elif target_position == "random":
      physics.named.model.geom_pos["target"][:3] = self.target_pos
    else:
      assert False

    if target_size is not None:
      assert type(target_size) == float
      physics.named.model.geom_size["target"][0] = target_size
    
    # print(physics.named.data.qpos)
    # print(physics.named.data.geom_xpos)
    # print(physics.named.model.geom_size)
    # print(physics.named.model.geom_pos)
    # pprint(dir(physics.named.model))
    # print(physics.named.model.cam_quat)
    # physics.named.model.cam_quat["cam0"][:2] = np.array([0, 0])
    # physics.named.model.cam_quat["fixed"] = np.array([0.707, 0.707, 0, 0])
    ###

    if self._randomize_gains:
      dir1 = self.random.randn(2)
      dir1 /= np.linalg.norm(dir1)
      # Find another actuation direction that is not 'too parallel' to dir1.
      parallel = True
      while parallel:
        dir2 = self.random.randn(2)
        dir2 /= np.linalg.norm(dir2)
        parallel = abs(np.dot(dir1, dir2)) > 0.9
      physics.model.wrap_prm[[0, 1]] = dir1
      physics.model.wrap_prm[[2, 3]] = dir2
    super().initialize_episode(physics)

  # def before_step(self, action, physics):
  #   data = receiver.get()
  #   print(data)
  #   if data is not None:
  #     physics.named.model.cam_pos["y side"] = list(data.values())[:3]
  #     # physics.named.model.cam_quat["y side"] = list(data.values())
    
  #   print(physics.named.model.cam_pos["y side"])
  #   print(physics.named.model.cam_quat["y side"])
  #   super().before_step(action, physics)

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()
    # Changed/added by Sugar
    obs['target_position'] = self.target_pos
    obs['relative_position'] = obs['position'] - obs['target_position']
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    target_size = physics.named.model.geom_size['target', 0]
    near_target = rewards.tolerance(physics.mass_to_target_dist(),
                                    bounds=(0, target_size), margin=target_size)
    control_reward = rewards.tolerance(physics.control(), margin=1,
                                       value_at_margin=0,
                                       sigmoid='quadratic').mean()
    small_control = (control_reward + 4) / 5
    return near_target * small_control

  # Changed/added by Sugar
  def _set_position(self):
    wall = 0.3
    self.init_self_pos = self.random.uniform(-wall/2, wall/2, size=3)
    self.init_self_pos[2] = self.random.uniform(wall*0.3, wall*0.75)
    # self.init_self_pos[2] = self.random.uniform(wall*0.8, wall*1)
    r = self.random.uniform(-0.1, 0.1)
    theta = self.random.uniform(0, 2*np.pi)
    phi = self.random.uniform(0, 2*np.pi)
    self.target_pos = self.init_self_pos + r * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    # self.target_pos = self.random.uniform(-wall, wall, size=2)
