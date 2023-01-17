# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Goal level 0."""

from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np
from safety_gymnasium.bases.base_task import BaseTask


class HumanrunLevel0(BaseTask):
    """A agent must navigate to a goal."""

    def __init__(self, config):
        super().__init__(config=config)

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        # xy_position_before = mass_center(self.model, self.data)
        # self.do_simulation(action, self.frame_skip)
        # xy_position_after = mass_center(self.model, self.data)

        # pylint: disable-next=wrong-spelling-in-comment
        # xy_velocity = (xy_position_after - xy_position_before) / self.dt
        # x_velocity, y_velocity = xy_velocity

        # ctrl_cost = self.control_cost(action)

        # forward_reward = self._forward_reward_weight * x_velocity
        # healthy_reward = self.healthy_reward

        # rewards = forward_reward + healthy_reward
        reward = 0
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    def build_observation_space(self):
        """Construct observation space.  Happens only once at during __init__ in Builder."""
        obs_space_dict = OrderedDict()  # See self.obs()

        obs_space_dict.update(self.agent.build_sensor_observation_space())

        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                name = obstacle.name + '_' + 'lidar'
                obs_space_dict[name] = gymnasium.spaces.Box(
                    0.0, 1.0, (self.lidar_conf.num_bins,), dtype=np.float64
                )
            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                gymnasium.spaces.Box(-1.0, 1.0, (self.compass_conf.shape,), dtype=np.float64)

        obs_space_dict['cinert'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.agent.body_info.nbody * 10,), dtype=np.float64
        )
        obs_space_dict['cvel'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.agent.body_info.nbody * 6,), dtype=np.float64
        )
        obs_space_dict['qfrc_actuator'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.agent.body_info.nv * 1,), dtype=np.float64
        )
        obs_space_dict['cfrc_ext'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.agent.body_info.nbody * 6,), dtype=np.float64
        )

        if self.observe_vision:
            width, height = self.vision_env_conf.vision_size
            rows, cols = height, width
            self.vision_env_conf.vision_size = (rows, cols)
            obs_space_dict['vision'] = gymnasium.spaces.Box(
                0, 255, self.vision_env_conf.vision_size + (3,), dtype=np.uint8
            )

        # Flatten it ourselves
        self.obs_info.obs_space_dict = obs_space_dict
        if self.observation_flatten:
            self.obs_info.obs_flat_size = sum(
                np.prod(i.shape) for i in self.obs_info.obs_space_dict.values()
            )
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, (self.obs_info.obs_flat_size,), dtype=np.float64
            )
        else:
            self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

    def obs(self):
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        obs = {}

        obs.update(self.agent.obs_sensor())

        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                obs[obstacle.name + '_lidar'] = self._obs_lidar(obstacle.pos, obstacle.group)
            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                obs[obstacle.name + '_comp'] = self._obs_compass(obstacle.pos[0])

        obs['cinert'] = self.data.cinert.flat.copy()
        obs['cvel'] = self.data.cvel.flat.copy()
        obs['qfrc_actuator'] = self.data.qfrc_actuator.flat.copy()
        obs['cfrc_ext'] = self.data.cfrc_ext.flat.copy()

        if self.observe_vision:
            obs['vision'] = self._obs_vision()
        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_info.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_info.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset : offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs
            assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
            assert (
                offset == self.obs_info.obs_flat_size
            ), 'Obs from mujoco do not match env pre-specifed lenth.'
        return obs

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return False

    # @property
    # def healthy_reward(self):
    #     """Return the reward for being healthy."""
    #     return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    # @property
    # def is_healthy(self):
    #     """Return whether the agent is healthy."""
    #     min_z, max_z = self._healthy_z_range
    #     is_healthy = min_z < self.data.qpos[2] < max_z

    #     return is_healthy
