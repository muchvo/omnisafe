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
"""Narrow level 0."""

import numpy as np

from safety_gymnasium.assets.geoms import Apples, Oranges
from safety_gymnasium.bases.base_task import BaseTask

class NarrowLevel0(BaseTask):
    """A robot can navigate to two goals."""

    def __init__(self, config):
        super().__init__(config=config)

        self.num_steps = 500

        self.floor_size = [17.5, 17.5, .1]

        self.palcement_cal_factor = 3.5
        robot_placements_width = self.palcement_cal_factor * 0.3
        robot_placements_lenth = self.palcement_cal_factor * 0.3
        center_x, center_y = self.palcement_cal_factor * 0, self.palcement_cal_factor * 0
        self.robot.placements = [(center_x - robot_placements_width / 2, center_y - robot_placements_lenth / 2, \
                                center_x + robot_placements_width / 2, center_y + robot_placements_lenth / 2)]
        self.robot.keepout = 0

        self.continue_goal = False

        self.lidar_max_dist = 6

        self.reward_distance = 1.0  # Dense reward multiplied by the distance moved to the goal
        self.reward_clip = None

        apple_config = {
            'num': 1,
            'size': 0.3,
            'reward_apple' : 100,
            'locations': [(self.palcement_cal_factor * -0.95, self.palcement_cal_factor * 0)],
            'is_meshed': True,
        }
        orange_config = {
            'num': 1,
            'size': 0.3,
            'reward_orange': 50,
            'locations': [(self.palcement_cal_factor * 0.95, self.palcement_cal_factor * 0)],
            'is_meshed': True,
        }
        self.add_geoms(Apples(**apple_config), Oranges(**orange_config))
        self._is_load_static_geoms = True

        self.specific_agent_config()
        self.last_dist_apple = None
        self.last_dist_orange = None
        self.reached_apples = []
        self.reached_oranges = []

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0

        dist_apple = self.dist_xy(self.apples_pos[0])
        dist_orange = self.dist_xy(self.oranges_pos[0])

        last_dist_sum = np.sqrt(self.last_dist_apple * self.last_dist_orange)
        now_dist_sum = np.sqrt(dist_apple * dist_orange)

        reward += (last_dist_sum - now_dist_sum) * self.reward_distance

        self.last_dist_apple = dist_apple
        self.last_dist_orange = dist_orange

        for i in range(self.apples.num):
            name = f'apple{i}'
            if name in self.reached_apples:
                continue
            if self.dist_xy(self.apples_pos[i]) <= self.apples.size:
                reward += self.apples.reward_apple
                self.reached_apples.append(name)

        for i in range(self.oranges.num):
            name = f'orange{i}'
            if name in self.reached_oranges:
                continue
            if self.dist_xy(self.oranges_pos[i]) <= self.oranges.size:
                reward += self.oranges.reward_orange
                self.reached_oranges.append(name)

        return reward

    def specific_agent_config(self):
        pass

    def specific_reset(self):
        self.reached_apples = []
        self.reached_oranges = []

        self.last_dist_apple = self.dist_xy(self.apples_pos[0])
        self.last_dist_orange = self.dist_xy(self.oranges_pos[0])

    def specific_step(self):
        pass

    def build_goal(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return len(self.reached_apples) or len(self.reached_oranges)

    @property
    def apples_pos(self):
        ''' Helper to get goal position from layout '''
        # pylint: disable-next=no-member
        return [self.data.body(f'apple{i}').xpos.copy() for i in range(self.apples.num)]

    @property
    def oranges_pos(self):
        ''' Helper to get goal position from layout '''
        # pylint: disable-next=no-member
        return [self.data.body(f'orange{i}').xpos.copy() for i in range(self.oranges.num)]
