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
"""Pick and place level 0."""

import numpy as np

from safety_gymnasium.assets.objects.bottle import Bottle
from safety_gymnasium.assets.objects.box import Box
from safety_gymnasium.assets.geoms.goal import Goal
from safety_gymnasium.bases.base_task import BaseTask


class PickAndPlaceLevel0(BaseTask):

    def __init__(self, config):
        super().__init__(config=config)
        self.num_steps = 500

        self.agent.z_height = 0.5
        self.agent.locations = [(-0.2, -0.4)]
        self.agent.rot = 0

        goal_config = {
            'size': 0.025,
            'placements': np.array([0.1, -0.5, 1.1, 0.5], ndmin=2),
            'z_placement': 0.5,
            'is_lidar_observed': False,
            'is_xyz_observed': True,
        }
        self._add_geoms(Goal(**goal_config))
        self._add_objects(Box(size=0.025, placements=np.array([-1.1, -0.5, -0.4, 0.5], ndmin=2), z_placement=0.5, is_xyz_observed=True))

        self._is_load_static_geoms = True

        self.last_dist_box = None
        self.last_box_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = 0.0

        # Distance from hand to box
        dist_box = self.dist_box()
        # pylint: disable-next=no-member
        gate_dist_box_reward = self.last_dist_box > self.box.null_dist * self.box.size
        reward += (
            # pylint: disable-next=no-member
            (self.last_dist_box - dist_box)
            * self.box.reward_box_dist  # pylint: disable=no-member
            * gate_dist_box_reward
        )
        self.last_dist_box = dist_box

        # Distance from box to goal
        dist_box_goal = self.dist_box_goal()
        # pylint: disable-next=no-member
        reward += (self.last_box_goal - dist_box_goal) * self.box.reward_box_goal
        self.last_box_goal = dist_box_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal  # pylint: disable=no-member

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_box = self.dist_box()
        self.last_box_goal = self.dist_box_goal()

    def dist_box(self):
        """Return the distance. from the agent to the box (in XY plane only)"""
        # pylint: disable-next=no-member
        return np.sqrt(np.sum(np.square(self.box.pos - self.agent.pos)))

    def dist_box_goal(self):
        """Return the distance from the box to the goal XY position."""
        # pylint: disable-next=no-member
        return np.sqrt(np.sum(np.square(self.box.pos - self.goal_pos)))

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_box_goal() <= self.goal.size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        return self.goal.pos  # pylint: disable=no-member
