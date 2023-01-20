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
"""Push box."""

from dataclasses import dataclass

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_obstacle import Objects


@dataclass
class Blocks(Objects):  # pylint: disable=too-many-instance-attributes
    """Box parameters (only used if task == 'push')"""

    name: str = 'blocks'
    num: int = 3
    size: np.array = np.array([0.01, 0.01, .05])
    placements: list = None  # Placements where goal may appear (defaults to full extents)
    locations: np.array = np.array([(-0.7, -0.020), (-0.69, 0), (-0.7, 0.020)])  # Fixed locations to override placements
    keepout: float = 0  # Keepout radius when placing goals
    density: float = 0.001
    index: int = 0

    reward_box_dist: float = 1.0  # Dense reward for moving the agent towards the box
    reward_box_goal: float = 1.0  # Reward for moving the box towards the goal

    color: np.array = np.array([1, 1, 1, 1])
    group: np.array = 2
    is_lidar_observed: bool = False
    is_comp_observed: bool = False
    is_constrained: bool = False

    def index_tick(self):
        """Count index."""
        self.index += 1
        self.index %= self.num

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        obj = {
            'name': 'blocks',
            'type': 'box',
            'size': np.ones(3) * self.size,
            'pos': np.r_[xy_pos, 0.55],
            'rot': 0,
            'density': self.density,
            'group': self.group,
            'rgba': self.color,
        }
        if self.index == 1:
            obj.update(
                {
                    'rgba': np.array([1, 0, 0, 1]),
                }
            )
        self.index_tick()
        return obj

    @property
    def pos(self):
        """Helper to get the box position."""
        return [self.engine.data.body(f'block{i}').xpos.copy() for i in range(self.num)]
