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
"""Bottle."""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_obstacle import Objects


@dataclass
class Bottle(Objects):  # pylint: disable=too-many-instance-attributes
    """Box parameters (only used if task == 'push')"""

    name: str = 'bottle'
    size: float = 0.05
    placements: np.array = None  # Box placements list (defaults to full extents)
    z_placement: float = 0.0
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.05  # Box keepout radius for placement
    null_dist: float = 2  # Within box_null_dist * box_size radius of box, no box reward given
    density: float = 0.001

    reward_bottle_dist: float = 1.0  # Dense reward for moving the agent towards the box
    reward_bottle_goal: float = 1.0  # Reward for moving the box towards the goal

    color: np.array = np.array([0.5, 0.5, 0.5, 1])
    group: np.array = 2
    is_lidar_observed: bool = False
    is_comp_observed: bool = False
    is_xyz_observed: bool = False
    is_constrained: bool = False

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        obj = {
            'name': 'bottle',
            'type': 'cylinder',
            'size': np.array([self.size, 2 * self.size]),
            'pos': np.r_[xy_pos, 2 * self.size + self.z_placement],
            'rot': rot,
            'density': self.density,
            'group': self.group,
            'rgba': self.color,
        }
        return obj

    @property
    def pos(self):
        """Helper to get the box position."""
        return self.engine.data.body('bottle').xpos.copy()
