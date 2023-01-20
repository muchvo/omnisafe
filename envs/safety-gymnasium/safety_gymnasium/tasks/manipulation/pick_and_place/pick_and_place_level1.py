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
"""Pick and place level 1."""

import numpy as np
from safety_gymnasium.assets.geoms.pillar import Pillars
from safety_gymnasium.tasks.manipulation.pick_and_place.pick_and_place_level0 import PickAndPlaceLevel0


class PickAndPlaceLevel1(PickAndPlaceLevel0):

    def __init__(self, config):
        super().__init__(config=config)
        self.num_steps = 20

        self.agent.z_height = 0.5
        self.agent.locations = [(-0.2, -0.4)]
        self.agent.rot = 0

        pillars_config = {
            'num': 1,
            'size': 0.1,
            'height': 1.0,
            'keepout': 0.1,
            'placements': np.array([-0.4, 0.0, -0.1, 0.5], ndmin=2),
            'z_placement': 0.5,
            'is_lidar_observed': False,
            'is_xyz_observed': True,
        }
        self._add_geoms(Pillars(**pillars_config))
