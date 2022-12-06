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
"""pillar"""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


@dataclass
class Pillars:
    # Pillars (immovable obstacles we should not touch)
    num: int = 0  # Number of pillars in the world
    placements: list = None  # Pillars placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.3  # Radius for placement of pillars
    cost: float = 1.0  # Cost (per step) for being in contact with a pillar

    def get_pillar(self, index, layout, rot, size=0.2, height=0.5):
        name = f'pillar{index}'
        geom = {
            'name': name,
            'size': [size, height],
            'pos': np.r_[layout[name], height],
            'rot': rot,
            'type': 'cylinder',
            'group': GROUP['pillar'],
            'rgba': COLOR['pillar'],
        }
        return geom