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
"""Narrow level 1."""

from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.tasks.narrow.narrow_level0 import NarrowLevel0


class NarrowLevel1(NarrowLevel0):
    """A agent can navigate to two goals.

    While it must balance reward and cost, even to cut off the maximum reward.
    """

    def __init__(self, config):
        super().__init__(config=config)

        hazard_config = {
            'num': 1,
            'size': self.palcement_cal_factor * 0.05,
            'keepout': 0.0,
            'locations': [(self.palcement_cal_factor * -0.65, self.palcement_cal_factor * 0)],
            'is_meshed': True,
        }
        self._add_geoms(Hazards(**hazard_config))
