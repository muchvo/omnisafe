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
"""Narrow level 2."""

from safety_gymnasium.tasks.narrow.narrow_level1 import NarrowLevel1


class NarrowLevel2(NarrowLevel1):
    """A agent can navigate to two goals, while it must balance reward and cost."""

    def __init__(self, config):
        super().__init__(config=config)


# self.hazards.locations = []
# self.hazards.placements = [(-1, self.palcement_cal_factor * 0.05, -0.2, self.palcement_cal_factor * 0.95),
#                             (-1, self.palcement_cal_factor * -0.95, -0.2, self.palcement_cal_factor * -0.05),
#                             # (-0.3, self.palcement_cal_factor * -0.95, -0.1, self.palcement_cal_factor * 0.95),
#                             (0.1, self.palcement_cal_factor * 0.05, 0.6, self.palcement_cal_factor * 0.95),
#                             (0.1, self.palcement_cal_factor * -0.95, 0.6, self.palcement_cal_factor * -0.05)]
# self.hazards.keepout = 0.05
# self.hazards.num = 10  # pylint: disable=wrong-spelling-in-comment
