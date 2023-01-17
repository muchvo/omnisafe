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
"""Humanoid."""

from safety_gymnasium.bases.base_agent import BaseAgent
from safety_gymnasium.utils.random_generator import RandomGenerator


class Humanoid(BaseAgent):
    """Humanoid is a more complex agent."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        random_generator: RandomGenerator,
        placements: list = None,
        locations: list = None,
        keepout: float = 0.4,
        rot: float = None,
    ):
        super().__init__(
            self.__class__.__name__, random_generator, placements, locations, keepout, rot
        )

    def is_alive(self):
        pass

    def reset(self):
        pass
