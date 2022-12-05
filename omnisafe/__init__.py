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
"""OmniSafe: A comprehensive and reliable benchmark for safe reinforcement learning."""

from omnisafe.algo_wrapper import AlgoWrapper as Agent
from omnisafe.algos.env_wrapper import EnvWrapper as Env
from omnisafe.algos.model_based.env_wrapper import EnvWrapper as EnvModelBased
from omnisafe.version import __version__