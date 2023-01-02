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
"""Examples for vision environments."""

import argparse
import os

import safety_gymnasium
from gymnasium.utils.save_video import save_video


DIR = os.path.join(os.path.dirname(__file__), 'cached_test_vision_video')


def run_random(env_name):
    """Random run."""
    env = safety_gymnasium.make(env_name)
    obs, _ = env.reset()
    # Use below to specify seed.
    # obs, _ = env.reset(seed=0)
    terminated, truncated = False, False
    ep_ret, ep_cost = 0, 0
    render_list = {'front': [], 'back': [], 'left': [], 'right': []}
    for i in range(1001):  # pylint: disable=unused-variable
        if terminated or truncated:
            print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()
            save_video(
                frames=render_list['front'],
                video_folder=DIR,
                name_prefix='test_front_vision_output',
                fps=30,
            )
            save_video(
                frames=render_list['back'],
                video_folder=DIR,
                name_prefix='test_back_vision_output',
                fps=30,
            )
            save_video(
                frames=render_list['left'],
                video_folder=DIR,
                name_prefix='test_left_vision_output',
                fps=30,
            )
            save_video(
                frames=render_list['right'],
                video_folder=DIR,
                name_prefix='test_right_vision_output',
                fps=30,
            )
            render_list = {'front': [], 'back': [], 'left': [], 'right': []}
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        # Use the environment's built_in max_episode_steps
        if hasattr(env, '_max_episode_steps'):  # pylint: disable=protected-access
            max_ep_len = env._max_episode_steps  # pylint: disable=unused-variable,protected-access
        render_list['front'].append(obs['vision_front'])
        render_list['back'].append(obs['vision_back'])
        render_list['left'].append(obs['vision_left'])
        render_list['right'].append(obs['vision_right'])
        # pylint: disable-next=unused-variable
        obs, reward, cost, terminated, truncated, info = env.step(act)

        ep_ret += reward
        ep_cost += cost


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyCarGoal2Vision-v0')
    args = parser.parse_args()
    run_random(args.env)
