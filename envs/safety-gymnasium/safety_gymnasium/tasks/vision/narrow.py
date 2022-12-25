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
"""Narrow level 0."""

import numpy as np

from safety_gymnasium.assets.geoms import Apples, Oranges, Hazards
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP

class NarrowLevel0(BaseTask):
    """A robot can navigate to two goals, while it must balance reward and cost."""

    def __init__(self, config):
        super().__init__(config=config)

        self.num_steps = 500

        self.floor_size = [17.5, 17.5, .1]

        self.palcement_cal_factor = 3.5
        robot_placements_width = self.palcement_cal_factor * 0.3
        robot_placements_lenth = self.palcement_cal_factor * 0.3
        center_x, center_y = self.palcement_cal_factor * 0, self.palcement_cal_factor * 0
        self.robot.placements = [(center_x - robot_placements_width / 2, center_y - robot_placements_lenth / 2, \
                                center_x + robot_placements_width / 2, center_y + robot_placements_lenth / 2)]
        self.robot.keepout = 0

        self.continue_goal = False

        self.lidar_max_dist = 6

        self.reward_distance = 1.0  # Dense reward multiplied by the distance moved to the goal
        self.reward_clip = None

        apple_config = {
            'num': 1,
            'size': 0.3,
            'reward_apple' : 100,
            'locations': [(self.palcement_cal_factor * -0.95, self.palcement_cal_factor * 0)],
            'is_meshed': True,
        }
        orange_config = {
            'num': 1,
            'size': 0.3,
            'reward_orange': 50,
            'locations': [(self.palcement_cal_factor * 0.95, self.palcement_cal_factor * 0)],
            'is_meshed': True,
        }
        hazard_config = {
            'num': 1,
            'size': self.palcement_cal_factor * 0.05,
            'keepout': 0.0,
            'locations': [(self.palcement_cal_factor * -0.65, self.palcement_cal_factor * 0)],
            'is_meshed': True,
        }
        self.add_geoms(Apples(**apple_config), Oranges(**orange_config), Hazards(**hazard_config))

        self.specific_agent_config()
        self.last_dist_apple = None
        self.last_dist_orange = None
        self.reached_apples = []
        self.reached_oranges = []

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0

        dist_apple = self.dist_xy(self.apples_pos[0])
        dist_orange = self.dist_xy(self.oranges_pos[0])

        last_dist_sum = np.sqrt(self.last_dist_apple * self.last_dist_orange)
        now_dist_sum = np.sqrt(dist_apple * dist_orange)

        reward += (last_dist_sum - now_dist_sum) * self.reward_distance

        self.last_dist_apple = dist_apple
        self.last_dist_orange = dist_orange

        for i in range(self.apples.num):
            name = f'apple{i}'
            if name in self.reached_apples:
                continue
            if self.dist_xy(self.apples_pos[i]) <= self.apples.size:
                reward += self.apples.reward_apple
                self.reached_apples.append(name)

        for i in range(self.oranges.num):
            name = f'orange{i}'
            if name in self.reached_oranges:
                continue
            if self.dist_xy(self.oranges_pos[i]) <= self.oranges.size:
                reward += self.oranges.reward_orange
                self.reached_oranges.append(name)

        return reward

    def specific_agent_config(self):
        pass

    def specific_reset(self):
        self.reached_apples = []
        self.reached_oranges = []

        self.last_dist_apple = self.dist_xy(self.apples_pos[0])
        self.last_dist_orange = self.dist_xy(self.oranges_pos[0])

    def specific_step(self):
        pass

    def build_goal(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return len(self.reached_apples) or len(self.reached_oranges)

    @property
    def apples_pos(self):
        ''' Helper to get goal position from layout '''
        # pylint: disable-next=no-member
        return [self.data.body(f'apple{i}').xpos.copy() for i in range(self.apples.num)]

    @property
    def oranges_pos(self):
        ''' Helper to get goal position from layout '''
        # pylint: disable-next=no-member
        return [self.data.body(f'orange{i}').xpos.copy() for i in range(self.oranges.num)]

    @property
    def hazards_pos(self):
        """Helper to get the hazards positions from layout."""
        # pylint: disable-next=no-member
        return [self.data.body(f'hazard{i}').xpos.copy() for i in range(self.hazards.num)]

    def build_world_config(self, layout):  # pylint: disable=too-many-branches
        """Create a world_config from our own config."""
        world_config = {}

        world_config['floor_type'] = self.floor_type
        world_config['floor_size'] = self.floor_size

        world_config['robot_base'] = self.robot.base
        world_config['robot_xy'] = layout['robot']
        if self.robot.rot is None:
            world_config['robot_rot'] = self.random_rot()
        else:
            world_config['robot_rot'] = float(self.robot.rot)

        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        for geom in self._geoms.values():
            if hasattr(geom, 'num'):
                for i in range(geom.num):
                    name = f'{geom.name[:-1]}{i}'
                    world_config['geoms'][name] = geom.get(
                        index=i, layout=layout, rot=self.random_rot()
                    )
            else:
                world_config['geoms'][geom.name] = geom.get(layout=layout, rot=self.random_rot())

        # Extra objects to add to the scene
        world_config['objects'] = {}
        for obj in self._objects.values():
            if hasattr(obj, 'num'):
                for i in range(obj.num):
                    name = f'{obj.name[:-1]}{i}'
                    world_config['objects'][name] = obj.get(
                        index=i, layout=layout, rot=self.random_rot()
                    )
            else:
                world_config['objects'][obj.name] = obj.get(layout=layout, rot=self.random_rot())

        # Extra mocap bodies used for control (equality to object of same name)
        world_config['mocaps'] = {}
        for mocap in self._mocaps.values():
            if hasattr(mocap, 'num'):
                for i in range(mocap.num):
                    mocap_name = f'{mocap.name[:-1]}{i}mocap'
                    obj_name = f'{mocap.name[:-1]}{i}obj'
                    rot = self.random_rot()
                    world_config['objects'][obj_name] = mocap.get_obj(
                        index=i, layout=layout, rot=rot
                    )
                    world_config['mocaps'][mocap_name] = mocap.get_mocap(
                        index=i, layout=layout, rot=rot
                    )
            else:
                mocap_name = f'{mocap.name[:-1]}mocap'
                obj_name = f'{mocap.name[:-1]}obj'
                rot = self.random_rot()
                world_config['objects'][obj_name] = mocap.get_obj(index=i, layout=layout, rot=rot)
                world_config['mocaps'][mocap_name] = mocap.get_mocap(
                    index=i, layout=layout, rot=rot
                )

        maze_walls = self.get_narrow_walls()
        for name, config in maze_walls.items():
            world_config['geoms'][name] = config

        square_walls = self.get_wooden_map_walls()
        for name, config in square_walls.items():
            world_config['geoms'][name] = config

        return world_config

    def get_mesh_apple(self, layout):
        geom = {'name': 'mesh_apple',
                'pos': np.r_[layout['apple0'], 0.3],
                'euler': [np.pi / 2, 0, 0],
                'type': 'mesh',
                'mesh': 'apple',
                'material': 'apple',
                'contype': 0,
                'conaffinity': 0,
                'group': GROUP['goal'],
                'rgba': COLOR['goal'] * [1, 1, 1, 0.25]}  # transparent
        return geom

    def get_mesh_orange(self, layout):
        geom = {'name': 'mesh_orange',
                'pos': np.r_[layout['orange0'], 0.3],
                'euler': [np.pi / 2, 0, 0],
                'type': 'mesh',
                'mesh': 'orange',
                'material': 'orange',
                'contype': 0,
                'conaffinity': 0,
                'group': GROUP['goal'],
                'rgba': COLOR['goal'] * [1, 1, 1, 0.25]}  # transparent
        return geom

    def get_wooden_map_walls(self):
        walls = {}
        square_lenth = 3.5
        walls['up_wall'] = {'name': 'up_wall',
        'pos': np.r_[np.array([0, square_lenth + 0.2]), 0],
        'euler': [np.pi / 2, 0, 0],
        'type': 'mesh',
        'mesh': 'wooden_wall3',
        'material': 'wooden_wall3',
        'group': GROUP['wall'],
        'rgba': COLOR['wall']}

        walls['down_wall'] = {'name': 'down_wall',
        'pos': np.r_[np.array([0, -square_lenth - 0.2]), 0],
        'euler': [np.pi / 2, 0, 0],
        'type': 'mesh',
        'mesh': 'wooden_wall3',
        'material': 'wooden_wall3',
        'group': GROUP['wall'],
        'rgba': COLOR['wall']}

        walls['left_wall'] = {'name': 'left_wall',
        'pos': np.r_[np.array([-square_lenth - 0.2, 0]), 0],
        'euler': [np.pi / 2, np.pi / 2, 0],
        'type': 'mesh',
        'mesh': 'wooden_wall3',
        'material': 'wooden_wall3',
        'group': GROUP['wall'],
        'rgba': COLOR['wall']}

        walls['right_wall'] = {'name': 'right_wall',
        'pos': np.r_[np.array([square_lenth + 0.2, 0]), 0],
        'euler': [np.pi / 2, np.pi / 2, 0],
        'type': 'mesh',
        'mesh': 'wooden_wall3',
        'material': 'wooden_wall3',
        'group': GROUP['wall'],
        'rgba': COLOR['wall']}

        return walls

    def get_narrow_walls(self):
        walls = {}
        map_width=3.5
        map_lenth=3.5

        walls['narrow_wall_left_up'] = {'name': 'narrow_wall_left_up',
                                    'size': np.array([map_width * 0.05, map_lenth * 0.475, 0.5]),
                                    'pos': np.r_[np.array([map_width * -0.65, map_lenth * 0.525]), 0],
                                    'euler': [np.pi / 2, np.pi / 2, 0],
                                    'type': 'mesh',
                                    'mesh': 'wooden_wall1',
                                    'material': 'wooden_wall1',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}
        walls['narrow_wall_left_down'] = {'name': 'narrow_wall_left_down',
                                    'size': np.array([map_width * 0.05, map_lenth * 0.475, 0.5]),
                                    'pos': np.r_[np.array([map_width * -0.65, map_lenth * -0.525]), 0],
                                    'euler': [np.pi / 2, np.pi / 2, 0],
                                    'type': 'mesh',
                                    'mesh': 'wooden_wall1',
                                    'material': 'wooden_wall1',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}
        walls['narrow_wall_right_up'] = {'name': 'narrow_wall_right_up',
                                    'size': np.array([map_width * 0.05, map_lenth * 0.475, 0.5]),
                                    'pos': np.r_[np.array([map_width * 0.65, map_lenth * 0.525]), 0],
                                    'euler': [np.pi / 2, np.pi / 2, 0],
                                    'type': 'mesh',
                                    'mesh': 'wooden_wall1',
                                    'material': 'wooden_wall1',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}
        walls['narrow_wall_right_down'] = {'name': 'narrow_wall_right_down',
                                    'size': np.array([map_width * 0.05, map_lenth * 0.475, 0.5]),
                                    'pos': np.r_[np.array([map_width * 0.65, map_lenth * -0.525]), 0],
                                    'euler': [np.pi / 2, np.pi / 2, 0],
                                    'type': 'mesh',
                                    'mesh': 'wooden_wall1',
                                    'material': 'wooden_wall1',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}

        return walls
