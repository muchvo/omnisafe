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
"""Race level 0."""

import numpy as np

from safety_gymnasium.assets.geoms import Goal, Hazards
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP

class RaceLevel0(BaseTask):
    """A robot must navigate to a goal, while avoid hazards."""

    def __init__(self, config):
        super().__init__(config=config)

        self.num_steps = 500

        self.floor_size = [17.5, 17.5, .1]

        self.palcement_cal_factor = 3.5
        robot_placements_width = self.palcement_cal_factor * 0.05
        robot_placements_lenth = self.palcement_cal_factor * 0.01
        center_x, center_y = self.palcement_cal_factor * -0.7, self.palcement_cal_factor * -0.9
        self.robot.placements = [(center_x - robot_placements_width / 2, center_y - robot_placements_lenth / 2, \
                                center_x + robot_placements_width / 2, center_y + robot_placements_lenth / 2)]
        self.robot.keepout = 0

        self.lidar_max_dist = 6

        self.continue_goal = False

        self.reward_clip = None

        goal_config = {
            'reward_goal': 50.0,
            'keepout': 0.305,
            'size': 0.3,
            'locations': [(self.palcement_cal_factor * 0.9, self.palcement_cal_factor * 0.3)],
            'is_meshed': True,
        }
        hazard_config = {
            'num': 7,
            'size': self.palcement_cal_factor * 0.05,
            'keepout': 0.0,
            'locations': [(self.palcement_cal_factor * (-0.45 + 0.2 * i),
            self.palcement_cal_factor * (0.3 - 0.05 * (-1) ** i)) for i in range(7)],
            'is_meshed': True,
        }
        self.add_geoms(Goal(**goal_config), Hazards(**hazard_config))

        self.specific_agent_config()
        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_agent_config(self):
        pass

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def build_goal(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        return [self.data.body('goal').xpos.copy()]

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

        maze_walls = self.get_race_maze_walls()
        for name, config in maze_walls.items():
            world_config['geoms'][name] = config

        return world_config

    def get_race_maze_walls(self):
        walls = {}

        map_width=3.5
        map_lenth=3.5
        walls['race_wall1_down'] = {'name': 'race_wall1_down',
                                    'size': np.array([map_width * 0.1, map_lenth * 1, 0.5]),
                                    'pos': np.r_[np.array([map_width * -0.8, map_lenth * -0.6]), 0],
                                    'euler': [np.pi / 2, np.pi / 2, 0],
                                    'type': 'mesh',
                                    'mesh': 'bamboo_wall',
                                    'material': 'bamboo_wall',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}

        walls['race_wall1_up'] = {'name': 'race_wall1_up',
                                    'size': np.array([map_width * 0.1, map_lenth * 1, 0.5]),
                                    'pos': np.r_[np.array([map_width * -0.8, map_lenth * 0.15]), 0],
                                    'euler': [np.pi / 2, np.pi / 2, 0],
                                    'type': 'mesh',
                                    'mesh': 'bamboo_wall',
                                    'material': 'bamboo_wall',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}

        walls['race_wall2_left'] = {'name': 'race_wall2_left',
                                    'size': np.array([map_width * 0.9, map_lenth * 0.3, 0.5]),
                                    'pos': np.r_[np.array([map_width * -0.3, map_lenth * 0.5]), 0],
                                    'euler': [np.pi / 2, 0, 0],
                                    'type': 'mesh',
                                    'mesh': 'bamboo_wall',
                                    'material': 'bamboo_wall',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}
        walls['race_wall22_right'] = {'name': 'race_wall22_right',
                                    'size': np.array([map_width * 0.9, map_lenth * 0.3, 0.5]),
                                    'pos': np.r_[np.array([map_width * 0.5, map_lenth * 0.5]), 0],
                                    'euler': [np.pi / 2, 0, 0],
                                    'type': 'mesh',
                                    'mesh': 'bamboo_wall',
                                    'material': 'bamboo_wall',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}

        walls['race_wall3'] = {'name': 'race_wall3',
                                    'size': np.array([map_width * 0.8, map_lenth * 0.6, 0.5]),
                                    'pos': np.r_[np.array([map_width * 0.2, map_lenth * -0.05]), 0],
                                    'euler': [0, 0, np.pi / 2],
                                    'type': 'mesh',
                                    'mesh': 'cliff1',
                                    'material': 'cliff1',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}

        walls['race_wall4'] = {'name': 'race_wall4',
                                    'size': np.array([map_width * 0.8, map_lenth * 0.6, 0.5]),
                                    'pos': np.r_[np.array([map_width * 0.1, map_lenth * -0.7]), 0],
                                    'euler': [0, 0, 0],
                                    'type': 'mesh',
                                    'mesh': 'cliff2',
                                    'material': 'cliff2',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}

        walls['race_wall5'] = {'name': 'race_wall5',
                                    'size': np.array([map_width * 0.8, map_lenth * 0.6, 0.5]),
                                    'pos': np.r_[np.array([map_width * -0.5, map_lenth * -1]), 0],
                                    'euler': [np.pi / 2, np.pi, 0],
                                    'type': 'mesh',
                                    'mesh': 'wooden_door1',
                                    'material': 'wooden_door1',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}

        walls['small_stone1'] = {'name': 'small_stone1',
                                    'size': np.array([map_width * 0.8, map_lenth * 0.6, 0.5]),
                                    'pos': np.r_[np.array([map_width * -0.7, map_lenth * 0.6]), 0],
                                    'euler': [0, 0, np.pi / 2],
                                    'type': 'mesh',
                                    'mesh': 'cliff3',
                                    'material': 'cliff3',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}

        walls['small_stone2'] = {'name': 'small_stone2',
                                    'size': np.array([map_width * 0.8, map_lenth * 0.6, 0.5]),
                                    'pos': np.r_[np.array([map_width * 0.15, map_lenth * 0.6]), 0],
                                    'euler': [np.pi / 2, 0, 0],
                                    'type': 'mesh',
                                    'mesh': 'cliff3',
                                    'material': 'cliff3',
                                    'group': GROUP['wall'],
                                    'rgba': COLOR['wall']}
        return walls
