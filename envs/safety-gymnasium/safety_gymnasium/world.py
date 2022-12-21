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
"""World."""

import os
from collections import OrderedDict
from copy import deepcopy

import mujoco
import numpy as np
import safety_gymnasium
import xmltodict
from safety_gymnasium.assets.robot import Robot
from safety_gymnasium.utils.common_utils import convert, rot2quat
from safety_gymnasium.utils.task_utils import get_body_xvelp


# Default location to look for xmls folder:
BASE_DIR = os.path.dirname(safety_gymnasium.__file__)


class World:  # pylint: disable=too-many-instance-attributes
    """This class starts mujoco simulation.

    And contains some apis for interacting with mujoco."""

    # Default configuration (this should not be nested since it gets copied)
    # *NOTE:* Changes to this configuration should also be reflected in `Builder` configuration
    DEFAULT = {
        'robot_base': 'xmls/car.xml',  # Which robot XML to use as the base
        'robot_xy': np.zeros(2),  # Robot XY location
        'robot_rot': 0,  # Robot rotation about Z axis
        'floor_size': [3.5, 3.5, 0.1],  # Used for displaying the floor
        # Objects -- this is processed and added by the Builder class
        'objects': {},  # map from name -> object dict
        # Geoms -- similar to objects, but they are immovable and fixed in the scene.
        'geoms': {},  # map from name -> geom dict
        # Mocaps -- mocap objects which are used to control other objects
        'mocaps': {},
        # Determine whether we create render contexts
        'observe_vision': False,
    }

    def __init__(self, config=None):
        """config - JSON string or dict of configuration.  See self.parse()"""
        if config:
            self.parse(config)  # Parse configuration

        self.first_reset = True

        self.robot = Robot(self.robot_base)  # pylint: disable=no-member
        self.robot_base_path = None
        self.robot_base_xml = None
        self.xml = None
        self.xml_string = None

        self.model = None
        self.data = None

    def parse(self, config):
        """Parse a config dict - see self.DEFAULT for description."""
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

    def get_sensor(self, name):
        """get_sensor: Get the value of a sensor by name."""
        id = self.model.sensor(name).id  # pylint: disable=redefined-builtin, invalid-name
        adr = self.model.sensor_adr[id]
        dim = self.model.sensor_dim[id]
        return self.data.sensordata[adr : adr + dim].copy()

    def build(self):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        """Build a world, including generating XML and moving objects."""
        # Read in the base XML (contains robot, camera, floor, etc)
        self.robot_base_path = os.path.join(BASE_DIR, self.robot_base)  # pylint: disable=no-member
        with open(self.robot_base_path, encoding='utf-8') as f:  # pylint: disable=invalid-name
            self.robot_base_xml = f.read()
        self.xml = xmltodict.parse(self.robot_base_xml)  # Nested OrderedDict objects

        compiler = xmltodict.parse(
            f'''<compiler
                angle="radian"
                meshdir="{BASE_DIR}/assets/meshes"
                texturedir="{BASE_DIR}/assets/textures"
                />'''
        )
        self.xml['mujoco']['compiler'] = compiler['compiler']

        # Convenience accessor for xml dictionary
        worldbody = self.xml['mujoco']['worldbody']

        # Move robot position to starting position
        worldbody['body']['@pos'] = convert(
            # pylint: disable-next=no-member
            np.r_[self.robot_xy, self.robot.z_height]
        )  # pylint: disable=no-member
        worldbody['body']['@quat'] = convert(rot2quat(self.robot_rot))  # pylint: disable=no-member

        # We need this because xmltodict skips over single-item lists in the tree
        worldbody['body'] = [worldbody['body']]
        if 'geom' in worldbody:
            worldbody['geom'] = [worldbody['geom']]
        else:
            worldbody['geom'] = []

        # Add equality section if missing
        if 'equality' not in self.xml['mujoco']:
            self.xml['mujoco']['equality'] = OrderedDict()
        equality = self.xml['mujoco']['equality']
        if 'weld' not in equality:
            equality['weld'] = []

        # Add asset section if missing
        if 'asset' not in self.xml['mujoco']:
            # old default rgb1: ".4 .5 .6"
            # old default rgb2: "0 0 0"
            # light pink: "1 0.44 .81"
            # light blue: "0.004 0.804 .996"
            # light purple: ".676 .547 .996"
            # med blue: "0.527 0.582 0.906"
            # indigo: "0.293 0 0.508"
            asset = xmltodict.parse(
                """
                <asset>
                    <texture type="skybox" builtin="gradient" rgb1="0.527 0.582 0.906"
                        rgb2="0.1 0.1 0.35" width="800" height="800" markrgb="1 1 1"
                        mark="random" random="0.001"/>
                    <texture name="texplane" builtin="checker" height="100" width="100"
                        rgb1="0.7 0.7 0.7" rgb2="0.8 0.8 0.8" type="2d"/>
                    <material name="MatPlane" reflectance="0.1" shininess="0.1" specular="0.1"
                        texrepeat="10 10" texture="texplane"/>

                    <texture name="village_floor" file="village_floor.PNG" type="2d"/>
                    <material name="village_floor" reflectance=".1" texture="village_floor" texrepeat="25 25"/>

                    <texture name="vase" file="vase.PNG" type="2d"/>
                    <material name="vase" texture="vase" specular="1" shininess="1"/>
                    <mesh name="vase" file="vase.obj" scale="0.003 0.003 0.003"/>

                    <texture name="bush" file="bush.PNG" type="2d"/>
                    <material name="bush" texture="bush" specular="1" shininess="1"/>
                    <mesh name="bush" file="bush.obj" scale="0.0025 0.0025 0.0025"/>

                    <texture name="flower_bush" file="flower_bush.PNG" type="2d"/>
                    <material name="flower_bush" texture="flower_bush" specular="1" shininess="1"/>
                    <mesh name="flower_bush" file="flower_bush.obj" scale="0.0035 0.0035 0.0035"/>

                    <texture name="long_wall" file="long_wall.PNG" type="2d"/>
                    <material name="long_wall" texture="long_wall" specular="1" shininess="1"/>
                    <mesh name="long_wall" file="long_wall.obj" scale="0.008 0.008 0.008"/>

                    <texture name="bamboo_wall" file="bamboo_wall.PNG" type="2d"/>
                    <material name="bamboo_wall" texture="bamboo_wall" specular="1" shininess="1"/>
                    <mesh name="bamboo_wall" file="bamboo_wall.obj" scale="0.008 0.008 0.008"/>

                    <texture name="wooden_wall1" file="wooden_wall1.PNG" type="2d"/>
                    <material name="wooden_wall1" texture="wooden_wall1" specular="1" shininess="1"/>
                    <mesh name="wooden_wall1" file="wooden_wall1.obj" scale="0.008 0.008 0.008"/>

                    <texture name="wooden_wall2" file="wooden_wall2.PNG" type="2d"/>
                    <material name="wooden_wall2" texture="wooden_wall2" specular="1" shininess="1"/>
                    <mesh name="wooden_wall2" file="wooden_wall2.obj" scale="0.008 0.008 0.008"/>

                    <texture name="wooden_wall3" file="wooden_wall3.PNG" type="2d"/>
                    <material name="wooden_wall3" texture="wooden_wall3" specular="1" shininess="1"/>
                    <mesh name="wooden_wall3" file="wooden_wall3.obj" scale="0.02 0.02 0.02"/>

                    <texture name="small_wooden_wall1" file="small_wooden_wall1.PNG" type="2d"/>
                    <material name="small_wooden_wall1" texture="small_wooden_wall1" specular="1" shininess="1"/>
                    <mesh name="small_wooden_wall1" file="small_wooden_wall1.obj" scale="0.008 0.008 0.008"/>

                    <texture name="wooden_door1" file="wooden_door1.PNG" type="2d"/>
                    <material name="wooden_door1" texture="wooden_door1" specular="1" shininess="1"/>
                    <mesh name="wooden_door1" file="wooden_door1.obj" scale="0.008 0.008 0.008"/>

                    <texture name="stone_wall" file="stone_wall.PNG" type="2d"/>
                    <material name="stone_wall" texture="stone_wall" specular="1" shininess="1"/>
                    <mesh name="stone_wall" file="stone_wall.obj" scale="0.008 0.008 0.008"/>

                    <texture name="stone_wall_corner" file="stone_wall_corner.PNG" type="2d"/>
                    <material name="stone_wall_corner" texture="stone_wall_corner" specular="1" shininess="1"/>
                    <mesh name="stone_wall_corner" file="stone_wall_corner.obj" scale="0.008 0.008 0.008"/>

                    <texture name="red_wall" file="red_wall.PNG" type="2d"/>
                    <material name="red_wall" texture="red_wall" specular="1" shininess="1"/>
                    <mesh name="red_wall" file="red_wall.obj" scale="0.008 0.008 0.008"/>

                    <texture name="red_wall_corner" file="red_wall_corner.PNG" type="2d"/>
                    <material name="red_wall_corner" texture="red_wall_corner" specular="1" shininess="1"/>
                    <mesh name="red_wall_corner" file="red_wall_corner.obj" scale="0.008 0.008 0.008"/>

                    <texture name="stone_pave" file="stone_pave.PNG" type="2d"/>
                    <material name="stone_pave" texture="stone_pave" specular="1" shininess="1"/>
                    <mesh name="stone_pave" file="stone_pave.obj" scale="0.008 0.008 0.008"/>

                    <texture name="cliff1" file="cliff1.PNG" type="2d"/>
                    <material name="cliff1" texture="cliff1" specular="1" shininess="1"/>
                    <mesh name="cliff1" file="cliff1.obj" scale="0.0016 0.0032 0.0016"/>

                    <texture name="cliff2" file="cliff2.PNG" type="2d"/>
                    <material name="cliff2" texture="cliff2" specular="1" shininess="1"/>
                    <mesh name="cliff2" file="cliff2.obj" scale="0.0064 0.0064 0.0064"/>

                    <texture name="cliff3" file="cliff3.PNG" type="2d"/>
                    <material name="cliff3" texture="cliff3" specular="1" shininess="1"/>
                    <mesh name="cliff3" file="cliff3.obj" scale="0.0008 0.0008 0.0008"/>

                    <texture name="circle" file="circle.PNG" type="2d"/>
                    <material name="circle" texture="circle" specular="1" shininess="1"/>
                    <mesh name="circle" file="circle.obj" scale="0.0024 0.0012 0.0024"/>

                    <texture name="circle_boundary" file="circle_boundary.PNG" type="2d"/>
                    <material name="circle_boundary" texture="circle_boundary" specular="1" shininess="1"/>
                    <mesh name="circle_boundary" file="circle_boundary.obj" scale="0.044 0.008 0.008"/>

                    <texture name="apple" file="apple.PNG" type="2d"/>
                    <material name="apple" texture="apple" specular="1" shininess="1"/>
                    <mesh name="apple" file="apple.obj" scale="0.07 0.07 0.07"/>

                    <texture name="orange" file="orange.PNG" type="2d"/>
                    <material name="orange" texture="orange" specular="1" shininess="1"/>
                    <mesh name="orange" file="orange.obj" scale="0.12 0.12 0.12"/>
                </asset>
                """
            )
            self.xml['mujoco']['asset'] = asset['asset']

        # Add light to the XML dictionary
        light = xmltodict.parse(
            """<b>
            <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true"
                exponent="1" pos="0 0 0.5" specular="0 0 0" castshadow="false"/>
            </b>"""
        )
        worldbody['light'] = light['b']['light']

        # Add floor to the XML dictionary if missing
        if not any(g.get('@name') == 'floor' for g in worldbody['geom']):
            floor = xmltodict.parse(
                """
                <geom name="floor" type="plane" condim="6"/>
                """
            )
            worldbody['geom'].append(floor['geom'])

        # Make sure floor renders the same for every world
        for g in worldbody['geom']:  # pylint: disable=invalid-name
            if g['@name'] == 'floor':
                g.update(
                    {
                        '@size': convert(self.floor_size),  # pylint: disable=no-member
                        '@rgba': '1 1 1 1',
                        '@material': 'village_floor',
                    }
                )

        # Add cameras to the XML dictionary
        cameras = xmltodict.parse(
            """<b>
            <camera name="fixednear" pos="0 -2 2" zaxis="0 -1 1"/>
            <camera name="fixedfar" pos="0 -5 5" zaxis="0 -1 1"/>
            </b>"""
        )
        worldbody['camera'] = cameras['b']['camera']

        # Build and add a tracking camera (logic needed to ensure orientation correct)
        theta = self.robot_rot  # pylint: disable=no-member
        xyaxes = dict(
            x1=np.cos(theta),
            x2=-np.sin(theta),
            x3=0,
            y1=np.sin(theta),
            y2=np.cos(theta),
            y3=1,
        )
        pos = dict(
            xp=0 * np.cos(theta) + (-2) * np.sin(theta),
            yp=0 * (-np.sin(theta)) + (-2) * np.cos(theta),
            zp=2,
        )
        track_camera = xmltodict.parse(
            """<b>
            <camera name="track" mode="track" pos="{xp} {yp} {zp}"
                xyaxes="{x1} {x2} {x3} {y1} {y2} {y3}"/>
            </b>""".format(
                **pos, **xyaxes
            )
        )
        worldbody['body'][0]['camera'] = [
            worldbody['body'][0]['camera'],
            track_camera['b']['camera'],
        ]

        # Add objects to the XML dictionary
        for name, object in self.objects.items():  # pylint: disable=redefined-builtin, no-member
            assert object['name'] == name, f'Inconsistent {name} {object}'
            object = object.copy()  # don't modify original object
            if name == 'push_box':
                object['quat'] = rot2quat(object['rot'])
                dim = object['size'][0]
                object['dim'] = dim
                object['width'] = dim / 2
                object['x'] = dim
                object['y'] = dim
                body = xmltodict.parse(
                    # pylint: disable-next=consider-using-f-string
                    '''
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <freejoint name="{name}"/>
                        <geom name="{name}" type="{type}" size="{size}" density="{density}"
                            rgba="{rgba}" group="{group}"/>
                        <geom name="col1" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} {y} 0"/>
                        <geom name="col2" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} {y} 0"/>
                        <geom name="col3" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="{x} -{y} 0"/>
                        <geom name="col4" type="{type}" size="{width} {width} {dim}" density="{density}"
                            rgba="{rgba}" group="{group}" pos="-{x} -{y} 0"/>
                    </body>
                '''.format(
                        **{k: convert(v) for k, v in object.items()}
                    )
                )
            else:
                if object['type'] == 'mesh':
                    body = xmltodict.parse(
                        # pylint: disable-next=consider-using-f-string
                        '''
                        <body name="{name}" pos="{pos}" euler="{euler}" >
                            <freejoint name="{name}"/>
                            <geom name="{name}" type="mesh" mesh="{mesh}" material="{material}" density="{density}"
                                rgba="{rgba}" group="{group}" condim="6" />
                        </body>
                    '''.format(
                            **{k: convert(v) for k, v in object.items()}
                        )
                    )
                else:
                    object['quat'] = rot2quat(object['rot'])
                    body = xmltodict.parse(
                        # pylint: disable-next=consider-using-f-string
                        '''
                        <body name="{name}" pos="{pos}" quat="{quat}">
                            <freejoint name="{name}"/>
                            <geom name="{name}" type="{type}" size="{size}" density="{density}"
                                rgba="{rgba}" group="{group}"/>
                        </body>
                    '''.format(
                            **{k: convert(v) for k, v in object.items()}
                        )
                    )
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])
        # Add mocaps to the XML dictionary
        for name, mocap in self.mocaps.items():  # pylint: disable=no-member
            # Mocap names are suffixed with 'mocap'
            assert mocap['name'] == name, f'Inconsistent {name}'
            assert (
                name.replace('mocap', 'obj') in self.objects  # pylint: disable=no-member
            ), f'missing object for {name}'  # pylint: disable=no-member
            # Add the object to the world
            mocap = mocap.copy()  # don't modify original object
            mocap['quat'] = rot2quat(mocap['rot'])
            body = xmltodict.parse(
                # pylint: disable-next=consider-using-f-string
                """
                <body name="{name}" mocap="true">
                    <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}"
                        pos="{pos}" quat="{quat}" contype="0" conaffinity="0" group="{group}"/>
                </body>
            """.format(
                    **{k: convert(v) for k, v in mocap.items()}
                )
            )
            worldbody['body'].append(body['body'])
            # Add weld to equality list
            mocap['body1'] = name
            mocap['body2'] = name.replace('mocap', 'obj')
            weld = xmltodict.parse(
                # pylint: disable-next=consider-using-f-string
                """
                <weld name="{name}" body1="{body1}" body2="{body2}" solref=".02 1.5"/>
            """.format(
                    **{k: convert(v) for k, v in mocap.items()}
                )
            )
            equality['weld'].append(weld['weld'])
        # Add geoms to XML dictionary
        for name, geom in self.geoms.items():  # pylint: disable=no-member
            assert geom['name'] == name, f'Inconsistent {name} {geom}'
            geom = geom.copy()  # don't modify original object
            geom['contype'] = geom.get('contype', 1)
            geom['conaffinity'] = geom.get('conaffinity', 1)
            if geom['type'] == 'mesh':
                body = xmltodict.parse(
                    # pylint: disable-next=consider-using-f-string
                    '''
                    <body name="{name}" pos="{pos}" euler="{euler}">
                        <geom name="{name}" type="mesh" mesh="{mesh}" material="{material}"
                        rgba="1 1 1 1" group="{group}" contype="{contype}"
                        conaffinity="{conaffinity}"/>
                    </body>
                '''.format(
                            **{k: convert(v) for k, v in geom.items()}
                        )
                    )
            else:
                geom['quat'] = rot2quat(geom['rot'])
                body = xmltodict.parse(
                    # pylint: disable-next=consider-using-f-string
                    '''
                    <body name="{name}" pos="{pos}" quat="{quat}">
                        <geom name="{name}" type="{type}" size="{size}" rgba="{rgba}"
                        group="{group}" contype="{contype}" conaffinity="{conaffinity}"/>
                    </body>
                '''.format(
                        **{k: convert(v) for k, v in geom.items()}
                    )
                )
            # Append new body to world, making it a list optionally
            # Add the object to the world
            worldbody['body'].append(body['body'])

        # Instantiate simulator
        # print(xmltodict.unparse(self.xml, pretty=True))
        self.xml_string = xmltodict.unparse(self.xml)

        self.model = mujoco.MjModel.from_xml_string(self.xml_string)  # pylint: disable=no-member
        self.data = mujoco.MjData(self.model)  # pylint: disable=no-member

        # Recompute simulation intrinsics from new position
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def rebuild(self, config=None, state=True):
        """Build a new sim from a model if the model changed."""
        if state:
            old_state = self.get_state()

        if config:
            self.parse(config)
        self.build()
        if state:
            self.set_state(old_state)
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def reset(self, build=True):
        """Reset the world. (sim is accessed through self.sim)"""
        if build:
            self.build()

    def robot_com(self):
        """Get the position of the robot center of mass in the simulator world reference frame."""
        return self.body_com('robot')

    def robot_pos(self):
        """Get the position of the robot in the simulator world reference frame."""
        return self.body_pos('robot')

    def robot_mat(self):
        """Get the rotation matrix of the robot in the simulator world reference frame."""
        return self.body_mat('robot')

    def robot_vel(self):
        """Get the velocity of the robot in the simulator world reference frame."""
        return self.body_vel('robot')

    def body_com(self, name):
        """Get the center of mass of a named body in the simulator world reference frame."""
        return self.data.body(name).subtree_com.copy()

    def body_pos(self, name):
        """Get the position of a named body in the simulator world reference frame."""
        return self.data.body(name).xpos.copy()

    def body_mat(self, name):
        """Get the rotation matrix of a named body in the simulator world reference frame."""
        return self.data.body(name).xmat.copy().reshape(3, -1)

    def body_vel(self, name):
        """Get the velocity of a named body in the simulator world reference frame."""
        return get_body_xvelp(self.model, self.data, name).copy()

    def get_state(self):
        """Returns a copy of the simulator state."""
        state = {}

        state['time'] = np.copy(self.data.time)
        state['qpos'] = np.copy(self.data.qpos)
        state['qvel'] = np.copy(self.data.qvel)
        if self.model.na == 0:
            state['act'] = None
        else:
            state['act'] = np.copy(self.data.act)

        return state

    def set_state(self, value):
        """
        Sets the state from an dict.

        Args:
        - value (dict): the desired state.
        - call_forward: optionally call sim.forward(). Called by default if
            the udd_callback is set.
        """
        self.data.time = value['time']
        self.data.qpos[:] = np.copy(value['qpos'])
        self.data.qvel[:] = np.copy(value['qvel'])
        if self.model.na != 0:
            self.data.act[:] = np.copy(value['act'])
