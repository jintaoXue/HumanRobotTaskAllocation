# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: class for nut-bolt env.

Inherits base class and abstract environment class. Inherited by nut-bolt task classes. Not directly executed.

Configuration defined in FactoryEnvNutBolt.yaml. Asset info defined in factory_asset_info_obj.yaml.
"""


import hydra
import numpy as np
import torch

from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.base.rl_task_v1 import RLTask
from omni.physx.scripts import physicsUtils, utils

from omniisaacgymenvs.robots.articulations.views.factory_franka_view import (
    FactoryFrankaView,
)
import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_base import FactoryBase
from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import (
    FactorySchemaConfigEnv,
)

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.articulations import ArticulationView

from omni.usd import get_world_transform_matrix, get_local_transform_matrix
from omniisaacgymenvs.utils.geometry import quaternion  
class Materials(object):

    def __init__(self, cube_list : list, hoop_list : list, bending_tube_list : list, upper_tube_list: list, product_list : list) -> None:

        self.cube_list = cube_list
        self.upper_tube_list = upper_tube_list
        self.hoop_list = hoop_list
        self.bending_tube_list = bending_tube_list
        self.product_list = product_list

        self.cube_state_dic = {-1:"done", 0:"wait", 1:"in_list", 2:"conveying", 3:"conveyed", 4:"cutting", 5:"cut_done", 6:"pick_up_place_cut", 
                                   7:"placed_station_inner", 8:"placed_station_outer", 9:"welding_left", 10:"welding_right", 11:"welding_upper",
                                   12:"process_done", 13:"pick_up_place_product"}
        self.hoop_state_dic = {-1:"done", 0:"wait", 1:"in_list", 2:"loading", 3:"loaded"}
        self.bending_tube_state_dic = {-1:"done", 0:"wait", 1:"in_list", 2:"loading", 3:"loaded"}
        self.upper_tube_state_dic = {}
        self.product_state_dic = {0:"waitng", 1:"placed", -1:"finished"}

        self.cube_states = [0]*len(self.cube_list)
        self.hoop_states = [0]*len(self.hoop_list)
        self.bending_tube_states = [0]*len(self.bending_tube_list)
        self.upper_tube_states = [0]*len(self.upper_tube_list)
        self.product_states = [0]*len(self.product_list)
        '''#for workers and agv to conveying the materials'''
        # self.cube_convey_states = [0]*len(self.cube_list)
        self.hoop_state_dic = {0:"wait", 1:"in_box", 2:"on_table"}
        self.bending_tube_state_dic = {0:"wait", 1:"in_box", 2:"on_table"}
        self.hoop_convey_states = [0]*len(self.hoop_list)
        self.bending_tube_convey_states = [0]*len(self.bending_tube_list)
        # self.upper_tube_convey_states = [0]*len(self.upper_tube_list)
        #for belt conveyor
        self.cube_convey_index = -1
        #cutting machine
        self.cube_cut_index = -1
        #grippers
        self.pick_up_place_cube_index = -1
        self.pick_up_place_upper_tube_index = -1
        #for inner station
        self.inner_hoop_processing_index = -1
        self.inner_cube_processing_index = -1
        self.inner_bending_tube_processing_index = -1
        self.inner_upper_tube_processing_index = -1
        #for outer station
        self.outer_hoop_processing_index = -1
        self.outer_cube_processing_index = -1
        self.outer_bending_tube_processing_index = -1
        self.outer_upper_tube_processing_index = -1

        self.initial_hoop_pose = None
        self.initial_bending_tube_pose = None

    def get_world_poses(self, list):
        poses = []
        for obj in list:
            poses.append(obj.get_world_poses())
        return poses
    
    def update_poses(self):
        pass

    def done(self):
        return max(self.product_states) == -1

    def find_next_raw_cube_index(self):
        # index 
        try:
            return self.cube_states.index(0)
        except:
            return -1
        # return self.cube_states.index(0)
    
    def find_next_raw_upper_tube_index(self):
        # index 
        try:
            return self.upper_tube_states.index(0)
        except:
            return -1
        # return self.upper_tube_states.index(0)
    
    def find_next_raw_hoop_index(self):
        # index 
        try:
            return self.hoop_states.index(0)
        except:
            return -1
        # return self.hoop_states.index(0)
    
    def find_next_raw_bending_tube_index(self):
        # index 
        try:
            return self.bending_tube_states.index(0)
        except:
            return -1
        # return self.bending_tube_states.index(0)


class Characters(object):

    def __init__(self, character_list) -> None:
        self.num = len(character_list)
        self.list = character_list
        self.state_character_dic = {0:"free", 1:"approaching", 2:"waiting_box"}
        self.task_character_dic = {0:"free", 1:"put_hoop_into_box", 2:"put_bending_into_box", 3:"cutting_machine"}
        self.states = [0]*self.num
        self.tasks = [0]*self.num
        self.corresp_agv_idxs = [-1]*self.num
        self.corresp_box_idxs = [-1]*self.num
        # self.corresp_agvs_idxs = [-1]*self.num
        self.x_paths = [[] for i in range(len(character_list))]
        self.y_paths = [[] for i in range(len(character_list))]
        self.yaws = [[] for i in range(len(character_list))]
        self.path_idxs = [0 for i in range(len(character_list))]

        self.picking_pose_hoop = [-0.09067, 6.48821, np.deg2rad(180)]
        self.picking_pose_bending_tube = [-0.09067, 13.12021, np.deg2rad(0)]

        self.LOADING_TIME = 5
        self.loading_operation_time_steps = [0 for i in range(len(character_list))]
        return
    
    def assign_task(self, task):
        #todo 
        idx = self.find_available_charac()
        if idx == -1:
            return idx
        if task == 'hoop_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 1 
        elif task == 'bending_tube_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 2
        return idx
    
    def find_available_charac(self):
        try:
            return self.tasks.index(0)
        except: 
            return -1

    def step_next_pose(self, charac_idx = 0):
        reaching_flag = False
        self.path_idxs[charac_idx] += 1
        path_idx = self.path_idxs[charac_idx]
        if path_idx == len(self.x_paths[charac_idx] - 1):
            reaching_flag = True
            self.reset_path(charac_idx)
        
        position = [self.x_paths[charac_idx][path_idx], self.y_paths[charac_idx][path_idx], 0]
        euler_angles = [0,0, self.yaws[charac_idx][path_idx]]
        orientation = quaternion.eulerAnglesToQuaternion(euler_angles)

        return position, orientation, reaching_flag
    
    def reset_path(self, charac_idx):
        self.x_paths[charac_idx] = []
        self.y_paths[charac_idx] = []
        self.yaws[charac_idx] = []
        self.path_idxs[charac_idx] = 0


class Agvs(object):

    def __init__(self, agv_list) -> None:
        self.list = agv_list
        self.num = len(agv_list)
        self.state_dic = {0:"free", 1:"moving_to_box", 2:"carrying_box", 3:"fulling"}
        self.task_dic = {0:"free", 1:"carry_box_to_hoop", 2:"carry_box_to_bending_tube", 3:"carry_box_to_hoop_table", 4:"carry_box_to_bending_tube_table"}
        self.states = [0]*self.num
        self.tasks = [0]*self.num
        self.corresp_charac_idxs = [-1]*self.num
        self.corresp_box_idxs = [-1]*self.num

        self.x_paths = [[] for i in range(len(agv_list))]
        self.y_paths = [[] for i in range(len(agv_list))]
        self.yaws = [[] for i in range(len(agv_list))]
        self.path_idxs = [0 for i in range(len(agv_list))]

        self.picking_pose_hoop = [-0.09067, 6.48821, np.deg2rad(180)]
        self.picking_pose_bending_tube = [-0.09067, 13.12021, np.deg2rad(0)]
        return
    
    def assign_task(self, task):
        #todo 
        idx = self.find_available_agv()
        if idx == -1:
            return idx
        if task == 'hoop_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 1 
        elif task == 'bending_tube_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 2
        return idx
    
    def find_available_agv(self):
        try:
            return self.tasks.index(0)
        except: 
            return -1
    
    def step_next_pose(self, agv_idx = 0):
        reaching_flag = False
        self.path_idxs[agv_idx] += 1
        path_idx = self.path_idxs[agv_idx]
        if path_idx == len(self.x_paths[agv_idx] - 1):
            reaching_flag = True
            self.reset_path(agv_idx)
        
        position = [self.x_paths[agv_idx][path_idx], self.y_paths[agv_idx][path_idx], 0]
        euler_angles = [0,0, self.yaws[agv_idx][path_idx]]
        orientation = quaternion.eulerAnglesToQuaternion(euler_angles)
        return position, orientation, reaching_flag
    
    def reset_path(self, agv_idx):
        self.x_paths[agv_idx] = []
        self.y_paths[agv_idx] = []
        self.yaws[agv_idx] = []
        self.path_idxs[agv_idx] = 0


class TransBoxs(object):

    def __init__(self, box_list) -> None:
        self.list = box_list
        self.num = len(box_list)
        self.state_dic = {0:"free", 1:"waiting", 2:"moving", 3:"loading", 4:"fulling"}
        self.task_dic = {0:"free", 1:"waiting_agv", 2:"moving_with_box"}
        self.state_boxs_dic = {}
        self.task_boxs_dic = {}
        self.states = [0]*self.num
        self.tasks = [0]*self.num
        self.corresp_agv_idxs = [-1]*self.num
        self.corresp_charac_idxs = [-1]*self.num

        self.picking_pose_hoop = [-0.09067, 6.48821, np.deg2rad(180)]
        self.picking_pose_bending_tube = [-0.09067, 13.12021, np.deg2rad(0)]

        self.hoop_idx_list =[[] for i in range(len(box_list))]
        self.bending_tube_idx_list = [[] for i in range(len(box_list))]
        self.capacity = 4

        return
    
    def assign_task(self, task):
        #todo 
        idx = self.find_available_box()
        if idx == -1:
            return idx
        if task == 'hoop_preparing' or task == 'bending_tube_preparing':
            # idx = self.find_available_charac()
            self.tasks[idx] = 1 
        return idx
    
    def find_available_box(self):
        try:
            return self.tasks.index(0)
        except: 
            return -1


class TaskManager(object):
    def __init__(self, character_list, agv_list, box_list) -> None:
        self.characters = Characters(character_list=character_list)
        self.agvs = Agvs(agv_list = agv_list)
        self.boxs = TransBoxs(box_list=box_list)
        self.task_dic = {'hoop_preparing': 0, 'bending_tube_preparing': 1}
        self.task_in_list = []
        self.task_in_dic = {}
        return
    
    def assign_task(self, task):
        
        charac_idx = self.characters.assign_task(task)
        agv_idx = self.agvs.assign_task(task)
        box_idx = self.boxs.assign_task(task)
        if charac_idx == -1 or agv_idx == -1 or box_idx == -1:
            lacking_resource = True            

        self.task_in_list.append(task)
        self.task_in_dic[task] = {'charac_idx': charac_idx, 'agv_idx': agv_idx, 'box_idx': box_idx, 'lacking_resource': lacking_resource}

        self.characters.corresp_agv_idxs[charac_idx] = agv_idx
        self.characters.corresp_box_idxs[charac_idx] = box_idx
        self.agvs.corresp_charac_idxs[agv_idx] = charac_idx
        self.agvs.corresp_box_idxs[agv_idx] = box_idx
        self.boxs.corresp_charac_idxs[box_idx] = charac_idx
        self.boxs.corresp_agv_idxs[box_idx] = agv_idx

        return True

    def step(self):
        for task in self.task_in_list:
            if self.task_in_dic[task]['lacking_resource']:
                if self.task_in_dic[task]['charac_idx'] == -1:
                    self.task_in_dic[task]['charac_idx'] = self.characters.assign_task()
                if self.task_in_dic[task]['agv_idx'] == -1:
                    self.task_in_dic[task]['agv_idx'] = self.agvs.assign_task()
                if self.task_in_dic[task]['box_idx'] == -1:
                    self.task_in_dic[task]['box_idx'] = self.boxs.assign_task()
                self.task_in_dic[task]['lacking_resource'] = list(self.task_in_dic.values()).index(-1) >= 0
        return 

class FactoryEnvTaskAlloc(FactoryBase, FactoryABCEnv):
    def __init__(self, name, sim_config, env) -> None:
        """Initialize base superclass. Initialize instance variables."""

        super().__init__(name, sim_config, env)

        self._get_env_yaml_params()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = (
            "task/FactoryEnvTaskAllocation.yaml"  # relative to Hydra search path (cfg dir)
        )
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_task_allocation.yaml"
        self.asset_info_obj = hydra.compose(config_name=asset_info_path)
        self.asset_info_obj = self.asset_info_obj[""][""][""]["tasks"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_observations = self._task_cfg["env"]["numObservations"]
        self._num_actions = self._task_cfg["env"]["numActions"]
        self._env_spacing = self.cfg_base["env"]["env_spacing"]

        self._get_env_yaml_params()

    def set_up_scene(self, scene) -> None:
        """Import assets. Add to scene."""
        # Increase buffer size to prevent overflow for Place and Screw tasks
        physxSceneAPI = self.world.get_physics_context()._physx_scene_api
        physxSceneAPI.CreateGpuCollisionStackSizeAttr().Set(256 * 1024 * 1024)

        # self.import_franka_assets(add_to_stage=True)
        # /obj/multi_people/F_Business_02/female_adult_business_02
        
        # from pxr import Sdf
        # # prim_path = Sdf.Path(f"/World/envs/env_0" + "/obj/multi_people/F_Business_02/female_adult_business_02/ManRoot/male_adult_construction_05")
        # prim_path = Sdf.Path(f"/World/envs/env_0" + "/obj/Characters/male_adult_construction_05/ManRoot/male_adult_construction_05")

        # # /obj/Characters/male_adult_construction_05/ManRoot/male_adult_construction_05
        # # from omniisaacgymenvs.robots.omni_anim_people.scripts.character_behavior import CharacterBehavior
        # # self.character_0 = CharacterBehavior(prim_path)
        # import omni.anim.graph.core as ag
        # self.character = ag.get_character(str(prim_path))


        self._stage = get_current_stage()
        # self.create_nut_bolt_material()
        RLTask.set_up_scene(self, scene, replicate_physics=False)
        self._import_env_assets(add_to_stage=True)

        # self.frankas = FactoryFrankaView(
        #     prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        # )

        #debug
        stage_utils.print_stage_prim_paths()
        
        # perspective = self._stage.GetPrimAtPath(f"/OmniverseKit_Persp")
        # ConveyorNode_0.GetAttribute('inputs:velocity').Set(100)
        # ConveyorNode_0.GetAttribute('inputs:animateTexture').Set(True)
        # perspective.GetAttributes()
        # perspective.GetAttribute('xformOp:translate').Set((36.0, 38.6, 16.8))
        # perspective.GetAttribute('xformOp:rotateXYZ').Set((63.8, 0, 141))
        # translate = perspective.GetAttribute('xformOp:translate')
        # result, prim_ConveyorBelt_A09_0_2 = commands.execute(
        #     "CreateConveyorBelt",
        #     prim_name="ConveyorActionGraph",
        #     conveyor_prim=self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_2/Belt")
        # )
        self.obj_belt_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A09_0_0/Belt",
            name="ConveyorBelt_A09_0_0/Belt",
            track_contact_forces=True,
        )
        self.obj_0_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/obj_0_1",
            name="obj_0_1",
            track_contact_forces=True,
        )
        self.obj_belt_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A09_0_2/Belt",
            name="ConveyorBelt_A09_0_2/Belt",
            track_contact_forces=True,
        )
        # self.obj_belt_2 = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/obj/ConveyorBelt_A08/Rollers",
        #     name="ConveyorBelt_A08/Rollers",
        #     track_contact_forces=True,
        # )
        self.obj_part_10 = ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part10", name="obj_part_10", reset_xform_properties=False
        )
        self.obj_part_7 = ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part7", name="obj_part_7", reset_xform_properties=False
        )
        self.obj_part_7_manipulator = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part7/manipulator2/robotiq_arg2f_base_link", name="obj_part_7_manipulator", reset_xform_properties=False
        )
        self.obj_part_9_manipulator = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part9/manipulator2/robotiq_arg2f_base_link", name="obj_part_9_manipulator", reset_xform_properties=False
        )
        self.obj_11_station_0 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0", name="obj_11_station_0", reset_xform_properties=False
        )
        self.obj_11_station_0_revolution = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/revolution", name="Station0/revolution", reset_xform_properties=False
        )
        self.obj_11_station_1_revolution = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/revolution", name="Station1/revolution", reset_xform_properties=False
        )
        self.obj_11_station_0_middle = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/middle_left", name="Station0/middle_left", reset_xform_properties=False
        )
        self.obj_11_station_1_middle = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/middle_left", name="Station1/middle_left", reset_xform_properties=False
        )
        self.obj_11_station_0_right = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0/right", name="Station0/right", reset_xform_properties=False
        )
        self.obj_11_station_1_right = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1/right", name="Station1/right", reset_xform_properties=False
        )
        self.obj_11_station_1 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station1", name="obj_11_station_1", reset_xform_properties=False
        )
        self.obj_11_welding_0 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Welding0", name="obj_11_welding_0", reset_xform_properties=False
        )
        self.obj_11_welding_1 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Welding1", name="obj_11_welding_1", reset_xform_properties=False
        )
        self.obj_2_loader_0 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part2/Loaders/Loader0", name="obj_2_loader_0", reset_xform_properties=False
        )
        self.obj_2_loader_1 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part2/Loaders/Loader1", name="obj_2_loader_1", reset_xform_properties=False
        )
        self.materials_cube_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_0",
            name="cube_0",
            track_contact_forces=True,
        )
        self.materials_hoop_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_0",
            name="hoop_0",
            track_contact_forces=True,
        )
        self.materials_bending_tube_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_0",
            name="bending_tube_0",
            track_contact_forces=True,
        )
        self.materials_upper_tube_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_0",
            name="upper_tube_0",
            track_contact_forces=True,
        )
        self.product_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_0",
            name="product_0",
            track_contact_forces=True,
        )
        self.materials_cube_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_01",
            name="cube_1",
            track_contact_forces=True,
        )
        self.materials_hoop_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_01",
            name="hoop_1",
            track_contact_forces=True,
        )
        self.materials_bending_tube_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_01",
            name="bending_tube_1",
            track_contact_forces=True,
        )
        self.materials_upper_tube_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_01",
            name="upper_tube_1",
            track_contact_forces=True,
        )
        self.product_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_01",
            name="product_1",
            track_contact_forces=True,
        )
        self.materials_cube_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/cubes/cube_02",
            name="cube_2",
            track_contact_forces=True,
        )
        self.materials_hoop_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoops/hoop_02",
            name="hoop_2",
            track_contact_forces=True,
        )
        self.materials_bending_tube_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tubes/bending_tube_02",
            name="bending_tube_2",
            track_contact_forces=True,
        )
        self.materials_upper_tube_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tubes/upper_tube_02",
            name="upper_tube_2",
            track_contact_forces=True,
        )
        self.product_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/products/product_02",
            name="product_2",
            track_contact_forces=True,
        )
        scene.add(self.obj_11_station_0)
        scene.add(self.obj_11_station_1)
        scene.add(self.obj_11_welding_0)
        scene.add(self.obj_11_welding_1)
        scene.add(self.obj_2_loader_0)
        scene.add(self.obj_2_loader_1)
        scene.add(self.obj_part_9_manipulator)
        scene.add(self.obj_part_10)
        scene.add(self.obj_part_7)
        # self.obj_cube = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/obj/obj_cube",
        #     name="obj_cube",
        #     track_contact_forces=True,
        # )
        scene.add(self.obj_0_1)
        scene.add(self.obj_belt_0)
        scene.add(self.obj_belt_1)
        scene.add(self.materials_cube_0)
        scene.add(self.materials_hoop_0)
        scene.add(self.materials_bending_tube_0)
        scene.add(self.materials_upper_tube_0)
        scene.add(self.product_0)        
        scene.add(self.materials_cube_1)
        scene.add(self.materials_hoop_1)
        scene.add(self.materials_bending_tube_1)
        scene.add(self.materials_upper_tube_1)
        scene.add(self.product_1)        
        scene.add(self.materials_cube_2)
        scene.add(self.materials_hoop_2)
        scene.add(self.materials_bending_tube_2)
        scene.add(self.materials_upper_tube_2)
        scene.add(self.product_2)
        #materials states
        self.materials : Materials = Materials(cube_list=[self.materials_cube_0, self.materials_cube_1, self.materials_cube_2], 
                hoop_list=[self.materials_hoop_0, self.materials_hoop_1, self.materials_hoop_2], 
                bending_tube_list=[self.materials_bending_tube_0, self.materials_bending_tube_1, self.materials_bending_tube_2], 
                upper_tube_list=[self.materials_upper_tube_0, self.materials_upper_tube_1, self.materials_upper_tube_2], 
                product_list = [self.product_0, self.product_1, self.product_2])
        # self.materials_flag_dic = {-1:"done", 0:"wait", 1:"conveying", 2:"conveyed", 3:"cutting", 4:"cut_done", 5:"pick_up_cut", 
        # 5:"down", 6:"combine_l", 7:"weld_l", 8:"combine_r", 9:"weld_r"}
        # conveyor
        self.obj_conveyor_0 = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/ConveyorBelt_A09_0_0").GetAttribute('xformOp:translate')
        self.obj_conveyor_1 = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/ConveyorBelt_A09_0_2").GetAttribute('xformOp:translate')
        self.obj_conveyor_2 = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/ConveyorBelt_A08").GetAttribute('xformOp:translate')
        #0 free 1 working
        self.convey_state = 0
        self.conveyor_pose_list = [self.obj_conveyor_0.Get(), self.obj_conveyor_1.Get(), self.obj_conveyor_2.Get()]
        #cutting machine
        #to do 
        self.cutting_state_dic = {0:"free", 1:"work", 2:"reseting"}
        self.cutting_machine_state = 0
        self.c_machine_oper_time = 0
        #gripper
        # self.max_speed_in_out = 0.1
        # self.max_speed_left_right = 0.1
        # self.max_speed_up_down = 0.1
        # self.max_speed_grip = 0.1
        speed = 0.3
        self.operator_gripper = torch.tensor([speed]*10, device='cuda:0')
        self.gripper_inner_task_dic = {0: "reset", 1:"pick_cut", 2:"place_cut_to_inner_station", 3:"place_cut_to_outer_station", 
                                    4:"pick_product_from_inner", 5:"pick_product_from_outer", 6:"place_product_from_inner", 7:"place_product_from_outer"}
        self.gripper_inner_task = 0
        self.gripper_inner_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_inner_state = 0

        self.gripper_outer_task_dic = {0: "reset", 1:"pick_upper_tube_for_inner_station", 2:"pick_upper_tube_for_outer_station", 3:"place_upper_tube_to_inner_station", 4:"place_upper_tube_to_outer_station"}
        self.gripper_outer_task = 0
        self.gripper_outer_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_outer_state = 0

        #welder 
        # self.max_speed_welder = 0.1
        self.welder_inner_oper_time = 0
        self.welder_outer_oper_time = 0
        self.operator_welder = torch.tensor([0.2], device='cuda:0')
        self.welder_task_dic = {0: "reset", 1:"weld_left", 2:"weld_right", 3:"weld_middle",}
        self.welder_state_dic = {0: "free_empty", 1: "moving_left", 2:"welding_left", 3:"welded_left", 4:"moving_right",
                                 5:"welding_right", 6:"rotate_and_welding", 7:"welded_right", 8:"welding_middle" , 9:"welded_upper"}
        
        self.welder_inner_task = 0
        self.welder_inner_state = 0
        
        self.welder_outer_task = 0
        self.welder_outer_state = 0
        
        #station
        # self.welder_inner_oper_time = 10
        self.operator_station = torch.tensor([0.1, 0.1, 0.1, 0.1], device='cuda:0')
        self.station_task_left_dic = {0: "reset", 1:"weld"}
        self.station_state_left_dic = {0: "reset_empty", 1:"loading", 2:"rotating", 3:"waiting", 4:"welding", 5:"welded", 6:"finished", -1:"resetting"}
        self.station_task_inner_left = 0
        self.station_task_outer_left = 0
        self.station_state_inner_left = -1
        self.station_state_outer_left = -1

        self.station_middle_task_dic = {0: "reset", 1:"weld_left", 2:"weld_middle", 3:"weld_right"}
        self.station_state_middle_dic = {-1:"resetting", 0: "reset_empty", 1:"placing", 2:"placed", 3:"moving_left", 4:"welding_left", 
                                         5:"welded_left", 6:"welding_right", 7:"welded_right", 8:"welding_upper", 9:"welded_upper"}
        self.station_state_inner_middle = 0
        self.station_state_outer_middle = 0
        self.station_task_inner_middle = 0
        self.station_task_outer_middle = 0
        
        self.station_right_task_dic = {0: "reset", 1:"weld"}
        self.station_state_right_dic = {0: "reset_empty", 1:"placing", 2:"placed", 3:"moving", 4:"welding_right", -1:"resetting"}
        self.station_state_inner_right = 0
        self.station_state_outer_right = 0
        self.station_task_outer_right = 0
        self.station_task_inner_right = 0
        
        self.process_groups_dict = {}
        self.proc_groups_inner_list = []
        self.proc_groups_outer_list = []
        hoop_world_pose_position, hoop_world_pose_orientation = self.obj_11_station_0_revolution.get_local_poses()

        '''side table state'''
        self.side_table_state_dic = {0: "empty", 1:"placing", 2:"placed"}
        self.table_capacity = 4
        self.num_side_table_hoop = 0
        self.num_side_table_bending_tube = 0
        self.state_side_table_hoop = 0
        self.state_side_table_bending_tube = 0

        '''for humans workers (characters) and robots (agv+boxs)'''
        self.character_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Characters/male_adult_construction_01",
            name="character_1",
            track_contact_forces=True,
        )
        self.character_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Characters/male_adult_construction_02",
            name="character_2",
            track_contact_forces=True,
        )
        scene.add(self.character_1)
        scene.add(self.character_2)
        character_list = [self.character_1, self.character_2]
        # self.characters = Characters(character_list)

        self.box_1 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/AGVs/box_01",
            name="box_1",
            track_contact_forces=True,
        )
        self.box_2 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/AGVs/box_02",
            name="box_2",
            track_contact_forces=True,
        )
        scene.add(self.box_1)
        scene.add(self.box_2)
        box_list = [self.box_1, self.box_2]
        # self.transboxs = TransBoxs(box_list)

        self.agv_1 = ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/AGVs/agv_01",
            name="agv_1",
            reset_xform_properties=False,
        )
        self.agv_2 = ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/AGVs/agv_02",
            name="agv_2",
            reset_xform_properties=False,
        )
        scene.add(self.agv_1)
        scene.add(self.agv_2)
        agv_list = [self.agv_1, self.agv_2]
        # self.agvs = Agvs(agv_list)

        self.task_manager : TaskManager = TaskManager(character_list, agv_list, box_list)
        '''Ending: for humans workers (characters) and robots (agv+boxs)'''
        # from omniisaacgymenvs.robots.omni_anim_people.scripts.character_behavior import CharacterBehavior
        # from pxr import Sdf
        # prim_path = Sdf.Path(f"/World/envs/env_0" + "/obj/Characters/male_adult_construction_05/ManRoot")
        # self.character_0 = CharacterBehavior(prim_path)
        
        # self.character_0.read_commands_from_file()
        # self.upper_tube_stationt_state_dic = {0:"is_not_full", 1:"fulled"}
        # self.station_state_tube_inner = 0

        # _, hoop_world_pose_orientation = self.materials.hoop_list[self.materials.inner_hoop_processing_index].get_world_poses()
        # from pxr import Gf, UsdGeom
        # self.inital_inner_revolution_matrix = Gf.Matrix4d()
        # position = hoop_world_pose_position.cpu()[0]
        # self.inital_inner_revolution_matrix.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        # orientation = hoop_world_pose_orientation.cpu()[0]
        # self.inital_inner_revolution_matrix.SetRotateOnly(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
        # prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/part9/manipulator2/robotiq_arg2f_base_link")
        # prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/part11/node/Station0")
        # matrix = get_world_transform_matrix(prim)
        # translate = matrix.ExtractTranslation()
        # rotation: Gf.Rotation = matrix.ExtractRotation()
        self.pre_progress_buf = 0
        # scene.add(self.obj_cube)
        # scene.add(self.obj_0_2_2)
        # scene.add(self.frankas)
        # scene.add(self.frankas._hands)
        # scene.add(self.frankas._lfingers)
        # scene.add(self.frankas._rfingers)
        # scene.add(self.frankas._fingertip_centered)
        return
    
    def post_next_group_to_be_processed_step(self):
        cube_index = self.materials.find_next_raw_cube_index()
        upper_tube_index = self.materials.find_next_raw_upper_tube_index()
        hoop_index = self.materials.find_next_raw_hoop_index()
        bending_tube_index = self.materials.find_next_raw_bending_tube_index()
        #todo find a way to better choose weld station 
        # station_inner_available = self.station_state_inner_middle <=0 and self.station_state_inner_left<=0 and self.station_state_inner_right<=0
        # station_outer_available = self.station_state_outer_middle <=0 and self.station_state_outer_left<=0 and self.station_state_outer_right<=0
        if cube_index<0 or upper_tube_index<0 or hoop_index<0 or bending_tube_index<0:
            return -1, -1, -1, -1
        self.materials.cube_states[cube_index] = 1
        self.materials.hoop_states[hoop_index] = 1
        self.materials.bending_tube_states[bending_tube_index] = 1
        self.materials.upper_tube_states[upper_tube_index] = 1
        _dict = {'cube_index':cube_index, 'upper_tube_index':upper_tube_index,  'hoop_index':hoop_index, 'bending_tube_index':bending_tube_index}
        if len(self.proc_groups_inner_list)<=len(self.proc_groups_outer_list):
            _dict['station'] = 'inner'
            self.proc_groups_inner_list.append(cube_index)
        else:
            self.proc_groups_outer_list.append(cube_index)
            _dict['station'] = 'outer'
        self.process_groups_dict[cube_index] = _dict
        return cube_index, upper_tube_index, hoop_index, bending_tube_index

    def initialize_views(self, scene) -> None:
        """Initialize views for extension workflow."""

        super().initialize_views(scene)

        self.import_franka_assets(add_to_stage=False)
        self._import_env_assets(add_to_stage=False)

        if scene.object_exists("frankas_view"):
            scene.remove_object("frankas_view", registry_only=True)
        if scene.object_exists("nuts_view"):
            scene.remove_object("nuts_view", registry_only=True)
        if scene.object_exists("bolts_view"):
            scene.remove_object("bolts_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("fingertips_view"):
            scene.remove_object("fingertips_view", registry_only=True)

        self.frankas = FactoryFrankaView(
            prim_paths_expr="/World/envs/.*/franka", name="frankas_view"
        )
        self.nuts = RigidPrimView(
            prim_paths_expr="/World/envs/.*/nut/factory_nut.*", name="nuts_view"
        )
        self.bolts = RigidPrimView(
            prim_paths_expr="/World/envs/.*/bolt/factory_bolt.*", name="bolts_view"
        )

        scene.add(self.nuts)
        scene.add(self.bolts)
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)

    def create_nut_bolt_material(self):
        """Define nut and bolt material."""

        self.nutboltPhysicsMaterialPath = "/World/Physics_Materials/NutBoltMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.nutboltPhysicsMaterialPath,
            density=self.cfg_env.env.nut_bolt_density,
            staticFriction=self.cfg_env.env.nut_bolt_friction,
            dynamicFriction=self.cfg_env.env.nut_bolt_friction,
            restitution=0.0,
        )

    def _import_env_assets(self, add_to_stage=True):
        """Import modular production assets."""

        self.obj_heights = []
        self.obj_widths_max = []
        self.thread_pitches = []
        self._stage = get_current_stage()
        assets_root_path = get_assets_root_path()

        for i in range(0, self._num_envs):
            # from omni.usd import get_context
            # usd_path = "/home/xue/work/Dataset/3D_model/xjt_v7.usd"
            # get_context().open_stage(usd_path)
            # omni.usd.get_context().open_stage(usd_path)
            # Wait two frames so that stage starts loading
            # self._simulation_app.update()
            # self._simulation_app.update()

            # print("Loading stage...")
            # from omni.isaac.core.utils.stage import is_stage_loading

            # while is_stage_loading():
            #     self._simulation_app.update()
            # print("Loading Complete")
            # omni.timeline.get_timeline_interface().play()
            
            # while self._simulation_app.is_running():
            #     # Run in realtime mode, we don't specify the step size
            #     self._simulation_app.update()
            # omni.timeline.get_timeline_interface().stop()
            # j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            for j in range(0, len(self.cfg_env.env.desired_subassemblies)):

                subassembly = self.cfg_env.env.desired_subassemblies[j]
                components = list(self.asset_info_obj[subassembly])

                obj_translation = torch.tensor(
                    [
                        i*10,
                        i*10,
                        0,
                    ],
                    device=self._device,
                )
                obj_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device)

                obj_height = self.asset_info_obj[subassembly][components[0]]["height"]
                obj_width_max = self.asset_info_obj[subassembly][components[0]][
                    "width_max"
                ]
                self.obj_heights.append(obj_height)
                self.obj_widths_max.append(obj_width_max)

                obj_file = (
                    self.asset_info_obj[subassembly][components[0]]["usd_path"]
                )

                if add_to_stage:
                    add_reference_to_stage(usd_path = obj_file, prim_path = f"/World/envs/env_{i}" + "/obj")
                    XFormPrim(
                        prim_path=f"/World/envs/env_{i}" + "/obj",
                        translation=obj_translation,
                        orientation=obj_orientation,
                    )
                    #debug
                    # stage_utils.print_stage_prim_paths()
                    # self._stage.GetPrimAtPath(
                    #     f"/World/envs/env_{i}" + f"/obj/factory_{components[0]}/collisions"
                    # ).SetInstanceable(
                    #     False
                    # )  # This is required to be able to edit physics material
                    # physicsUtils.add_physics_material_to_prim(
                    #     self._stage,
                    #     self._stage.GetPrimAtPath(
                    #         f"/World/envs/env_{i}"
                    #         + f"/nut/factory_{components[0]}/collisions/mesh_0"
                    #     ),
                    #     self.nutboltPhysicsMaterialPath,
                    # )

                    # applies articulation settings from the task configuration yaml file
                    self._sim_config.apply_articulation_settings(
                        "obj",
                        self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj"),
                        self._sim_config.parse_actor_config("obj"),
                    )
                thread_pitch = self.asset_info_obj[subassembly]["thread_pitch"]
                self.thread_pitches.append(thread_pitch)
            # from omni.kit import commands
            # result, prim_ConveyorBelt_A09_0_0  = commands.execute(
            #     "CreateConveyorBelt",
            #     prim_name="ConveyorActionGraph",
            #     conveyor_prim=self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_0/Belt")
            # )
            # self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_0/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:enabled').Set(False)

            # self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_2/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:enabled').Set(False)
            # self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_2/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:velocity').Set(0)
            # self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_0/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:enabled').Set(False)
            # self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_0/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:velocity').Set(0)


            # self._stage.GetPrimAtPath(f"/World/envs/env_1" + "/obj/ConveyorBelt_A09_0_2/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:enabled').Set(False)
            # self._stage.GetPrimAtPath(f"/World/envs/env_1" + "/obj/ConveyorBelt_A09_0_2/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:velocity').Set(0)
            # self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/ConveyorBelt_A09_0_0/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:enabled').Set(False)
            # self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/ConveyorBelt_A09_0_0/ConveyorBeltGraph/ConveyorNode").GetAttribute('inputs:velocity').Set(0)
            # ConveyorNode_0 = self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_0/ConveyorBeltGraph/ConveyorNode")
            # ConveyorNode_0.GetAttribute('inputs:velocity').Set(100)
            # ConveyorNode_0.GetAttribute('inputs:animateTexture').Set(True)
            # ConveyorNode_0.GetAttribute('inputs:enabled').Set(False)
            # result, prim_ConveyorBelt_A09_0_2 = commands.execute(
            #     "CreateConveyorBelt",
            #     prim_name="ConveyorActionGraph",
            #     conveyor_prim=self._stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/obj/ConveyorBelt_A09_0_2/Belt")
            # )
            

        # For computing body COM pos
        self.obj_heights = torch.tensor(
            self.obj_heights, device=self._device
        ).unsqueeze(-1)

        # For setting initial state
        self.obj_widths_max = torch.tensor(
            self.obj_widths_max, device=self._device
        ).unsqueeze(-1)

        self.thread_pitches = torch.tensor(
            self.thread_pitches, device=self._device
        ).unsqueeze(-1)

    def refresh_env_tensors(self):
        """Refresh tensors."""

        # Nut tensors
        self.nut_pos, self.nut_quat = self.nuts.get_world_poses(clone=False)
        self.nut_pos -= self.env_pos

        self.nut_com_pos = fc.translate_along_local_z(
            pos=self.nut_pos,
            quat=self.nut_quat,
            offset=self.bolt_head_heights + self.nut_heights * 0.5,
            device=self.device,
        )
        self.nut_com_quat = self.nut_quat  # always equal

        nut_velocities = self.nuts.get_velocities(clone=False)
        self.nut_linvel = nut_velocities[:, 0:3]
        self.nut_angvel = nut_velocities[:, 3:6]

        self.nut_com_linvel = self.nut_linvel + torch.cross(
            self.nut_angvel, (self.nut_com_pos - self.nut_pos), dim=1
        )
        self.nut_com_angvel = self.nut_angvel  # always equal

        self.nut_force = self.nuts.get_net_contact_forces(clone=False)

        # Bolt tensors
        self.bolt_pos, self.bolt_quat = self.bolts.get_world_poses(clone=False)
        self.bolt_pos -= self.env_pos

        self.bolt_force = self.bolts.get_net_contact_forces(clone=False)
