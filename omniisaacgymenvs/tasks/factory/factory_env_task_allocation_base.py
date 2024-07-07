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

class Materials(object):
    def __init__(self, cube_list : list, hoop_list : list, bending_tube_list : list, upper_tube_list: list) -> None:

        self.cube_list = cube_list
        self.upper_tube_list = upper_tube_list
        self.hoop_list = hoop_list
        self.bending_tube_list = bending_tube_list

        self.materials_state_dic = {-1:"done", 0:"wait", 1:"conveying", 2:"conveyed", 3:"cutting", 4:"cut_done", 5:"picking_up_cut", 
                                   6:"placed_station_inner", 7:"placed_station_outer", 7:"weld_l", 8:"combine_r", 9:"weld_r"}

        self.cube_states = [0]*len(self.cube_list)
        self.hoop_states = [0]*len(self.hoop_list)
        self.bending_tube_states = [0]*len(self.bending_tube_list)
        self.upper_tube_states = [0]*len(self.upper_tube_list)

        # self.cube_poses = self.get_world_poses(self.cube_list)
        # self.hoop_poses = self.get_world_poses(self.hoop_list)
        # self.bending_tube_poses = self.get_world_poses(self.bending_tube_list)

        # self.raw_cube_index = -1
        # self.process_groups = {}

        self.cube_convey_index = -1
        self.cube_cut_index = -1
        self.pick_up_place_cut_index = -1

    def get_world_poses(self, list):
        poses = []
        for obj in list:
            poses.append(obj.get_world_poses())
        return poses
    
    def update_poses(self):
        pass

    def done(self):
        return max(self.cube_states) == -1

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
        self.obj_part_9_manipulator = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/part9/manipulator2/robotiq_arg2f_base_link", name="obj_part_9_manipulator_", reset_xform_properties=False
        )
        self.obj_11_station_0 =  ArticulationView(
            prim_paths_expr="/World/envs/.*/obj/part11/node/Station0", name="obj_11_station_0", reset_xform_properties=False
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
            prim_paths_expr="/World/envs/.*/obj/Materials/cube_0",
            name="cube_0",
            track_contact_forces=True,
        )
        self.materials_hoop_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/hoop_0",
            name="hoop_0",
            track_contact_forces=True,
        )
        self.materials_bending_tube_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/bending_tube_0",
            name="bending_tube_0",
            track_contact_forces=True,
        )
        self.materials_upper_tube_0 = RigidPrimView(
            prim_paths_expr="/World/envs/.*/obj/Materials/upper_tube_0",
            name="upper_tube_0",
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
        #materials states
        self.materials : Materials = Materials(cube_list=[self.materials_cube_0], hoop_list=[self.materials_hoop_0], 
                                               bending_tube_list=[self.materials_bending_tube_0], upper_tube_list=[self.materials_upper_tube_0])
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
        self.gripper_inner_task_dic = {0: "reset", 1:"pick_cut", 2:"place_cut_to_inner_station", 3:"place_cut_to_outer_station"}
        self.gripper_inner_task = 0
        self.gripper_inner_state_dic = {0: "free_empty", 1:"picking", 2:"placing"}
        self.gripper_inner_state = 0

        self.gripper_outer_task_dic = {0: "reset", 1:"move_inner", 2:"down", 3:"grip", 4:"lifting"}
        self.gripper_outer_task = 0
        self.gripper_outer_state_dic = {0: "empty", 1:"picked"}
        self.gripper_outer_state = 0

        #welder 
        # self.max_speed_welder = 0.1
        self.welder_oper_time = 10
        self.operator_welder = torch.tensor([0.1], device='cuda:0')
        self.welder_task_dic = {0: "reset", 1:"weld_middle", 2:"weld_left", 3:"weld_right"}
        self.welder_state_dic = {0: "free_empty", 1:"welding_middle", 2:"welded_middle", 3:"welding_left", 4:"welded_left", 5:"welding_right", 6:"welded_right"}
        
        self.welder_inner_task = 0
        self.welder_inner_state = 0
        
        self.welder_outer_task = 0
        self.welder_outer_state = 0
        
        #station
        # self.welder_inner_oper_time = 10
        self.operator_station = torch.tensor([0.1, 0.1, 0.1, 0.1], device='cuda:0')
        self.station_task_left_dic = {0: "reset", 1:"weld"}
        self.station_state_left_dic = {0: "reset_empty", 1:"loaded", 2:"rotating", 3:"waiting", 4:"finished", -1:"resetting"}
        self.station_task_inner_left = 0
        self.station_task_outer_left = 0
        self.station_state_inner_left = 0
        self.station_state_outer_left = 0

        self.station_middle_task_dic = {0: "reset", 1:"weld_left", 2:"weld_middle", 3:"weld_right"}
        self.station_state_middle_dic = {-1:"resetting", 0: "reset_empty", 1:"placed", 2:"welding_left", 3:"welding_right", 4:"welding_up", 6:"finished"}
        self.station_state_inner_middle = 0
        self.station_state_outer_middle = 0
        self.station_task_inner_middle = 0
        self.station_task_outer_middle = 0
        
        self.station_right_task_dic = {0: "reset", 1:"weld"}
        self.station_state_right_dic = {0: "reset_empty", 1:"placed", 2:"moving", 3:"waiting", 4:"finished", -1:"resetting"}
        self.station_state_inner_right = 0
        self.station_state_outer_right = 0
        self.station_task_outer_right = 0
        self.station_task_inner_right = 0
        
        self.process_groups = {}
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
    
    def add_next_group_to_be_processed(self, cube_index : int):
        upper_tube_index = self.materials.find_next_raw_upper_tube_index()
        self.materials.upper_tube_states[upper_tube_index] = 1
        hoop_index = self.materials.find_next_raw_hoop_index()
        self.materials.hoop_states[hoop_index] = 1
        bending_tube_index = self.materials.find_next_raw_bending_tube_index()
        self.materials.bending_tube_states[bending_tube_index] = 1
        #todo
        station_inner_available = self.station_state_inner_middle <=0 and self.station_state_inner_left<=0 and self.station_state_inner_right<=0
        # station_outer_available = self.station_state_outer_middle <=0 and self.station_state_outer_left<=0 and self.station_state_outer_right<=0
        weld_station = 'inner' if station_inner_available else 'outer'
        self.process_groups[cube_index] = {'upper_tube_index':upper_tube_index,  'hoop_index':hoop_index, 'bending_tube_index':bending_tube_index, 
                                           'weld_station': weld_station}
        return 

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
