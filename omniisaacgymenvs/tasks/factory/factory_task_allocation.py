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

"""Factory: Class for nut-bolt pick task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
PYTHON_PATH omniisaacgymenvs/scripts/rlgames_train.py task=FactoryTaskNutBoltPick
"""

import asyncio

import hydra
import omegaconf
import torch
import omni.kit
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.torch.transformations import tf_combine
from typing import Tuple

import omni.isaac.core.utils.torch as torch_utils
import omniisaacgymenvs.tasks.factory.factory_control as fc
from omniisaacgymenvs.tasks.factory.factory_env_task_allocation_base import FactoryEnvTaskAlloc
from omniisaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from omniisaacgymenvs.tasks.factory.factory_schema_config_task import (
    FactorySchemaConfigTask,
)

from omni.isaac.core.prims import RigidPrimView
# import numpy as np
class FactoryTaskAlloc(FactoryEnvTaskAlloc, FactoryABCTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        """Initialize environment superclass. Initialize instance variables."""

        super().__init__(name, sim_config, env)

        self._get_task_yaml_params()

    def _get_task_yaml_params(self) -> None:
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self._task_cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        asset_info_path = "../tasks/factory/yaml/factory_asset_info_task_allocation.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt[""][""][""]["tasks"][
            "factory"
        ][
            "yaml"
        ]  # strip superfluous nesting

        ppo_path = "train/FactoryTaskAllocationPPO.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def post_reset(self) -> None:
        """Reset the world. Called only once, before simulation begins."""

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        # self.acquire_base_tensors()
        self._acquire_task_tensors()

        # self.refresh_base_tensors()
        # self.refresh_env_tensors()
        # self._refresh_task_tensors()

        # Reset all envs
        # indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        # asyncio.ensure_future(
        #     self.reset_idx_async(indices, randomize_gripper_pose=False)
        # )

    def _acquire_task_tensors(self) -> None:
        """Acquire tensors."""
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device
        )

    def pre_physics_step(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self.world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
            # self.reset_idx(env_ids, randomize_gripper_pose=True)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        # self._apply_actions_as_ctrl_targets(
        #     actions=self.actions,
        #     ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
        #     do_scale=True,
        # )

    async def pre_physics_step_async(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self.world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            await self.reset_idx_async(env_ids, randomize_gripper_pose=True)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            do_scale=True,
        )

    def reset_idx(self, env_ids, randomize_gripper_pose) -> None:
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        if randomize_gripper_pose:
            self._randomize_gripper_pose(
                env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
            )

        self._reset_buffers(env_ids)

    async def reset_idx_async(self, env_ids, randomize_gripper_pose) -> None:
        """Reset specified environments."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        if randomize_gripper_pose:
            await self._randomize_gripper_pose_async(
                env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
            )

        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids) -> None:
        """Reset DOF states and DOF targets of Franka."""

        self.dof_pos[env_ids] = torch.cat(
            (
                torch.tensor(
                    self.cfg_task.randomize.franka_arm_initial_dof_pos,
                    device=self.device,
                ),
                torch.tensor(
                    [self.asset_info_franka_table.franka_gripper_width_max],
                    device=self.device,
                ),
                torch.tensor(
                    [self.asset_info_franka_table.franka_gripper_width_max],
                    device=self.device,
                ),
            ),
            dim=-1,
        )  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_positions(self.dof_pos[env_ids], indices=indices)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

    def _reset_object(self, env_ids) -> None:
        """Reset root states of nut and bolt."""

        # Randomize root state of nut
        nut_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        nut_noise_xy = nut_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.nut_pos_xy_noise, device=self.device)
        )

        self.nut_pos[env_ids, 0] = (
            self.cfg_task.randomize.nut_pos_xy_initial[0] + nut_noise_xy[env_ids, 0]
        )
        self.nut_pos[env_ids, 1] = (
            self.cfg_task.randomize.nut_pos_xy_initial[1] + nut_noise_xy[env_ids, 1]
        )
        self.nut_pos[
            env_ids, 2
        ] = self.cfg_base.env.table_height - self.bolt_head_heights.squeeze(-1)

        self.nut_quat[env_ids, :] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        ).repeat(len(env_ids), 1)

        self.nut_linvel[env_ids, :] = 0.0
        self.nut_angvel[env_ids, :] = 0.0

        indices = env_ids.to(dtype=torch.int32)
        self.nuts.set_world_poses(
            self.nut_pos[env_ids] + self.env_pos[env_ids],
            self.nut_quat[env_ids],
            indices,
        )
        self.nuts.set_velocities(
            torch.cat((self.nut_linvel[env_ids], self.nut_angvel[env_ids]), dim=1),
            indices,
        )

        # Randomize root state of bolt
        bolt_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        bolt_noise_xy = bolt_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.bolt_pos_xy_noise, device=self.device)
        )

        self.bolt_pos[env_ids, 0] = (
            self.cfg_task.randomize.bolt_pos_xy_initial[0] + bolt_noise_xy[env_ids, 0]
        )
        self.bolt_pos[env_ids, 1] = (
            self.cfg_task.randomize.bolt_pos_xy_initial[1] + bolt_noise_xy[env_ids, 1]
        )
        self.bolt_pos[env_ids, 2] = self.cfg_base.env.table_height

        self.bolt_quat[env_ids, :] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        ).repeat(len(env_ids), 1)

        indices = env_ids.to(dtype=torch.int32)
        self.bolts.set_world_poses(
            self.bolt_pos[env_ids] + self.env_pos[env_ids],
            self.bolt_quat[env_ids],
            indices,
        )

    def _reset_buffers(self, env_ids) -> None:
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ) -> None:
        """Apply actions from policy as position/rotation/force/torque targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.fingertip_midpoint_pos + pos_actions
        )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_midpoint_quat
        )

        if self.cfg_ctrl["do_force_ctrl"]:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.force_action_scale, device=self.device
                    )
                )

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.torque_action_scale, device=self.device
                    )
                )

            self.ctrl_target_fingertip_contact_wrench = torch.cat(
                (force_actions, torque_actions), dim=-1
            )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def post_physics_step(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1
        move_horizontal = False
        move_vertical = False
        if self.world.is_playing():
            # In this policy, episode length is constant
            is_last_step = self.progress_buf[0] == self.max_episode_length - 1
            #initial pose: self.obj_0_3.get_world_poses() (tensor([[-8.3212,  2.2496,  2.7378]], device='cuda:0'), tensor([[ 0.9977, -0.0665,  0.0074,  0.0064]], device='cuda:0'))
            if not self.materials.done():
                self.post_conveyor_step()
                self.post_cutting_machine_step()
                self.post_grippers_step()
                self.post_weld_station_step()
                self.post_welder_step()
                # world_pose = (torch.tensor([[-21.2700,   2.5704,   1.9564]], device='cuda:0'), torch.tensor([[ 0.9589, -0.2821, -0.00,  0.000]], device='cuda:0'))
                # self.obj_part_9_manipulator.get_world_poses()
                # self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/part9/manipulator2/robotiq_arg2f_base_link").GetAttribute('xformOp:translate')
                # dof_pos_7 = torch.tensor([[-2.4175e-07, -0.01148,  2.1637e-10, 0, 0, -1.36, 0,
                #     0, 0, 0, 0,  
                #     0.045, -0.045, -0.045, -0.045, 
                #     0, 0,  
                #     0.045,  0.045]], device='cuda:0')
                # self.obj_part_7.set_joint_efforts(torch.zeros(dof_pos_7.shape, device='cuda:0')[0])
                # self.obj_0_3.set_world_poses(positions=world_pose[0], orientations=world_pose[1])
                # if self.progress_buf[0] >= 180:

            # self.refresh_base_tensors()
            # self.refresh_env_tensors()
            # self._refresh_task_tensors()
            # self.get_observations()
            self.get_states()
            # self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_conveyor_step(self):
        '''material long cube'''
        #first check the state
        if  self.convey_state == 0:
            #conveyor is free, we can process it
            raw_cube_index = self.materials.find_next_raw_cube_index()
            #todo check convey startup
            if raw_cube_index>=0 and self.put_cube_on_conveyor(raw_cube_index):
                self.add_next_group_to_be_processed(raw_cube_index)
                self.materials.cube_states[raw_cube_index] = 1
                self.materials.cube_convey_index = raw_cube_index
                self.convey_state = 1
            else:
                #todo
                pass
        elif self.convey_state == 1:
            #the threhold means the cube is right under the gripper
            threhold = -20.99342
            obj_index = self.materials.cube_convey_index
            obj :RigidPrimView = self.materials.cube_list[obj_index]
            obj_state = self.materials.cube_states[obj_index]
            obj_world_pose = obj.get_world_poses()
            # delta_pose = obj_world_pose[0] - torch.tensor(self.conveyor_pose_list[-1], device='cuda:0') 
            #check the threhold to know whether the cude arrived at cutting position
            if obj_state == 1:
                if obj_world_pose[0][0][0] <= threhold:
                    #conveyed to the cutting machine
                    self.materials.cube_states[obj_index] = 2
                    self.materials.cube_cut_index = obj_index
                else:
                    #keep conveying the cube
                    obj_world_pose[0][0][0] -=  0.2
                obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
                obj.set_velocities(torch.zeros((1,6), device='cuda:0'))
                return
            elif obj_state in range(2,4):
                #2:"conveyed", 3:"cutting", 4:"cut_done",
                # obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
                # obj.set_velocities(torch.zeros((1,6), device='cuda:0'))
                obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
                obj.set_velocities(torch.zeros((1,6), device='cuda:0'))
                return
            elif obj_state == 5:
                self.convey_state = 0
                self.materials.cube_convey_index = -1
        return        

    def put_cube_on_conveyor(self, cude_index) -> bool:
        #todo 
        return True

    def post_cutting_machine_step(self):
        dof_pos_10 = None
        dof_vel_10 = torch.tensor([[0., 0]], device='cuda:0')
        initial_pose = torch.tensor([[-5., 0]], device='cuda:0')
        end_pose = torch.tensor([[-5, 0.35]], device='cuda:0')
        cube_cut_index = self.materials.cube_cut_index
        if self.cutting_machine_state == 0:
            '''reseted laser cutter'''
            # delta_position = torch.tensor([[0., 0]], device='cuda:0')
            dof_pos_10 = initial_pose
            if cube_cut_index>=0:
                self.cutting_machine_state = 1
        elif self.cutting_machine_state == 1:
            '''cutting cube'''
            if self.c_machine_oper_time < 10:
                self.c_machine_oper_time += 1
                dof_pos_10 = (end_pose - initial_pose)*self.c_machine_oper_time/10 + initial_pose
                self.materials.cube_states[cube_cut_index] = 3
            elif self.c_machine_oper_time == 10:
                self.c_machine_oper_time = 0
                self.cutting_machine_state = 2
                dof_pos_10 = end_pose
                #sending picking flag to gripper
        elif self.cutting_machine_state == 2:
            '''reseting machine'''
            if self.c_machine_oper_time < 5:
                self.c_machine_oper_time += 1
                dof_pos_10 = (initial_pose - end_pose)*self.c_machine_oper_time/5 + end_pose
            elif self.c_machine_oper_time == 5:
                self.c_machine_oper_time = 0
                self.cutting_machine_state = 0
                dof_pos_10 = initial_pose
                #for inner gripper start picking the cut cube
                self.materials.cube_states[cube_cut_index] = 4
                self.materials.pick_up_place_cut_index = cube_cut_index
                self.materials.cube_cut_index = -1
        self.obj_part_10.set_joint_positions(dof_pos_10[0])
        self.obj_part_10.set_joint_velocities(dof_vel_10[0])   

    def post_grippers_step(self):
        next_pos_inner = None
        next_pos_outer = None
        delta_pos = None
        next_gripper_pose = torch.zeros(size=(20,), device='cuda:0')
        dof_vel = torch.zeros(size=(1,20), device='cuda:0')
        inner_initial_pose = torch.zeros(size=(1,10), device='cuda:0')
        outer_initial_pose = torch.zeros(size=(1,10), device='cuda:0')
        # outer_end_pose = torch.tensor([[-5, 0.35]], device='cuda:0')
        # outer_end_pose = torch.tensor([[-5, 0.35]], device='cuda:0')
        gripper_pose = self.obj_part_7.get_joint_positions(clone=False)
        gripper_pose_inner = torch.index_select(gripper_pose, 1, torch.tensor([1,3,5,7,12,13,14,15,18,19], device='cuda'))
        gripper_pose_outer = torch.index_select(gripper_pose, 1, torch.tensor([0,2,4,6,8,9,10,11,16,17], device='cuda'))
        pick_up_place_cut_index = self.materials.pick_up_place_cut_index
        if self.gripper_inner_state == 0:
            #gripper is free and empty todo
            stations_are_full = self.station_state_inner_middle and self.station_state_outer_middle
            if (pick_up_place_cut_index>=0) and not stations_are_full:
                # making sure station is available before start picking
                self.gripper_inner_task = 1
                self.gripper_inner_state = 1    
                #moving inner 
                target_pose = torch.tensor([[0.36, 0, -0.8, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_inner_state = 2
                    #choose which station to place on the cube
                    self.gripper_inner_task = 3 if self.station_state_inner_middle else 2
                    # self.materials.cube_states[pick_up_place_cut_index] = 5
                # dof_pos_7_inner = inner_initial_pose + delta_pos
            else:
                #no task to do, reset
                self.gripper_inner_task = 0
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], inner_initial_pose[0], 'reset')
        elif self.gripper_inner_state == 1:
            #gripper is picking
            if self.gripper_inner_task == 1:
                #picking cut cube
                target_pose = torch.tensor([[0.36, 0, -0.8, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_inner_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_inner_task = 3 if self.station_state_inner_middle else 2
            else:
                #other picking task
                a = 1
        elif self.gripper_inner_state == 2:
            #gripper is placeing
            self.materials.cube_states[pick_up_place_cut_index] = 5
            if self.gripper_inner_task == 2:
                #place_cut_to_inner_station
                target_pose = torch.tensor([[-2.5, 0, -1.3, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    self.gripper_inner_state = 0
                    self.materials.cube_states[pick_up_place_cut_index] = 6
                    self.station_state_inner_middle = 1
                    self.materials.pick_up_place_cut_index = -1
            elif self.gripper_inner_task == 3:
                #place_cut_to_outer_station
                #todo checking collision with outer gripper
                target_pose = torch.tensor([[-2.5 - 3.34, 0, -1.3, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    self.gripper_inner_state = 0
                    self.materials.cube_states[pick_up_place_cut_index] = 7
                    self.station_state_outer_middle = 1
                    self.materials.pick_up_place_cut_index = -1
            else:
                #other placing task
                a = 1
            ref_pose = self.obj_part_9_manipulator.get_world_poses()
            # ref_pose[0] += torch.tensor([[0,   0,   -0.3]], device='cuda:0')
            self.materials.cube_list[pick_up_place_cut_index].set_world_poses(
                positions=ref_pose[0]+torch.tensor([[-0.5,   1.3,   -1.6]], device='cuda:0'), orientations=ref_pose[1])
            self.materials.cube_list[pick_up_place_cut_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            self.materials.cube_list[pick_up_place_cut_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            # self.materials.cube_list[pick_up_place_cut_index].set_world_poses(
            #     positions=ref_pose[0]+torch.tensor([[-1,   0,   0]], device='cuda:0'), 
            #     orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
            a = 1
        # elif self.gripper_inner_state == 3:
        #     #gripper picked material
        #     a = 1
            # if self.gripper_inner_task == 
            
        #merge inner and outer pose
        next_pos_outer = gripper_pose_outer[0]
        # import copy
        next_gripper_pose = self.merge_two_grippers_pose(next_gripper_pose, next_pos_inner, next_pos_outer)
        self.obj_part_7.set_joint_positions(next_gripper_pose)
        self.obj_part_7.set_joint_velocities(dof_vel[0])
        self.obj_part_7.set_joint_efforts(dof_vel[0])
        return 
    
    def merge_two_grippers_pose(self, pose, pose_inner, pose_outer):
        pose[0:7:2] = pose_outer[:4]
        pose[1:8:2] = pose_inner[:4]
        pose[8:12] = pose_outer[4:8]
        pose[12:16] = pose_inner[4:8]
        pose[16:18] = pose_outer[8:]
        pose[18:] = pose_inner[8:]
        return pose
    # def get_material_pose_by_ref_pose(self, ref_pose, delta_pos):

    def get_gripper_moving_pose(self, gripper_pose : torch.Tensor, target_pose : torch.Tensor, task):
        #for one env pose generation
        ####debug
        # gripper_pose = torch.zeros((10), device='cuda:0')
        # target_pose = torch.tensor([-0.01148, 0, -1.36, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045], device='cuda:0')

        #warning, revolution is 0 when doing picking task
        THRESHOLD_A = 0.05
        THRESHOLD_B = 0.04
        threshold = torch.tensor([THRESHOLD_A]*3 + [THRESHOLD_B]*7, device='cuda:0')
        dofs = gripper_pose.shape[0]
        next_gripper_pose = torch.zeros(dofs, device='cuda:0')
        new_target_pose = torch.zeros(dofs, device='cuda:0')
        delta_pose = target_pose - gripper_pose
        move_done = torch.where(torch.abs(delta_pose)<threshold, True, False)
        new_target_pose = torch.zeros(dofs, device='cuda:0')
        next_gripper_pose = torch.zeros(dofs, device='cuda:0')
        next_delta_pose = torch.zeros(dofs, device='cuda:0')
        #todo
        # manipulator_reset_pose = torch.zeros(6, device='cuda:0')
        manipulator_reset_pose = torch.tensor([1.8084e-02, -2.9407e-02, -2.6935e-02, -1.6032e-02,  3.3368e-02,  3.2771e-02], device='cuda:0')
        delta_m = manipulator_reset_pose - gripper_pose[4:]
        reset_done_m = torch.where(torch.abs(delta_m)<THRESHOLD_B, True, False).all()
        '''todo, manipulator faces reseting and move done problems'''
        reset_done_m = True
        move_done[4:] = True
        '''todo, manipulator faces reseting and move done problems'''
        reset_done_revolution = torch.abs(gripper_pose[3]-0)<THRESHOLD_B
        reset_done_up_down = torch.abs(gripper_pose[2]-0)<THRESHOLD_A

        if task == 'place':
            reset_done_m = True
            reset_done_revolution = True
        if move_done.all():
            next_gripper_pose = target_pose
            return next_gripper_pose, next_delta_pose, True
        elif move_done[:4].all():
            #if in out, left right, up down, revolution done, control manipulator
            new_target_pose = target_pose
            # return self.get_gripper_pose_helper(gripper_pose, target_pose), False
        elif move_done[:3].all():
            if  reset_done_m:
                #move revolution, freeze others
                new_target_pose[:3] = gripper_pose[:3]
                new_target_pose[4:] = manipulator_reset_pose
                new_target_pose[3] = target_pose[3]
            else:
                #freeze [:4], do the manipulator reset
                new_target_pose[:4] = gripper_pose[:4]
                new_target_pose[4:] = manipulator_reset_pose
        elif move_done[:2].all():
            #check manipulator reset done
            if reset_done_m:
                new_target_pose[4:] = manipulator_reset_pose
                #check revolution reset done
                if reset_done_revolution:
                    # do the up down 
                    new_target_pose[:4] = gripper_pose[:4]
                    new_target_pose[2] = target_pose[2] 
                    new_target_pose[3] = 0
                else:
                    #freeze [:3] and reset revolution
                    new_target_pose[:3] = gripper_pose[:3]
                    new_target_pose[3] = 0
            else:
                #freeze [:4], do the manipulator reset
                new_target_pose[:4] = gripper_pose[:4]
                new_target_pose[4:] = manipulator_reset_pose
        else:
            #check manipulator reset done
            if reset_done_m:
                new_target_pose[4:] = manipulator_reset_pose
                #check revolution reset done
                if reset_done_revolution:
                    # new_target_pose[3] = gripper_pose[3]
                    new_target_pose[3] = 0
                    # check the up down 
                    if reset_done_up_down:
                        #do in out and left right
                        # new_target_pose[2] = gripper_pose[2]
                        new_target_pose[2] = 0
                        new_target_pose[:2] = target_pose[:2]
                    else:
                        new_target_pose[:2] = gripper_pose[:2]
                        new_target_pose[2] = 0
                else:
                    #freeze [:3]
                    new_target_pose[:3] = gripper_pose[:3]
                    new_target_pose[3] = 0
            else:
                #freeze [:4], do the manipulator reset
                new_target_pose[:4] = gripper_pose[:4]
                new_target_pose[4:] = manipulator_reset_pose
        next_gripper_pose, delta_pose = self.get_gripper_pose_helper(gripper_pose, new_target_pose)
        return next_gripper_pose, delta_pose, False

    
    def get_gripper_pose_helper(self, gripper_pose, target_pose):
        delta_pose = target_pose - gripper_pose
        sign = torch.sign(delta_pose)
        next_gripper_pose = sign*self.operator_gripper + gripper_pose
        next_pose_not_reach_target = torch.where((target_pose - next_gripper_pose)*delta_pose>0, True, False)
        next_gripper_pose = torch.where(next_pose_not_reach_target, next_gripper_pose, target_pose)
        delta_pose = next_gripper_pose - gripper_pose
        return next_gripper_pose, delta_pose
    
    def post_weld_station_step(self):
        #todo materials need to be enough
        #left station step
        if self.station_state_inner_left == 0:
            raw_cube_index = self.materials.find_next_raw_cube_index()
            if raw_cube_index>=0:
                raw_upper_upper_tube_index = self.materials.find_next_raw_upper_tube_index()

        #right station step
        #middle station step
        return
    
    def post_welder_step(self):
        return
    

    async def post_physics_step_async(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        if self.world.is_playing():
            # In this policy, episode length is constant
            is_last_step = self.progress_buf[0] == self.max_episode_length - 1

            if self.cfg_task.env.close_and_lift:
                # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
                if is_last_step:
                    await self._close_gripper_async(
                        sim_steps=self.cfg_task.env.num_gripper_close_sim_steps
                    )
                    await self._lift_gripper_async(
                        sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps
                    )

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pose of nut grasping frame
        self.nut_grasp_quat, self.nut_grasp_pos = tf_combine(
            self.nut_quat,
            self.nut_pos,
            self.nut_grasp_quat_local,
            self.nut_grasp_pos_local,
        )

        # Compute pos of keypoints on gripper and nut in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_gripper[:, idx] = tf_combine(
                self.fingertip_midpoint_quat,
                self.fingertip_midpoint_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            self.keypoints_nut[:, idx] = tf_combine(
                self.nut_grasp_quat,
                self.nut_grasp_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]

    def get_observations(self) -> dict:
        """Compute observations."""

        # Shallow copies of tensors
        obs_tensors = [
            self.fingertip_midpoint_pos,
            self.fingertip_midpoint_quat,
            self.fingertip_midpoint_linvel,
            self.fingertip_midpoint_angvel,
            self.nut_grasp_pos,
            self.nut_grasp_quat,
        ]

        self.obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  # shape = (num_envs, num_observations)

        observations = {self.frankas.name: {"obs_buf": self.obs_buf}}

        return observations

    def calculate_metrics(self) -> None:
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self) -> None:
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def _update_rew_buf(self) -> None:
        """Compute reward at current timestep."""

        keypoint_reward = -self._get_keypoint_dist()
        action_penalty = (
            torch.norm(self.actions, p=2, dim=-1)
            * self.cfg_task.rl.action_penalty_scale
        )

        self.rew_buf[:] = (
            keypoint_reward * self.cfg_task.rl.keypoint_reward_scale
            - action_penalty * self.cfg_task.rl.action_penalty_scale
        )

        # In this policy, episode length is constant across all envs
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1

        if is_last_step:
            # Check if nut is picked up and above table
            lift_success = self._check_lift_success(height_multiple=3.0)
            self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
            self.extras["successes"] = torch.mean(lift_success.float())

    def _get_keypoint_offsets(self, num_keypoints) -> torch.Tensor:
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = (
            torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5
        )

        return keypoint_offsets

    def _get_keypoint_dist(self) -> torch.Tensor:
        """Get keypoint distance."""

        keypoint_dist = torch.sum(
            torch.norm(self.keypoints_nut - self.keypoints_gripper, p=2, dim=-1), dim=-1
        )

        return keypoint_dist

    def _close_gripper(self, sim_steps=20) -> None:
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20) -> None:
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # No hand motion
        self._apply_actions_as_ctrl_targets(
            delta_hand_pose, gripper_dof_pos, do_scale=False
        )

        # Step sim
        for _ in range(sim_steps):
            SimulationContext.step(self.world, render=True)

    def _lift_gripper(
        self, franka_gripper_width=0.0, lift_distance=0.3, sim_steps=20
    ) -> None:
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, franka_gripper_width, do_scale=False
            )
            SimulationContext.step(self.world, render=True)

    async def _close_gripper_async(self, sim_steps=20) -> None:
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        await self._move_gripper_to_dof_pos_async(
            gripper_dof_pos=0.0, sim_steps=sim_steps
        )

    async def _move_gripper_to_dof_pos_async(
        self, gripper_dof_pos, sim_steps=20
    ) -> None:
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )  # No hand motion
        self._apply_actions_as_ctrl_targets(
            delta_hand_pose, gripper_dof_pos, do_scale=False
        )

        # Step sim
        for _ in range(sim_steps):
            self.world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

    async def _lift_gripper_async(
        self, franka_gripper_width=0.0, lift_distance=0.3, sim_steps=20
    ) -> None:
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, franka_gripper_width, do_scale=False
            )
            self.world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

    def _check_lift_success(self, height_multiple) -> torch.Tensor:
        """Check if nut is above table by more than specified multiple times height of nut."""

        lift_success = torch.where(
            self.nut_pos[:, 2]
            > self.cfg_base.env.table_height
            + self.nut_heights.squeeze(-1) * height_multiple,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device),
        )

        return lift_success

    def _randomize_gripper_pose(self, env_ids, sim_steps) -> None:
        """Move gripper to random pose."""

        # step once to update physx with the newly set joint positions from reset_franka()
        SimulationContext.step(self.world, render=True)

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device
            )
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device
            )
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        # Step sim and render
        for _ in range(sim_steps):
            if not self.world.is_playing():
                return

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                do_scale=False,
            )

            SimulationContext.step(self.world, render=True)

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # step once to update physx with the newly set joint velocities
        SimulationContext.step(self.world, render=True)

    async def _randomize_gripper_pose_async(self, env_ids, sim_steps) -> None:
        """Move gripper to random pose."""

        # step once to update physx with the newly set joint positions from reset_franka()
        await omni.kit.app.get_app().next_update_async()

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device
            )
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device
            )
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                do_scale=False,
            )

            self.world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # step once to update physx with the newly set joint velocities
        self.world.physics_sim_view.flush()
        await omni.kit.app.get_app().next_update_async()
