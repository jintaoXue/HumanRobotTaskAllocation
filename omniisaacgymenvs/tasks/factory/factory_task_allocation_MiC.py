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
import torch
from typing import Tuple
from omniisaacgymenvs.tasks.factory.factory_task_allocation import FactoryTaskAlloc
from omni.isaac.core.prims import RigidPrimView
from omni.physx.scripts import utils
from pxr import Gf, Sdf, Usd, UsdPhysics, UsdGeom, PhysxSchema
from omni.usd import get_world_transform_matrix

MAX_FLOAT = 3.40282347e38
# import numpy as np
class FactoryTaskAllocMiC(FactoryTaskAlloc):
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
                self.post_characters_step()
                self.post_agvs_step()
                self.post_conveyor_belt_step()
                self.post_cutting_machine_step()
                self.post_grippers_step()
                self.post_weld_station_step()
                self.post_welder_step()

                # self.post_characters_step()
            # self.refresh_base_tensors()
            # self.refresh_env_tensors()
            # self._refresh_task_tensors()
            # self.get_observations()
            self.get_states()
            # self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_characters_step(self):
        # self.character_0.get_simulation_commands()
        # self.world.get_physics_dt()
        # self.world.current_time
        # self.character_0.on_update(self.world.current_time, self.world.get_physics_dt())
        for charac_idx in range(0, self.characters.num):
            self.post_character_step(charac_idx)
        return
    
    def post_character_step(self, idx):
        charac = self.characters.list[idx]
        state = self.characters.states[idx]
        task = self.characters.tasks[idx]
        # corresp_agv_idx = self.characters.corresp_agvs_idxs[idx] 
        corresp_box_idx = self.characters.corresp_boxs_idxs[idx] 
        if state == 0: #"worker is free" 
            if self.state_side_table_hoop == 0:
                corresp_box_idx = self.transboxs.find_corresp_box_idx_for_charac(charac_idx = idx)
                if corresp_box_idx >= 0:
                    task = 1 #putting 
                    state = 1 
                    self.characters.corresp_boxs_idxs[idx] = corresp_box_idx
                    self.transboxs.corresp_charac_idxs[corresp_box_idx] = idx
                
        if state == 1: #worker is approaching 
            if task == 1:
        
        return

    def post_agvs_step(self):

        return
    
    def post_conveyor_belt_step(self):
        '''material long cube'''
        #first check the state
        if  self.convey_state == 0:
            #conveyor is free, we can process it
            raw_cube_index, upper_tube_index, hoop_index, bending_tube_index = self.post_next_group_to_be_processed_step()
            if raw_cube_index >=0 :
                #todo check convey startup
                self.materials.cube_convey_index = raw_cube_index
                self.convey_state = 1
                self.materials.cube_states[raw_cube_index] = 2
        elif self.convey_state ==1:
            #conveyor is waiting for the cube 
            if self.put_cube_on_conveyor(self.materials.cube_convey_index):
                self.convey_state = 2
                self.materials.cube_list[self.materials.cube_convey_index].set_world_poses(positions=torch.tensor([[-10.3648,   1.9675,   2.9610]], device='cuda:0'), orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
                self.materials.cube_list[self.materials.cube_convey_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        elif self.convey_state == 2:
            #start conveying, the threhold means the cube arrived cutting area
            threhold = -20.99342
            obj_index = self.materials.cube_convey_index
            obj :RigidPrimView = self.materials.cube_list[obj_index]
            obj_state = self.materials.cube_states[obj_index]
            obj_world_pose = obj.get_world_poses()
            # delta_pose = obj_world_pose[0] - torch.tensor(self.conveyor_pose_list[-1], device='cuda:0') 
            #check the threhold to know whether the cude arrived at cutting position
            if obj_state == 2:
                if obj_world_pose[0][0][0] <= threhold:
                    #conveyed to the cutting machine
                    self.materials.cube_states[obj_index] = 3
                    self.materials.cube_cut_index = obj_index
                else:
                    #keep conveying the cube
                    obj_world_pose[0][0][0] -=  0.2
                obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
                obj.set_velocities(torch.zeros((1,6), device='cuda:0'))
                return
            elif obj_state in range(3,6):
                #3:"conveyed", 4:"cutting", 5:"cut_done",
                # obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
                # obj.set_velocities(torch.zeros((1,6), device='cuda:0'))
                obj.set_world_poses(positions=obj_world_pose[0], orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
                obj.set_velocities(torch.zeros((1,6), device='cuda:0'))
                return
            elif obj_state == 6:
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
                self.materials.cube_states[cube_cut_index] = 4
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
                self.materials.cube_states[cube_cut_index] = 5
                self.materials.pick_up_place_cube_index = cube_cut_index
                self.materials.cube_cut_index = -1
        self.obj_part_10.set_joint_positions(dof_pos_10[0])
        self.obj_part_10.set_joint_velocities(dof_vel_10[0])   

    def post_grippers_step(self):
        next_pos_inner = None
        next_pos_outer = None
        next_gripper_pose = torch.zeros(size=(20,), device='cuda:0')
        dof_vel = torch.zeros(size=(1,20), device='cuda:0')

        gripper_pose = self.obj_part_7.get_joint_positions(clone=False)
        gripper_pose_inner = torch.index_select(gripper_pose, 1, torch.tensor([1,3,5,7,12,13,14,15,18,19], device='cuda'))
        gripper_pose_outer = torch.index_select(gripper_pose, 1, torch.tensor([0,2,4,6,8,9,10,11,16,17], device='cuda'))

        next_pos_inner = self.post_inner_gripper_step(next_pos_inner, gripper_pose_inner)
        next_pos_outer = self.post_outer_gripper_step(next_pos_outer, gripper_pose_outer, gripper_pose_inner)
        next_gripper_pose = self.merge_two_grippers_pose(next_gripper_pose, next_pos_inner, next_pos_outer)

        self.obj_part_7.set_joint_positions(next_gripper_pose)
        self.obj_part_7.set_joint_velocities(dof_vel[0])
        self.obj_part_7.set_joint_efforts(dof_vel[0])
        return 
    
    def post_inner_gripper_step(self, next_pos_inner, gripper_pose_inner):
        pick_up_place_cube_index = self.materials.pick_up_place_cube_index
        inner_initial_pose = torch.zeros(size=(1,10), device='cuda:0')
        positions, orientations = self.obj_part_9_manipulator.get_world_poses()
        translate_tensor = torch.tensor([[-0.65,   1.3,   -1.6]], device='cuda:0') #default as cube
        if self.gripper_inner_state == 0:
            #gripper is free and empty todo
            self.gripper_inner_task = 0
            # stations_are_full = self.station_state_inner_middle and self.station_state_outer_middle #only state == 0 means free, -1 and >= 0 means full
            if (pick_up_place_cube_index>=0): #pick cut cube by cutting machine
                if self.process_groups_dict[pick_up_place_cube_index]['station'] == 'inner':
                    if self.station_state_inner_middle == 1: #station is waiting
                        self.gripper_inner_task = 1
                        self.gripper_inner_state = 1
                elif self.process_groups_dict[pick_up_place_cube_index]['station'] == 'outer':
                    if self.station_state_outer_middle == 1 and self.gripper_outer_state == 0: #station is wating
                        self.gripper_inner_task = 1
                        self.gripper_inner_state = 1
            next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], inner_initial_pose[0], 'reset')
        elif self.gripper_inner_state == 1:
            #gripper is picking
            if self.gripper_inner_task == 1: #picking cut cube
                target_pose = torch.tensor([[0.5, 0, -0.65, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'pick')
                if move_done: 
                    #choose which station to place on the cube
                    self.gripper_inner_state = 2
                    if self.process_groups_dict[pick_up_place_cube_index]['station'] == 'inner':
                        self.gripper_inner_task = 2 #place on inner station
                    elif self.process_groups_dict[pick_up_place_cube_index]['station'] == 'outer':
                        self.gripper_inner_task = 3 #place on outer station
            elif self.gripper_inner_task == 4: #pick_product_from_inner
                target_pose = torch.tensor([[-2.55, -1, -0.8, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_inner_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_inner_task = 6 #place_product from inner
                    self.station_state_inner_middle = -1
            elif self.gripper_inner_task == 5: #pick_product_from_outer
                target_pose = torch.tensor([[-2.55-3.44, -1, -0.8, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_inner_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_inner_task = 7 #place_product from outer
                    self.station_state_outer_middle = -1
        elif self.gripper_inner_state == 2: #gripper is placeing
            placing_material = None
            if self.gripper_inner_task == 2: #place_cut_to_inner_station
                self.materials.cube_states[pick_up_place_cube_index] = 6
                placing_material = self.materials.cube_list[pick_up_place_cube_index]
                target_pose = torch.tensor([[-2.4, 0, -1.25, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    self.gripper_inner_state = 0
                    self.materials.cube_states[pick_up_place_cube_index] = 7
                    self.station_state_inner_middle = 2
                    self.materials.pick_up_place_cube_index = -1
            elif self.gripper_inner_task == 3: #place_cut_to_outer_station
                self.materials.cube_states[pick_up_place_cube_index] = 6
                placing_material = self.materials.cube_list[pick_up_place_cube_index]
                target_pose = torch.tensor([[-5.7, 0, -1.3, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    self.gripper_inner_state = 0
                    self.materials.cube_states[pick_up_place_cube_index] = 8
                    self.station_state_outer_middle = 2
                    self.materials.pick_up_place_cube_index = -1
            elif self.gripper_inner_task == 6: #place_product_from_inner
                placing_material = self.materials.product_list[self.materials.inner_cube_processing_index]
                orientations = torch.tensor([[ 9.9921e-01, -8.5482e-04, -3.9828e-02, -5.8904e-04]], device='cuda:0')
                translate_tensor = torch.tensor([[10.1983,   -6.4176,   -2.4110]], device='cuda:0')
                target_pose = torch.tensor([[-9.7, -1, -1, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    "a product is produced and placed on the robot, then do resetting"
                    self.gripper_inner_state = 0
                    self.materials.product_states[self.materials.inner_cube_processing_index] = 1 # product is placed
                    self.proc_groups_inner_list.pop(0)
                    self.materials.inner_cube_processing_index = -1
            elif self.gripper_inner_task == 7: #place_product_from_outer
                placing_material = self.materials.product_list[self.materials.outer_cube_processing_index]
                orientations = torch.tensor([[ 9.9921e-01, -8.5482e-04, -3.9828e-02, -5.8904e-04]], device='cuda:0')
                translate_tensor = torch.tensor([[10.1983,   -6.4176,   -2.4110]], device='cuda:0')
                target_pose = torch.tensor([[-9.7, -1, -1, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0')
                next_pos_inner, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_inner[0], target_pose[0], 'place')
                if move_done:
                    "a product is produced and placed on the robot, then do resetting"
                    self.gripper_inner_state = 0
                    self.materials.product_states[self.materials.inner_cube_processing_index] = 1 # product is placed
                    self.proc_groups_outer_list.pop(0)
                    self.materials.outer_cube_processing_index = -1
            # ref_pose[0] += torch.tensor([[0,   0,   -0.3]], device='cuda:0')
            placing_material.set_world_poses(positions=positions+translate_tensor, orientations=orientations)
            placing_material.set_velocities(torch.zeros((1,6), device='cuda:0'))
            # self.materials.cube_list[pick_up_place_cube_index].set_world_poses(
            #     positions=ref_pose[0]+torch.tensor([[-1,   0,   0]], device='cuda:0'), 
            #     orientations=torch.tensor([[ 9.9490e-01, -1.0071e-01, -5.6209e-04,  5.7167e-03]], device='cuda:0'))
        # elif self.gripper_inner_state == 3:
        #     #gripper picked material
        #     a = 1
            # if self.gripper_inner_task == 
        return next_pos_inner
    
    def post_outer_gripper_step(self, next_pos_outer, gripper_pose_outer, gripper_pose_inner):
        '''if any station have weld upper tube request, activate the outer gripper'''
        pick_up_upper_tube_index = self.materials.pick_up_place_upper_tube_index
        outer_initial_pose = torch.zeros(size=(1,10), device='cuda:0')
        target_pose = torch.zeros(size=(1,10), device='cuda:0')
        if self.gripper_outer_state == 0:
            self.gripper_outer_task = 0
            if self.station_state_inner_middle == 7 and self.check_grippers_in_safe_distance(gripper_pose_inner, gripper_pose_outer): #welded_right
                # check inner gripper reset and wont collide with outer gripper
                self.materials.pick_up_place_upper_tube_index = self.materials.inner_upper_tube_processing_index
                self.gripper_outer_state = 1 #Picking
                self.gripper_outer_task = 1 #pick_upper_tube for inner station
            elif self.station_state_outer_middle == 7 and self.check_grippers_in_safe_distance(gripper_pose_inner, gripper_pose_outer):
                self.materials.pick_up_place_upper_tube_index = self.materials.outer_upper_tube_processing_index
                self.gripper_outer_state = 1 #Picking
                self.gripper_outer_task = 2 #pick_upper_tube for inner station
            next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], outer_initial_pose[0], 'reset')
        elif self.gripper_outer_state == 1:
            #gripper is picking
            if self.gripper_outer_task == 1:
                #picking upper cube for inner station
                target_pose = torch.tensor([[6.45, 0, -1.5, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_outer_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_outer_task = 3
            if self.gripper_outer_task == 2:
                #picking upper cube for outer station
                target_pose = torch.tensor([[6.45, 0, -1.5, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], target_pose[0], 'pick')
                if move_done: 
                    self.gripper_outer_state = 2
                    #check available laser station(always true, making sure station is available before start picking)
                    self.gripper_outer_task = 4
        elif self.gripper_outer_state == 2:
            #gripper is placeing
            position, orientation = self.obj_part_7_manipulator.get_world_poses()
            if self.gripper_outer_task == 3:
                #place upper tube to_inner_station
                target_pose = torch.tensor([[8.3, 0, -0.25, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], target_pose[0], 'place')
                if move_done:
                    if self.station_state_inner_middle == 7:
                        self.station_state_inner_middle = 8 #welding_middle
                    elif self.station_state_inner_middle == 9: #welded_middle 
                        #if done reset the outer gripper
                        self.gripper_outer_state = 0
                # orientation = torch.tensor([[ 1.0000e+00,  9.0108e-17, -1.9728e-17,  1.0443e-17]], device='cuda:0')
                orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device='cuda:0')
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(
                    positions=position+torch.tensor([[0,   0,   -2]], device='cuda:0'), orientations=orientation)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            elif self.gripper_outer_task == 4:
                #place upper tube to outer station
                target_pose = torch.tensor([[5.2, 0, -0.25, 0, 0.045, -0.045, -0.045, -0.045, 0.045,  0.045]], device='cuda:0')
                next_pos_outer, delta_pos, move_done = self.get_gripper_moving_pose(gripper_pose_outer[0], target_pose[0], 'place')
                if move_done:
                    if self.station_state_outer_middle == 7:
                        self.station_state_outer_middle = 8 #welding_middle
                    elif self.station_state_outer_middle == 9: #welded_middle 
                        #if done reset the outer gripper
                        self.gripper_outer_state = 0
                # orientation = torch.tensor([[ 1.0000e+00,  9.0108e-17, -1.9728e-17,  1.0443e-17]], device='cuda:0')
                orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device='cuda:0')
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(
                    positions=position+torch.tensor([[0,   0,   -2]], device='cuda:0'), orientations=orientation)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))

        return next_pos_outer
    
    def check_grippers_in_safe_distance(self, gripper_inner, gripper_outer):
        
        return (torch.abs(gripper_inner[0][0] + 10.73 - gripper_outer[0][0]) > 4 and self.gripper_inner_state == 0)

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
        next_gripper_pose, delta_pose = self.get_next_pose_helper(gripper_pose, new_target_pose, self.operator_gripper)
        return next_gripper_pose, delta_pose, False

    def get_next_pose_helper(self, pose, target_pose, operator_gripper):
        delta_pose = target_pose - pose
        sign = torch.sign(delta_pose)
        next_pose = sign*operator_gripper + pose
        next_pose_not_reach_target = torch.where((target_pose - next_pose)*delta_pose>0, True, False)
        next_pose = torch.where(next_pose_not_reach_target, next_pose, target_pose)
        delta_pose = next_pose - pose
        return next_pose, delta_pose
    
    def post_weld_station_step(self):
        #inner station step
        weld_station_inner_pose = self.obj_11_station_0.get_joint_positions()
        inner_revolution_target = self.post_weld_station_inner_hoop_step(weld_station_inner_pose[:, 0])
        inner_mid_target_A, inner_mid_target_B = self.post_weld_station_inner_cube_step(weld_station_inner_pose[:, 1], weld_station_inner_pose[:, 2])
        inner_right_target = self.post_weld_station_inner_bending_tube_step(weld_station_inner_pose[:, 3])
        self.post_weld_station_inner_tube_step()

        target_pose = torch.tensor([[inner_revolution_target,  inner_mid_target_A,  inner_mid_target_B, inner_right_target]], device='cuda:0')
        next_pose, _ = self.get_next_pose_helper(weld_station_inner_pose, target_pose, self.operator_station)
        self.obj_11_station_0.set_joint_positions(next_pose)
        self.obj_11_station_0.set_joint_velocities(torch.zeros(4, device='cuda:0'))

        #outer station step
        weld_station_outer_pose = self.obj_11_station_1.get_joint_positions()
        outer_revolution_target = self.post_weld_station_outer_hoop_step(weld_station_outer_pose[:, 0])
        outer_mid_target_A, outer_mid_target_B = self.post_weld_station_outer_cube_step(weld_station_outer_pose[:, 1], weld_station_outer_pose[:, 2])
        outer_right_target = self.post_weld_station_outer_bending_tube_step(weld_station_outer_pose[:, 3])
        self.post_weld_station_outer_tube_step()

        target_pose = torch.tensor([[outer_revolution_target,  outer_mid_target_A,  outer_mid_target_B, outer_right_target]], device='cuda:0')
        next_pose, _ = self.get_next_pose_helper(weld_station_outer_pose, target_pose, self.operator_station)
        self.obj_11_station_1.set_joint_positions(next_pose)
        self.obj_11_station_1.set_joint_velocities(torch.zeros(4, device='cuda:0'))

        return
    
    def post_weld_station_inner_hoop_step(self, dof_inner_revolution):
        THRESHOLD = 0.1
        reset_revolution_pose = 1.5
        inner_revolution_target = 1.5
        inner_hoop_index = self.materials.inner_hoop_processing_index
        # hoop_world_pose_position, hoop_world_pose_orientation = self.materials_hoop_0.get_world_poses()
        if self.station_state_inner_left == 0: #reset_empty
            #station is free now, find existing process group task
            # inner_revolution_target = 1.5
            if len(self.proc_groups_inner_list) > 0:
                raw_cube_index = self.proc_groups_inner_list[0]
                raw_hoop_index = self.process_groups_dict[raw_cube_index]["hoop_index"]
                if self.materials.hoop_states[raw_hoop_index] == 1:
                    self.station_state_inner_left = 1
                    self.materials.inner_hoop_processing_index = raw_hoop_index
                    self.materials.hoop_states[raw_hoop_index] = 2
        elif self.station_state_inner_left == 1: #loading
            # inner_revolution_target = 1.5
            if self.put_hoop_on_weld_station_inner(self.materials.inner_hoop_processing_index):
                self.station_state_inner_left = 2
                self.materials.hoop_states[self.materials.inner_hoop_processing_index] = 3
        elif self.station_state_inner_left == 2: #rotating
            #the station start to rotating 
            inner_revolution_target = 0.0
            delta_pose = torch.abs(dof_inner_revolution[0] - inner_revolution_target)
            if delta_pose < THRESHOLD:
                self.station_state_inner_left = 3
        elif self.station_state_inner_left == 3: #waiting
            #waiting for the station middle is prepared well (and the cube is already placed on the middle station)
            if self.welder_inner_task == 1:
                #the welder task is to weld the left part
                self.station_state_inner_left = 4
        elif self.station_state_inner_left == 4:
            "welding the left part"
        elif self.station_state_inner_left == 5:
            "welded the left part"
        elif self.station_state_inner_left == -1:
            # the station is resetting
            delta_pose = torch.abs(dof_inner_revolution[0] - reset_revolution_pose)
            if delta_pose < THRESHOLD:
                self.station_state_inner_left = 0
            # inner_revolution_target = 1.5
        # ref_pose[0] += torch.tensor([[0,   0,   -0.3]], device='cuda:0')
        if self.station_state_inner_left in range(1,6) and inner_hoop_index >= 0:
            hoop_world_pose_position, hoop_world_pose_orientation = self.obj_11_station_0_revolution.get_world_poses()
            matrix = Gf.Matrix4d()
            orientation = hoop_world_pose_orientation.cpu()[0]
            matrix.SetRotateOnly(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
            translate = Gf.Vec4d(0,0,0.6,0)*matrix
            translate_tensor = torch.tensor(translate[:3], device='cuda:0')
            self.materials.hoop_list[self.materials.inner_hoop_processing_index].set_world_poses(
                positions=hoop_world_pose_position+translate_tensor, orientations=hoop_world_pose_orientation)
            self.materials.hoop_list[self.materials.inner_hoop_processing_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        elif self.station_state_inner_left == 6:
            #set the hoop underground
            self.materials.hoop_list[self.materials.inner_hoop_processing_index].set_world_poses(positions=torch.tensor([[0,0,-100]], device='cuda:0'))
            self.materials.hoop_list[self.materials.inner_hoop_processing_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            self.materials.inner_hoop_processing_index = -1
        if self.station_state_inner_left in range(2,6):
            inner_revolution_target = 0.0
       
        return inner_revolution_target
            
    def put_hoop_on_weld_station_inner(self, raw_hoop_index) -> bool:
        #todo 
        return True
    
    def post_weld_station_inner_cube_step(self, dof_inner_middle_A, dof_inner_middle_B):
        THRESHOLD = 0.05
        welding_left_pose_A = 0.0 
        welding_left_pose_B = 0.0 
        target_inner_middle_A = 0.0 #default is reseted state
        target_inner_middle_B = 0.0
        if self.station_state_inner_middle in range(3,8):
            target_inner_middle_A, target_inner_middle_B = welding_left_pose_A, welding_left_pose_B
        if self.station_state_inner_middle == 0:
            if len(self.proc_groups_inner_list) > 0:
                raw_cube_index = self.proc_groups_inner_list[0]
                if self.materials.cube_states[raw_cube_index] in range(1, 6): #1:"in_list", 2:"conveying", 3:"conveyed", 4:"cutting", 5:"cut_done", 6:"pick_up_place_cut",
                    self.station_state_inner_middle = 1
                    self.materials.inner_cube_processing_index = raw_cube_index
        elif self.station_state_inner_middle == 1:
            "#waiting for the gripper to place the cut cube on station inner middle"
        elif self.station_state_inner_middle == 2: #placed
            #waiting for the station inner left loaded hoop
            if self.station_state_inner_left == 3: #waiting
                self.station_state_inner_middle = 3
        elif self.station_state_inner_middle == 3: #moving left
            #moving left to start welding left part
            # target_inner_middle_A, target_inner_middle_B = welding_left_pose_A, welding_left_pose_B
            if torch.abs(dof_inner_middle_A[0] - welding_left_pose_A) <= THRESHOLD:
                self.station_state_inner_middle = 4
                self.welder_inner_task = 1
                # self.station_state_inner_left = 4
        elif self.station_state_inner_middle == 4: #welding left
            #moved left and wating for the welder finished
            # target_inner_middle_A, target_inner_middle_B = welding_left_pose_A, welding_left_pose_B
            a = 1
        elif self.station_state_inner_middle == 5: #welded_left
            #finished welding left and waiting for the starion right is prepared well
            # target_inner_middle_A, target_inner_middle_B = welding_left_pose_A, welding_left_pose_B
            if self.station_state_inner_right == 4: #welding_right
                #start welding right
                self.station_state_inner_middle = 6 
                self.welder_inner_task = 2
        elif self.station_state_inner_middle == 6: #welding_right
            #welding right waiting for the welder finish
            a = 1
        elif self.station_state_inner_middle == 7: #welded_right
            "post_outer_gripper_step to place the upper tube on cube"
            #change the bending tube pose 
        elif self.station_state_inner_middle == 8: #welding_upper
            "welding upper waiting for the welder finish"
            self.welder_inner_task = 3
        elif self.station_state_inner_middle == 9: #welded_upper
            "finished welding and do the materials merge waiting for the inner gripper to pick up the product"
            
            #set cube to underground 
            self.materials.cube_list[self.materials.inner_cube_processing_index].set_world_poses(positions=torch.tensor([[0,0,-100]], device='cuda:0'))
            self.materials.cube_list[self.materials.inner_cube_processing_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            #set product position
            position, orientation= (torch.tensor([[-12.3281,  -2.8855,  -0.1742]], device='cuda:0'), torch.tensor([[ 9.9978e-01, -6.1045e-04, -2.0787e-02, -4.5162e-05]], device='cuda:0'))
            self.materials.product_list[self.materials.inner_cube_processing_index].set_world_poses(positions=position, orientations=orientation)
        elif self.station_state_inner_middle == -1: #resetting middle part
            if torch.abs(dof_inner_middle_A[0] - target_inner_middle_A) <= THRESHOLD:
                self.station_state_inner_middle = 0
        if self.station_state_inner_middle in range(2,9):
            #set cube pose
            ref_position, ref_orientation = self.obj_11_station_0_middle.get_world_poses()
            cube_index = self.materials.inner_cube_processing_index
            self.materials.cube_list[cube_index].set_world_poses(positions=ref_position+torch.tensor([[-2.4, -4.24,   -0.45]], device='cuda:0'), orientations=ref_orientation)
            self.materials.cube_list[cube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))

        return target_inner_middle_A, target_inner_middle_B
    
    def post_weld_station_inner_bending_tube_step(self, dof_inner_right):
        THRESHOLD = 0.05
        welding_right_pose = -2.6
        target_inner_right = 0.0 
        if self.station_state_inner_right == 0: #reset_empty
            if len(self.proc_groups_inner_list) > 0:
                raw_cube_index = self.proc_groups_inner_list[0]
                raw_bending_tube_index = self.process_groups_dict[raw_cube_index]["bending_tube_index"]
                if self.materials.bending_tube_states[raw_bending_tube_index] == 1: #0:"wait"
                    self.station_state_inner_right = 1
                    self.materials.inner_bending_tube_processing_index = raw_bending_tube_index      
                    self.materials.bending_tube_states[raw_bending_tube_index] = 2    
        elif self.station_state_inner_right == 1: #placing
            #place bending tube on the station right 
            if self.put_bending_tube_on_weld_station_inner(self.materials.inner_bending_tube_processing_index):
                self.station_state_inner_right = 2
                self.materials.bending_tube_states[self.materials.inner_bending_tube_processing_index] = 3
        elif self.station_state_inner_right == 2: #placed
            "waiting for the middle part and welding left task finished"
        elif self.station_state_inner_right == 3: #moving
            #moving
            if torch.abs(dof_inner_right[0] - welding_right_pose) <= THRESHOLD:
                self.station_state_inner_right = 4
                # self.station_state_inner_left = 4
        elif self.station_state_inner_right == 4: #welding_right_step_1
            "wating for the welding right task finished"
        elif self.station_state_inner_right == -1:
            #do the resetting 
            if torch.abs(dof_inner_right[0] - target_inner_right) <= THRESHOLD:
                self.station_state_inner_right = 0
        if self.station_state_inner_right in range(2, 5):
            raw_orientation = torch.tensor([[-0.0017, -0.0032,  0.9999,  0.0163]], device='cuda:0')
            ref_position, _ = self.obj_11_station_0_right.get_world_poses()
            raw_bending_tube_index = self.materials.inner_bending_tube_processing_index
            self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(
                positions=ref_position+torch.tensor([[-2.5,   -3.25,   2]], device='cuda:0'), orientations = raw_orientation)
            self.materials.bending_tube_list[raw_bending_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        if self.station_state_inner_right in range(3,5):
            target_inner_right = welding_right_pose
        return target_inner_right
    
    def put_bending_tube_on_weld_station_inner(self, inner_bending_tube_processing_index):
        return True
    
    def post_weld_station_inner_tube_step(self):
        if len(self.proc_groups_inner_list) > 0:
            raw_cube_index = self.proc_groups_inner_list[0]
            upper_tube_index = self.process_groups_dict[raw_cube_index]['upper_tube_index']
            if self.materials.upper_tube_states[upper_tube_index] == 0: #0:"wait", 1:"conveying", 2:"conveyed", 3:"cutting", 4:"cut_done", 5:"pick_up_place_cut",
                self.materials.upper_tube_states[upper_tube_index] = 1
                self.materials.inner_upper_tube_processing_index = upper_tube_index

    def put_upper_tube_on_station(self, inner_upper_tube_processing_index):
        return True

    def post_welder_step(self):
        self.post_inner_welder_step()
        self.post_outer_welder_step()
        return 

    def post_inner_welder_step(self):
        THRESHOLD = 0.05
        welding_left_pose = -3.5
        welding_middle_pose = -2
        welding_right_pose = 0
        target = 0.0 
        welder_inner_pose = self.obj_11_welding_0.get_joint_positions()
        if self.welder_inner_state == 0: #free_empty
            #waiting for the weld station prepared well 
            if self.welder_inner_task == 1:
                #welding left part task
                self.welder_inner_state = 1
        elif self.welder_inner_state == 1: #moving_left
            #moving to the welding_left_pose
            target = welding_left_pose
            if torch.abs(welder_inner_pose[0] - target) <= THRESHOLD:
                self.welder_inner_state = 2
        elif self.welder_inner_state == 2: #welding_left
            #start welding left
            target = welding_left_pose
            self.welder_inner_oper_time += 1
            if self.welder_inner_oper_time > 10:
                #task finished
                self.welder_inner_oper_time = 0
                self.welder_inner_state = 3 
                self.station_state_inner_left = 5 #welded
                self.station_state_inner_middle = 5 #welded_left
                self.station_state_inner_right = 3 #moving right
                # cube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/cube_" + "{}".format(self.materials.inner_cube_processing_index))
                # hoop_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/hoop_" + "{}".format(self.materials.inner_hoop_processing_index))
                # product_prim = self.create_fixed_joint(self.materials.hoop_list[self.materials.inner_hoop_processing_index], 
                #                                        self.materials.cube_list[self.materials.inner_cube_processing_index], 
                #                                        torch.tensor([[-4.9,   -2,   -3.05]], device='cuda:0'), joint_name='Hoop')
        elif self.welder_inner_state == 3: #welded_left
            target = welding_left_pose
            if self.welder_inner_task == 2:
                self.welder_inner_state = 4
        elif self.welder_inner_state == 4: #moving_right
            #moving to the welding_right_pose
            target = welding_right_pose
            if torch.abs(welder_inner_pose[0] - target) <= THRESHOLD:
                self.welder_inner_state = 5
        elif self.welder_inner_state == 5: #welding_right
            target = welding_right_pose
            self.welder_inner_oper_time += 1
            if self.welder_inner_oper_time > 10:
                #task finished
                self.welder_inner_oper_time = 0
                self.welder_inner_state = 6 
                self.station_state_inner_middle = 7 #welded_right
                self.station_state_inner_right = -1 #welded_right
        elif self.welder_inner_state == 6: #rotate_and_welding
            target = welding_right_pose
            self.welder_inner_oper_time += 1
            # ref_position, ref_orientation = self.materials.cube_list[self.materials.inner_cube_processing_index].get_world_poses()
            # raw_bending_tube_index = self.materials.inner_bending_tube_processing_index
            #initial pose bending tube (tensor([[-23.4063,   3.5746,   2.6908]], device='cuda:0'), tensor([[-0.0154, -0.0033,  0.9999,  0.0044]], device='cuda:0'))
            # https://www.cnblogs.com/meteoric_cry/p/7987548.html
            # _, prev_orientation =  self.materials.bending_tube_list[raw_bending_tube_index].get_world_poses()
            # initial_orientation = torch.tensor([[-0.0154, -0.0033,  0.9999,  0.0044]], device='cuda:0')
            # next_orientation_tensor = self.welder_inner_oper_time*(ref_orientation - initial_orientation)/10 + initial_orientation
            # prev_matrix = Gf.Matrix4d()
            # prev_matrix.SetRotateOnly(Gf.Quatd(float(prev_orientation[0][0]), float(prev_orientation[0][1]), float(prev_orientation[0][2]), float(prev_orientation[0][3])))
            # rot_theta =  9.0*self.welder_inner_oper_time
            # rot_matrix =  Gf.Matrix4d(0.0, 0.0, 1, 0.0,
            #     0,  np.cos(rot_theta), np.sin(rot_theta), 0.0,
            #     0, -np.sin(rot_theta), np.cos(rot_theta), 0.0,
            #     0,0,0, 1.0)
            # next_rot_matrix = prev_matrix*rot_matrix
            # next_orientation_quat = next_rot_matrix.ExtractRotationQuat()
            # real, imaginary = next_orientation_quat.GetReal(), next_orientation_quat.GetImaginary()
            # next_orientation_tensor = torch.tensor([[real, imaginary[0], imaginary[1], imaginary[2]]])
            # self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(
            # positions=ref_position+torch.tensor([[ 1.3123, -1.0179, -2.4780]], device='cuda:0'), orientations=next_orientation_tensor)
            # self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(torch.tensor([[-23.4193,   4.5691,   1.4]], device='cuda:0') ,
            #                                                                           orientations=torch.tensor([[ 0.0051,  0.0026, -0.7029,  0.7113]], device='cuda:0'))
            # self.materials.bending_tube_list[raw_bending_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            if self.welder_inner_oper_time > 10:
                #task finished
                self.welder_inner_oper_time = 0
                self.welder_inner_state = 7 
                self.station_state_inner_middle = 7 #welded_right
                # self.station_state_inner_right = -1 #welded_right
                # cube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/cube_" + "{}".format(self.materials.inner_cube_processing_index))
                # bending_tube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/bending_tube_" + "{}".format(self.materials.inner_bending_tube_processing_index))
                # product_prim = self.create_fixed_joint(self.materials.bending_tube_list[self.materials.inner_bending_tube_processing_index], 
                #                         self.materials.cube_list[self.materials.inner_cube_processing_index], 
                #                         torch.tensor([[0,   0,   0]], device='cuda:0'), joint_name='BendingTube')
        elif self.welder_inner_state == 7: #welded_right
            target= welding_middle_pose
            if torch.abs(welder_inner_pose[0] - target) <= THRESHOLD and self.welder_inner_task == 3:
                self.welder_inner_state = 8
        elif self.welder_inner_state == 8: #welding_upper
            pick_up_upper_tube_index = self.materials.inner_upper_tube_processing_index
            target = welding_middle_pose
            self.welder_inner_oper_time += 1
            if self.welder_inner_oper_time > 10:
                #task finished
                self.welder_inner_oper_time = 0
                self.welder_inner_state = 9
                self.station_state_inner_middle = 9 #welded_upper
                self.station_state_inner_left = -1
                self.welder_inner_task =0
                self.gripper_inner_task = 4
                self.gripper_inner_state = 1
                # cube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/cube_" + "{}".format(self.materials.inner_cube_processing_index))
                # upper_tube_prim = self._stage.GetPrimAtPath(f"/World/envs/env_0" + "/obj/Materials/upper_tube_" + "{}".format(self.materials.inner_upper_tube_processing_index))
                # product_prim = self.create_fixed_joint(self.materials.upper_tube_list[self.materials.inner_upper_tube_processing_index], 
                #                         self.materials.cube_list[self.materials.inner_cube_processing_index], 
                #                         torch.tensor([[0,   0,   0]], device='cuda:0'), joint_name='UpperTube')
                # product_prim = utils.createJoint(self._stage, "Fixed", upper_tube_prim, cube_prim)
                '''set the upper tube under the ground'''
                position = torch.tensor([[0,   0,   -100]], device='cuda:0')
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(positions=position)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            else:
                position = torch.tensor([[-21.5430,   3.4130,   1.1422]], device='cuda:0')
                orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device='cuda:0')
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(positions=position, orientations=orientation)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        elif self.welder_inner_state == 9: #welded_upper
            #do the reset
            if torch.abs(welder_inner_pose[0] - target) <= THRESHOLD:
                self.welder_inner_state = 0
            #set bending tube to underground
            bending_tube_index = self.materials.inner_bending_tube_processing_index
            self.materials.bending_tube_list[bending_tube_index].set_world_poses(torch.tensor([[0,   0,   -100]], device='cuda:0'))
            self.materials.bending_tube_list[bending_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            #set upper tube to under ground
            upper_tube_index = self.materials.inner_upper_tube_processing_index
            self.materials.upper_tube_list[upper_tube_index].set_world_poses(torch.tensor([[0,   0,   -100]], device='cuda:0'))
            self.materials.upper_tube_list[upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            self.materials.inner_bending_tube_processing_index = -1
            self.materials.inner_upper_tube_processing_index = -1   
        if self.welder_inner_state in range(6, 9):
            bending_tube_index = self.materials.inner_bending_tube_processing_index
            self.materials.bending_tube_list[bending_tube_index].set_world_poses(torch.tensor([[-23.4193,   4.5691,   1.35]], device='cuda:0') ,
                                                                                      orientations=torch.tensor([[ 0.0051,  0.0026, -0.7029,  0.7113]], device='cuda:0'))
            self.materials.bending_tube_list[bending_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            # ref_position, ref_orientation = self.materials.cube_list[self.materials.inner_cube_processing_index].get_world_poses()
            # raw_bending_tube_index = self.materials.inner_bending_tube_processing_index
            # self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(
            #     positions=ref_position+torch.tensor([[0.0,   0,   0.5]], device='cuda:0'), 
            #     orientations=ref_orientation+torch.tensor([[0.0,   0,   0.0, 0.0]], device='cuda:0'))
            # self.materials.bending_tube_list[raw_bending_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        # if self.welder_inner_state in range(8, 10):
        # if self.welder_inner_state in range(8, 9):
        #     pick_up_upper_tube_index = self.materials.inner_upper_tube_processing_index
        #     position = torch.tensor([[-21.5430,   3.4130,   1.1422]], device='cuda:0')
        #     orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device='cuda:0')
        #     self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(positions=position, orientations=orientation)
        #     self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        next_pose, _ = self.get_next_pose_helper(welder_inner_pose[0], target, self.operator_welder)
        self.obj_11_welding_0.set_joint_positions(next_pose)
        self.obj_11_welding_0.set_joint_velocities(torch.zeros(1, device='cuda:0'))

    def get_trans_matrix_from_pose(self, position, orientation):
        matrix = Gf.Matrix4d()
        matrix.SetTranslateOnly(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        matrix.SetRotateOnly(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
        return matrix

    def create_fixed_joint(self, from_prim_view: RigidPrimView, to_prim_view: RigidPrimView, translate, joint_name):
        utils.createJoint
        # from_prim_view.disable_rigid_body_physics()
        # from_prim_view.prims[0].GetAttribute('physics:collisionEnabled').Set(False)
        from_position, from_orientation = from_prim_view.get_world_poses()
        to_position, to_orientation = to_prim_view.get_world_poses()
        # to_position = from_position
        # from_position = to_position
        # from_position += torch.tensor([[5,   0,   0]], device='cuda:0')
        to_position += translate
        from_matrix = self.get_trans_matrix_from_pose(from_position[0].cpu(), from_orientation[0].cpu())
        # from_matrix.SetRotateOnly(Gf.Quatd(-0.5, Gf.Vec3d(-0.5, 0.5, 0.5)))
        to_matrix = self.get_trans_matrix_from_pose(to_position[0].cpu(), to_orientation[0].cpu())
        # to_matrix.SetRotateOnly(Gf.Quatd(0.0, Gf.Vec3d(0.0, -0.7071067811865475, 0.7071067811865476)))
        # translate = Gf.Vec3d(0,0,0)*to_matrix.ExtractRotationMatrix()
        # translate = Gf.Vec3d(-5.5, 1, -1)
        # to_matrix.SetTranslateOnly(translate + to_matrix.ExtractTranslation())
        rel_pose = to_matrix * from_matrix.GetInverse()
        rel_pose = rel_pose.RemoveScaleShear()
        # rel_pose.SetRotateOnly(Gf.Quatd(0.0, Gf.Vec3d(0.7071067811865475, 0.0, 0.7071067811865476)))
        # _matrix.ExtractRotationQuat()
        # _matrix = Gf.Matrix4d(0.0, 0.0, 1, 0.0,
        #     0, -1, 0, 0.0,
        #     1, 0, 0, 0.0,
        #     0.8399995477320195, -1.4471829022912717, 2.065864107281353, 1.0)
        pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
        rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())

        from_path = from_prim_view.prim_paths[0]
        to_path = to_prim_view.prim_paths[0]

        joint_path = to_path + '/FixedJoint' + joint_name
        component = UsdPhysics.FixedJoint.Define(self._stage, joint_path)

        # joint_path = to_path + '/PrismaticJoint'
        # component = UsdPhysics.PrismaticJoint.Define(self._stage, joint_path)
        # component.CreateAxisAttr("X")
        # component.CreateLowerLimitAttr(0.0)
        # component.CreateUpperLimitAttr(0.0)
        component.CreateBody0Rel().SetTargets([Sdf.Path(from_path)])
        component.CreateBody1Rel().SetTargets([Sdf.Path(to_path)])
        component.CreateLocalPos0Attr().Set(pos1)
        component.CreateLocalRot0Attr().Set(rot1)
        component.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        component.CreateBreakForceAttr().Set(MAX_FLOAT)
        component.CreateBreakTorqueAttr().Set(MAX_FLOAT)

        return self._stage.GetPrimAtPath(joint_path)
    
    def post_weld_station_outer_hoop_step(self, dof_outer_revolution):
        THRESHOLD = 0.1
        reset_revolution_pose = 1.5
        outer_revolution_target = 1.5
        outer_hoop_index = self.materials.outer_hoop_processing_index
        # hoop_world_pose_position, hoop_world_pose_orientation = self.materials_hoop_0.get_world_poses()
        if self.station_state_outer_left == 0: #reset_empty
            #station is free now, find existing process group task
            # outer_revolution_target = 1.5
            if len(self.proc_groups_outer_list) > 0:
                raw_cube_index = self.proc_groups_outer_list[0]
                raw_hoop_index = self.process_groups_dict[raw_cube_index]["hoop_index"]
                if self.materials.hoop_states[raw_hoop_index] == 1:
                    self.station_state_outer_left = 1
                    self.materials.outer_hoop_processing_index = raw_hoop_index
                    self.materials.hoop_states[raw_hoop_index] = 2
        elif self.station_state_outer_left == 1: #loading
            # outer_revolution_target = 1.5
            if self.put_hoop_on_weld_station_outer(self.materials.outer_hoop_processing_index):
                self.station_state_outer_left = 2
                self.materials.hoop_states[self.materials.outer_hoop_processing_index] = 3
        elif self.station_state_outer_left == 2: #rotating
            #the station start to rotating 
            outer_revolution_target = 0.0
            delta_pose = torch.abs(dof_outer_revolution[0] - outer_revolution_target)
            if delta_pose < THRESHOLD:
                self.station_state_outer_left = 3
        elif self.station_state_outer_left == 3: #waiting
            #waiting for the station middle is prepared well (and the cube is already placed on the middle station)
            if self.welder_outer_task == 1:
                #the welder task is to weld the left part
                self.station_state_outer_left = 4
        elif self.station_state_outer_left == 4:
            "welding the left part"
        elif self.station_state_outer_left == 5:
            "welded the left part"
        elif self.station_state_outer_left == -1:
            # the station is resetting
            delta_pose = torch.abs(dof_outer_revolution[0] - reset_revolution_pose)
            if delta_pose < THRESHOLD:
                self.station_state_outer_left = 0
            # outer_revolution_target = 1.5
        # ref_pose[0] += torch.tensor([[0,   0,   -0.3]], device='cuda:0')
        if self.station_state_outer_left in range(1,6) and outer_hoop_index >= 0:
            hoop_world_pose_position, hoop_world_pose_orientation = self.obj_11_station_1_revolution.get_world_poses()
            matrix = Gf.Matrix4d()
            orientation = hoop_world_pose_orientation.cpu()[0]
            matrix.SetRotateOnly(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
            translate = Gf.Vec4d(0,0,0.6,0)*matrix
            translate_tensor = torch.tensor(translate[:3], device='cuda:0')
            self.materials.hoop_list[self.materials.outer_hoop_processing_index].set_world_poses(
                positions=hoop_world_pose_position+translate_tensor, orientations=hoop_world_pose_orientation)
            self.materials.hoop_list[self.materials.outer_hoop_processing_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        elif self.station_state_outer_left == 6:
            #set the hoop underground
            self.materials.hoop_list[self.materials.outer_hoop_processing_index].set_world_poses(positions=torch.tensor([[0,0,-100]], device='cuda:0'))
            self.materials.hoop_list[self.materials.outer_hoop_processing_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            self.materials.outer_hoop_processing_index = -1
        if self.station_state_outer_left in range(2,6):
            outer_revolution_target = 0.0
       
        return outer_revolution_target
            
    def put_hoop_on_weld_station_outer(self, raw_hoop_index) -> bool:
        #todo 
        return True
    
    def post_weld_station_outer_cube_step(self, dof_outer_middle_A, dof_outer_middle_B):
        THRESHOLD = 0.05
        welding_left_pose_A = 0.0 
        welding_left_pose_B = 0.0 
        target_outer_middle_A = 0.0 #default is reseted state
        target_outer_middle_B = 0.0
        if self.station_state_outer_middle in range(3,8):
            target_outer_middle_A, target_outer_middle_B = welding_left_pose_A, welding_left_pose_B
        if self.station_state_outer_middle == 0:
            if len(self.proc_groups_outer_list) > 0:
                raw_cube_index = self.proc_groups_outer_list[0]
                if self.materials.cube_states[raw_cube_index] in range(1, 6): 
                    self.station_state_outer_middle = 1
                    self.materials.outer_cube_processing_index = raw_cube_index
        elif self.station_state_outer_middle == 1:
            "#waiting for the gripper to place the cut cube on station outer middle"
        elif self.station_state_outer_middle == 2: #placed
            #waiting for the station outer left loaded hoop
            if self.station_state_outer_left == 3: #waiting
                self.station_state_outer_middle = 3
        elif self.station_state_outer_middle == 3: #moving left
            #moving left to start welding left part
            # target_outer_middle_A, target_outer_middle_B = welding_left_pose_A, welding_left_pose_B
            if torch.abs(dof_outer_middle_A[0] - welding_left_pose_A) <= THRESHOLD:
                self.station_state_outer_middle = 4
                self.welder_outer_task = 1
                # self.station_state_outer_left = 4
        elif self.station_state_outer_middle == 4: #welding left
            #moved left and wating for the welder finished
            # target_outer_middle_A, target_outer_middle_B = welding_left_pose_A, welding_left_pose_B
            a = 1
        elif self.station_state_outer_middle == 5: #welded_left
            #finished welding left and waiting for the starion right is prepared well
            # target_outer_middle_A, target_outer_middle_B = welding_left_pose_A, welding_left_pose_B
            if self.station_state_outer_right == 4: #welding_right
                #start welding right
                self.station_state_outer_middle = 6 
                self.welder_outer_task = 2
        elif self.station_state_outer_middle == 6: #welding_right
            #welding right waiting for the welder finish
            a = 1
        elif self.station_state_outer_middle == 7: #welded_right
            "post_outer_gripper_step to place the upper tube on cube"
            #change the bending tube pose 
        elif self.station_state_outer_middle == 8: #welding_upper
            "welding upper waiting for the welder finish"
            self.welder_outer_task = 3
        elif self.station_state_outer_middle == 9: #welded_upper
            "finished welding and do the materials merge waiting for the outer gripper to pick up the product"
            
            #set cube to underground 
            self.materials.cube_list[self.materials.outer_cube_processing_index].set_world_poses(positions=torch.tensor([[0,0,-100]], device='cuda:0'))
            self.materials.cube_list[self.materials.outer_cube_processing_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            #set product position
            position, orientation= (torch.tensor([[-12.3281,  0.4,  -0.1742]], device='cuda:0'), torch.tensor([[ 9.9978e-01, -6.1045e-04, -2.0787e-02, -4.5162e-05]], device='cuda:0'))
            self.materials.product_list[self.materials.outer_cube_processing_index].set_world_poses(positions=position, orientations=orientation)
        elif self.station_state_outer_middle == -1: #resetting middle part
            if torch.abs(dof_outer_middle_A[0] - target_outer_middle_A) <= THRESHOLD:
                self.station_state_outer_middle = 0
        if self.station_state_outer_middle in range(2,9):
            #set cube pose
            ref_position, ref_orientation = self.obj_11_station_1_middle.get_world_poses()
            cube_index = self.materials.outer_cube_processing_index
            self.materials.cube_list[cube_index].set_world_poses(positions=ref_position+torch.tensor([[-2.4, -4.25,   -0.45]], device='cuda:0'), orientations=ref_orientation)
            self.materials.cube_list[cube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))

        return target_outer_middle_A, target_outer_middle_B
    
    def post_weld_station_outer_bending_tube_step(self, dof_outer_right):
        THRESHOLD = 0.05
        welding_right_pose = -2.6
        target_outer_right = 0.0 
        if self.station_state_outer_right == 0: #reset_empty
            if len(self.proc_groups_outer_list) > 0:
                raw_cube_index = self.proc_groups_outer_list[0]
                raw_bending_tube_index = self.process_groups_dict[raw_cube_index]["bending_tube_index"]
                if self.materials.bending_tube_states[raw_bending_tube_index] == 1: #0:"wait"
                    self.station_state_outer_right = 1
                    self.materials.outer_bending_tube_processing_index = raw_bending_tube_index      
                    self.materials.bending_tube_states[raw_bending_tube_index] = 2  
        elif self.station_state_outer_right == 1: #placing
            #place bending tube on the station right 
            if self.put_bending_tube_on_weld_station_outer(self.materials.outer_bending_tube_processing_index):
                self.station_state_outer_right = 2
                self.materials.bending_tube_states[self.materials.outer_bending_tube_processing_index] = 3
        elif self.station_state_outer_right == 2: #placed
            "waiting for the middle part and welding left task finished"
        elif self.station_state_outer_right == 3: #moving
            #moving
            if torch.abs(dof_outer_right[0] - welding_right_pose) <= THRESHOLD:
                self.station_state_outer_right = 4
                # self.station_state_outer_left = 4
        elif self.station_state_outer_right == 4: #welding_right_step_1
            "wating for the welding right task finished"
        elif self.station_state_outer_right == -1:
            #do the resetting 
            if torch.abs(dof_outer_right[0] - target_outer_right) <= THRESHOLD:
                self.station_state_outer_right = 0
        if self.station_state_outer_right in range(2, 5):
            raw_orientation = torch.tensor([[-0.0017, -0.0032,  0.9999,  0.0163]], device='cuda:0')
            ref_position, _ = self.obj_11_station_1_right.get_world_poses()
            raw_bending_tube_index = self.materials.outer_bending_tube_processing_index
            self.materials.bending_tube_list[raw_bending_tube_index].set_world_poses(
                positions=ref_position+torch.tensor([[-2.5,   -3.25,   2]], device='cuda:0'), orientations = raw_orientation)
            self.materials.bending_tube_list[raw_bending_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        if self.station_state_outer_right in range(3,5):
            target_outer_right = welding_right_pose
        return target_outer_right
    
    def put_bending_tube_on_weld_station_outer(self, outer_bending_tube_processing_index):
        return True
    
    def post_weld_station_outer_tube_step(self):
        if len(self.proc_groups_outer_list) > 0:
            raw_cube_index = self.proc_groups_outer_list[0]
            upper_tube_index = self.process_groups_dict[raw_cube_index]['upper_tube_index']
            if self.materials.upper_tube_states[upper_tube_index] == 0: #0:"wait", 1:"conveying", 2:"conveyed", 3:"cutting", 4:"cut_done", 5:"pick_up_place_cut",
                self.materials.upper_tube_states[upper_tube_index] = 1
                self.materials.outer_upper_tube_processing_index = upper_tube_index

    def put_upper_tube_on_station(self, outer_upper_tube_processing_index):
        return True

    def post_outer_welder_step(self):
        THRESHOLD = 0.05
        welding_left_pose = -3.1
        welding_middle_pose = -2
        welding_right_pose = 0
        target = 0.0 
        welder_outer_pose = self.obj_11_welding_1.get_joint_positions()
        if self.welder_outer_state == 0: #free_empty
            #waiting for the weld station prepared well 
            if self.welder_outer_task == 1:
                #welding left part task
                self.welder_outer_state = 1
        elif self.welder_outer_state == 1: #moving_left
            #moving to the welding_left_pose
            target = welding_left_pose
            if torch.abs(welder_outer_pose[0] - target) <= THRESHOLD:
                self.welder_outer_state = 2
        elif self.welder_outer_state == 2: #welding_left
            #start welding left
            target = welding_left_pose
            self.welder_outer_oper_time += 1
            if self.welder_outer_oper_time > 10:
                #task finished
                self.welder_outer_oper_time = 0
                self.welder_outer_state = 3 
                self.station_state_outer_left = 5 #welded
                self.station_state_outer_middle = 5 #welded_left
                self.station_state_outer_right = 3 #moving right
        elif self.welder_outer_state == 3: #welded_left
            target = welding_left_pose
            if self.welder_outer_task == 2:
                self.welder_outer_state = 4
        elif self.welder_outer_state == 4: #moving_right
            #moving to the welding_right_pose
            target = welding_right_pose
            if torch.abs(welder_outer_pose[0] - target) <= THRESHOLD:
                self.welder_outer_state = 5
        elif self.welder_outer_state == 5: #welding_right
            target = welding_right_pose
            self.welder_outer_oper_time += 1
            if self.welder_outer_oper_time > 10:
                #task finished
                self.welder_outer_oper_time = 0
                self.welder_outer_state = 6 
                self.station_state_outer_middle = 7 #welded_right
                self.station_state_outer_right = -1 #welded_right
        elif self.welder_outer_state == 6: #rotate_and_welding
            target = welding_right_pose
            self.welder_outer_oper_time += 1
            if self.welder_outer_oper_time > 10:
                #task finished
                self.welder_outer_oper_time = 0
                self.welder_outer_state = 7 
                self.station_state_outer_middle = 7 #welded_right
        elif self.welder_outer_state == 7: #welded_right
            target= welding_middle_pose
            if torch.abs(welder_outer_pose[0] - target) <= THRESHOLD and self.welder_outer_task == 3:
                self.welder_outer_state = 8
        elif self.welder_outer_state == 8: #welding_upper
            pick_up_upper_tube_index = self.materials.outer_upper_tube_processing_index
            target = welding_middle_pose
            self.welder_outer_oper_time += 1
            if self.welder_outer_oper_time > 10:
                #task finished
                self.welder_outer_oper_time = 0
                self.welder_outer_state = 9
                self.station_state_outer_middle = 9 #welded_upper
                self.station_state_outer_left = -1
                self.welder_outer_task =0
                self.gripper_inner_task = 5 
                self.gripper_inner_state = 1
                '''set the upper tube under the ground'''
                position = torch.tensor([[0,   0,   -100]], device='cuda:0')
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(positions=position)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            else:
                position = torch.tensor([[-21.5430,   6.7130,   1.1422]], device='cuda:0')
                orientation = torch.tensor([[ 7.0711e-01, -6.5715e-12,  1.3597e-12,  7.0711e-01]], device='cuda:0')
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_world_poses(positions=position, orientations=orientation)
                self.materials.upper_tube_list[pick_up_upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        elif self.welder_outer_state == 9: #welded_upper
            #do the reset
            if torch.abs(welder_outer_pose[0] - target) <= THRESHOLD:
                self.welder_outer_state = 0
            #set bending tube to underground
            bending_tube_index = self.materials.outer_bending_tube_processing_index
            self.materials.bending_tube_list[bending_tube_index].set_world_poses(torch.tensor([[0,   0,   -100]], device='cuda:0'))
            self.materials.bending_tube_list[bending_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            #set upper tube to under ground
            upper_tube_index = self.materials.outer_upper_tube_processing_index
            self.materials.upper_tube_list[upper_tube_index].set_world_poses(torch.tensor([[0,   0,   -100]], device='cuda:0'))
            self.materials.upper_tube_list[upper_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
            self.materials.outer_bending_tube_processing_index = -1
            self.materials.outer_upper_tube_processing_index = -1   
        if self.welder_outer_state in range(6, 9):
            bending_tube_index = self.materials.outer_bending_tube_processing_index
            self.materials.bending_tube_list[bending_tube_index].set_world_poses(torch.tensor([[-23.4193,   7.9,   1.35]], device='cuda:0') ,
                                                                                      orientations=torch.tensor([[ 0.0051,  0.0026, -0.7029,  0.7113]], device='cuda:0'))
            self.materials.bending_tube_list[bending_tube_index].set_velocities(torch.zeros((1,6), device='cuda:0'))
        next_pose, _ = self.get_next_pose_helper(welder_outer_pose[0], target, self.operator_welder)
        self.obj_11_welding_1.set_joint_positions(next_pose)
        self.obj_11_welding_1.set_joint_velocities(torch.zeros(1, device='cuda:0'))
    
