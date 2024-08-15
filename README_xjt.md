# Objects info in task allocation 
Dataset/3D_model/Part1/1_1.usd
是上料管道的运输架



omniisaacgymenvs/cfg/task/FactoryEnvTaskAllocation.yaml
omniisaacgymenvs/cfg/task/FactoryTaskAllocation.yaml
omniisaacgymenvs/tasks/factory/yaml/factory_asset_info_task_allocation.yaml
omniisaacgymenvs/cfg/train/FactoryTaskAllocationPPO.yaml

# 进度
issac-sim 中旋转 90 90 -90

# 下一步尝试用代码控制物体
core API:
https://docs.omniverse.nvidia.com/py/isaacsim/index.html
https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_omnigraph_shortcuts.html?highlight=control%20graph#controller-graphs controller
 

Part1 上料输送轨道+激光除锈工位+下料输送轨道
Part2 法兰料架x2，左+右
Part3 固定物体
Part4 龙门架静态
Part5 运料框
Part6 固定物体 气罐
Part7 龙门架，下料抓手
Part8 龙门架中间放料区
Part9 上料抓手 
Part10 激光切料区
Part11 激光焊接区



Part10 激光切料区
    '''reset laser cutter'''
    dof_pos_10 = torch.tensor([[0., 0]], device='cuda:0')
    dof_vel_10 = torch.tensor([[0., 0]], device='cuda:0')
    self.obj_part_10.set_joint_positions(dof_pos_10[0])
    self.obj_part_10.set_joint_velocities(dof_vel_10[0])
    '''start laser cutter'''
    dof_pos_10 = torch.tensor([[-5, 0.35]], device='cuda:0')
    dof_vel_10 = torch.tensor([[0., 0]], device='cuda:0')
    self.obj_part_10.set_joint_positions(dof_pos_10[0])
    self.obj_part_10.set_joint_velocities(dof_vel_10[0])

#7.12
还差什么部分：
self.station_state_inner_left = 4结束之后怎么复原

如果inner station 发出welding upper 的请求，先确认inner gripper是resetting 状态
prim_transformation:
https://docs.omniverse.nvidia.com/isaacsim/latest/how_to_guides/environment_setup.html?highlight=transform%2520matrix#adding-a-transform-matrix-to-a-prim

#7.14 local_pose transformation to global

https://docs.omniverse.nvidia.com/isaacsim/latest/features/warehouse_logistics/ext_omni_anim_people.html


# character_prim_path可能需要注意
question: https://forums.developer.nvidia.com/t/how-to-control-peoples-animation-in-isaac-using-python-api-rather-than-the-omni-anim-people-extension-with-ui/301824

# 8.6

route planning: https://developer.nvidia.com/blog/optimizing-robot-route-planning-with-nvidia-cuopt-for-isaac-sim/

# 8.12
occupancy map
x-50 30
y-30 40

# 8.13
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
