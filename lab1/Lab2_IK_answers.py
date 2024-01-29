import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    # Get joint data from meta_data
    path, _, _, _ = meta_data.get_path_from_root_to_end()
    joint_parent = meta_data.joint_parent
    joint_offset = [meta_data.joint_initial_position[i] - meta_data.joint_initial_position[joint_parent[i]] for i in range(len(joint_positions))]
    joint_offset[0] = np.array([.0, .0, .0])
    joint_rotations = [R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i]) for i in range(len(joint_orientations))]
    joint_rotations[0] = R.from_quat(joint_orientations[0])
    
    # Gradient Descent
    offset_t = [torch.tensor(off) for off in joint_offset]
    positions_t = [torch.tensor(pos) for pos in joint_positions]
    orientations_t = [torch.tensor(R.from_quat(ori).as_matrix()) for ori in joint_orientations]
    rotations_t = [torch.tensor(rot.as_matrix(), requires_grad=True) for rot in joint_rotations]
    target_pose_t = torch.tensor(target_pose)
    iterations = 150
    alpha = 0.02
    for _ in range(iterations):
        for i in range(len(path)):
            if i == 0:
                continue
            curr = path[i]
            prev = path[i - 1]
            # rotations_t is relative rotation
            # From waist to hand
            if prev == joint_parent[curr]:
                orientations_t[curr] = orientations_t[prev] @ rotations_t[curr]
                positions_t[curr] = positions_t[prev] + orientations_t[prev] @ offset_t[curr]
            # From foot to waist
            else:
                orientations_t[curr] = orientations_t[prev] @ rotations_t[prev].transpose(0, 1)
                positions_t[curr] = positions_t[prev] + orientations_t[curr] @ (-offset_t[prev])
        # Optimize
        opti_function = torch.norm(positions_t[path[-1]] - target_pose_t)
        opti_function.backward()
        # Update rotations
        for joint in path:
            if rotations_t[joint].grad is not None:
                rotations_t[joint] = rotations_t[joint] - alpha * rotations_t[joint].grad
    
    # Update joint_positions and joint_orientations
    for i in range(len(path)):
        if i == 0:
            continue
        curr = path[i]
        prev = path[i - 1]
        # From waist to hand
        if prev == joint_parent[curr]:
            joint_orientations[curr] = (R.from_quat(joint_orientations[prev]) * 
                                        R.from_matrix(rotations_t[curr].detach().numpy())).as_quat()
            joint_positions[curr] = joint_positions[prev] + R.from_quat(joint_orientations[prev]).as_matrix() @ np.array(joint_offset[curr])
        # From foot to waist
        else:
            joint_orientations[curr] = (R.from_quat(joint_orientations[prev]) * 
                                        R.from_matrix(rotations_t[prev].detach().numpy()).inv()).as_quat()
            joint_positions[curr] = joint_positions[prev] + R.from_quat(joint_orientations[curr]).as_matrix() @ np.array(-joint_offset[prev])

    # Update other joints
    ik_path_set = set(path)
    for i in range(len(joint_positions)):
        if i in ik_path_set:
            continue
        else:
            joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * joint_rotations[i]).as_quat()
            joint_positions[i] = joint_positions[joint_parent[i]] + R.from_quat(joint_orientations[i]).as_matrix() @ np.array(joint_offset[i])

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations