import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("Frame Time"):
                break
        motion_data = []
        for line in lines[i + 1 :]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    lines = None
    # Read the file
    with open(bvh_file_path, "r") as f:
        lines = f.readlines()
    # Get the joint name, parent and offset
    joint_name = []
    joint_parent = []
    joint_offset = []
    stack = []
    for i in range(len(lines)):
        line = lines[i].lstrip()
        if line.startswith("ROOT"):
            joint_name.append(line.split()[1])
            joint_parent.append(-1)
            joint_offset.append(np.array([float(x) for x in lines[i + 2].split()[1:]]))
            stack.append(0)
        elif line.startswith("JOINT"):
            joint_name.append(line.split()[1])
            joint_parent.append(stack[-1])
            joint_offset.append(np.array([float(x) for x in lines[i + 2].split()[1:]]))
            stack.append(len(joint_name) - 1)
        elif line.startswith("}"):
            stack.pop()
        elif line.startswith("End"):
            joint_name.append(joint_name[stack[-1]] + "_end")
            joint_parent.append(stack[-1])
            joint_offset.append(np.array([float(x) for x in lines[i + 2].split()[1:]]))
            stack.append(len(joint_name) - 1)
        elif line.startswith("MOTION"):
            break
    return joint_name, joint_parent, np.array(joint_offset)


def part2_forward_kinematics(
    joint_name, joint_parent, joint_offset, motion_data, frame_id
):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    # Parse the frame data
    frame_data = motion_data[frame_id].reshape(-1, 3)
    root_pos = frame_data[0]
    rot_data = R.from_euler("XYZ", frame_data[1:], degrees=True)

    # Calculate the joint positions and orientations
    joint_positions = []
    joint_orientations = []
    end_count = 0
    for i, (name, parent, offset) in enumerate(
        zip(joint_name, joint_parent, joint_offset)
    ):
        if name == "RootJoint":
            joint_positions.append(root_pos)
            joint_orientations.append(rot_data[0].as_quat())
        elif name.endswith("_end"):
            end_count += 1
            quat = R.from_quat([0, 0, 0, 1])
            quat_parent = R.from_quat(joint_orientations[parent])
            joint_orientations.append((quat_parent * quat).as_quat())
            joint_positions.append(joint_positions[parent] + quat_parent.apply(offset))
        else:
            quat = rot_data[i - end_count]
            quat_parent = R.from_quat(joint_orientations[parent])
            joint_orientations.append((quat_parent * quat).as_quat())
            joint_positions.append(joint_positions[parent] + quat_parent.apply(offset))

    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出:
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    joint_remove_A, joint_remove_T = [], []

    joint_name_T, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_A, _, _ = part1_calculate_T_pose(A_pose_bvh_path)

    motion_data_A = load_motion_data(A_pose_bvh_path)
    root_position, motion_data_A = motion_data_A[:, :3], motion_data_A[:, 3:]

    motion_data = np.zeros_like(motion_data_A)

    for joint_list, joint_remove in zip(
        [joint_name_A, joint_name_T], [joint_remove_A, joint_remove_T]
    ):
        for i in joint_list:
            if "_end" not in i:
                joint_remove.append(i)

    motion_dict = {
        name: motion_data_A[:, 3 * index : 3 * (index + 1)]
        for index, name in enumerate(joint_remove_A)
    }

    for index, name in enumerate(joint_remove_T):
        if name == "lShoulder":
            motion_dict[name][:, 2] -= 45
        elif name == "rShoulder":
            motion_dict[name][:, 2] += 45
        motion_data[:, 3 * index : 3 * (index + 1)] = motion_dict[name]

    motion_data = np.concatenate([root_position, motion_data], axis=1)
    return motion_data
