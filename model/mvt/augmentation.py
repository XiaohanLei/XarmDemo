# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
import torch
import model.mvt.aug_utils as aug_utils
import libs.torch3d_transforms  as torch3d_tf
from scipy.spatial.transform import Rotation


# def perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds):
#     """Perturb point clouds with given transformation.
#     :param pcd:
#         Either:
#         - list of point clouds [[bs, 3, H, W], ...] for N cameras
#         - point cloud [bs, 3, H, W]
#         - point cloud [bs, 3, num_point]
#         - point cloud [bs, num_point, 3]
#     :param trans_shift_4x4: translation matrix [bs, 4, 4]
#     :param rot_shift_4x4: rotation matrix [bs, 4, 4]
#     :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
#     :param bounds: metric scene bounds [bs, 6]
#     :return: peturbed point clouds in the same format as input
#     """
#     # batch bounds if necessary

#     # for easier compatibility
#     single_pc = False
#     if not isinstance(pcd, list):
#         single_pc = True
#         pcd = [pcd]

#     bs = pcd[0].shape[0]
#     if bounds.shape[0] != bs:
#         bounds = bounds.repeat(bs, 1)

#     perturbed_pcd = []
#     for p in pcd:
#         p_shape = p.shape
#         permute_p = False
#         if len(p.shape) == 3:
#             if p_shape[-1] == 3:
#                 num_points = p_shape[-2]
#                 p = p.permute(0, 2, 1)
#                 permute_p = True
#             elif p_shape[-2] == 3:
#                 num_points = p_shape[-1]
#             else:
#                 assert False, p_shape

#         elif len(p.shape) == 4:
#             assert p_shape[-1] != 3, p_shape[-1]
#             assert p_shape[-2] != 3, p_shape[-2]
#             num_points = p_shape[-1] * p_shape[-2]

#         else:
#             assert False, len(p.shape)

#         action_trans_3x1 = (
#             action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
#         )
#         trans_shift_3x1 = (
#             trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
#         )

#         # flatten point cloud
#         p_flat = p.reshape(bs, 3, -1)
#         p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

#         # shift points to have action_gripper pose as the origin
#         p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

#         # apply rotation
#         perturbed_p_flat_4x1_action_origin = torch.bmm(
#             p_flat_4x1_action_origin.transpose(2, 1), rot_shift_4x4
#         ).transpose(2, 1)

#         # apply bounded translations
#         bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
#         bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
#         bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

#         action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
#         action_then_trans_3x1_x = torch.clamp(
#             action_then_trans_3x1[:, 0], min=bounds_x_min, max=bounds_x_max
#         )
#         action_then_trans_3x1_y = torch.clamp(
#             action_then_trans_3x1[:, 1], min=bounds_y_min, max=bounds_y_max
#         )
#         action_then_trans_3x1_z = torch.clamp(
#             action_then_trans_3x1[:, 2], min=bounds_z_min, max=bounds_z_max
#         )
#         action_then_trans_3x1 = torch.stack(
#             [action_then_trans_3x1_x, action_then_trans_3x1_y, action_then_trans_3x1_z],
#             dim=1,
#         )

#         # shift back the origin
#         perturbed_p_flat_3x1 = (
#             perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1
#         )
#         if permute_p:
#             perturbed_p_flat_3x1 = torch.permute(perturbed_p_flat_3x1, (0, 2, 1))
#         perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
#         perturbed_pcd.append(perturbed_p)

#     if single_pc:
#         perturbed_pcd = perturbed_pcd[0]

#     return perturbed_pcd


def perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds):
    """Perturb point clouds with given transformation, handling [num_pts, 3] inputs.
    :param pcd:
        - list of point clouds [tensor(num_pts, 3), tensor(num_pts, 3), ...]
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: perturbed point clouds in the same format as input
    """
    single_pc = False

    bs = len(pcd)

    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for i, p in enumerate(pcd):
        p_shape = p.shape
        permute_p = False

        if len(p.shape) == 2:
            # Handle [num_pts, 3] shape
            num_points = p_shape[0]
            p = p.T  # Shape [3, num_points]
            # p = p.unsqueeze(0)  # Shape [1, 3, num_points]


        action_trans_3x1 = (
            action_gripper_4x4[i, 0:3, 3].unsqueeze(-1).repeat(1, num_points) # 3, npts
        )
        trans_shift_3x1 = (
            trans_shift_4x4[i, 0:3, 3].unsqueeze(-1).repeat(1, num_points)
        )
        c_rot_shift_4x4 = rot_shift_4x4[i]

        # Flatten point cloud
        p_flat = p.reshape(3, -1)
        p_flat_4x1_action_origin = torch.ones(4, p_flat.shape[-1], device=p_flat.device)

        # Shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:3, :] = p_flat - action_trans_3x1 # 4, npts

        # Apply rotation
        perturbed_p_flat_4x1_action_origin = torch.mm(
            p_flat_4x1_action_origin.transpose(1, 0), c_rot_shift_4x4
        ).transpose(1, 0) # 4, npts

        # Apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(
            action_then_trans_3x1[0], min=bounds_x_min, max=bounds_x_max
        )
        action_then_trans_3x1_y = torch.clamp(
            action_then_trans_3x1[1], min=bounds_y_min, max=bounds_y_max
        )
        action_then_trans_3x1_z = torch.clamp(
            action_then_trans_3x1[2], min=bounds_z_min, max=bounds_z_max
        )
        action_then_trans_3x1 = torch.stack(
            [action_then_trans_3x1_x, action_then_trans_3x1_y, action_then_trans_3x1_z],
            dim=0,
        ) # 3, npts

        # Shift back the origin
        perturbed_p_flat_3x1 = (
            perturbed_p_flat_4x1_action_origin[:3, :] + action_then_trans_3x1
        ) # 3, npts

        # if len(p.shape) == 2:
        #     perturbed_p = perturbed_p_flat_3x1.squeeze(0).T  # Shape [num_points, 3]
        # else:
        #     if permute_p:
        #         perturbed_p_flat_3x1 = torch.permute(perturbed_p_flat_3x1, (0, 2, 1))
        #     perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)

        perturbed_pcd.append(perturbed_p_flat_3x1.T)

    if single_pc:
        perturbed_pcd = perturbed_pcd[0]

    return perturbed_pcd



# version copied from peract:
# https://github.com/peract/peract/blob/a3b0bd855d7e749119e4fcbe3ed7168ba0f283fd/voxel/augmentation.py#L68
def apply_se3_augmentation(
    pcd,
    action_gripper_pose,
    action_trans,
    action_rot_grip,
    bounds,
    layer,
    trans_aug_range,
    rot_aug_range,
    rot_aug_resolution,
    voxel_size,
    rot_resolution,
    device,
):
    """Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param action_rot_grip: discretized rotation and gripper action [bs, 4]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    # bs = pcd[0].shape[0]
    bs = len(pcd)

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat(
        (action_gripper_pose[:, 6].unsqueeze(1), action_gripper_pose[:, 3:6]), dim=1
    )
    action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    perturbed_trans = torch.full_like(action_trans, -1.0)
    perturbed_rot_grip = torch.full_like(action_rot_grip, -1.0)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            raise Exception("Failing to perturb action and keep it within bounds.")

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(
            device=device
        )
        trans_shift = trans_range * aug_utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = aug_utils.rand_discrete(
            (bs, 1), min=-roll_aug_steps, max=roll_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        pitch = aug_utils.rand_discrete(
            (bs, 1), min=-pitch_aug_steps, max=pitch_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        yaw = aug_utils.rand_discrete(
            (bs, 1), min=-yaw_aug_steps, max=yaw_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
            torch.cat((roll, pitch, yaw), dim=1), "XYZ"
        )
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            perturbed_action_gripper_4x4[:, :3, :3]
        )
        perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    perturbed_action_quat_wxyz[:, 1:],
                    perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies, rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            trans_idx = aug_utils.point_to_voxel_index(
                perturbed_action_trans[b], voxel_size, bounds_np
            )
            trans_indicies.append(trans_idx.tolist())

            quat = perturbed_action_quat_xyzw[b]
            quat = aug_utils.normalize_quaternion(perturbed_action_quat_xyzw[b])
            if quat[-1] < 0:
                quat = -quat
            disc_rot = aug_utils.quaternion_to_discrete_euler(quat, rot_resolution)
            rot_grip_indicies.append(
                disc_rot.tolist() + [int(action_rot_grip[b, 3].cpu().numpy())]
            )

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        perturbed_rot_grip = torch.from_numpy(np.array(rot_grip_indicies)).to(
            device=device
        )

    action_trans = perturbed_trans
    action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return action_trans, action_rot_grip, pcd


# def apply_se3_aug_con(
#     pcd,
#     action_gripper_pose,
#     bounds,
#     trans_aug_range,
#     rot_aug_range,
#     scale_aug_range=False,
#     single_scale=True,
#     ver=2,
# ):
#     """Apply SE3 augmentation to a point clouds and actions.
#     :param pcd: [bs, num_points, 3]
#     :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
#     :param bounds: metric scene bounds
#         Either:
#         - [bs, 6]
#         - [6]
#     :param trans_aug_range: range of translation augmentation
#         [x_range, y_range, z_range]; this is expressed as the percentage of the scene bound
#     :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
#     :param scale_aug_range: range of scale augmentation [x_range, y_range, z_range]
#     :param single_scale: whether we preserve the relative dimensions
#     :return: perturbed action_gripper_pose,  pcd
#     """

#     # batch size
#     bs = pcd.shape[0]
#     device = pcd.device

#     if len(bounds.shape) == 1:
#         bounds = bounds.unsqueeze(0).repeat(bs, 1).to(device)
#     if len(trans_aug_range.shape) == 1:
#         trans_aug_range = trans_aug_range.unsqueeze(0).repeat(bs, 1).to(device)
#     if len(rot_aug_range.shape) == 1:
#         rot_aug_range = rot_aug_range.unsqueeze(0).repeat(bs, 1)

#     # identity matrix
#     identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

#     # 4x4 matrix of keyframe action gripper pose
#     action_gripper_trans = action_gripper_pose[:, :3]

#     if ver == 1:
#         action_gripper_quat_wxyz = torch.cat(
#             (action_gripper_pose[:, 6].unsqueeze(1), action_gripper_pose[:, 3:6]), dim=1
#         )
#         action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)

#     elif ver == 2:
#         # applying gimble fix to calculate a new action_gripper_rot
#         r = Rotation.from_quat(action_gripper_pose[:, 3:7].cpu().numpy())
#         euler = r.as_euler("xyz", degrees=True)
#         euler = aug_utils.sensitive_gimble_fix(euler)
#         action_gripper_rot = torch.tensor(
#             Rotation.from_euler("xyz", euler, degrees=True).as_matrix(),
#             device=action_gripper_pose.device,
#         )
#     else:
#         assert False

#     action_gripper_4x4 = identity_4x4.detach().clone()
#     action_gripper_4x4[:, :3, :3] = action_gripper_rot
#     action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

#     # sample translation perturbation with specified range
#     # augmentation range is a percentage of the scene bound
#     trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
#     # rand_dist samples value from -1 to 1
#     trans_shift = trans_range * aug_utils.rand_dist((bs, 3)).to(device=device)

#     # apply bounded translations
#     bounds_x_min, bounds_x_max = bounds[:, 0], bounds[:, 3]
#     bounds_y_min, bounds_y_max = bounds[:, 1], bounds[:, 4]
#     bounds_z_min, bounds_z_max = bounds[:, 2], bounds[:, 5]

#     trans_shift[:, 0] = torch.clamp(
#         trans_shift[:, 0],
#         min=bounds_x_min - action_gripper_trans[:, 0],
#         max=bounds_x_max - action_gripper_trans[:, 0],
#     )
#     trans_shift[:, 1] = torch.clamp(
#         trans_shift[:, 1],
#         min=bounds_y_min - action_gripper_trans[:, 1],
#         max=bounds_y_max - action_gripper_trans[:, 1],
#     )
#     trans_shift[:, 2] = torch.clamp(
#         trans_shift[:, 2],
#         min=bounds_z_min - action_gripper_trans[:, 2],
#         max=bounds_z_max - action_gripper_trans[:, 2],
#     )

#     trans_shift_4x4 = identity_4x4.detach().clone()
#     trans_shift_4x4[:, 0:3, 3] = trans_shift

#     roll = np.deg2rad(rot_aug_range[:, 0:1] * aug_utils.rand_dist((bs, 1)))
#     pitch = np.deg2rad(rot_aug_range[:, 1:2] * aug_utils.rand_dist((bs, 1)))
#     yaw = np.deg2rad(rot_aug_range[:, 2:3] * aug_utils.rand_dist((bs, 1)))
#     rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
#         torch.cat((roll, pitch, yaw), dim=1), "XYZ"
#     )
#     rot_shift_4x4 = identity_4x4.detach().clone()
#     rot_shift_4x4[:, :3, :3] = rot_shift_3x3

#     if ver == 1:
#         # rotate then translate the 4x4 keyframe action
#         perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
#     elif ver == 2:
#         perturbed_action_gripper_4x4 = identity_4x4.detach().clone()
#         perturbed_action_gripper_4x4[:, 0:3, 3] = action_gripper_4x4[:, 0:3, 3]
#         perturbed_action_gripper_4x4[:, :3, :3] = torch.bmm(
#             rot_shift_4x4.transpose(1, 2)[:, :3, :3], action_gripper_4x4[:, :3, :3]
#         )
#     else:
#         assert False

#     perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

#     # convert transformation matrix to translation + quaternion
#     perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
#     perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
#         perturbed_action_gripper_4x4[:, :3, :3]
#     )
#     perturbed_action_quat_xyzw = (
#         torch.cat(
#             [
#                 perturbed_action_quat_wxyz[:, 1:],
#                 perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
#             ],
#             dim=1,
#         )
#         .cpu()
#         .numpy()
#     )

#     # TODO: add scale augmentation

#     # apply perturbation to pointclouds
#     # takes care for not moving the point out of the image
#     pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

#     return perturbed_action_trans, perturbed_action_quat_xyzw, pcd


def apply_se3_aug_con(
    pcd,
    action_trans_con,  # [bs, 3], translation in meters
    action_rot,        # [bs, 3], Euler angles in degrees
    bounds,
    trans_aug_range,
    rot_aug_range,
    scale_aug_range=False,
    single_scale=True,
    ver=2,
):
    """Apply SE3 augmentation to point clouds and actions.

    :param pcd: [bs, num_points, 3]
    :param action_trans_con: translation of keyframe action [bs, 3] in meters
    :param action_rot: Euler angles of keyframe action [bs, 3] in degrees
    :param bounds: metric scene bounds [bs, 6] or [6]
    :param trans_aug_range: translation augmentation ranges as percentages of the scene bounds
    :param rot_aug_range: rotation augmentation ranges in degrees
    :param scale_aug_range: scale augmentation ranges (unused in this function)
    :param single_scale: whether to preserve relative dimensions (unused in this function)
    :param ver: version indicator (ver=2 is used here)
    :return: perturbed action translation, perturbed action Euler angles (degrees), perturbed point cloud
    """

    # batch size and device
    bs = len(pcd)
    device = pcd[0].device

    # Ensure inputs are on the correct device
    action_trans_con = action_trans_con.to(device)
    action_rot = action_rot.to(device)
    
    if len(bounds.shape) == 1:
        bounds = bounds.unsqueeze(0).repeat(bs, 1).to(device)
    if len(trans_aug_range.shape) == 1:
        trans_aug_range = trans_aug_range.unsqueeze(0).repeat(bs, 1).to(device)
    if len(rot_aug_range.shape) == 1:
        rot_aug_range = rot_aug_range.unsqueeze(0).repeat(bs, 1).to(device)

    # Identity matrix [bs, 4, 4]
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device)

    # Compute the action gripper transformation matrix
    action_gripper_trans = action_trans_con  # [bs, 3]
    action_rot_rad = torch.deg2rad(action_rot)  # Convert degrees to radians
    action_gripper_rot = torch3d_tf.euler_angles_to_matrix(action_rot_rad, convention="XYZ")  # [bs, 3, 3]

    action_gripper_4x4 = identity_4x4.clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, :3, 3] = action_gripper_trans

    # Compute translation perturbation
    trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device)
    trans_shift = trans_range * aug_utils.rand_dist((bs, 3)).to(device)

    # Apply bounded translations
    bounds_min = bounds[:, :3]  # [bs, 3]
    bounds_max = bounds[:, 3:]  # [bs, 3]
    trans_shift = torch.min(
        torch.max(trans_shift, bounds_min - action_gripper_trans), bounds_max - action_gripper_trans
    )

    trans_shift_4x4 = identity_4x4.clone()
    trans_shift_4x4[:, :3, 3] = trans_shift

    # Compute rotation perturbation
    roll = torch.deg2rad(rot_aug_range[:, 0:1] * aug_utils.rand_dist((bs, 1)).to(device))
    pitch = torch.deg2rad(rot_aug_range[:, 1:2] * aug_utils.rand_dist((bs, 1)).to(device))
    yaw = torch.deg2rad(rot_aug_range[:, 2:3] * aug_utils.rand_dist((bs, 1)).to(device))
    rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
        torch.cat((roll, pitch, yaw), dim=1), convention='XYZ'
    )
    rot_shift_4x4 = identity_4x4.clone()
    rot_shift_4x4[:, :3, :3] = rot_shift_3x3

    # Apply rotation and translation perturbations based on version
    if ver == 1:
        # Rotate then translate the keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
    elif ver == 2:
        # Rotate around the action gripper's own frame
        perturbed_action_gripper_4x4 = identity_4x4.clone()
        perturbed_action_gripper_4x4[:, :3, 3] = action_gripper_4x4[:, :3, 3]
        perturbed_action_gripper_4x4[:, :3, :3] = torch.bmm(
            rot_shift_4x4.transpose(1, 2)[:, :3, :3], action_gripper_4x4[:, :3, :3]
        )
    else:
        raise ValueError("Unknown version number")

    perturbed_action_gripper_4x4[:, :3, 3] += trans_shift

    # Extract perturbed action translation and rotation matrices
    perturbed_action_trans = perturbed_action_gripper_4x4[:, :3, 3]  # [bs, 3]
    perturbed_action_rot_matrices = perturbed_action_gripper_4x4[:, :3, :3]  # [bs, 3, 3]

    # Convert rotation matrices back to Euler angles (in degrees)
    perturbed_action_rot_euler_rad = torch3d_tf.matrix_to_euler_angles(
        perturbed_action_rot_matrices, convention='XYZ'
    )  # [bs, 3] in radians
    perturbed_action_rot_euler = torch.rad2deg(perturbed_action_rot_euler_rad)  # [bs, 3] in degrees

    # Apply perturbation to point clouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return perturbed_action_trans, perturbed_action_rot_euler, pcd