import numpy as np
import open3d as o3d

import sys
sys.path.append('D:\\Codes\\XarmDemo\\utils')

from dataset_utils import get_dataset
import time

def filter_out_plane(point_cloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    # Segment the plane
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
    
    # Extract the planar and non-planar points
    planar_cloud = point_cloud.select_by_index(inliers)
    non_planar_cloud = point_cloud.select_by_index(inliers, invert=True)
    
    return non_planar_cloud, planar_cloud

if __name__ == '__main__':

    dataset = get_dataset(bs=1)
    for batch in dataset:
        pts = batch['current_pts'][0].cpu().numpy()
        cols = batch['current_cols'][0].cpu().numpy()

        # gripper_trans = batch['current_gripper_pos'][0].cpu().numpy() # 3 for x, y, z
        # gripper_rot = batch['current_gripper_rot'][0].cpu().numpy() # 3 for pitch, roll, yaw
        gripper_trans = batch['next_gripper_pos'][0].cpu().numpy() # 3 for x, y, z
        gripper_rot = batch['next_gripper_rot'][0].cpu().numpy() # 3 for pitch, roll, yaw
        print(gripper_trans)
        print(gripper_rot)

        # Create gripper coordinate frame
        gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

        # Convert Euler angles (pitch, roll, yaw) to rotation matrix
        R = o3d.geometry.get_rotation_matrix_from_xyz(gripper_rot * np.pi / 180)

        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = gripper_trans

        # Apply transformation to gripper frame
        gripper_frame.transform(T)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        flip_z = np.eye(4)
        flip_z[2, 2] = -1  # Flip the z-axis
        flip_z[0, 3] = 0.5
        # Apply the transformation to the coordinate frame
        coor.transform(flip_z)

        o3d.visualization.draw_geometries([pcd, coor, gripper_frame])
        time.sleep(2)