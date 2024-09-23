import numpy as np
import open3d as o3d

import sys
sys.path.append('D:\\Codes\\XarmDemo\\utils')

from dataset_utils import get_dataset
import time
from sklearn.cluster import DBSCAN

def filter_out_plane(point_cloud, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    # Segment the plane
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
    
    # Extract the planar and non-planar points
    planar_cloud = point_cloud.select_by_index(inliers)
    non_planar_cloud = point_cloud.select_by_index(inliers, invert=True)
    
    return non_planar_cloud, planar_cloud


def filter_largest_cluster(pts, cols, eps=0.01, min_samples=100):
    # 执行DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(cols)
    
    # 找出最大的类别（不包括噪声点，噪声点的标签为-1）
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        print("警告：没有找到任何聚类，返回原始数据。")
        return pts, cols
    
    largest_cluster = unique_labels[np.argmax(counts)]
    
    # 创建掩码，True表示要保留的点
    mask = labels != largest_cluster
    
    # 应用掩码到点和颜色
    fil_pts = pts[mask]
    fil_cols = cols[mask]
    
    return fil_pts, fil_cols

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

        import time
        t1 = time.time()
        # pts, cols = filter_largest_cluster(pts, cols)
        print(time.time() - t1)

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