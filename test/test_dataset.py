import numpy as np
import open3d as o3d

import sys
sys.path.append('D:\\Codes\\XarmDemo\\utils')

from dataset_utils import get_dataset
import time

if __name__ == '__main__':

    dataset = get_dataset(bs=1)
    for batch in dataset:
        pts = batch['current_pts'][0].cpu().numpy()
        cols = batch['current_cols'][0].cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([pcd, coor])
        time.sleep(10)