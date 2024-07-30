import numpy as np
import open3d as o3d

data_path = 'data/'

if __name__ == '__main__':

    data = np.load('data/pick the lemon/episode_001.npz', allow_pickle=True)
    frame_0 = data['frames'][0]
    camera_data = frame_0['camera_data'][0]
    color = camera_data['color'][..., ::-1]
    depth = camera_data['depth'] * 0.00025 # from mm to m
    intrinsics = camera_data['intrinsics']
    extrinsics = camera_data['extrinsics']

    K = np.array(intrinsics)
    extrinsics = np.array(extrinsics) # cam 2 base
    print(extrinsics)

    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid_mask = z > 0  # Check for valid depth values

    x = (i - K[0, 2]) * z / K[0, 0]
    y = (j - K[1, 2]) * z / K[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3) / 255.0

    # Filter out invalid points and points with depth greater than 0.5 meters
    mask = valid_mask.reshape(-1) & (points[:, 2] <= 1)
    points = points[mask]
    colors = colors[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Apply extrinsics
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    R_inv = np.linalg.inv(R)
    t_inv = -R_inv @ t
    points_transformed = np.dot(R, points.T).T + t
    pcd.points = o3d.utility.Vector3dVector(points_transformed)

    x_bounds = (0.2, 0.6)
    y_bounds = (-0.2, 0.2)
    z_bounds = (-0.2, 0.8)

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[x_bounds[0], y_bounds[0], z_bounds[0]],
        max_bound=[x_bounds[1], y_bounds[1], z_bounds[1]]
    )
    pcd_cropped = pcd.crop(bbox)

    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    o3d.visualization.draw_geometries([pcd, coor])
    o3d.visualization.draw_geometries([pcd_cropped])