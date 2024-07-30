import open3d as o3d
import numpy as np

def create_frame(size=1.0, origin=[0, 0, 0]):
    """Creates a coordinate frame mesh."""
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=origin
    )
    return mesh_frame

def transform_frame(frame, transformation):
    """Applies a transformation to a coordinate frame."""
    frame.transform(transformation)
    return frame

# Define the base frame
base_frame = create_frame(size=0.3, origin=[0, 0, 0])

# Define the transformation matrix (4x4) from camera to base
camera_to_base_transformation = np.array([[ 0.09529217, 0.47956549, -0.87231666, 0.83413991],
                           [ 0.99378559, 0.00481251, 0.11120721, 0.11541528],
                           [0.05752917, -0.8774929,  -0.47612667, 0.40279357],
                           [0, 0, 0, 1]])

# Define the camera frame and apply the transformation
camera_frame = create_frame(size=0.1)
camera_frame = transform_frame(camera_frame, camera_to_base_transformation)

# Visualize both frames
o3d.visualization.draw_geometries([base_frame, camera_frame])

