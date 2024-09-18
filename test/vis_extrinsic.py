import pyrealsense2 as rs
import numpy as np
import cv2
import os

import open3d as o3d
import numpy as np


def capture_and_save_aligned_images():
    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color and depth streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    try:
        # Wait for a coherent pair of frames: depth and color
        for i in range(30):  # Warm-up frames
            pipeline.wait_for_frames()
        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("Could not acquire depth or color frame")
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Align depth to color frame
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
        # Save images
        cv2.imwrite('tmp/color_image.png', color_image)
        # Save depth image as 16-bit PNG
        np.save('tmp/aligned_depth_image.npy', aligned_depth_image)
        
        print("Images saved successfully.")
        
        # Save camera intrinsics
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        np.save('tmp/camera_intrinsics.npy', np.array([
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy
        ]))
        
        print("Camera intrinsics saved.")
        
    finally:
        # Stop streaming
        pipeline.stop()

def create_coordinate_frame(size=1.0, origin=[0, 0, 0]):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

if __name__ == "__main__":
    # capture_and_save_aligned_images()

    import open3d as o3d
    import numpy as np

    # Load the extrinsic matrix
    extrinsic = np.load('extrinsic.npy')

    # Assume you have captured RGB and depth images from the L515 camera
    # Replace these with your actual image capture code
    color_image = o3d.io.read_image("tmp/color_image.png")
    depth_image = o3d.geometry.Image(np.load("tmp/aligned_depth_image.npy"))

    # Load camera intrinsics
    intrinsic_params = np.load('tmp/camera_intrinsics.npy')
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(intrinsic_params[0]),
        height=int(intrinsic_params[1]),
        fx=intrinsic_params[2],
        fy=intrinsic_params[3],
        cx=intrinsic_params[4],
        cy=intrinsic_params[5]
    )

    # Create an RGB-D image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, 
        depth_scale=4000.0,  # Adjust this based on your depth image scale
        depth_trunc=3.0,     # Maximum depth in meters
        convert_rgb_to_intensity=False)

    # Create point cloud from RGB-D image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    
    # Transform the point cloud using the extrinsic matrix
    pcd.transform(extrinsic)

    # Create a coordinate frame for the camera (origin)
    camera_frame = create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    # Create a coordinate frame for the extrinsic frame
    extrinsic_frame = create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    extrinsic_frame.transform(extrinsic)

    # Visualize the point cloud and coordinate frames
    o3d.visualization.draw_geometries([pcd, camera_frame, extrinsic_frame])

    # # Transform the point cloud using the extrinsic matrix
    # pcd.transform(extrinsic)

    # # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])