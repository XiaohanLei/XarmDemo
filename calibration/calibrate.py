## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

ip = "192.168.1.197"
# ARUCO_DICT = cv2.aruco.DICT_6X6_250
ARUCO_DICT = cv2.aruco.DICT_4X4_50
PARAMS = cv2.aruco.DetectorParameters()

def detect_pose_aruco(image, intrinsics, distortion, dist=0.05):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # lists of ids and the corners beloning to each id# We call the function 'cv2.aruco.detectMarkers()'
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary, PARAMS)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray_frame)

    # Draw detected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=image, corners=corners, ids=ids, borderColor=(0, 255, 0))

    # # Draw rejected markers:
    # frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))

    success = False
    if ids is not None:
        # rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in corners.
        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, dist, intrinsics, distortion)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, intrinsics, distortion, rvec, tvec, dist)
            success = True
    if success:
        return frame, rvecs[0], tvecs[0]
    else:
        return None
    

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def main():

    arm = XArmAPI(ip)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)


    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)


    # Start streaming
    profile = pipeline.start(config)

    # Get the intrinsics
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    INTRINSICS = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1]])
    print('intrinsics: ', INTRINSICS)

    DISTORTION = np.array(intrinsics.coeffs)


    eye_on_hand = False
    counter = 0
    rmat_marker2camera_list = []
    tvec_marker2camera_list = []
    rmat_ee2base_list = []
    tvec_ee2base_list = []
    if not eye_on_hand:
        trans_camera2marker_list = []
        trans_base2ee_list = []

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            _ , pos = arm.get_position(is_radian=True)

            if eye_on_hand:
                results = detect_pose(color_image)
            else:
                results = detect_pose_aruco(color_image, INTRINSICS, DISTORTION)
            if results is not None:
                vis_images, rvec_marker2camera, tvec_marker2camera = results
                # print(tvec_marker2camera)
            else:
                vis_images = color_image

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', vis_images)
            k = cv2.waitKey(1)
            
            if k == 27:         # wait for ESC key to exit    
                cv2.destroyAllWindows()
                trans_cam2gripper = np.concatenate([R_cam2gripper, t_cam2gripper], axis=1)
                trans_cam2gripper = np.concatenate([trans_cam2gripper, np.array([[0, 0, 0, 1]])], axis=0)
                np.save('extrinsic.npy', trans_cam2gripper)
                break

            elif k == ord('s'):  
                if results is not None:
                    rmat_marker2camera = cv2.Rodrigues(rvec_marker2camera)[0]
                    if not eye_on_hand:
                        # tvec_marker2camera = tvec_marker2camera[:, None]
                        trans_marker2camera = np.concatenate((rmat_marker2camera, tvec_marker2camera), axis=1)
                        trans_marker2camera = np.concatenate((trans_marker2camera, np.array([[0, 0, 0, 1]])), axis=0)
                        trans_camera2marker = np.linalg.inv(trans_marker2camera)
                        trans_camera2marker_list.append(trans_camera2marker)

                    rmat_marker2camera_list.append(rmat_marker2camera)
                    tvec_marker2camera_list.append(tvec_marker2camera.reshape((3, 1)))

                    r_scipy = R.from_euler('xyz', pos[-3:], degrees=False)
                    rmat_ee2base = r_scipy.as_matrix()
                    tvec_ee2base = np.array(pos[:3]) / 1000.
                    # print(tvec_ee2base)
                    # rvec_ee2base = cv2.Rodrigues(rmat_ee2base)[0]
                    rmat_ee2base_list.append(rmat_ee2base)
                    tvec_ee2base_list.append(tvec_ee2base.reshape((3, 1)))

                    if len(rmat_ee2base_list) > 3:
                        if eye_on_hand:
                            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                                    rmat_ee2base_list,
                                    tvec_ee2base_list,
                                    rmat_marker2camera_list,
                                    tvec_marker2camera_list,
                                    method=0
                                ) # method 0 - 5
                            print(R_cam2gripper, t_cam2gripper)
                        else:
                            cal_trans_camera2marker_list = [elem[:3, :3] for elem in trans_camera2marker_list]
                            cal_trans_camera2marker_list = [elem[:3, 3] for elem in trans_camera2marker_list]
                            cal_rmat_base2ee_list = [elem[:3, :3].T for elem in rmat_ee2base_list]
                            cal_tvec_base2ee_list = [-rmat.T @ tvec for rmat, tvec in zip(rmat_ee2base_list, tvec_ee2base_list)]
                            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                                    cal_rmat_base2ee_list,
                                    cal_tvec_base2ee_list,
                                    rmat_marker2camera_list,
                                    tvec_marker2camera_list,
                                    method=0
                                ) # method 0 - 5
                            print(R_cam2gripper, t_cam2gripper)

    finally:

        # Stop streaming
        pipeline.stop()

if __name__=="__main__":
    main()

