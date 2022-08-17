import cv2
from webcam_handler import WebcamHandler
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt

def get_aruko_poses(img, K, aruco_dict, aruko_sq_size):
    ar_params = aruco.DetectorParameters_create()
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    dist = np.zeros(5)
    aruko_poses = []
    if marker_ids is not None and len(marker_ids) > 0:
        img = aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        for single_marker_corner, marker_id in zip(marker_corners, marker_ids):
            rvec,tvec,obj_p_corners = aruco.estimatePoseSingleMarkers(single_marker_corner, aruko_sq_size, K, dist)
            aruko_poses.append((marker_id, rvec,tvec))
    return aruko_poses



def draw_markers(img, K, aruco_dict):
    ar_params = aruco.DetectorParameters_create()
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    dist = np.zeros(5)
    if marker_ids is not None and len(marker_ids) > 0:
        img = aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        for single_marker_corner, marker_id in zip(marker_corners, marker_ids):
            rvec,tvec,obj_p_corners = aruco.estimatePoseSingleMarkers(single_marker_corner, 66.0*1e-3, K, dist)
            cv2.drawFrameAxes(img, K, dist, rvec,tvec,0.05)
    return img



def run_webcam(cam, aruko_dict, K, dist):
    i = 0
    while(True):
        ret, img = cam.get_camera_image()
        undist_img = cv2.undistort(img, K, dist)
        show_img = draw_markers(undist_img, K, aruko_dict)



        cv2.imshow("show_img", show_img)
        i+=1
        if cv2.waitKey(1)& 0xFF ==ord("q"):
            break


if __name__ == '__main__':
    cam_conf = {
        "valid_frame_sizes": {
            "3840x2160": (8000,8000),
            "1920x1080": (1920,1080),
            "1280x720": (1280,720),
            "1500x1500":(1500,1500),
            "640x480": (640,480),
            "320x240": (320,240),
            "160x120": (160,120)
        },
        "capture_resolution": "1920x1080",
        "focus_distance": 0,
        "use_autofocus": False,
        "crop_size": (1080,1080),
        "resize_to": None,
    }
    K = np.load("K.npy")
    dist = np.load("dist_coeffs.npy")


    cam =WebcamHandler(0, cam_conf)

    aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_APRILTAG_16h5)
    #aruco_dict.bytesList=aruco_dict.bytesList[30:,:,:]
    #draw_marker(aruco_dict, 0)
    #board = aruco.CharucoBoard_create(6, 8, 32.521*1e-3, 32.521*1e-3/2.0, aruco_dict)
    run_webcam(cam, aruco_dict, K, dist)
