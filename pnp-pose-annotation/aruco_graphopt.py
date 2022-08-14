import cv2
from cv2 import aruco as aruco
import numpy as np


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



def draw_markers(img, K, aruco_dict_str):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    ar_params = aruco.DetectorParameters_create()
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    dist = np.zeros(5)
    if marker_ids is not None and len(marker_ids) > 0:
        img = aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        for single_marker_corner, marker_id in zip(marker_corners, marker_ids):
            rvec,tvec,obj_p_corners = aruco.estimatePoseSingleMarkers(single_marker_corner, 66.0*1e-3, K, dist)
            cv2.drawFrameAxes(img, K, dist, rvec,tvec,0.05)
    return img





if __name__ == '__main__':
    pass
