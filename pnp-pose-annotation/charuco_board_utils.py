import os
import cv2
from cv2 import aruco
import numpy as np
from gui_utils import read_rgb


def create_board(squares_x, squares_y, cb_sq_width, aruco_sq_width, aruco_dict_str, start_id):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    aruco_dict.bytesList=aruco_dict.bytesList[start_id:,:,:]
    board = aruco.CharucoBoard_create(squares_x,squares_y,cb_sq_width,aruco_sq_width,aruco_dict)
    return board, aruco_dict

def get_aruco_board_pose(img, K, board,aruco_dict, use_cr_at=False):
    ar_params = aruco.DetectorParameters_create()
    if use_cr_at:
        ar_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    dist = np.zeros(5)
    T_CB = None
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    if marker_ids is not None and len(marker_ids) > 0:
        num, char_corners, char_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board, cameraMatrix=K, distCoeffs=np.zeros(5))
        if(char_ids is not None and len(char_ids)>0):
            valid, rvec,tvec = aruco.estimatePoseCharucoBoard(char_corners, char_ids, board, K, dist, np.empty(1), np.empty(1))
            if valid:
                T_CB = np.identity(4)
                R, _ = cv2.Rodrigues(rvec)
                T_CB[:3,:3] = R
                T_CB[:3,3] = tvec.flatten()
    return T_CB

def T_to_opencv_rvec_tvec(T):
    R = T[:3,:3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = T[:3,3]
    return rvec,tvec


def get_all_image_board_pose_dict(img_paths, K, aruco_dict_str, aruco_sq_size, arucos_per_board=4):
    aruko_poses_all_imgs = {}
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        img = read_rgb(img_path)
        aruco_board_poses = get_aruco_board_pose_dict(img, K, aruco_dict_str, aruco_sq_size)
        if len(aruco_board_poses) > 0:
            aruko_poses_all_imgs[basename] = aruco_board_poses
    return aruko_poses_all_imgs

def get_aruco_board_pose_dict(img, K, aruco_dict_str, aruko_sq_size, arucos_per_board=4):
    ar_params = aruco.DetectorParameters_create()
    ar_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    aruko_poses = {}
    num_search = 30//arucos_per_board
    print("num search", num_search)
    for i in range(num_search):
        board, aruco_dict = create_board(3,3,55.8*1e-3,36.27*1e-3, aruco_dict_str, int(i*arucos_per_board))
        T = get_aruco_board_pose(img, K, board, aruco_dict)
        if T is not None:
            dict_key = f'{[i]}'
            print(dict_key)
            aruko_poses[dict_key] = T
    return aruko_poses


def draw_markers_board(img, K, aruco_dict_str, arucos_per_board=4):
    dist = np.zeros(5)
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    num_search = 7
    for i in range(num_search):
        board, aruco_dict = create_board(3,3,55.8*1e-3,36.27*1e-3, aruco_dict_str, int(i*arucos_per_board))
        T = get_aruco_board_pose(img, K, board, aruco_dict, use_cr_at=False)
        if T is not None:
            rvec,tvec = T_to_opencv_rvec_tvec(T)
            cv2.drawFrameAxes(img, K, dist, rvec,tvec, 55.8*3*1e-3)
    return img

    
