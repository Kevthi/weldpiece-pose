import os
import cv2
from cv2 import aruco
import numpy as np
from gui_utils import read_rgb


"""
These codes is used for camera calibration and pose estimation using ChArUco boards.
"""


def create_board(squares_x: int, squares_y: int, cb_sq_width: float, 
                 aruco_sq_width: float, aruco_dict_str: str, start_id: int) -> tuple:
    """Creates a ChArUco board with defined squares and ArUco markers.

    Args:
        squares_x (int): Number of squares in X direction
        squares_y (int): Number of squares in Y direction
        cb_sq_width (float): Size of each chessboard square (in desired units)
        aruco_sq_width (float): Size of each ArUco marker (in desired units)
        aruco_dict_str (str): ArUco dictionary name (e.g. 'DICT_6X6_250')
        start_id (int): Starting ID for ArUco markers subset

    Returns:
        tuple: A tuple containing:
            - board: The created ChArUco board object
            - aruco_dict: The modified ArUco dictionary

    Example:
        >>>> board, aruco_dict = create_board(5, 7, 0.04, 0.02, 'DICT_6X6_250', 0)
    """
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    aruco_dict.bytesList = aruco_dict.bytesList[start_id:,:,:]
    board = aruco.CharucoBoard_create(squares_x, squares_y, cb_sq_width, 
                                     aruco_sq_width, aruco_dict)
    return board, aruco_dict

def get_aruco_board_pose(img: np.ndarray, K: np.ndarray, board, aruco_dict: aruco.Dictionary, use_cr_at=False):
    """
    Estimate the pose of a ChArUco board in an image.

    Args:
        img (np.ndarray): The input image containing the ChArUco board.
        camera_int (np.ndarray): The camera intrinsic parameters.
        board: The ChArUco board object.
        aruco_dict (aruco.Dictionary): The ArUco dictionary used for marker detection.
        use_cr_at (bool, optional): Whether to use corner refinement. Defaults to False.

    Returns:
        np.ndarray or None: The 4x4 transformation matrix representing the pose of the ChArUco board 
                            in the camera frame if detection is successful, otherwise None.
    """
    ar_params = aruco.DetectorParameters_create()
    if use_cr_at:
        ar_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    dist = np.zeros(5)
    T_CB = None #Placeholder for the transformation matrix and by default it is None
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
    return T_CB #The transformation matrix of the board in the camera frame

def T_to_opencv_rvec_tvec(T):
    #Seperate rotation and translation from the 4x4 transformation matrix
    R = T[:3,:3]
    rvec, _ = cv2.Rodrigues(R)
    tvec = T[:3,3]
    return rvec,tvec


def get_all_image_board_pose_dict(img_paths : str, K : np.array, aruco_dict_str : str, aruco_sq_size : int, arucos_per_board=4) -> dict:
    """
    Stores the pose of the ChArUco board in all images in a dictionary.
    This is done by iterating over all images and calling get_aruco_board_pose_dict() for each image.
    """
    aruko_poses_all_imgs = {}
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        img = read_rgb(img_path)
        aruco_board_poses = get_aruco_board_pose_dict(img, K, aruco_dict_str, aruco_sq_size)
        if len(aruco_board_poses) > 0:
            aruko_poses_all_imgs[basename] = aruco_board_poses
    return aruko_poses_all_imgs

def get_aruco_board_pose_dict(img, K, aruco_dict_str, aruko_sq_size, arucos_per_board=4):
    """
    Detects the pose of a ChArUco board in a single image and returns a dictionary of poses.
    """
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
    return aruko_poses #Return the 


def draw_markers_board(img, K, aruco_dict_str, arucos_per_board=4):
    """
    Detects ChArUco boards in an image, estimates their poses, and draws the coordinate axes on the image.
    """
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


