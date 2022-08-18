import cv2
from webcam_handler import WebcamHandler
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt


def create_board(squares_x, squares_y, cb_sq_width, aruco_sq_width, aruco_dict_str, start_id):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    aruco_dict.bytesList=aruco_dict.bytesList[start_id:,:,:]
    board = aruco.CharucoBoard_create(squares_x,squares_y,cb_sq_width,aruco_sq_width,aruco_dict)
    return board, aruco_dict





#aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_5X5_250)
#aruco_dict.bytesList=aruco_dict.bytesList[30:,:,:]
#board = aruco.CharucoBoard_create(6, 8, 32.521*1e-3, 32.521*1e-3/2.0, aruco_dict)
#plt.imshow(board.draw((500,500)))
#plt.show()

def draw_charuco_corners(img, board, aruco_dict):
    dist = np.zeros(5)
    img = img.copy()
    ar_params = cv2.aruco.DetectorParameters_create()
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    if marker_ids is not None and len(marker_ids) > 0:
        img = aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        num, char_corners, char_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board, cameraMatrix=K, distCoeffs=dist)
        if(char_ids is not None and len(char_ids)>0):
            aruco.drawDetectedCornersCharuco(img, char_corners, char_ids, (0,0,255))
    return img


def get_charuco_pose(img, board, K, aruco_dict):
    dist = np.zeros(5)
    ar_params = cv2.aruco.DetectorParameters_create()
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    rvec = None
    tvec = None
    valid = False
    if marker_ids is not None and len(marker_ids) > 0:
        num, char_corners, char_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board, cameraMatrix=K, distCoeffs=dist)
        if(char_ids is not None and len(char_ids)>0):
            rvec = np.array([0,0,0])
            tvec = np.array([0,0,0])
            valid, rvec,tvec = aruco.estimatePoseCharucoBoard(char_corners, char_ids, board, K, dist, np.empty(1), np.empty(1))

    return valid, rvec,tvec

def cv2_pose_to_T(rvec,tvec):
    T_CA = np.identity(4)
    R, _ = cv2.Rodrigues(rvec)
    T_CA[:3,:3] = R
    T_CA[:3,3] = tvec.flatten()
    return T_CA


def detect_charuko(cam, K, dist_coeffs, win_name):
    i = 0
    board_id0, aruco_dict_id0 = create_board(4,3, 37.5*1e-3, 24.375*1e-3, "DICT_APRILTAG_16H5", 0)
    board_id6, aruco_dict_id6 = create_board(4,3, 37.5*1e-3, 24.375*1e-3, "DICT_APRILTAG_16H5", 6)



    while(True):
        ret, img = cam.get_camera_image()

        img = cv2.undistort(img, K, dist_coeffs)

        valid_0,rvec,tvec = get_charuco_pose(img, board_id0, K, aruco_dict_id0)
        if(valid_0):
            T_C0 = cv2_pose_to_T(rvec,tvec)
            cv2.drawFrameAxes(img, K, dist, rvec,tvec, 0.15)
            draw_charuco_corners(img, board_id0, aruco_dict_id0)


        valid_1,rvec,tvec = get_charuco_pose(img, board_id6, K, aruco_dict_id6)
        if(valid_1):
            T_C1 = cv2_pose_to_T(rvec,tvec)
            cv2.drawFrameAxes(img, K, dist, rvec,tvec, 0.15)
            draw_charuco_corners(img, board_id0, aruco_dict_id6)

        if valid_0 and valid_1:
            T_01 = np.linalg.inv(T_C0)@T_C1
            print(T_01[:3,3])




        img = cv2.resize(img, (1080,1080))
        cv2.imshow("show_img", img)
        i+=1
        if cv2.waitKey(1)& 0xFF ==ord("q"):
            break


if __name__ == '__main__':
    import json
    with open('brio-2160.json') as json_file:
        cam_conf  = json.load(json_file)
    print(cam_conf)
    cam =WebcamHandler(0, cam_conf)
    K = np.load("K.npy")
    print("K")
    print(K)
    dist = np.load("dist_coeffs.npy")
    print("dist coeffs")
    print(dist)

    #detect_charuko(cam, board, aruco_dict, "board idx 6")
    detect_charuko(cam, K, dist, "board idx 0")


        




