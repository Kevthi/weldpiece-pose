import cv2
from webcam_handler import WebcamHandler
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt

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

def create_board(squares_x, squares_y, cb_sq_width, aruco_sq_width, aruco_dict_str, start_id):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    aruco_dict.bytesList=aruco_dict.bytesList[start_id:,:,:]
    board = aruco.CharucoBoard_create(squares_x,squares_y,1,aruco_sq_width,aruco_dict)
    return board, aruco_dict




cam =WebcamHandler(0, cam_conf)

#aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_5X5_250)
#aruco_dict.bytesList=aruco_dict.bytesList[30:,:,:]
#board = aruco.CharucoBoard_create(6, 8, 32.521*1e-3, 32.521*1e-3/2.0, aruco_dict)
board, aruco_dict = create_board(4,3, 37.5*1e-3, 24.375*1e-3, "DICT_APRILTAG_16H5", 6)
#plt.imshow(board.draw((500,500)))
#plt.show()

def detect_charuko(cam, board, aruco_dict, win_name):
    i = 0
    ar_params = cv2.aruco.DetectorParameters_create()

    while(True):
        ret, img = cam.get_camera_image()
        show_img = img
        img_copy = img.copy()

        (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img_copy, aruco_dict, parameters=ar_params)
        if marker_ids is not None and len(marker_ids) > 0:
            img_copy = aruco.drawDetectedMarkers(img_copy, marker_corners, marker_ids)
            num, char_corners, char_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board, cameraMatrix=K, distCoeffs=dist)
            if(char_ids is not None and len(char_ids)>0):
                aruco.drawDetectedCornersCharuco(img_copy, char_corners, char_ids, (0,0,255))
                rvec = np.array([0,0,0])
                tvec = np.array([0,0,0])
                valid, rvec,tvec = aruco.estimatePoseCharucoBoard(char_corners, char_ids, board, K, dist, np.empty(1), np.empty(1))
                if(valid):
                    print("valid")
                    cv2.drawFrameAxes(img_copy, K, dist, rvec,tvec, 1.0)
                    show_img = img_copy
        print(i)
        cv2.imshow("show_img", show_img)
        i+=1
        if cv2.waitKey(1)& 0xFF ==ord("q"):
            break


board_id0, aruco_dict_id0 = create_board(4,3, 37.5*1e-3, 24.375*1e-3, "DICT_APRILTAG_16H5", 0)

detect_charuko(cam, board, aruco_dict, "board idx 6")
detect_charuko(cam, board_id0, aruco_dict_id0, "board idx 0")


        




