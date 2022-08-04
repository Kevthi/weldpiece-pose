import cv2
from webcam_handler import WebcamHandler
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt


def run_webcam(cam, aruko_dict):
    ar_params = aruco.DetectorParameters_create()
    i = 0
    while(True):
        ret, img = cam.get_camera_image()
        show_img = img
        img_copy = img.copy()

        (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img_copy, aruco_dict, parameters=ar_params)
        if marker_ids is not None and len(marker_ids) > 0:
            img_copy = aruco.drawDetectedMarkers(img_copy, marker_corners, marker_ids)
            show_img = img_copy

        cv2.imshow("show_img", show_img)
        i+=1
        if cv2.waitKey(1)& 0xFF ==ord("q"):
            break

def viz_board(aruco_dict):
    board = aruco.GridBoard_create(2,3,1,1,aruco_dict)
    img = board.draw((500,500))
    plt.imshow(img)
    plt.show()

def draw_marker(aruco_dict, m_id):
    img = aruco.drawMarker(aruco_dict, m_id, 200)
    plt.imshow(img)
    plt.show()


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

    aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_5X5_250)
    #aruco_dict.bytesList=aruco_dict.bytesList[30:,:,:]
    viz_board(aruco_dict)
    #draw_marker(aruco_dict, 0)
    #board = aruco.CharucoBoard_create(6, 8, 32.521*1e-3, 32.521*1e-3/2.0, aruco_dict)
    #run_webcam(cam, aruco_dict)
    print("main")
