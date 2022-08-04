import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import numpy as np

A4_Y = 210
A4_X = 297
A4_XY_RATIO = A4_X/A4_Y


def create_printable_aruco_grid(aruco_dict, px_width, squares_x,squares_y, spacing_ratio):
    squares_xy_ratio = squares_x/squares_y
    px_per_mm = px_width/A4_X
    print("px per mm", px_per_mm)
    board = aruco.GridBoard_create(squares_x,squares_y,1,spacing_ratio,aruco_dict)
    px_height = np.round(px_width/A4_XY_RATIO, 0)



    if squares_xy_ratio > A4_XY_RATIO:
        norm_width = (squares_x*spacing_ratio+squares_x)
        padding = px_width*1.0/norm_width*spacing_ratio/2
        img = board.draw((px_width,int(px_height)), marginSize=int(padding))
        aruco_size = np.round((px_width/norm_width)/px_per_mm, 2)
    else:
        norm_height = ((squares_y+1)*spacing_ratio+squares_y)
        padding = px_height*1.0/norm_height*spacing_ratio
        img = board.draw((px_width,int(px_height)), marginSize=int(padding))
        aruco_size = np.round((px_height/norm_height)/px_per_mm, 2)


    print("aruco dict", aruco_dict)
    label = "APRILTAG_16H5" + f' SZ:{aruco_size}mm' + f' pad:{spacing_ratio*aruco_size}mm'
    imboard = cv2.putText(img, label, (100,100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(0,0,0), thickness=5)
    print("aruco_size mm", aruco_size)
    img = img.T
    img = cv2.flip(img,1)
    return img












if __name__ == '__main__':
    aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_16h5)
    px_width = 4000
    squares_x = 3
    squares_y = 2
    spacing_ratio = 0.5
    img = create_printable_aruco_grid(aruco_dict, px_width, squares_x, squares_y, spacing_ratio)
    cv2.imwrite(f'/home/ola/Pictures/aruco_{str(squares_x)+"x"+str(squares_y)}_apriltag16h5.png', img)

    pass
