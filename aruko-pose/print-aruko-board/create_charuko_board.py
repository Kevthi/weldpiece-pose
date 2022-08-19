import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import numpy as np

#A4_Y = 210
#A4_X = 297
A4_Y = 195.0 # printable area
A4_X = 276.0 # printable area

#A4_X = 420.0
#A4_Y = 297.0

A4_XY_RATIO = A4_X/A4_Y

def create_board(squares_x, squares_y, cb_sq_width, aruco_sq_width, aruco_dict_str, start_id):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    aruco_dict.bytesList=aruco_dict.bytesList[start_id:,:,:]
    board = aruco.CharucoBoard_create(squares_x,squares_y,cb_sq_width,aruco_sq_width,aruco_dict)
    return board


def create_printable_aruco_grid(aruco_dict_str, px_width, squares_x,squares_y, spacing_ratio, start_id, padding):
    #aruco_dict.bytesList=aruco_dict.bytesList[start_id:,:,:]
    squares_xy_ratio = squares_x/squares_y
    px_per_mm = px_width/A4_X
    print("px per mm", px_per_mm)
    #board = aruco.CharucoBoard_create(squares_x,squares_y,1,spacing_ratio,aruco_dict)
    board = create_board(squares_x, squares_y, 1, spacing_ratio, aruco_dict_str, start_id)
    px_height = np.round(px_width/A4_XY_RATIO, 0)



    if squares_xy_ratio > A4_XY_RATIO:
        norm_width = (squares_x*spacing_ratio+squares_x)
        #padding = px_width*1.0/norm_width*spacing_ratio/2
        img = board.draw((px_width,int(px_height)), marginSize=int(padding))
        ch_board_sq_size = ((px_width-2*padding)/squares_x)/px_per_mm
        aruko_size = spacing_ratio*ch_board_sq_size
        print("ch_board_size", ch_board_sq_size)
        print("aruko_size", aruko_size)

    else:
        norm_height = ((squares_y+1)*spacing_ratio+squares_y)
        #padding = px_height*1.0/norm_height*spacing_ratio
        img = board.draw((px_width,int(px_height)), marginSize=int(padding))
        ch_board_sq_size = ((px_height-2*padding)/squares_y)/px_per_mm
        aruko_size = spacing_ratio*ch_board_sq_size
        print("ch_board_size", ch_board_sq_size)
        print("aruko_size", aruko_size)


    label = "APRILTAG_16H5" + f' SZ_CH_SQ:{np.round(ch_board_sq_size, 3)}mm' 
    label += f' AR_SZ:{np.round(aruko_size, 3)}mm' + f' start id: {str(start_id)}'
    imboard = cv2.putText(img, label, (100,100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(0,0,0), thickness=5)
    img = img.T
    img = cv2.flip(img,1)
    return img












if __name__ == '__main__':
    #aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_16h5)
    aruco_dict_str = "DICT_APRILTAG_16H5"
    px_width = 4000
    squares_x = 3
    squares_y = 3
    spacing_ratio = 0.65
    for i in range(0, 7):
        start_id = i*4
        print(start_id)
        img = create_printable_aruco_grid(aruco_dict_str, px_width, squares_x, squares_y, spacing_ratio, start_id, 200)
        plt.imshow(img)
        plt.show()
        cv2.imwrite(f'/home/ola/Pictures/AT3x3/charuco_stID{start_id}_{str(squares_x)+"x"+str(squares_y)}_apriltag16h5.png', img)

    pass
