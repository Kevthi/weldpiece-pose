import cv2
from cv2 import aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from .AOTGAN import get_model, run_inpaint

def create_board(squares_x, squares_y, cb_sq_width, aruco_sq_width, aruco_dict_str, start_id):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    aruco_dict.bytesList=aruco_dict.bytesList[start_id:,:,:]
    board = aruco.CharucoBoard_create(squares_x,squares_y,cb_sq_width,aruco_sq_width,aruco_dict)
    return board, aruco_dict


def corners_as_2d_arr(char_corners, cb_size):
    corners_2d = np.zeros((cb_size[0]-1, cb_size[1]-1, 2))
    i = 0
    for y in range(cb_size[1]-1):
        for x in range(cb_size[0]-1):
            corners_2d[x,y,:] = np.array(char_corners[i])
            i+=1
    return corners_2d

def pix_within_img(pix, img):
    pix_x = pix[0]
    pix_y = pix[1]
    pix_x = max(pix_x, 0)
    pix_x = min(pix_x, img.shape[1]-1)
    pix_y = max(pix_y, 0)
    pix_y = min(pix_y, img.shape[0]-1)
    return [int(pix_x), int(pix_y)]

def draw_mask_charuco(img, char_corners, board, pad_sq_ratio=0.0):
    print("draw mask")
    print("char corners", char_corners)
    print("")
    cb_size = board.getChessboardSize()
    if char_corners is None:
        return

    corners_2d = corners_as_2d_arr(char_corners, cb_size)
    dx = corners_2d[1,0] - corners_2d[0,0]
    dy = corners_2d[0,1] - corners_2d[0,0]



    print("dx", dx, "dy", dy)
    print("corners 2d 0 0", corners_2d[0,0])


    p1 = corners_2d[0,0] -dx -dy + pad_sq_ratio*(-dx-dy)
    p1 = pix_within_img(p1, img)
    #cv2.putText(img, f'p1', p1, cv2.FONT_HERSHEY_PLAIN, 1, color=(255,0,0), thickness=2)
    p2 = corners_2d[-1,0] +dx -dy + pad_sq_ratio*(+dx-dy)
    p2 = pix_within_img(p2, img)
    #cv2.putText(img, f'p2', p2, cv2.FONT_HERSHEY_PLAIN, 1, color=(255,0,0), thickness=2)
    p3 = corners_2d[-1,-1] +dx +dy + pad_sq_ratio*(+dx+dy)
    p3 = pix_within_img(p3, img)
    #cv2.putText(img, f'p3', p3, cv2.FONT_HERSHEY_PLAIN, 1, color=(255,0,0), thickness=2)

    p4 = corners_2d[0,-1] -dx +dy + pad_sq_ratio*(-dx+dy)
    p4 = pix_within_img(p4, img)
    #cv2.putText(img, f'p4', p4, cv2.FONT_HERSHEY_PLAIN, 1, color=(255,0,0), thickness=2)
    print(p1)
    pts = [np.array([p1,p2,p3,p4]).astype(np.int32)]
    print(type(pts[0][0]))
    #pts = np.array([[0,0], [100,100], [0,100]])
    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillPoly(mask, pts=pts, color=(255,255,255)) 
    return mask



    """
    for x_idx, x in enumerate(corners_2d):
        for y_idx,y in enumerate(x):
            cv2.putText(img, f'{x_idx},{y_idx}', y.astype(np.uint32), cv2.FONT_HERSHEY_PLAIN, 1, color=(255,0,0), thickness=2)
    """

    
def get_cb_corners(img, board, aruco_dict):
    dist = np.zeros(5)
    cb_size = board.getChessboardSize()
    num_corners = (cb_size[0]-1)*(cb_size[1]-1)
    ar_params = aruco.DetectorParameters_create()
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    if marker_ids is not None and len(marker_ids)>0:
        num, char_corners, char_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)
        if char_ids is not None and num==num_corners:
            return char_corners.squeeze()
    return None



def remove_aruco_boards_handler(image: np.ndarray, aruco_info: dict) -> np.ndarray:
    """
    Removes Aruco boards from the image using inpainting.

    Args:
        image (np.ndarray): The input image.
        aruco_info (dict): Information about detected Aruco markers.

    Returns:
        np.ndarray: The inpainted image.
    """
    if not aruco_info:
        return image

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for marker in aruco_info.get('markers', []):
        corners = marker.get('corners', [])
        if len(corners) == 4:
            pts = np.array(corners, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    inpaint_img = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return inpaint_img


if __name__ == '__main__':
    img_path = "img_0-undist.png"
    aruco_dict_str = "DICT_APRILTAG_16H5" 
    #board, aruco_dict = create_board(3,3,55.8*1e-3,36.27*1e-3, aruco_dict_str, 0)
    board, aruco_dict = create_board(3,3,55.8*1e-3,36.27*1e-3, aruco_dict_str, 0)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = np.zeros((img.shape[0], img.shape[1]))
    char_corners = get_cb_corners(img, board, aruco_dict)
    mask = draw_mask_charuco(img, char_corners, board, 0.8)
    mask = np.where(mask>0, 1, 0).astype(np.uint8)
    model = get_model()
    print("img shape", img.shape)
    inpaint_img = run_inpaint(img, mask, model)
    #inpaint_img = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)


    fig,ax = plt.subplots()
    fig.set_size_inches(18.5, 18.5)

    plt.imshow(img)
    plt.show()
    #plt.imshow(mask)
    #plt.show()
    fig,ax = plt.subplots()
    fig.set_size_inches(18.5, 18.5)
    plt.imshow(inpaint_img)
    plt.show()







