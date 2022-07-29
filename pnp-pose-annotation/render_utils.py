import cv2
import numpy as np
import trimesh as tm
import spatialmath as sm
from se3_helpers import look_at_SE3


def crop_square_bounding_box(img, mask, padding, reshape=None):
    if reshape is None:
        reshape = img.shape[:2]
    img_w, img_h = img.shape[:2]
    x,y,w,h = cv2.boundingRect(mask)

    cx = x+w/2
    cy = y+h/2

    square_size = max(w,h) + padding*2

    start_x = int(max(cx - square_size/2, 0))
    start_y = int(max(cy - square_size/2, 0))
    end_x = int(min(cx+square_size/2, img_w-1))
    end_y = int(min(cy+square_size/2, img_h-1))
    
    square_crop = img[start_y:end_y, start_x:end_x]
    square_crop = cv2.resize(square_crop, reshape)
    return square_crop

def img_float2uint8(img):
    img = (img*255.0).astype(np.uint8)
    return img

def get_optimal_camera_pose(mesh_path):
    mesh = tm.load(mesh_path)
    centroid = mesh.centroid
    mesh = center_mesh_to_centroid(mesh)
    bounds = mesh.bounds
    rx = sm.SE3.Rz(20, unit='deg')
    max_bound = np.max(np.abs(bounds))
    origin = centroid + max_bound*3.5
    T = look_at_SE3(origin, centroid, [0,0,1])
    T = rx@T
    T = T.inv().data[0]
    return T





    return 2*max_bound


def center_mesh_to_centroid(mesh):
    c = mesh.centroid
    transf = np.eye(4)
    transf[:3, 3] = -c
    mesh.apply_transform(transf)
    return mesh


def get_centroid(mesh_path):
    mesh = tm.load(mesh_path)
    return mesh.centroid

    
    
if __name__ == '__main__':
    mesh_path = "corner.ply"
    opt = get_optimal_camera_pose(mesh_path)
    print(opt)
    


