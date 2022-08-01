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

def get_optimal_camera_pose(mesh_path, scale=3.5):
    mesh = tm.load(mesh_path)
    centroid = mesh.centroid
    #mesh = center_mesh_to_centroid(mesh)
    mesh = center_mesh_bb(mesh)
    bounds = mesh.bounds
    rx = sm.SE3.Rz(20, unit='deg')
    max_bound = np.max(np.abs(bounds))
    origin =  np.ones(3)*max_bound*scale
    T = look_at_SE3(origin, [0,0,0], [0,0,1])
    #T = T@rx
    #T = T.inv().data[0]
    return T




def center_mesh_to_centroid(mesh):
    c = mesh.centroid
    transf = np.eye(4)
    transf[:3, 3] = -c
    mesh.apply_transform(transf)
    return mesh

def center_mesh_bb(mesh):
    bb_bounds = mesh.bounds
    c = np.mean(bb_bounds, axis=0)
    transf = np.eye(4)
    transf[:3, 3] = -c
    mesh.apply_transform(transf)
    return mesh


def get_centroid(mesh_path):
    mesh = tm.load(mesh_path)
    return mesh.centroid


def convert_cam_mat(cam_mat, current_size, new_width):
    c_width = current_size[0]
    c_height = current_size[1]

    #new_height = int((cam_mat[1,2]/cam_mat[0,2])*new_width)
    new_height = int((c_height*1.0/c_width)*new_width)
    new_size = (new_width, new_height)
    old_x = cam_mat[0,2]*2
    scale_x = (new_width*1.0)/old_x
    new_K = cam_mat*scale_x
    new_K[2,2] = 1.0
    return new_K, new_size
    
    
if __name__ == '__main__':
    mesh_path = "corner.ply"
    opt = get_optimal_camera_pose(mesh_path)
    print(opt)
    


