import os
#os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'glfw'
from render_utils import crop_square_bounding_box, center_mesh_to_centroid, img_float2uint8, get_optimal_camera_pose, convert_cam_mat
from render_utils import center_mesh_bb
import numpy as np
import spatialmath as sm
import trimesh as tm
from se3_helpers import *
import matplotlib.pyplot as plt
import pyrender
from PIL import Image
from se3_helpers import get_T_CO_init_and_gt, look_at_SE3
import cv2
import rhovee

class PersistentRenderer():
    def __init__(self, model_path, K, T_WC, img_size):
        print("Model path", model_path)
        print("K", K)
        print("T_WC", T_WC)
        print("img size", img_size)
        #K = get_camera_matrix(36,18,img_size[0], img_size[1])

        self.scene = pyrender.Scene(ambient_light=[0.3,0.3,0.3,1.0], bg_color=[1.0,1.0,1.0])
        trimesh_mesh = tm.load(model_path)
        trimesh_mesh = center_mesh_bb(trimesh_mesh)
        self.mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
        self.scene.add(self.mesh)
        fx,fy, ux,uy = K[0,0], K[1,1], K[0,2], K[1,2]
        self.camera = pyrender.IntrinsicsCamera(fx, fy, ux,uy)
        self.renderer = pyrender.OffscreenRenderer(img_size[0], img_size[1])
        self.cam_node = pyrender.Node(camera=self.camera, matrix=np.identity(4))
        self.scene.add_node(self.cam_node)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
        self.light_node = pyrender.Node(light=light, matrix=np.identity(4))
        self.scene.add_node(self.light_node)
        #self.scene.add(self.camera, pose=T_WC)
        #self.scene.add(light, pose=T_WC)
        print("scene meshes", self.scene.meshes)
        #print("camera nodes", self.scene.camera_nodes[0])
        
        #pyrender.Viewer(self.scene, use_raymond_lighting=True)

    def render(self, T_wc):
        T_wc = self.convert_camera_frame(T_wc)

        self.scene.set_pose(self.cam_node, T_wc)
        self.scene.set_pose(self.light_node, T_wc)
        rgb,depth = self.renderer.render(self.scene)
        return rgb,depth

    def convert_camera_frame(self, T_WC):
        T_CO = invert_T(T_WC)
        T_CO = sm.SE3.Rx(180, unit='deg').data[0]@T_CO # convert from OpenCV camera frame to OpenGL camera frame
        T_WC = invert_T(T_CO) 
        return T_WC










def get_camera_matrix(focal_len, sensor_width, width, height):
    pix_per_mm = sensor_width/(width*1.0)
    fx = fy = focal_len/pix_per_mm
    vx = width/2
    vy = height/2
    K = np.array([[fx, 0, vx],[0, fy, vy],[0,0,1]])
    return K


def add_object(scene, path):
    trimesh_mesh = tm.load(path)
    trimesh_mesh = center_mesh_bb(trimesh_mesh)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
    scene.add(mesh)

def add_light(scene, T_CO):
    assert T_CO.shape == (4,4)
    T_OC = np.linalg.inv(T_CO)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
    scene.add(light, pose=T_OC)

def add_camera(scene, T_CO, K):
    assert T_CO.shape == (4,4)
    T_OC = np.linalg.inv(T_CO)
    fx,fy, ux,uy = K[0,0], K[1,1], K[0,2], K[1,2]
    camera = pyrender.IntrinsicsCamera(fx, fy, ux,uy)
    scene.add(camera, pose=T_OC)

def render(scene, img_size):
    r = pyrender.OffscreenRenderer(img_size[0], img_size[1])
    color, depth = r.render(scene)
    r.delete()
    return color, depth


def render_scene(object_path, T_CO, K, img_size, bg_color=(1.0,1.0,1.0)):
    assert T_CO.shape == (4,4)
    print(T_CO)
    T_CO = sm.SE3.Rx(180, unit='deg').data[0]@T_CO # convert from OpenCV camera frame to OpenGL camera frame
    T_CO = rhovee.SE3.SVDO(T_CO)
    scene = pyrender.Scene()
    scene.bg_color = bg_color
    add_object(scene, object_path)
    add_light(scene, T_CO)
    add_camera(scene, T_CO, K)
    img, depth = render(scene, img_size)
    return img, depth

def normalize_depth(depth_img):
    mean_val = np.mean(depth_img[depth_img>0.01])
    std = np.std(depth_img[depth_img>0.01])
    normalized = np.where(depth_img>0.01, (depth_img-mean_val)/std, 0.0)
    return normalized.astype(np.float32)

def render_thumbnail(object_path, img_size=(840,840)):
    K = get_camera_matrix(36,18,img_size[0], img_size[1])
    T = get_optimal_camera_pose(object_path).inv().data[0]
    img, depth = render_scene(object_path, T, K, img_size)
    mask = np.where(depth>0, 255, 0).astype(np.uint8)
    square_crop = crop_square_bounding_box(img, mask, 50, (612,612))
    return img_float2uint8(square_crop)



     
       


if __name__ == '__main__':
    model_path = "corner.ply"
    thumbnail = render_thumbnail(model_path)
    plt.imshow(thumbnail)
    plt.show()










