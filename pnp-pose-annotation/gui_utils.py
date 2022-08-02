import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
import cv2
import matplotlib.pyplot as plt
import colorsys


def read_rgb(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)



def ask_directory():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory()
    return file_path

def ask_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def get_image_paths_from_dir(img_dir, allowed_formats=("png", "jpg")):
    return [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if filename.endswith((allowed_formats))]

def get_files_from_dir(filedir, allowed_formats):
    return [os.path.join(filedir, filename) for filename in os.listdir(filedir) if filename.endswith((allowed_formats))]


def read_json_to_dict(json_path):
    with open(json_path) as json_file:
        py_dict = json.load(json_file)
    return py_dict

def get_camera_matrix_json(json_path, image_path):
    image_name = os.path.basename(image_path)
    json_dict = read_json_to_dict(json_path)
    if image_name in json_dict:
        return convert_cam_mat_format(json_dict[image_name]["K"])
    elif "default" in json_dict:
        return convert_cam_mat_format(json_dict["default"])
    else:
        return None


def is_valid_cam_info_dict(cam_info_dict):
    for key in cam_info_dict:
        if (key == "default") and (cam_info_dict["default"] is None):
            continue
        potential_K = cam_info_dict[key]
        if type(potential_K) == dict:
            if ("K" in potential_K):
                K = potential_K["K"]
        if is_valid_camera_matrix(K):
            continue
        else:
            return False
    return True

def is_valid_camera_matrix(K):
    print("type K", type(K))
    if type(K) is list:
        K = np.array(K)
    if type(K) is np.ndarray:
        K_flat = K.flatten() 
        if len(K_flat) == 9:
            print(K_flat)
            first_correct =  K_flat[0]>0.0 and K_flat[1] == 0.0 and K_flat[2] > 0.0
            mid_correct = K_flat[3] == 0 and K_flat[4] > 0 and K_flat[5] > 0.0
            last_correct = K_flat[6] == 0 and K_flat[7] == 0 and np.isclose(K_flat[8], 1)
            print(first_correct, mid_correct, last_correct)
            if first_correct and mid_correct and last_correct:
                return True
        else:
            return False
    return False

def create_camera_matrix(fx,fy,cx,cy):
    return np.array([fx,.0,cx,.0,fy,cy,.0,.0,1.0]).reshape((3,3))


def search_dir_camera_info(search_dir):
    valid_camera_info_paths = []
    valid_camera_info_paths += get_valid_camera_info_json(search_dir)
    valid_camera_info_paths += get_valid_camera_info_npy(search_dir)
    return valid_camera_info_paths

def get_valid_camera_info_json(search_dir):
    valid_cam_info_json_paths = []
    json_file_paths = [os.path.join(search_dir, filename) for filename in os.listdir(search_dir) if filename.endswith(".json")]
    if len(json_file_paths) == 0:
        return None

    for json_file_path in json_file_paths:
        cam_info_dict = read_json_to_dict(json_file_path)
        if is_valid_cam_info_dict(cam_info_dict):
            valid_cam_info_json_paths.append(json_file_path)
    return valid_cam_info_json_paths

def get_valid_camera_info_npy(search_dir):
    valid_npys = []
    npy_file_paths = [os.path.join(search_dir, filename) for filename in os.listdir(search_dir) if filename.endswith(".npy")]
    for npy_file_path in npy_file_paths:
        np_array = np.load(npy_file_path)
        if is_valid_camera_matrix(np_array):
            valid_npys.append(npy_file_path)
    return valid_npys


def get_camera_matrix_from_dict(camera_info_dict, image_basename):
    if image_basename in camera_info_dict:
        return camera_info_dict[image_basename]
    elif camera_info_dict["default"] is not None:
        return camera_info_dict["default"]
    return None

    

def init_camera_info_dict(default=None):
    camera_info = {
        "default": default,
    }
    return camera_info

def convert_cam_mat_format(K_multiformat):
    print("K multi", K_multiformat)
    K_format = type(K_multiformat)
    K_np = np.array(K_multiformat)
    K_np = np.reshape(K_np, (3,3))
    return K_np


def draw_marker(img, coord, color, marker_size):
    img = img.copy()
    x_first = (coord[1], coord[0])
    cv2.drawMarker(img, x_first, color, markerType=cv2.MARKER_CROSS, thickness=marker_size, markerSize=marker_size*8)
    return img

def get_hue_color(hue):
    rgb_col = list(np.array(colorsys.hsv_to_rgb(hue, 1.0,1.0)))
    rgb_col = tuple([int(val*255) for val in rgb_col])
    return rgb_col

def get_color_list(length):
    hue = 0
    hue_incr = 1.0/7.0
    color_list = []
    for i in range(length):
        rgb = get_hue_color(hue)
        color_list.append(rgb)
        hue+=hue_incr
    return color_list


def draw_corresps(img, corr_list, additional=None, marker_size=5):
    img = img.copy()
    col_list = get_color_list(len(corr_list)+1)
    idx = 0
    for corr in corr_list:
        rgb_col = col_list[idx]
        img = draw_marker(img, corr, rgb_col, marker_size)
        idx +=1
    if additional is not None:
        rgb_col = col_list[idx]
        img = draw_marker(img, additional, rgb_col, marker_size)
    return img

def draw_corresps_both(cam_img, rend_img, both_corresps, img_select, rend_select, marker_size=5):
    img_corrs, rend_corrs = split_corresps(both_corresps)
    c_width = cam_img.shape[0]
    r_width = rend_img.shape[0]
    if c_width>r_width:
        cam_marker_size = int(c_width*1.0/r_width*marker_size)
        rend_marker_size = int(marker_size)
    else:
        rend_marker_size = int(r_width*1.0/c_width*marker_size)
        cam_marker_size = int(marker_size)

    cam_img = draw_corresps(cam_img, img_corrs, img_select, cam_marker_size)
    rend_img = draw_corresps(rend_img, rend_corrs, rend_select, rend_marker_size)
    return cam_img, rend_img

def split_corresps(both_corresps):
    img_corrs = [img_corr for img_corr,_ in both_corresps]
    rend_corrs = [rend_corr for _, rend_corr in both_corresps]
    return img_corrs, rend_corrs









if __name__ == '__main__':
    img1 = read_rgb("corner1-undist.png")
    rgb_col = get_hue_color(0.33)
    print(rgb_col)
    img = draw_marker(img1, (500,500), rgb_col, 3)
    plt.imshow(img)
    plt.show()
    plt.imshow(img1)
    plt.show()

