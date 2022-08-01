import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
import cv2


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




if __name__ == '__main__':
    K = np.identity(3)
    search_dir = "/home/ola/projects/weldpiece-pose-datasets/ds-projects/office-corner/captures"

    print("is valid", search_dir_camera_info(search_dir))

