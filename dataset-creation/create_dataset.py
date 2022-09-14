import numpy as np
import math
import os
import json
import cv2
import matplotlib.pyplot as plt
import yaml
from pyrender_render import render_scene
import trimesh as tm
import random
from se3_helpers import apply_small_random_rotation_translation
from parser_config import add_config_cli_input, get_config_from_args, get_dict_from_cli

def get_vertices(mesh_path):
    mesh = tm.load(mesh_path, force='mesh')
    vertices = mesh.vertices
    return vertices

def sample_vertices(mesh_path, num_verts=1000):
    verts = get_vertices(mesh_path)
    sampled_verts = []
    for i in range(num_verts):
        vert = np.array(random.choice(verts))
        sampled_verts.append(vert)
    sampled_verts = np.array(sampled_verts, dtype=np.float32)
    return sampled_verts


def write_yaml(write_path, yaml_dict):
    with open(write_path, 'w') as outfile:
        yaml.dump(yaml_dict, outfile, default_flow_style=False)

def read_rgb(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def write_rgb(save_path, img):
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def resize_camera_matrix(current_img_size, new_img_size, current_K):
    ratio = new_img_size*1.0/current_img_size
    new_K = current_K*ratio
    new_K[2,2] = 1.0
    return new_K

def read_json_to_dict(json_path):
    with open(json_path) as json_file:
        py_dict = json.load(json_file)
    return py_dict


def create_unix_mesh_path(config):
    dataset_name = config["dataset_name"]
    mesh_filename = config["mesh_filename"]
    return os.path.join(dataset_name, mesh_filename)


def create_metadata_json(config):
    metadata = {}
    ds_name = config["3d_dataset_name"]
    mesh_class = config["da"]
    metadata["dataset_name"] = config["3d_dataset_name"]
    metadata["mesh_class"] = config["dataset_class"]
    metadata["mesh_filename"] = config["mesh_filename"]
    metadata["train_or_test"] = "train"
    metadata["unix_mesh_path"] = os.path.join(config["3d_dataset_name"], config["mesh_filename"])
    return metadata



def crop_square_bounding_box(img, mask, crop_ratio):
    mask = mask.astype(np.uint8)
    img_h, img_w = img.shape[:2]
    x,y,w,h = cv2.boundingRect(mask)
    cx = x+w/2
    cy = y+h/2
    padding = max(w,h)*(crop_ratio-1)/2
    square_size = max(w,h) + padding*2
    start_x = int(max(cx - square_size/2, 0))
    start_y = int(max(cy - square_size/2, 0))
    end_x = int(min(cx+square_size/2, img_w-1))
    end_y = int(min(cy+square_size/2, img_h-1))
    w = end_x-start_x
    h = end_y-start_y
    square_crop = img[start_y:end_y, start_x:end_x]
    return square_crop, (start_x, start_y, w, h)

def crop_camera_matrix(K, crop_x, crop_y):
    new_K = K
    new_K[0,2] = K[0,2] - crop_x
    new_K[1,2] = K[1,2] - crop_y
    return new_K



def create_dataset(config, json_path):
    json_path = os.path.join(config["project_name"], json_path)
    pose_dict = read_json_to_dict(json_path)
    project_name = config["project_name"]
    #print(pose_dict)
    poses = pose_dict["poses"]
    model_path = pose_dict["model_path"]
    img_dir = pose_dict["img_dir"]
    new_img_size = config["img_size"]
    orig_img_size = config["orig_img_size"]

    dataset_name = config["dataset_name"]
    dataset_path = os.path.join(project_name, dataset_name)
    #os.makedirs(dataset_name, exist_ok=True)

    split_poses_dict = split_dataset(poses, config["dataset_split"])
    crop_ratio_range = config["crop"]["crop_ratio_range"]
    use_crop = config["crop"]["use_crop"]
    for dataset_type in split_poses_dict:
        poses = split_poses_dict[dataset_type]
        for idx,img_basename in enumerate(poses):
            crop_ratio = np.random.uniform(crop_ratio_range[0], crop_ratio_range[1])
            print("crop ratio", crop_ratio)
            img_dict = poses[img_basename]
            T_CO = np.array(img_dict["T_CO"]).astype(np.float32)
            K = np.array(img_dict["K"])
            real_img_path = os.path.join(img_dir, img_basename)
            #image_handler(real_img_path)
            img = read_rgb(real_img_path)
            resize_from_img_size = orig_img_size
            if use_crop:
                print("Cropping image")
                _, rend_depth = render_scene(model_path, T_CO, K=K, img_size=orig_img_size)
                mask = np.where(rend_depth>0, True, False)
                img, (crop_x,crop_y,crop_w,crop_h) = crop_square_bounding_box(img, mask, crop_ratio)
                print("Uncropped K")
                print(K)
                K = crop_camera_matrix(K, crop_x,crop_y)
                print("crop_w", crop_w)
                print("Cropped K")
                print(K)
                resize_from_img_size = crop_w
            rz_img = cv2.resize(img, (new_img_size, new_img_size))
            rz_K = resize_camera_matrix(resize_from_img_size, new_img_size, K)
            img_example_dir = os.path.join(config["project_name"], config["dataset_name"], config["dataset_class"], dataset_type, f'ex{idx}')
            os.makedirs(img_example_dir, exist_ok=True)
            np.save(os.path.join(img_example_dir, "K.npy"), rz_K)
            np.save(os.path.join(img_example_dir, "T_CO_gt.npy"), T_CO)
            write_rgb(os.path.join(img_example_dir, "real.png"), rz_img)
            metadata = config["metadata"]
            write_yaml(os.path.join(img_example_dir, "metadata.yml"), metadata)
            T_CO_init = np.linalg.inv(apply_small_random_rotation_translation(np.linalg.inv(T_CO), 20, 0.05))
            np.save(os.path.join(img_example_dir, "T_CO_init.npy"), T_CO_init.astype(np.float32))
            rend_img,ren_dep = render_scene(model_path, T_CO_init, K=rz_K, img_size=new_img_size)
            np.save(os.path.join(img_example_dir, "init_depth.npy"), ren_dep)
            rend_img_gt, _ = render_scene(model_path, T_CO, K=rz_K, img_size=new_img_size)
            write_rgb(os.path.join(img_example_dir, "gt_rend.png"), np.uint8(rend_img_gt*255))
            write_rgb(os.path.join(img_example_dir, "init.png"), np.uint8(rend_img*255))
            verts = sample_vertices(model_path)
            np.save(os.path.join(img_example_dir, "vertices.npy"), verts)







def split_list(original_list, weight_list):
    sublists = []
    prev_index = 0
    for weight in weight_list:
        next_index = prev_index + math.ceil( (len(original_list) * weight) )

        sublists.append( original_list[prev_index : next_index] )
        prev_index = next_index

    return sublists




    

def split_dataset(ds_dict, dataset_split):
    out_dict = {}
    len_ds = len(ds_dict)
    indices = list(np.arange(len_ds))
    np.random.shuffle(indices)
    weights = []
    for key in dataset_split:
        weight = dataset_split[key]
        weights.append(weight)
    ind_split = split_list(indices, weights)
    for split_idx, dataset_type in enumerate(dataset_split):
        out_dict[dataset_type] = {}
        for ex_idx,ds_key in enumerate(ds_dict):
            if ex_idx in ind_split[split_idx]:
                out_dict[dataset_type][ds_key] = ds_dict[ds_key]
    return out_dict

    

    



def create_ds_test_dict():
    out = {}
    for i in range(10):
        out["img"+str(i)] = {"idx":i}
    return out


if __name__ == '__main__':
    json_path = "pose_dict.json"
    config_dict = get_dict_from_cli()
    print(get_dict_from_cli())

    #test_ds_dict = create_ds_test_dict()

    #split_dataset(test_ds_dict, config_dict["dataset_split"])
    create_dataset(config_dict, json_path)

