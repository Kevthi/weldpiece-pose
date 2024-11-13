import graphslam
from graphslam.graph import Graph
from graphslam.pose.se3 import PoseSE3
from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from collections import OrderedDict
import itertools
import cv2
from cv2 import aruco as aruco
import numpy as np
from gui_utils import read_rgb, get_image_paths_from_dir
import matplotlib.pyplot as plt
import os
import spatialmath as sm
from scipy.spatial.transform import Rotation as spR
from utils_aruco import set_axes_equal, plot_3d_graph, TwoWayDict
import pickle
from renderer import render_scene
import numpy as np
from debug import nice_print_dict
from gui_utils import blend_imgs, read_rgb
from g2o_pose_opt import PoseGraphOptimization
from charuco_board_utils import *
import g2o

"""
This Python script is designed for optimizing the poses of ArUco markers 
and cameras using graph-based optimization techniques.

1. Detect ArUco Markers:
detect_aruco_markers(image, aruco_dict, aruco_params): 
Detects ArUco markers in the image and returns their corners and IDs.

2. Estimate Poses:
estimate_pose(corners, ids, camera_matrix, dist_coeffs, marker_length): 
Estimates the poses of the detected markers relative to the camera and returns a dictionary of poses.

3. Build Initial Graph:
build_graph(poses, g2o_graph): 
Adds nodes to the graph for each detected marker pose.

4. Add Edges:
add_edges(g2o_graph, id_vert_dict, relative_poses): 
Adds edges to the graph representing the relative poses between the markers. For simplicity, 
this example assumes identity matrices for relative poses.

5. Optimize Graph:
optimize_graph(g2o_graph): 
Optimizes the graph and returns the optimized poses.
"""


def add_model_to_poses(img_basename : str, 
                       pose_dict : dict, 
                       poses : np.array) -> np.array:
    """
    Add the model pose (work piece) to the poses dictionary.
    model_pose is a nested dictionary
    example of pose_dict = {"img1.jpg": {"T_CO": np.array}, 
                             "img2.jpg: "{"T_CO": np.array},
    ...}.
    """
    if "T_CO" in pose_dict[img_basename]:
        poses["model"] =  pose_dict[img_basename]["T_CO"]
    return poses

from typing import Tuple

def contains_idx_in_graph(graph : dict, ar_poses : dict) -> Tuple[str, np.array] | Tuple[None, None]:
    """
    Identifies the AruCo marker index contained from ar_poses in contained in the graph.
    """
    for (board_idx, T_CA) in ar_poses.items():
        if str(board_idx) in graph:
            return str(board_idx), T_CA
    return None, None

def init_aruco_markers_graph(graph : dict, aruco_poses_all_imgs : dict) -> dict:
    """
    Initalize the graph with Aruco markers poses from multiple images.
    """
    graph["[0]"] = np.identity(4)
    graph_changed = True
    while graph_changed:
        graph_changed = False
        for img_basename_key in aruco_poses_all_imgs:
            ar_poses = aruco_poses_all_imgs[img_basename_key]
            contained_idx, T_CA = contains_idx_in_graph(graph, ar_poses)
            if contained_idx:
                for (marker_idx, T_CA_c) in ar_poses.items():
                    if not str(marker_idx) in graph:
                        graph[str(marker_idx)] = graph[str(contained_idx)]@np.linalg.inv(T_CA)@T_CA_c
                        graph_changed=True
    return OrderedDict(sorted(graph.items()))

def add_camera_positions_to_graph(outgraph, aruco_poses_all_imgs):
    """
    Adds the positions of cameras to the graph based on the poses of ArUco markers 
    detected in the images.
    """
    for img_basename in aruco_poses_all_imgs:
        poses = aruco_poses_all_imgs[img_basename]
        contained_idx, T_WA = contains_idx_in_graph(outgraph, poses)
        if contained_idx is not None:
            outgraph["cam_"+img_basename] = outgraph[str(contained_idx)]@np.linalg.inv(T_WA)
        else:
            print("Contained idx is none")
            print("img_basename", img_basename)
            print("poses")
            nice_print_dict(poses)
            print("outgraph")
            print(outgraph)

    return OrderedDict(sorted(outgraph.items()))

def add_model_poses_to_graph(outgraph, pose_dict):
    """
    Adds model poses to the graph based on the provided pose dictionary.
    """
    for img_basename in pose_dict:
        if "T_CO" in pose_dict[img_basename]:
            T_CO = pose_dict[img_basename]["T_CO"]
            outgraph["model"] = outgraph["cam_"+img_basename]@T_CO
    return OrderedDict(sorted(outgraph.items()))


def init_aruco_pose_graph(img_paths, aruko_poses_all_imgs, pose_dict):
    """
    Initializes the graph with ArUco marker poses, camera positions, and model poses.
    """
    outgraph = {}
    print("aruko_poses_all_imgs")
    nice_print_dict(aruko_poses_all_imgs)
    outgraph = init_aruco_markers_graph(outgraph, aruko_poses_all_imgs)
    print("Outgraph after init markers")
    nice_print_dict(outgraph)
    outgraph = add_camera_positions_to_graph(outgraph, aruko_poses_all_imgs)
    outgraph = add_model_poses_to_graph(outgraph, pose_dict)
    return OrderedDict(sorted(outgraph.items()))


def create_edges_graphslam(image_pose_dict, id_vert_dict, pose_dict, g2o_graph):
    """
    Creates edges in the graph for graph-based SLAM.
    """
    for key in id_vert_dict:
        print(key, id_vert_dict[key])

    print("#¤%&/()      CREATE EDGES GRAPHSLAM")
    nice_print_dict(image_pose_dict)
    aruko_info_mat = np.identity(6).astype(np.float32)*3.0
    model_info_mat = np.identity(6).astype(np.float32)*1.0
    #inf_mat_model = np.identity(6).astype(np.float32)
    for image_basename in image_pose_dict:
        image_poses = image_pose_dict[image_basename]
        print(image_pose_dict[image_basename])
        image_poses = add_model_to_poses(image_basename, pose_dict, image_poses)
        for (marker_idx, T_CA) in image_poses.items():
            cam_label = "cam_"+image_basename
            idx_cam = int(id_vert_dict[cam_label])
            marker_idx = int(id_vert_dict[str(marker_idx)])
            print("cam_label", cam_label, "mark idx", marker_idx)
            if marker_idx == 'model':
                info_mat = model_info_mat
            else:
                info_mat = aruko_info_mat/np.linalg.norm(T_CA[:3,3])
            g2o_graph.add_edge([idx_cam, marker_idx], g2o.Isometry3d(T_CA[:3,:3], T_CA[:3,3]), info_mat, g2o.RobustKernelHuber(np.sqrt(5.991)))

def init_vertices_graphslam(init_graph, g2o_graph):
    """
    Initializes the vertices in the g2o graph for graph-based SLAM.
    """
    id_vert_dict = TwoWayDict()
    for i,marker_idx in enumerate(init_graph):
        fixed = (i == 0)
        id_vert_dict[marker_idx] = str(i)
        T_WA = init_graph[marker_idx]
        g2o_graph.add_vertex(i, g2o.Isometry3d(T_WA[:3,:3], T_WA[:3,3]), fixed)
    return id_vert_dict

def optimize_aruko_graph(init_graph, image_pose_dict, pose_dict):
    """
    Optimizes the graph using g2o for graph-based SLAM.
    """
    g2o_graph = PoseGraphOptimization()
    id_vert_dict = init_vertices_graphslam(init_graph, g2o_graph)
    nice_print_dict(id_vert_dict)
    create_edges_graphslam(image_pose_dict, id_vert_dict, pose_dict, g2o_graph)
    g2o_graph.optimize(max_iterations=100)
    num_poses = len(id_vert_dict)
    out_graph = {}
    for idx in range(num_poses):
        str_repr = id_vert_dict[str(idx)]
        out_graph[str_repr]= np.array(g2o_graph.get_pose(idx).matrix())
    return out_graph


def get_T_CO(graph, img_basename):
    """
    Extracts the transformation matrix from the graph for the camera to the object.
    """
    T_WC = graph["cam_"+img_basename]
    T_WO = graph["model"]
    T_CO = np.linalg.inv(T_WC)@T_WO
    return T_CO

def aruko_optimize_handler(img_paths, K, aruco_dict_str, aruco_sq_size, pose_dict):
    """
    Handles the entire process of optimizing the graph using ArUco markers.
    
    Parameters:
    - img_paths: List of image paths.
    - K: Camera intrinsic matrix.
    - aruco_dict_str: ArUco dictionary string.
    - aruco_sq_size: Size of the ArUco squares.
    - pose_dict: Dictionary of model poses.
    
    Returns:
    - A dictionary with optimized transformation matrices for each image.
    """

    aruko_poses_all_imgs = get_all_image_board_pose_dict(img_paths, K, aruco_dict_str, aruco_sq_size, 6)
    out_dict = {}
    init_graph = init_aruco_pose_graph(img_paths, aruko_poses_all_imgs, pose_dict)
    out_graph = optimize_aruko_graph(init_graph, aruko_poses_all_imgs, pose_dict)

    #fig = plt.figure()
    #ax1 = fig.add_subplot(1,2,1,projection='3d')
    #ax2 = fig.add_subplot(1,2,2,projection='3d')
    #plot_3d_graph(ax1, init_graph)
    #plot_3d_graph(ax2, out_graph)
    #set_axes_equal(ax1)
    #set_axes_equal(ax2)
    ## EXIT ##
    #plt.show()
    for img_path in img_paths:
        img_basename = os.path.basename(img_path)
        out_dict[img_basename] = {}
        out_dict[img_basename]["T_CO_opt"] = get_T_CO(out_graph, img_basename)
    return out_dict
    



if __name__ == '__main__':
    pass

