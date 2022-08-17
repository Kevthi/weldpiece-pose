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



def get_aruko_poses(img, K, aruco_dict_str, aruko_sq_size):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    ar_params = aruco.DetectorParameters_create()
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    dist = np.zeros(5)
    aruko_poses = []
    if marker_ids is not None and len(marker_ids) > 0:
        img = aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        for single_marker_corner, marker_id in zip(marker_corners, marker_ids):
            rvec,tvec,obj_p_corners = aruco.estimatePoseSingleMarkers(single_marker_corner, aruko_sq_size, K, dist)
            T_CA = np.identity(4)
            R, _ = cv2.Rodrigues(rvec)
            T_CA[:3,:3] = R
            T_CA[:3,3] = tvec
            aruko_poses.append((marker_id, T_CA))
    return aruko_poses



def draw_markers(img, K, aruco_dict_str):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    ar_params = aruco.DetectorParameters_create()
    (marker_corners, marker_ids, rejected) = aruco.detectMarkers(img, aruco_dict, parameters=ar_params)
    dist = np.zeros(5)
    if marker_ids is not None and len(marker_ids) > 0:
        img = aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        for single_marker_corner, marker_id in zip(marker_corners, marker_ids):
            rvec,tvec,obj_p_corners = aruco.estimatePoseSingleMarkers(single_marker_corner, 66.0*1e-3, K, dist)
            cv2.drawFrameAxes(img, K, dist, rvec,tvec,0.05)
    return img

def add_model_to_poses(img_basename, pose_dict, poses):
    if "T_CO" in pose_dict[img_basename]:
        print(img_basename)
        if pose_dict[img_basename]["pose_set_with_pnp"]:
            poses.append(["model", pose_dict[img_basename]["T_CO"]])
    return poses

def init_aruco_pose_graph(img_paths, K, aruco_dict_str, aruko_sq_size, pose_dict):
    outgraph = {}
    for img_path in img_paths:
        img = read_rgb(img_path)
        poses = get_aruko_poses(img, K, aruco_dict_str, aruko_sq_size)
        for (marker_idx, T_CA) in poses:
            outgraph[str(marker_idx)] = np.identity(4)

    for img_basename in pose_dict:
        if "T_CO" in pose_dict[img_basename]:
            outgraph["model"] = np.identity(4)
    return OrderedDict(sorted(outgraph.items()))

"""
def init_aruco_pose_graph(img_paths, K, aruco_dict_str, aruko_sq_size, pose_dict):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    img_paths.sort()
    lowest_idx = np.infty
    img_nodes = {}
    for img_path in img_paths:
        img = read_rgb(img_path)
        poses = get_aruko_poses(img, K, aruco_dict_str, aruko_sq_size)
        img_basename = os.path.basename(img_path)
        for (marker_idx, T_CA) in poses:
            if marker_idx < lowest_idx:
                lowest_idx = marker_idx
        poses = add_model_to_poses(img_basename, pose_dict, poses)
        img_nodes[img_basename] = poses
    graph = {
        str(lowest_idx): {
            "parent": None,
            "dist_to_orig": 0.0,
            "T_WA":np.identity(4)
        }
    }
    graph_updated = True
    while graph_updated:
        graph_updated = False
        for key in img_nodes:
            poses = img_nodes[key]
            closest_idx, T_clos_cam, closest_dist = get_closest_idx_in_graph(graph, poses)
            if closest_idx is not None:
                for (marker_idx, T_CA) in poses:
                    T_clos_A = T_clos_cam@T_CA
                    T_WA = graph[str(closest_idx)]["T_WA"]@T_clos_A
                    actual_dist_to_orig = np.linalg.norm(T_WA[:3,3])
                    if str(marker_idx) in graph:
                        if graph[str(marker_idx)]["dist_to_orig"] > (actual_dist_to_orig+closest_dist):
                            graph_updated = True
                            graph[str(marker_idx)] = {
                                "parent": closest_idx,
                                "dist_to_orig": np.linalg.norm(T_WA[:3,3]),
                                #"dist_to_orig": 0.0,
                                "T_WA": T_WA, }
                    else:
                        graph_updated = True
                        graph[str(marker_idx)] = {
                            "parent": closest_idx,
                            "dist_to_orig": np.linalg.norm(T_WA[:3,3]),
                            "T_WA": T_WA,
                        }
    out_graph = {}
    for key in graph:
        out_graph[key] = graph[key]["T_WA"]
    return OrderedDict(sorted(out_graph.items()))
"""

def get_closest_idx_in_graph(graph, poses):
    closest_dist = np.infty
    closest_idx = None
    T_clos_cam = None
    for (marker_idx, T_CA) in poses:
        if str(marker_idx) in graph:
            if graph[str(marker_idx)]["dist_to_orig"] < closest_dist:
                closest_dist = graph[str(marker_idx)]["dist_to_orig"]
                closest_idx = marker_idx
                T_clos_cam = np.linalg.inv(T_CA)
    return closest_idx, T_clos_cam, closest_dist


def create_image_pose_dict(img_paths, K, aruco_dict_str, aruko_sq_size):
    aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_str))
    img_paths.sort()
    img_pose_dict = {}
    for img_path in img_paths:
        img = read_rgb(img_path)
        poses = get_aruko_poses(img, K, aruco_dict_str, aruko_sq_size)
        img_pose_dict[os.path.basename(img_path)] = poses
    return OrderedDict(sorted(img_pose_dict.items()))

def T_to_PoseSE3(T):
    R = T[:3,:3] 
    t = T[:3,3]
    q = spR.from_matrix(R).as_quat()
    T_PoseSE3 = PoseSE3(t, q)
    return T_PoseSE3


def create_graphslam_vertex(idx, T_W):
    T_W_gs = T_to_PoseSE3(T_W)
    return Vertex(idx, T_W_gs)

def create_graphslam_edge(idx_k, idx_l, T_kl, info_mat):
    T_kl_gs = T_to_PoseSE3(T_kl)
    edge = EdgeOdometry([int(idx_k), int(idx_l)], info_mat, T_kl_gs)
    return edge

def init_vertices_graphslam(init_graph):
    id_vert_dict = TwoWayDict()
    gs_vertices = []
    for i,marker_idx in enumerate(init_graph):
        id_vert_dict[marker_idx] = str(i)
        T_WA = init_graph[marker_idx]
        gs_vertices.append(create_graphslam_vertex(i, T_WA))
    return gs_vertices, id_vert_dict

"""
def create_edges_graphslam(image_pose_dict, id_vert_dict, pose_dict):
    gs_edges = []
    inf_mat = np.identity(6).astype(np.float32)*100
    inf_mat_model =np.identity(6).astype(np.float32)
    for image_basename in image_pose_dict:
        image_poses = image_pose_dict[image_basename]
        image_poses = add_model_to_poses(image_basename, pose_dict, image_poses)
        image_poses_cp = image_poses.copy()
        for (marker_idx, T_CA) in image_poses:
            for (marker_idx_other, T_CA_other) in image_poses_cp:
                if marker_idx != marker_idx_other:
                    T_A_Aother = np.linalg.inv(T_CA)@T_CA_other
                    idx_A = id_vert_dict[str(marker_idx)]
                    idx_other = id_vert_dict[str(marker_idx_other)]
                    if marker_idx == 'model' or marker_idx_other == 'model':
                        info_mat = inf_mat_model
                    else: 
                        info_mat = inf_mat
                    edge = create_graphslam_edge(idx_A, idx_other, T_A_Aother, info_mat)
                    gs_edges.append(edge)
                    print(marker_idx, marker_idx_other)
            image_poses_cp.pop(0)
    return gs_edges
"""


def create_edges_graphslam(image_pose_dict, id_vert_dict, pose_dict):
    gs_edges = []
    inf_mat = np.identity(6).astype(np.float32)*100
    inf_mat_model = np.identity(6).astype(np.float32)
    for image_basename in image_pose_dict:
        image_poses = image_pose_dict[image_basename]
        image_poses = add_model_to_poses(image_basename, pose_dict, image_poses)
        for (marker_idx, T_CA) in image_poses:
            cam_label = "cam_"+image_basename
            idx_cam = id_vert_dict(cam_label)
            marker_idx = str(marker_idx)
            edge = create_graphslam_edge(idx_cam, idx_cam, T_CA, info_mat)
            gs_edges.append(edge)
    return gs_edges
    






def optimize_aruko_graph(init_graph, image_pose_dict, pose_dict):
    gs_vertices, id_vert_dict = init_vertices_graphslam(init_graph)
    gs_edges = create_edges_graphslam(image_pose_dict, id_vert_dict, pose_dict)
    gs_graph = Graph(gs_edges, gs_vertices)
    gs_graph.optimize(tol=1e-16, max_iter=40)
    out_graph = {}
    for vert in gs_graph._vertices:
        out_graph[id_vert_dict[str(vert.id)]]= vert.pose.to_matrix()
    return out_graph




if __name__ == '__main__':
    with open('pose_dict.pkl', 'rb') as handle:
        pose_dict = pickle.load(handle)
    K = np.array([[1166.0, .0, 509],[0, 1166.0, 546.0],[0,0,1.0]])
    img_dir = "/home/ola/projects/weldpiece-pose-datasets/ds-projects/office-corner-brio/captures"
    img_paths = get_image_paths_from_dir(img_dir)
    aruco_dict_str = "DICT_APRILTAG_16H5"
    aruco_sq_size = 66.0*1e-3
    init_graph = init_aruco_pose_graph(img_paths, K, aruco_dict_str, aruco_sq_size, pose_dict)
    fig = plt.figure()
    #ax1 = fig.add_subplot(1,1,1,projection='3d')
    #plot_3d_graph(ax1, init_graph)
    #set_axes_equal(ax1)
    #plt.show()
    img_pose_dict = create_image_pose_dict(img_paths, K, aruco_dict_str, aruco_sq_size)
    out_graph = optimize_aruko_graph(init_graph, img_pose_dict, pose_dict)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    plot_3d_graph(ax1, init_graph)
    plot_3d_graph(ax2, out_graph)
    set_axes_equal(ax1)
    set_axes_equal(ax2)
    plt.show()
