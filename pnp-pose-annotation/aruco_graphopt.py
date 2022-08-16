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
        poses.append(["model", pose_dict[img_basename]["T_CO"]])
    return poses



def contains_idx_in_graph(graph, ar_poses):
    for (marker_idx, T_CA) in ar_poses:
        if str(marker_idx) in graph:
            return str(marker_idx), T_CA
    return None

def init_aruco_graph(graph, img_paths, K, aruco_dict_str, aruko_sq_size):

    graph["[0]"] = np.identity(4)

    graph_changed = True
    while graph_changed:
        graph_changed = False
        for img_path in img_paths:
            img = read_rgb(img_path)
            ar_poses = get_aruko_poses(img, K, aruco_dict_str, aruko_sq_size)
            contained_idx, T_CA = contains_idx_in_graph(graph, ar_poses)
            if contained_idx:
                for (marker_idx, T_CA_c) in ar_poses:
                    if not str(marker_idx) in graph:
                        print("contained idx", contained_idx, "marker_idx", marker_idx)
                        graph[str(marker_idx)] = graph[str(contained_idx)]@np.linalg.inv(T_CA)@T_CA_c
                        graph_changed=True
    return graph






def init_aruco_pose_graph(img_paths, K, aruco_dict_str, aruko_sq_size, pose_dict):
    outgraph = {}
    outgraph = init_aruco_graph(outgraph, img_paths, K, aruco_dict_str, aruko_sq_size)
    for img_path in img_paths:
        img = read_rgb(img_path)
        poses = get_aruko_poses(img, K, aruco_dict_str, aruko_sq_size)
        img_basename = os.path.basename(img_path)
        contained_idx, T_WA = contains_idx_in_graph(outgraph, poses)
        outgraph["cam_"+img_basename] = outgraph[str(contained_idx)]@np.linalg.inv(T_WA)

        #for (marker_idx, T_CA) in poses:
            #outgraph[str(marker_idx)] = np.identity(4)

    for img_basename in pose_dict:
        if "T_CO" in pose_dict[img_basename]:
            T_CO = pose_dict[img_basename]["T_CO"]
            outgraph["model"] = outgraph["cam_"+img_basename]@T_CO
            break
    return OrderedDict(sorted(outgraph.items()))



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



def create_edges_graphslam(image_pose_dict, id_vert_dict, pose_dict):
    for key in id_vert_dict:
        print(key, id_vert_dict[key])
    gs_edges = []
    aruko_info_mat = np.identity(6).astype(np.float32)*10.0
    model_info_mat = np.identity(6).astype(np.float32)*1.0
    #inf_mat_model = np.identity(6).astype(np.float32)
    for image_basename in image_pose_dict:
        image_poses = image_pose_dict[image_basename]
        image_poses = add_model_to_poses(image_basename, pose_dict, image_poses)
        for (marker_idx, T_CA) in image_poses:
            cam_label = "cam_"+image_basename
            idx_cam = int(id_vert_dict[cam_label])
            marker_idx = int(id_vert_dict[str(marker_idx)])
            print("cam_label", cam_label, "mark idx", marker_idx)
            if marker_idx == 'model':
                info_mat = model_info_mat
            else:
                info_mat = aruko_info_mat
            edge = create_graphslam_edge(idx_cam, marker_idx, T_CA, info_mat)
            gs_edges.append(edge)
    return gs_edges
    






def optimize_aruko_graph(init_graph, image_pose_dict, pose_dict):
    gs_vertices, id_vert_dict = init_vertices_graphslam(init_graph)
    gs_edges = create_edges_graphslam(image_pose_dict, id_vert_dict, pose_dict)
    gs_graph = Graph(gs_edges, gs_vertices)
    gs_graph.optimize(tol=1e-16, max_iter=100)
    out_graph = {}
    for vert in gs_graph._vertices:
        out_graph[id_vert_dict[str(vert.id)]]= vert.pose.to_matrix()
    return out_graph


def get_T_CO(graph, img_basename):
    T_WC = graph["cam_"+img_basename]
    T_WO = graph["model"]
    T_CO = np.linalg.inv(T_WC)@T_WO
    return T_CO

def aruko_optimize_handler(img_paths, K, aruco_dict_str, aruco_sq_size, pose_dict):

    out_dict = {}
    init_graph = init_aruco_pose_graph(img_paths, K, aruco_dict_str, aruco_sq_size, pose_dict)
    img_pose_dict = create_image_pose_dict(img_paths, K, aruco_dict_str, aruco_sq_size)
    out_graph = optimize_aruko_graph(init_graph, img_pose_dict, pose_dict)

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
    with open('pose_dict.pkl', 'rb') as handle:
        pose_dict = pickle.load(handle)
    K = np.array([[1166.3, .0, 509],[0, 1166.0, 546.0],[0,0,1.0]])
    print(type(K[0,0]))
    K_load = np.load("K.npy")
    print(type(K_load[0,0]))
    print("K load")
    print(np.load("K.npy"))
    
    #print("K_old", K_old)
    print("K", K)
    K = K.astype(np.float64)
    img_dir = "/home/ola/projects/weldpiece-pose-datasets/ds-projects/office-corner-brio/captures"
    img_paths = get_image_paths_from_dir(img_dir)
    aruco_dict_str = "DICT_APRILTAG_16H5"
    aruco_sq_size = 66.0*1e-3

    opt_dict = aruko_optimize_handler(img_paths, K, aruco_dict_str, aruco_sq_size, pose_dict)

    obj_path = "/home/ola/projects/weldpiece-pose-datasets/3d-models/corner.ply"

    nice_print_dict(opt_dict)

    for key in opt_dict:
        T_CO = opt_dict[key]["T_CO_opt"]
        print("T_CO render")
        print(T_CO)
        img, dep = render_scene(obj_path, T_CO, K, (1080,1080))
        plt.imshow(img)
        plt.show()


