import numpy as np
import os
import open3d as o3d
import json
import spatialmath as sm
import copy
import trimesh as tm
import random
print(o3d.__version__)

def read_json_file(file_name):
    with open(file_name, 'r') as json_file:
        return json.load(json_file)

"""
def draw_registration_result(source, target, transformation):
    source_temp = source.clone()
    target_temp = target.clone()

    source_temp.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source_temp.to_legacy(),
         target_temp.to_legacy()],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])
"""

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

def calc_add_l1_loss(verts, T_CO, T_CO_gt):
    print(verts.shape)
    R_CO = T_CO[:3,:3]
    t_CO = T_CO[:3,3]
    R_CO_gt = T_CO_gt[:3,:3]
    t_CO_gt = T_CO_gt[:3,3]
    losses = ((R_CO@verts.T).T+t_CO)-((R_CO_gt@verts.T).T+t_CO_gt)
    loss = np.mean(np.abs(losses))
    return loss




def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp, mesh_frame],
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[0, 0, 3.0],
                                        up=[0, 0, 1])


def run_icp(mesh_path, target_mesh_path, init_T=np.identity(4)):
    source = o3d.io.read_point_cloud(mesh_path)
    target = o3d.io.read_point_cloud(target_mesh_path)
    rx = sm.SE3.Rx(180, unit='deg').data[0]
    target = target.transform(rx)
    #source.point["colors"] = source.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
    #target.point["colors"] = target.point["colors"].to(o3d.core.Dtype.Float32) / 255.0
    threshold = 2.0
    evalutation = o3d.pipelines.registration.evaluate_registration(source,target,threshold, init_T)
    #draw_registration_result(source,target, init_T)
    p2pl =  o3d.pipelines.registration.TransformationEstimationPointToPlane()
    #p2pl = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    conv_crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold, init_T,p2pl, conv_crit)
    transf_reg = reg_p2l.transformation
    #draw_registration_result(source, target, transf_reg)
    return transf_reg


def icp_handler(target_mesh_dir, source_mesh_path, pose_dict):
    target_mesh_paths = [os.path.join(target_mesh_dir, filename) for filename in os.listdir(target_mesh_dir)]
    target_mesh_paths.sort()
    add_l1_losses = []
    for target_mesh_path in target_mesh_paths:
        #print(target_mesh_path)
        T_gt = pose_dict["poses"][f'img{os.path.basename(target_mesh_path)[4]}_Color-undist.png']["T_CO"]
        T_gt = np.array(T_gt)
        T_init = T_gt@sm.SE3.Ry(30, unit='deg').data[0]
        #print(T_init)
        T_reg = run_icp(source_mesh_path, target_mesh_path, T_init)
        #print("T_gt")
        #print(T_gt)
        #print("T_reg")
        #print(T_reg)
        verts = sample_vertices(source_mesh_path)
        addl1_loss = calc_add_l1_loss(verts, T_reg, T_gt)
        print("add l1 loss", addl1_loss)
        add_l1_losses.append(addl1_loss)

    print("Mean loss")
    print(np.mean(np.array(add_l1_losses)))
    print("Std")
    print(np.std(add_l1_losses))













if __name__ == '__main__':
    target_mesh_dir = "mesh"
    mesh_path = "alu-corner-remesh.ply"
    target_mesh_path = "mesh/mesh0.ply"
    pose_json = read_json_file("pose_dict.json")
    print(pose_json)
    init_T = pose_json["poses"]["img0_Color-undist.png"]["T_CO"]
    init_T = np.array(init_T)
    print(init_T)
    #run_icp(mesh_path, target_mesh_path, init_T)
    icp_handler(target_mesh_dir, mesh_path, pose_json)


