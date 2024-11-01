import trimesh as tm
import os
import numpy as np
import spatialmath as sm


def crop_mesh(mesh):
    box = tm.creation.box(extents=[1.5, 1.5, 1.5])
    transl = np.identity(4)
    transl[:3,3] = [0,0,2]
    box = box.apply_transform(transl)
    print(type(box))
    mesh = mesh.slice_plane(box.facets_origin, -box.facets_normal)
    return mesh


def crop_pc_handler(pc_dir : str, export_dir : str, xy_lim=1):
    """Crop point clouds in the given directory and save them in the export directory."""
    mesh_paths = [os.path.join(pc_dir, filename) for filename in os.listdir(pc_dir)]
    os.makedirs(export_dir, exist_ok=True)
    for mesh_path in mesh_paths:
        print(mesh_path)
        mesh = tm.load(mesh_path)
        print(mesh.vertices)
        rx = sm.SE3.Rx(180, unit='deg').data[0]
        print(rx)
        mesh = mesh.apply_transform(rx)
        print("After transf")
        print(mesh.vertices)
        mesh = crop_mesh(mesh)
        out_file = os.path.join(export_dir, os.path.basename(mesh_path))
        tm.exchange.export.export_mesh(mesh, out_file)



if __name__ == '__main__':
    pc_dir = "mesh"
    export_dir = "mesh-crop"
    crop_pc_handler(pc_dir, export_dir)
