import argparse


# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
parser.add_argument("--input", type=str, help="The path to the input mesh file.")
parser.add_argument("--output", type=str, help="The path to store the USD file.")
args_cli = parser.parse_args()

import os
import glob
from tqdm import tqdm
import random
import open3d as o3d
import numpy as np


def obj_mesh(path):
  return o3d.io.read_triangle_mesh(path)

def draw_mesh(mesh):

  mesh.paint_uniform_color([1., 0., 0.])
  mesh.compute_vertex_normals()
  o3d.visualization.draw_geometries([mesh])

def mesh2ply(mesh):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
  return pcd


def main():
    # check valid file path
    mesh_dir = args_cli.input

    OBJ_PATH = os.path.join(mesh_dir, "*", "*", "*", "*_015.obj")
    obj_list = glob.glob(OBJ_PATH, recursive=True)
    print(obj_list)

    dest_dir = args_cli.output

    for mesh_path in tqdm(obj_list, total=len(obj_list)):
        mesh_path = os.path.abspath(mesh_path)

        # dest_path = os.path.abspath(os.path.join(dest_dir, (os.path.basename(mesh_path)).replace('.obj', '.npy')))
        dest_path = os.path.abspath(mesh_path.replace('.obj', '.npy'))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        mesh = obj_mesh(mesh_path)

        pcd = mesh2ply(mesh)
        print(len(pcd.points))
        pcd_npy = np.asarray(pcd.points).copy()
        np.save(dest_path, pcd_npy)
        # print(pcd_npy.size)
        # o3d.io.write_point_cloud(dest_path, pcd)

if __name__ == "__main__":
    # run the main function
    main()
