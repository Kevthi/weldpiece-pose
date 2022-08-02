import numpy as np
import cv2
import os
import spatialmath as sm
import scipy

def project_to_3d(pixel_coords, depth_map, K):

    print("pixel_coords")
    print(tuple(pixel_coords))
    indices = [(val[0], val[1]) for val in pixel_coords]
    print(indices)
    print(depth_map[indices])
    pixel_coords = np.array(pixel_coords).transpose()
    depth_vals = depth_map[pixel_coords]
    print(depth_vals.shape)
    homg_pix_coords = np.vstack((pixel_coords, np.ones(pixel_coords.shape[1])))
    K_inv = np.linalg.inv(K)
    norm_img_corrds = K_inv@homg_pix_coords


if __name__ == '__main__':
    pixel_coords = [np.array([1,2]), np.array([3,4]), np.array([5,6])]
    K = np.random.random((3,3))
    depth_map = np.random.random((20,20))
    project_to_3d(pixel_coords, depth_map, K)
    print(pixel_coords)
