import numpy as np
import cv2
import os
import spatialmath as sm
import scipy
from se3_helpers import look_at_SE3

"""
def project_to_3d(pixel_coords, depth_map, K):
    indices = tuple(np.transpose([(int(val[0]), int(val[1])) for val in pixel_coords]))
    pixel_coords = np.array(pixel_coords).transpose()
    depth_vals = depth_map[indices]
    homg_pix_coords = np.vstack((pixel_coords[1,:], pixel_coords[0,:], np.ones(pixel_coords.shape[1])))
    K_inv = np.linalg.inv(K)
    norm_img_coords = K_inv@homg_pix_coords
    points_3d = depth_vals*norm_img_coords
    return points_3d.T
"""

def convert_indices_to_pix_coord(inds):
    pix_coords = []
    for y,x in inds:
        pix_coords.append([x,y])
    return np.array(pix_coords)



def project_to_3d(points, depth, K):
    points_3d = []
    K_inv = np.linalg.inv(K)
    for x,y in points:
        pix = np.array([x,y,1], dtype=np.float64)
        depth_xy = depth[y,x]
        s = K_inv@pix
        x = s*depth_xy
        points_3d.append(x)
    return np.array(points_3d)


def transform_points(points_cw, T):
    R = T[:3,:3,]
    t = T[:3,3]
    points_transf = ((R@points_cw).T + t).T
    return points_transf

def reproject(points_3d, K):
    assert points_3d.shape[0] == 3
    assert K.shape == (3,3)
    s = K@points_3d
    s = s/s[2,:]
    return s[:2,:]

    


def solve_pnp_ransac(points_W, img_coords, cam_K, reproj_tol=8, T_CW_guess=None):
    img_coords = np.array(img_coords).astype(np.float64)
    #img_coords = np.flip(img_coords, axis=0)
    cam_K = cam_K.astype(np.float64)
    points_W = points_W.astype(np.float64)
    assert points_W.shape[1] == 3
    assert cam_K.shape == (3,3)
    assert img_coords.shape[1] == 2

    if T_CW_guess is None:
        retval, rodr_CW, tvec, inliers = cv2.solvePnPRansac(points_W, img_coords, cam_K, np.array([]), reprojectionError=reproj_tol)
    else:
        assert T_CW_guess.shape ==(4,4)
        R_CW_g = T_CW_guess[:3,:3]
        t_CW_g = T_CW_guess[:3,3].flatten()
        rodr_CW_g,_ = cv2.Rodrigues(R_CW_g)
        retval, rodr_CW, tvec, inliers = cv2.solvePnPRansac(points_W, img_coords, cam_K, np.array([]), reprojectionError=reproj_tol, rvec=rodr_CW_g, tvec=t_CW_g)


    rodr_CW = rodr_CW.transpose()[0]
    R_CW,_ = cv2.Rodrigues(rodr_CW)
    T_CW = np.identity(4)
    T_CW[:3,:3] = R_CW
    T_CW[:3, 3] = tvec.flatten()
    return T_CW, inliers


@staticmethod
def solve_pnp_smcv(points_W, pixels, T_CO_current, K):
    assert points_W.shape[1] == 3
    assert K.shape == (3,3)
    assert T_CO_current.shape == (4,4)

    print("solve_pnp_smcv")
    print(type(points_W))
    print(type(pixels))
    print(type(T_CO_current))
    print(type(K))
    print(points_W.shape)
    print(pixels.shape)
    print(points_W)
    print(pixels)

    if len(points_W) < 5:
        print("POINTS_W < 5, returning identity")
        return sm.SE3.Rx(0)

    R_current = T_CO_current[:3,:3]
    t_c = T_CO_current[:3, 3].reshape((3,1))

    r_c,_ = cv2.Rodrigues(R_current)

    _, rodr_CW, transl,_ = cv2.solvePnPRansac(points_W, pixels, K, np.array([]), rvec=r_c, tvec=t_c, reprojectionError=5.0, useExtrinsicGuess=True)
    #_,rodr_CW, transl = cv2.solvePnP(points_W, pixels, K, np.array([]))
    rodr_CW = rodr_CW.transpose()[0]
    #R_CW = R.from_mrp(rodr_CW).as_matrix()
    R_CW,_ = cv2.Rodrigues(rodr_CW)

    SO3_CW = sm.SO3(R_CW, check=True)
    T_CW = sm.SE3.Rt(SO3_CW, transl)
    return T_CW






if __name__ == '__main__':
    points_W = np.random.random((3,10))
    T_WC = look_at_SE3(origin=[5,5,5], target=[3,3,3], up=[0,0,1])
    T_CW = T_WC.inv().data[0]
    T_WC = T_WC.data[0]
    points_C = transform_points(points_W, T_CW)
    K = np.array([[3600,0,600],[0,3600,600],[0,0,1]])
    pixel_coords = reproject(points_C, K)

    depth_map = (np.random.random((1000,1000))+2)*3
    #points_3d = project_to_3d(pixel_coords, depth_map, K)
    T_CW_pnp = solve_pnp_ransac(points_W.T, pixel_coords.T, K, T_CW_guess=T_CW)
    print("T_CW_pnp")
    print(T_CW_pnp)
    print("T_CW")
    print(T_CW)
    print(np.allclose(T_CW_pnp, T_CW, atol=0.000001))



