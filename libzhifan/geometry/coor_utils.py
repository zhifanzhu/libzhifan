""" Utility function for coordinate system. """
from typing import Callable
import numpy as np
import torch


def nptify(x) -> Callable:
    """ 
    Example:
    a = torch.Tensor([1])
    b = np.array([])
    To convert b to the type of a:
    b_out = nptify(a)(b)

    Returns:
        A Callable that converts its input to x's type
    """
    if isinstance(x, torch.Tensor):
        return lambda a: torch.as_tensor(a, dtype=x.dtype, device=x.device)
    elif isinstance(x, np.ndarray):
        return lambda a: np.asarray(a)


""" Functions dealing with homogenous coordiate transforms. """


def to_homo_xn(pts):
    """ assume [x, n], output [x+1, n]"""
    n = pts.shape[1]
    return np.vstack((pts, np.ones([1, n])))


def to_homo_nx(pts):
    """ [n,x] -> [n,x+1] """
    return to_homo_xn(pts.T).T


def from_home_xn(pts):
    """ [x+1, n] -> [x, n] """
    return pts[:-1, :]


def from_home_nx(pts):
    """ [n, x+1] -> [n, x] """
    return from_home_xn(pts.T).T


def normalize_homo_xn(x_h):
    return x_h / x_h[-1, :]


def normalize_homo_nx(x_h):
    return x_h / x_h[:, -1]


def normalize_and_drop_homo_xn(x_h):
    """ x_h: (c, n) -> (c-1, n) """
    return x_h[:-1, :] / x_h[-1, :]


def normalize_and_drop_homo_nx(x_h):
    """ x_h: (n, c) -> (n, c-1) """
    return normalize_and_drop_homo_xn(x_h.T).T


def transform_nx3(transform_matrix, x):
    """

    Args:
        transform_matrix: (4, 4)
        x: (n, 3)

    Returns: (n, 3)

    """
    return transform_3xn(transform_matrix, x.T).T


def transform_3xn(transform_matrix, x):
    """

    Args:
        transform_matrix: (4, 4)
        x: (3, n)

    Returns: (3, n)

    """
    x2 = transform_matrix @ to_homo_xn(x)
    return from_home_xn(x2)


def concat_rot_transl_3x4(rot, transl):
    """
    Args:
        rot: (3, 3)
        transl: (3, 1) or (3, )

    Returns: (3, 4)

    """
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = rot
    Rt[0:3, -1] = transl.squeeze()
    return Rt


def concat_rot_transl_4x4(rot, transl):
    """
    Args:
        rot: (3, 3)
        transl: (3, 1) or (3, )

    Returns: (4, 4)

    """
    Rt = np.zeros([4, 4])
    Rt[0:3, 0:3] = rot
    Rt[0:3, -1] = transl.squeeze()
    Rt[-1, -1] = 1.0
    return Rt


def rotation_epfl(alpha, beta, gamma):
    R = np.zeros([3, 3])
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)
    R[0, 0] = cos_a * cos_g - cos_b * sin_a * sin_g
    R[1, 0] = cos_g * sin_a + cos_a * cos_b * sin_g
    R[2, 0] = sin_b * sin_g

    R[0, 1] = -cos_b * cos_g * sin_a - cos_a * sin_g
    R[1, 1] = cos_a * cos_b * cos_g - sin_a * sin_g
    R[2, 1] = cos_g * sin_b

    R[0, 2] = sin_a * sin_b
    R[1, 2] = -cos_a * sin_b
    R[2, 2] = cos_b

    return R


# def rotation_xyz_from_euler(x_rot, y_rot, z_rot):
#     rz = np.float32([
#         [np.cos(z_rot), np.sin(z_rot), 0],
#         [-np.sin(z_rot), np.cos(z_rot), 0],
#         [0, 0, 1]
#     ])
#     ry = np.float32([
#         [np.cos(y_rot), 0, -np.sin(y_rot)],
#         [0, 1, 0],
#         [np.sin(y_rot), 0, np.cos(y_rot)],
#     ])
#     rx = np.float32([
#         [1, 0, 0],
#         [0, np.cos(x_rot), np.sin(x_rot)],
#         [0, -np.sin(x_rot), np.cos(x_rot)],
#     ])
#     return rz @ ry @ rx


""" Primitive Transforms """


def translate(points, x, y, z):
    """ 
    Args:
        points: (n, 3)
        x, y, z: scalar
    Returns:
        (n, 3)
    """
    out_type = nptify(points)
    translation = out_type([[x, y, z]])
    return points + translation


def scale(points, x, y, z):
    """ 
    Args:
        points: (n, 3)
        x, y, z: scalar
    Returns:
        (n, 3)
    """
    out_type = nptify(points)
    scaling = out_type([[x, y, z]])
    return points * scaling




""" Camera """


def camera_matrix(fx, fy, cx, cy):
    """

    Returns:
        K: (3, 3) ndarray
    """
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0,  1]
    ], dtype=type(fx))
    return K


def project_3d_2d(x3d, A=None, fx=None, fy=None, cx=None, cy=None):
    """

    Args:
        x3d: (3, n)
        A: (3, 3)
        fx, fy, cx, cy: scalar
            either A or (fx, fy, cx , cy) must be supplied.

    Returns: (2, n)
    """
    if A is not None:
        assert A.ndim == 2 and A.shape[-1] == 3
        x2d_h = A @ x3d
    else:
        assert fx and fy and (cx is not None) and (cy is not None)
        K = camera_matrix(fx, fy, cx, cy)
        x2d_h = nptify(x3d)(K) @ x3d
    return extract_pixel_homo_xn(x2d_h)


def extract_pixel_homo_xn(x2d_h):
    """ x2d_h: [3, n] -> [2, n] """
    return normalize_and_drop_homo_xn(x2d_h)


def extract_pixel_homo_nx(x2d_h):
    """ x2d_h: [n, 3] -> [n, 2] """
    return normalize_and_drop_homo_nx(x2d_h.T).T


def world_to_camera_pipeline(x3d, M_intrinsic, M_offset=None, Tmw=None):
    """
    P_2d = M_offset @ divide_by_z() @ M_intrinsic @ P_3d
        where P_3d = (x, y, z)

    Args:
        x3d: (n, 3)
        M_offset: (3, 3)
        M_intrinsic: (3, 3) or (3, 4)
        Tmw: (4, 4)

    Returns:

    """
    pts_h = to_homo_nx(x3d).T  # pts_h: (4, n)
    if M_offset is None:
        M_offset = np.identity(3)
    if Tmw is None:
        Tmw = np.identity(4)
    if M_intrinsic.shape == (3, 3):
        _M_intrinsic = np.zeros([3, 4])
        _M_intrinsic[:3, :3] = M_intrinsic
    elif M_intrinsic.shape == (3, 4):
        _M_intrinsic = np.zeros([3, 4])
        _M_intrinsic[:3, :] = M_intrinsic
    else:
        raise ValueError(f"Unexpected M_intrinsic shape: {M_intrinsic.shape}")

    M_transformations = _M_intrinsic @ Tmw
    pts_h = M_transformations @ pts_h
    pts_h = normalize_homo_xn(pts_h)
    pts_h = M_offset @ pts_h
    pts_2d = from_home_xn(pts_h).transpose()  # (3,n) -> (n,2)

    return pts_2d


def world_to_camera_pipeline_test():
    x3d = np.float32([
        [10, 20, 10],
        [0, -10, 2],
        [-30, 0, 3],
        [-5, -5, 1],
    ])
    fx = 4
    fy = 2
    M_intrinsic = np.float32([
        [fx, 0, 0],
        [0, fy, 0],
        [0, 0,  1],
    ])
    x2d_true = np.float32([
        [4, 4],
        [0, -10],
        [-40, 0],
        [-20, -10]
    ])
    x2d_est = world_to_camera_pipeline(x3d, M_intrinsic)
    np.testing.assert_almost_equal(x2d_true, x2d_est, verbose=True)


def reverse_offset(x2d, M_offset):
    """
    A typical pipeline of projecting 3d points to 2d is:
    P_2d = M_offset @ divide_by_z() @ M_intrinsic @ P_3d
         = M_offset @ P_2d'

    This function calculate P_2d' given P_2d and M_offset

    Args:
        x2d: (n, 2), above P_2d
        M_offset: (3, 3)

    Returns: (n, 2)

    """
    M_offset_inv = np.linalg.inv(M_offset)
    x2d_h = to_homo_nx(x2d)
    x2d_prim_h = x2d_h @ M_offset_inv.T
    return from_home_nx(x2d_prim_h)


""" Change of coordinate system. """

def points_opencv_to_opengl(pts):
    """
    Args:
        (n, 3)

    Returns:
        (n, 3)
    """
    pts = pts.copy()
    pts[:, 1] = - pts[:, 1]
    return pts

