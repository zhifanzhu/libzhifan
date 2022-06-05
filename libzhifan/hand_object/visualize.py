import numpy as np
import trimesh
import torch

from libzhifan.geometry.coor_utils import nptify


def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().squeeze().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor


def visualize_hand_object(hand_verts=None,
                          hand_faces=None,
                          hand_poses=None,
                          hand_betas=None,
                          hand_glb_orient=None,
                          hand_transl=None,
                          use_pca=True,
                          obj_verts=None,
                          obj_faces=None,
                          show_axis=True):
    """

    If `hand_verts` are supplied, the rest of hand_* will be ignored.
    Otherwise, will try to construct hand_verts using hand_{poses, betas, glb_orient}

    Args:
        hand_verts: (778, 3)
        hand_faces: (F, 3)
        hand_poses: (45, 3)
        hand_betas: (10,)
        obj_verts: (V_o, 3)

    Returns:
        trimesh.Scene
    """
    s = trimesh.Scene()

    if hand_verts is not None:
        assert hand_faces is not None
        hand_mesh = trimesh.Trimesh(hand_verts, hand_faces)
        s.add_geometry(hand_mesh)
    elif hand_poses is not None:
        assert hand_faces is not None

    if obj_verts is not None:
        if obj_faces is not None:
            obj = trimesh.Trimesh(obj_verts, obj_faces)
        else:
            obj = trimesh.points.PointCloud(
                obj_verts, colors=np.tile(np.array([0, 0, 0, 1]), (len(obj_verts), 1))
            )
        s.add_geometry(obj)
    
    if show_axis:
        axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.004, axis_length=0.4)
        s.add_geometry(axis)
    
    return s


def visualize_mesh(mesh, show_axis=True):
    """ 
    Args:
        mesh: pytorch3d Mesh

    Return:
        trimesh.Scene
    """
    s = trimesh.Scene()
    s.add_geometry(
        trimesh.Trimesh(
            vertices=numpify(mesh.verts_packed()),
            faces=numpify(mesh.faces_packed()),
        )
    )

    if show_axis:
        axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.004, axis_length=0.4)
        s.add_geometry(axis)
    
    return s
    
