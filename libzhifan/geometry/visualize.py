import numpy as np
from typing import Union
import trimesh
import torch
from trimesh.transformations import rotation_matrix
from pytorch3d.structures import Meshes
from libzhifan.geometry.numeric import numpify

_Ry = rotation_matrix(np.pi, [0, 1, 0])  # rotate pi around y-axis


def trimesh_from_pytorch3d(mesh_in: Meshes) -> trimesh.Trimesh:
    return trimesh.Trimesh(
                vertices=numpify(mesh_in.verts_packed()),
                faces=numpify(mesh_in.faces_packed()))


def visualize_hand_object(hand_verts=None,
                          hand_faces=None,
                          hand_poses=None,
                          hand_betas=None,
                          hand_glb_orient=None,
                          hand_transl=None,
                          use_pca=True,
                          obj_verts=None,
                          obj_faces=None,
                          show_axis=True,
                          viewpoint='pytorch3d'):
    """

    If `hand_verts` are supplied, the rest of hand_* will be ignored.
    Otherwise, will try to construct hand_verts using hand_{poses, betas, glb_orient}

    Args:
        hand_verts: (778, 3)
        hand_faces: (F, 3)
        hand_poses: (45, 3)
        hand_betas: (10,)
        obj_verts: (V_o, 3)
        viewpoint: str, one of {'pytorch3d', 'opengl'}

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

    if viewpoint == 'pytorch3d':
        s.apply_transform(_Ry)

    return s


def visualize_mesh(mesh_data,
                   show_axis=True,
                   viewpoint='pytorch3d'):
    """
    Args:
        mesh: pytorch3d Mesh or a list of Meshes
        viewpoint: str, one of {'pytorch3d', 'opengl'}

    Return:
        trimesh.Scene
    """
    s = trimesh.Scene()
    if isinstance(mesh_data, Meshes):
        s.add_geometry(
            trimesh_from_pytorch3d(mesh_data))
    elif isinstance(mesh_data, list):
        assert isinstance(mesh_data[0], Meshes)
        for _m in mesh_data:
            s.add_geometry(trimesh_from_pytorch3d(_m))

    if show_axis:
        axis = trimesh.creation.axis(origin_size=0.01, axis_radius=0.004, axis_length=0.4)
        s.add_geometry(axis)

    if viewpoint == 'pytorch3d':
        s.apply_transform(_Ry)

    return s

