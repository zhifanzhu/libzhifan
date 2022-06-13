from typing import Union

import numpy as np
import torch

from trimesh import PointCloud, Trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from .numeric import numpize


def _drop_dim0(tensor):
    if not hasattr(tensor, 'shape'):
        return tensor
    if len(tensor.shape) == 2:
        return tensor
    elif len(tensor.shape) == 3 and tensor.shape[0] == 1:
        return tensor[0]
    else:
        raise ValueError("Input shape must be (1, ?, ?) or (?, ?)")


class SimpleMesh(Trimesh):

    """
    A unoptimized wrapper class that simplifies the conversion to pytorch3d.Meshes.

    When the underlying pytorch3d.Meshes is need,
    one should call self.synced_mesh to generate the pytorch3d.Meshes dynamically.

    self.synced_mesh will always be on CUDA.

    """

    def __init__(self,
                 verts: Union[np.ndarray, torch.Tensor],
                 faces: Union[np.ndarray, torch.Tensor],
                 copy_orig=True):
        """
        Args:
            verts: (V, 3) float32
            faces: (F, 3) int
            copy_orig: copy original input
                so self.synced_mesh holds original input
                We have to clone original input, as trimesh may alter `verts``
        """
        if copy_orig:
            self.orig_verts = torch.as_tensor(
                _drop_dim0(verts)).clone()
            self.orig_faces = torch.as_tensor(
                _drop_dim0(faces)).clone()
        else:
            self.orig_verts = self.orig_faces = None

        verts = numpize(_drop_dim0(verts))
        faces = numpize(_drop_dim0(faces))

        super(SimpleMesh, self).__init__(
            vertices=verts,
            faces=faces)

    def copy(self):
        copied = super(SimpleMesh, self).copy()
        return SimpleMesh(copied.vertices, copied.faces)

    @property
    def synced_mesh(self):
        device = 'cuda'
        if self.orig_verts is None:
            verts = torch.as_tensor(self.vertices, device=device, dtype=torch.float32)
            faces = torch.as_tensor(self.faces, device=device)
        else:
            verts = torch.as_tensor(self.orig_verts, device=device, dtype=torch.float32)
            faces = torch.as_tensor(self.orig_faces, device=device)
        verts_rgb = torch.ones_like(verts) * \
            torch.as_tensor([0.65, 0.74, 0.86], device=device)
        textures = TexturesVertex(verts_features=verts_rgb[None].to(device))
        return Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )

    def apply_translation_(self, translation):
        """ Copied from Trimesh

        In-place translate the current mesh.

        Parameters
        ----------
        translation : (3,) float
          Translation in XYZ
        """
        translation = np.asanyarray(translation, dtype=np.float64)
        if translation.shape == (2,):
            # create a planar matrix if we were passed a 2D offset
            return self.apply_transform(
                tf.planar_matrix(offset=translation))
        elif translation.shape != (3,):
            raise ValueError('Translation must be (3,) or (2,)!')

        # manually create a translation matrix
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        return self.apply_transform(matrix)

    def apply_translation(self, translation):
        """
        Returns a copy of translated mesh.
        """
        return self.copy().apply_translation_(translation)


# class SimplePCD(PointCloud):

#     """
#     A wrapper for PointCloud, for easier visualization.
#     """

#     def __init__(self, verts):
#         verts = _drop_dim0(verts)
#         super(SimplePCD, self).__init__(
#             verts,
#             colors=np.tile(np.array([0, 0, 0, 1]), (len(verts), 1))
#         )
