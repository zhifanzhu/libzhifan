from typing import Tuple
import torch
import torch.nn as nn

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer
)


class InstanceIDRenderer(nn.Module):
    def __init__(self, 
                 cameras: CamerasBase, 
                 image_size: Tuple[int],
                 blur_radius=1e-7,
                 max_faces_per_bin=None,
                 bin_size=None) -> None:
        super().__init__()
        self.image_size = image_size
        raster_settings = RasterizationSettings(
            image_size=image_size, blur_radius=blur_radius, 
            faces_per_pixel=1, max_faces_per_bin=max_faces_per_bin,
            bin_size=bin_size)
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    def to(self, device):
        self.rasterizer.to(device)
        return self

    def forward(self, 
                meshes: list, 
                mesh_to_id: list = None, 
                **kwargs) -> torch.Tensor:
        """ Render meshes in a single scene as instance id.

        Args:
            meshes: list. 
            mesh_to_id: Assign instance id to each mesh.
                e.g. mesh_to_id = [1, 2, 5].
                if None, mesh will be assigned to id [1, 2, ...]

        Returns:
            (H, W) int32
        """
        assert type(meshes) == list, "Must be a list to assign valid instance id!"
        num_faces = [len(v.faces_packed()) for v in meshes]
        faceid_to_id = [torch.ones(1)*0]
        if mesh_to_id is not None:
            assert len(mesh_to_id) == len(num_faces), "Mismatch between mesh_to_id and meshes!"
            for i,f in enumerate(num_faces):
                faceid_to_id.append(torch.ones(f)*mesh_to_id[i])
        else:
            for i,f in enumerate(num_faces):
                faceid_to_id.append(torch.ones(f)*(i+1))
        faceid_to_id = torch.cat(faceid_to_id)  # face-0 for bg, actual faces starts from 1
        faceid_to_id = faceid_to_id.int()

        mesh_joined = join_meshes_as_scene(meshes=meshes)
        fragments = self.rasterizer(mesh_joined, **kwargs)
        instance_id_mask = faceid_to_id[fragments.pix_to_face + 1]  # 0 for bg
        return instance_id_mask.view(*self.image_size)
        