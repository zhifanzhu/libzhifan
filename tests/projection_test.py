import numpy as np
from tokenize import Name
import torch
import pytorch3d
from typing import NamedTuple, Sequence

from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, PointLights,
    MeshRasterizer, SoftPhongShader, MeshRenderer,
    TexturesUV,
)
from pytorch3d.structures import Meshes

from libzhifan.geometry import example_meshes
from libzhifan.epylab import eimshow


def visualize_cube_with_unit_camera():
    """ 
    Note:
    In pytorch3d, `in_ndc=False` means the units are defined in screen space,
    which means the units are pixels;
    However, we normally define the units in world coordinates,
    which hardly can have pixels units, therefore `in_ndc` should be set to True.
    """
    IN_NDC = True  # Setting in_ndc=True is very important
    image_size = (200, 200)
    verts, faces = example_meshes.canonical_cuboids(
        x=0, y=0, z=3,
        w=2, h=2, d=2,
        convention='opencv'
    )
    verts, faces = map(torch.from_numpy, (verts, faces))

    device = 'cuda'
    # R, T = pytorch3d.renderer.look_at_view_transform(-1, 0, 0)
    cameras = PerspectiveCameras(
        focal_length=[(1, 1)],
        principal_point=[(0, 0)],
        in_ndc=IN_NDC,
        # R=R,
        # T=T,
        image_size=[image_size],
    )

    # Equivalently 1:
    # TODO: why their full_projection_matrix differs?
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=90 ,R=R, T=T)  

    
    # Equivalently 2:
    # for K, fx=fy=cx=cy= W/2

    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0, faces_per_pixel=1)
    lights = PointLights(location=[[0, 0, 0]])
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftPhongShader(cameras=cameras, lights=lights)
    renderer = MeshRenderer(
        rasterizer=rasterizer, shader=shader).to(device)

    V, F = verts.shape[0], faces.shape[0]
    cube_map = torch.ones([1, 1, 3]) * torch.Tensor([0.65, 0.74, 0.86])
    verts = verts / 1
    cube_faceuv = torch.zeros([F, 3]).long()
    cube_vertuv = torch.zeros([1, 2])
    cube = Meshes(
        verts=[verts], faces=[faces],
        textures=TexturesUV(
            maps=[cube_map], faces_uvs=[cube_faceuv], verts_uvs=[cube_vertuv])
    ).to(device)

    images = renderer(cube)

    eimshow(images[0, :, :,  :])

    

if __name__ == '__main__':
    visualize_cube_with_unit_camera()