import unittest

import numpy as np
from libzhifan.geometry import example_meshes
from libzhifan.geometry import projection
from libzhifan.geometry import CameraManager


def project_by_camera(mesh_data, camera):
    fx = camera.fx
    fy = camera.fy
    cx = camera.cx
    cy = camera.cy
    img_h = camera.img_h
    img_w = camera.img_w
    img = projection.perspective_projection(
        mesh_data,
        cam_f=(fx, fy),
        cam_p=(cx, cy),
        method=dict(
            name='pytorch3d',
            in_ndc=False,
        ),
        img_h=img_h,
        img_w=img_w,
    )
    return img

class CameraManagerTest(unittest.TestCase):
    
    def test_crop_and_resize(self):
        H, W = 200, 400
        image_size = (H, W)

        global_cam = CameraManager(
            fx=10, fy=20, cx=0, cy=0, img_h=H, img_w=W,
            in_ndc=True)

        H1, W1 = 200, 200
        local_box_1 = np.asarray([0, 0, H1, W1]) # xywh
        local_cam_1_exp = CameraManager(
            fx=20, fy=20, cx=1, cy=0, img_h=H1, img_w=W1,
            in_ndc=True)

        H2, W2 = 100, 100
        local_box_2 = np.asarray([200, 100, H2, W2]) # xywh
        local_cam_2_exp = CameraManager(
            fx=40, fy=40, cx=-1, cy=-1, img_h=H2, img_w=W2,
            in_ndc=True)

        cube_1 = example_meshes.canonical_cuboids(
            x=0.5, y=0, z=10.25,
            w=0.5, h=0.5, d=0.5,
            convention='pytorch3d'
        )
        cube_2 = example_meshes.canonical_cuboids(
            x=-0.375, y=-0.125, z=10.125,
            w=0.25, h=0.25, d=0.25,
            convention='pytorch3d'
        )

        np.testing.assert_allclose(
            local_cam_1_exp.get_K(),
            global_cam.crop(local_box_1).get_K())
        np.testing.assert_allclose(
            local_cam_2_exp.get_K(),
            global_cam.crop(local_box_2).get_K())
        
        """ image rendered by Local camera 1 
        x=0.5 => x_pix=100
        x=0.75 => x_pix=150 (=50 after flip)

        """

        img_global = project_by_camera(
            [cube_1, cube_2],
            global_cam)
        img_1 = project_by_camera(
            [cube_1, cube_2],
            global_cam.crop(local_box_1))
        img_2 = project_by_camera(
            [cube_1, cube_2],
            global_cam.crop(local_box_2))


if __name__ == '__main__':
    unittest.main()
        