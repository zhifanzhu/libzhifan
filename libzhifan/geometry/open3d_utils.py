from typing import List, Union
import numpy as np
from open3d.visualization import rendering
import PIL

from libzhifan.geometry import CameraManager, SimpleMesh


""" Implementing open3d equivalence of projection.py """

class RenderContext(object):
    """ 
    Define a Context Manager, because open3d allows at most one renderer 
    Warning: Do instantiate low-level OffcreeenRenderer() when using this
    """
    def __init__(self, width, height):
        self._render = rendering.OffscreenRenderer(width, height)

    def __enter__(self):
        """ Can't return self.render, to ensure proper releasing """
        return self

    def __exit__(self, type, value, traceback):
        del self._render

    def capture_rgba(self):
        img = np.asarray(self._render.render_to_image())
        depth = np.asarray(self._render.render_to_depth_image())
        ret = np.zeros([img.shape[0], img.shape[1], 4], dtype=np.uint8)
        ret[:, :, :3] = img
        ret[:, :, 3][depth<0.99] = 255
        return ret

    def scene_add(self, name, geom, mat):
        if self._render.scene.has_geometry(name):
            return
        self._render.scene.add_geometry(name, geom, mat)
    
    def render_by_camera(self, 
                         mesh_data: List[SimpleMesh], 
                         camera: CameraManager,
                         in_coor_sys: str,
                         # image=None,
                         **kwargs) -> np.ndarray:
        """
        Args:
            in_coor_sys: 'pytorch3d' or 'open3d'. 
                See projection.py for coordinate system convention
        """
        # fx, fy, cx, cy, img_h, img_w = camera.unpack()
        # assert method.get('in_ndc', False) == False, "in_ndc Must be False for CamaraManager"

        scene_transform = np.eye(4)
        if in_coor_sys == 'pytorch3d':
            R_pth_to_gl = np.float32([
                [-1, 0, 0, 0],
                [0,  1, 0, 0],
                [0,  0, -1, 0],
                [0,  0, 0, 1]])  # this is equiv to R_y_180
            scene_transform = R_pth_to_gl

        # cam = CameraManager(fx=100, fy=100, cx=100, cy=100, img_h=200, img_w=200)
        intrinsic = camera.get_K()

        near, far = 0.0, 1e9
        self._render.scene.camera.set_projection(
            intrinsic, near, far, camera.img_h, camera.img_w) # intrinsic_4x4, near, far, height, width

        self._render.scene.clear_geometry()
        background_color = [1, 1, 1, 1.0]
        self._render.scene.set_background(background_color)

        # m = _shapes.cube3.as_open3d
        # mat = get_material('red', shader='defaultLitTransparency')
        mat = get_material('red', shader='defaultLit')
        for i, mesh in enumerate(mesh_data):
            mesh = mesh.apply_transform(scene_transform)
            m = mesh.as_open3d
            m.compute_triangle_normals()
            self.scene_add(f'm{i}', m, mat)
        # cube3_gl = _shapes.cube3.apply_transform(R_pth_to_gl)
        # m = cube3_gl.as_open3d
        # m.compute_triangle_normals()
        # mat = open3d_utils.get_material('red')

        # self.render.scene.add_geometry('m1', m, mat)
        img = np.asarray(self._render.render_to_image())
        depth = np.asarray(self._render.render_to_depth_image())
        # ret = np.zeros([img.shape[0], img.shape[1], 4], dtype=np.uint8)
        # ret[:, :, :3] = img
        # ret[:, :, 3][depth<0.99] = 255
        return img#, depth
        # return self.capture_rgba()


def get_material(color, shader="defaultUnlit", alpha=1.0, point_size=1.0) -> rendering.MaterialRecord:
    """
    Args:
        shader: e.g.
            'defaultUnlit', 'defaultLit', 'depth', 'normal', 
            'defaultLitTransparency', 'defaultUnlitTransparency' (buggy)
            see Open3D: cpp/open3d/visualization/rendering/filament/FilamentScene.cpp#L1109
        alpha: float. 0.0 (transparent) to 1.0 (opaque)
        point_size: float. Not unused unless PointCloud
    
    Returns:
        (H, W, 4) RGBA image
    """
    import PIL.ImageDraw as ImageDraw  # needed for getrgb()
    material = rendering.MaterialRecord()
    material.shader = shader
    if isinstance(color, str):
        color_rgb = PIL.ImageColor.getrgb(color)  # tuple
        color_rgb = [c / 255.0 for c in color_rgb]
        material.base_color = color_rgb + [alpha]
    else:
        material.base_color = list(color) + [alpha]
    material.point_size = point_size
    return material


""" Obtain Viewpoint from Open3D GUI """
def parse_o3d_gui_view_status(status: dict, render: rendering.OffscreenRenderer):
    """ Parse open3d GUI's view status and convert to OffscreenRenderer format.
    This will do the normalisation of front and compute eye vector (updated version of front)


    Args:
        status: Ctrl-C output from Open3D GUI
        render: OffscreenRenderer
    Output:
       params for render.setup_camera(fov, lookat, eye, up)
    """
    cam_info = status['trajectory'][0]
    fov = cam_info['field_of_view']
    lookat = np.asarray(cam_info['lookat'])
    front = np.asarray(cam_info['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(cam_info['up'])
    zoom = cam_info['zoom']
    """
    See Open3D/cpp/open3d/visualization/visualizer/ViewControl.cpp#L243:
        void ViewControl::SetProjectionParameters()
    """
    right = np.cross(up, front) / np.linalg.norm(np.cross(up, front))
    view_ratio = zoom * render.scene.bounding_box.get_max_extent()
    distance = view_ratio / np.tan(fov * 0.5 / 180.0 * np.pi)
    eye = lookat + front * distance
    return fov, lookat, eye, up


def set_offscreen_as_gui(render: rendering.OffscreenRenderer, status: dict):
    """ Set offscreen renderer as GUI's view status
    """
    fov, lookat, eye, up = parse_o3d_gui_view_status(status, render)
    render.setup_camera(fov, lookat, eye, up)
