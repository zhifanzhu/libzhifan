import numpy as np

from .coor_utils import points_opencv_to_opengl


def pivot_simplex(
        x=0,
        y=0,
        z=0,
        xlen=1,
        ylen=1.5,
        zlen=2):
    """
    Generate a simplex in 3D.

    Returns:
        vertices: (4, 3)
        faces: (4, 3)
    """
    pts = np.float32([
        [0, 0, 0],
        [xlen, 0, 0],
        [0, ylen, 0],
        [0, 0, zlen]
    ]) + np.float32([x, y, z])
    faces = np.int32([
        [0, 2, 1],
        [0, 3, 2],
        [0, 1, 3],
        [1, 2, 3]
    ])
    return pts, faces


def canonical_cuboids(
        x=0,
        y=0,
        z=0,
        w=1,
        h=1,
        d=1,
        convention='opengl'):
    """
    Generate a Cuboid/Cube.

         4 +-----------------+ 5
          /     TOP         /|
         /                 / |
      0 +-----------------+ 1|
        |      FRONT      |  |
        |                 |  |
        |  x <--+         |  |
        |       |         |  |
        |       v         |  + 6
        |        y        | /
        |                 |/
      3 +-----------------+ 2


    Args:
        convention: one of {'opencv', 'opengl'}  TODO(check this)

    Returns:
        vertices: (8, 3)
        faces: (12, 3)
    """
    # NOTE: following use OpenCV's coordinate convention,

    # X axis point to the right
    right = x + w / 2.0
    _left = x - w / 2.0
    # Y axis point downward
    top = y - h / 2.0
    bottom = y + h / 2.0
    # Z axis point outward
    front = z + d / 2.0
    rear = z - d / 2.0

    # List of 8 vertices of the box
    pts = np.float32([
        [_left, top, front],     # Front Top Left
        [right, top, front],     # Front Top Right
        [right, bottom, front],  # Front Bottom Right
        [_left, bottom, front],  # Front Bottom Left
        [_left, top, rear],      # Rear Top Left
        [right, top, rear],      # Rear Top Right
        [right, bottom, rear],   # Rear Bottom Right
        [_left, bottom, rear],   # Rear Bottom Left
    ])
    faces = np.int32([
        [1, 3, 2],
        [0, 3, 1],
        [2, 6, 5],
        [5, 1, 2],
        [6, 7, 4],
        [4, 5, 6],
        [7, 3, 0],
        [0, 4, 7],
        [0, 1, 5],
        [5, 4, 0],
        [7, 6, 2],
        [2, 3, 7]
    ])
    if convention == 'opencv':
        pass
    elif convention == 'opengl':
        pts = points_opencv_to_opengl(pts)

    return pts, faces


def canonical_pivots(
    width=1,
    height=1,
    depth=1):
    """ a.k.a XYZ-axes

    Returns: (4, 3)
    """
    pts = np.float32([
        [0, 0, 0],
        [width, 0, 0],   # X-axis
        [0, height, 0],  # Y-axis
        [0, 0, depth],   # Z-axis
    ])
    return pts
