""" A helper class that manages camera parameters 

Ref on crop and resize:
https://github.com/BerkeleyAutomation/perception/blob/0.0.1/perception/camera_intrinsics.py#L176-#L236
"""

import numpy as np


class CameraManager:

    """ 
    By default, the parameter is in 
    conventional non-NDC representation.

    use 

    ```fx, fy, cx, cy = self.to_ndc()```
    or 
    ```fx, fy, cx, cy = self.to_nr(orig_size)```

    to convert camera parameters to pytorch3d's NDC or neural_renderer's 
    representation.

    """
    
    def __init__(self,
                 fx,
                 fy,
                 cx,
                 cy,
                 img_h,
                 img_w,
                 in_ndc=False):
        """

        Args:
            in_ndc (bool): 
                If True, will assume {fx,fy,cx,cy} are in ndc format.
            
        """
        if in_ndc:
            half_h = img_h / 2
            half_w = img_w / 2
            fx = fx * half_w
            fy = fy * half_h
            cx = half_w * (cx + 1)  # W/2 * cx + W/2
            cy = half_h * (cy + 1)

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.img_h = int(img_h)
        self.img_w = int(img_w)

    def __repr__(self):
        return f"CameraManager\n K (non-NDC) = \n {self.get_K()}"

    def get_K(self, in_ndc=True):
        """ Returns: (3, 3) """
        K = np.float32([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        return K

    def unpack(self):
        return self.fx, self.fy, self.cx, self.cy, self.img_h, self.img_w

    def to_ndc(self):
        half_h, half_w = self.img_h / 2, self.img_w / 2
        fx, fy = self.fx / half_w, self.fy / half_h
        cx, cy = self.cx/half_w - 1, self.cy/half_h - 1
        return fx, fy, cx, cy, self.img_h, self.img_w

    def to_nr(self, orig_size):
        """ Convert to neural renderer format. """
        fx, fy = self.fx / self.img_w, self.fy / self.img_h
        cx, cy = self.cx / self.img_w, self.cy / self.img_h
        return fx, fy, cx, cy, self.img_h, self.img_w

    def crop(self, crop_bbox):
        """ 
        Args:
            crop_bbox: (4,) x0y0wh
        """
        x0, y0, w_crop, h_crop = crop_bbox
        crop_center_x = x0 + w_crop/2
        crop_center_y = y0 + h_crop/2
        cx_updated = self.cx + w_crop/2 - crop_center_x
        cy_updated = self.cy + h_crop/2 - crop_center_y
        return CameraManager(
            fx=self.fx, fy=self.fy,
            cx=cx_updated, cy=cy_updated,
            img_h=h_crop, img_w=w_crop,
            in_ndc=False
        )

    def resize(self, new_h, new_w):
        scale_x = new_w / self.img_w
        scale_y = new_h / self.img_h
        fx = scale_x * self.fx
        fy = scale_y * self.fy
        cx = scale_x * self.cx
        cy = scale_y * self.cy
        return CameraManager(
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_h=new_h, img_w=new_w,
            in_ndc=False
        )

    def crop_and_resize(self, crop_bbox, output_size):
        """ 
        Crop a window (changing the center & boundary of scene),
        and resize the output image (implicitly chaning intrinsic matrix)

        Args:
            crop_bbox: (4,) x0y0wh
            output_size: tuple of (new_h, new_w) or int

        Returns:
            CameraManager
        """

        if isinstance(output_size, int):
            new_h = new_w = output_size
        elif len(output_size) == 2:
            new_h, new_w = output_size
        else:
            raise ValueError("output_size not understood.")

        return self.crop(crop_bbox).resize(new_h, new_w)
    
        