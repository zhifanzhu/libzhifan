import numpy as np
import cv2
from PIL import Image
from .def_palette import default_palette


def overlay_mask(image, 
                 mask, 
                 mask_ids: list = None,
                 blend_mode: str = 'bright',
                 weight_mask=0.5) -> np.ndarray:
    """  
    Args:
        image: (H, W, 3) 
        mask: one of
            - PIL.Image that is colour already coded
            - (H, W) of categorical numbers {0, 1, 2, ...}
        mask_ids: Selected mask ids for overlaying.

        blend_mode:
            - 'bright': 
                avoid output unmasked regions in image being too dim.

            - 'default': simply addWeighted(image, mask),
                the unmasked region maybe todim.

            - 'highlight_only': highlight foreground by 
                making background more transparent

            - 'highlight_mask': highlight foreground and apply mask color

        weight_mask: float. 0-1
    """
    if isinstance(mask, Image.Image):
        mask_clr = np.asarray(mask.convert('RGB'))
        mask_int = np.asarray(mask.convert('P'))
    elif isinstance(mask, np.ndarray):
        mask_int = mask
        mask_clr = default_palette[mask_int]
    else:
        raise ValueError(f"Unknown type {type(mask)=}")

    if mask_ids is not None:
        mask_int_list = np.unique(mask_int)
        for m_id in mask_int_list:
            if m_id not in mask_ids:
                mask_clr[mask_int == m_id] = [0, 0, 0]
                mask_int[mask_int == m_id] = 0
    
    if blend_mode == 'bright':
        ind_bg = mask_int == 0
        mask_clr[ind_bg] = image[ind_bg]
        mask_rgb = mask_clr

    elif blend_mode == 'default':
        mask_rgb = mask_clr

    elif blend_mode == 'highlight_only':
        mask_rgb = np.ones_like(image) * 255
        ind_fg = mask_int != 0
        mask_rgb[ind_fg] = image[ind_fg]

    elif blend_mode == 'highlight_mask':
        mask_rgb = np.ones_like(image) * 255
        ind_fg = mask_int != 0
        mask_rgb[ind_fg] = mask_clr[ind_fg]
    else:
        raise ValueError(f"Unknown {blend_mode=}")

    weight_img = 1 - weight_mask
    overlay = cv2.addWeighted(image, weight_img, mask_rgb, weight_mask, 1.0)
    return overlay


""" Alias of overlay_mask() """
blend_image_mask = overlay_mask


""" Used by heatmap_on_image() """
def _rescale_img_pixel(img):
    """ Rescale pixel value from [-1, 1] float to [0, 255] uint8.

    Args:
        img: ndarray, range from [-1, 1] float

    Return:
        ndarray, [0, 255] uint8
    """
    return np.uint8((img + 1) * 255. / 2)


def heatmap_on_image(heatmap, image, weight_hm=0.5, weight_img=None):
    """ Draw heatmap on image.
        output pixel = weight_hm * hm + weight_img * image

    Args:
        heatmap: [H, W] np.float32 image, value range [-1, 1],
            the value of heatmap will be scaled to [0, 255] and convert to uint8
        image: [H, W, 3] np.uint8.
        weight_hm: float
        weight_img: None or float, if None, weight_img = 1 - weight_hm

    Return:
        [H, W, 3]
    """
    img_h, img_w, _ = image.shape
    hm_resize = cv2.resize(heatmap, (img_w, img_h))
    hm_resize = _rescale_img_pixel(hm_resize)
    hm_color = cv2.applyColorMap(hm_resize, cv2.COLORMAP_JET)
    if weight_img is None:
        weight_img = 1 - weight_hm
    heatmapped_img = cv2.addWeighted(hm_color, weight_hm, image, weight_img, 0)
    return heatmapped_img