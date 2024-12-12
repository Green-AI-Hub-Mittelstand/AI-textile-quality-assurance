import cv2
from data_transfer import dtos
import copy

def coordinate_transformation_rot_image(image : cv2.typing.MatLike,
                            distance_measurements : list[dtos.DistanceMeasurement],
                            rotation : int):
    """
    Rotates image and measurements

    Args:
        image (cv2.typing.MatLike): The image to rotate.
        boxes (list[dtos.distance_measurement]): The boxes that must be transformed
        rotation (int): cv2 rotation constant.

    Returns:
        tuple[cv2.typing.MatLike,list[dtos.distance_measurement]]:  The rotated image and measurements.
    """    
    cp_img = image.copy()
    cp_boxes = copy.deepcopy(distance_measurements)
    s_height = cp_img.shape[0] - 1
    s_width = cp_img.shape[1] - 1
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        cp_img = cv2.rotate(cp_img,rotation)
        for b in cp_boxes:
             p_1 = b.p_1
             p_2 = b.p_2
             b.p_1 = (s_height - p_1[1], p_1[0])
             b.p_2 = (s_height - p_2[1], p_2[0])
    
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        cp_img = cv2.rotate(cp_img,rotation)
        for b in cp_boxes:
            p_1 = b.p_1
            p_2 = b.p_2
            b.p_1 = (p_1[1],s_width - p_1[0])
            b.p_2 = (p_2[1],s_width - p_2[0])
    elif rotation == cv2.ROTATE_180:
        cp_img = cv2.rotate(cp_img,rotation)
        for b in cp_boxes:
            p_1 = b.p_1
            p_2 = b.p_2
            b.p_1 = (s_width - p_1[0], s_height - p_1[1])
            b.p_2 = (s_width - p_2[0], s_height - p_2[1])
    else:
        pass
    return cp_img, cp_boxes

def coordinate_transformation_mirror_image(image : cv2.typing.MatLike,
                            distance_measurements : list[dtos.DistanceMeasurement],
                            y_axes : bool = True)->tuple[cv2.typing.MatLike,list[dtos.DistanceMeasurement]]:
    """
    Mirror the image and measurements.

    Args:
        image (cv2.typing.MatLike): The image to rotate.
        distance_measurements (list[dtos.distance_measurement]): The measured distances to change.
        y_axes (bool, optional): If is True the image is mirrored at y-axe else x. Defaults to True.

    Returns:
        tuple[cv2.typing.MatLike,list[dtos.distance_measurement]]: The mirrored image and measurements.
    """    
    
    cp_img = image.copy()
    cp_boxes = copy.deepcopy(distance_measurements)
    s_height = cp_img.shape[0] - 1
    s_width = cp_img.shape[1] - 1
    if y_axes:
        cp_img =cv2.flip(cp_img,1)
        for b in cp_boxes:
             p_1 = b.p_1
             p_2 = b.p_2
             b.p_1 = (s_width - p_1[0], p_1[1])
             b.p_2 = (s_width - p_2[0], p_2[1])
    
    else:
        cp_img =cv2.flip(cp_img,0)
        for b in cp_boxes:
            p_1 = b.p_1
            p_2 = b.p_2
            b.p_1 = (p_1[0],s_height - p_1[1])
            b.p_2 = (p_2[0],s_height - p_2[1])
     
    return cp_img, cp_boxes