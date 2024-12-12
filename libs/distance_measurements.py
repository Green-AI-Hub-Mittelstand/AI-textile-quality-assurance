import cv2
import numpy as np
from data_transfer import dtos as dtos


def get_distance(p_1 : tuple[int, int], p_2 : tuple[int, int], dpi : int = -1):
    '''
    get distance between point 1 and point 2

    '''
    output  = float(np.linalg.norm(np.array(p_1) - np.array(p_2), ord=2))
    if dpi > 0:
        output = distance_pixel_to_mm(output,dpi)
    return output

def distance_mm_to_px(distance_mm : float,dpi : int = 600):
    return (distance_mm*dpi)/25.4

def distance_pixel_to_mm(distance_px: float, dpi = 600) -> float:
    '''
    transfers pixel distance to mm distance in reality.

    Parameters
    ----------
        distance_px : number pixel of the distance
        dpi: dpi of teh scanner, default 600.

    Returns
    -------
        distance_mm

    '''
    distance_mm = (distance_px / dpi) * 25.4
    return distance_mm


def switch_axes(point):
    return int(point[1]),int(point[0])









    
    
    

