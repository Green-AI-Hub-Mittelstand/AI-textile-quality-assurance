"""
Created on Mon Sept 16 14:53:00 2024

@author: Kai  Krämer
"""

import numpy as np
import cv2 as cv
import gudhi as gd
from data_transfer.dtos import OffsetImage, BoxFeature


def get_top_border(gray_image : cv.typing.MatLike, thickness : int = 250):
    '''
    Crop the top boundaries of the image.

    Parameter:
        img: The image to process.
        thickness: The thickness of the croped boundary

    Returns:
        image: The cropped top boundary with offset.
    '''
    #find start of the top border,
    for i in range(0, gray_image.shape[0]):
        if sum(gray_image[i,:])/gray_image.shape[0] < 240:
            break        
    if i > thickness//5:
        x = i-thickness//5
        y = i+4*(thickness//5)
        output = gray_image[x:y,:]
    else:
        x = 0
        y = thickness
        output = gray_image[x:y,:]
    return (x,y),output


def get_bottom_border(gray_image : cv.typing.MatLike, thickness : int = 250):
    '''
    Crop the bottom boundaries of the image.

    Parameter:
        img: The image to process.
        thickness: The thickness of the croped boundary

    Returns:
        image: The cropped bottom boundary with offset.
    '''
    for i in reversed(range(gray_image.shape[0])):
        if sum(gray_image[i,:])/gray_image.shape[0] < 240:
            break
    if gray_image.shape[0] - i > thickness//5:
        x = i-4*(thickness//5)
        y = i+(thickness//5)
        output = gray_image[x:y,:]
    else:
        x = -thickness
        y = 0
        output = gray_image[-thickness:,:]
    return (x,y),output

def _get_boundaries(gray_img : cv.typing.MatLike, thickness : int = 250) -> list[OffsetImage]:
    '''
    Crop the boundaries of the image.

    Parameter:
        img: The image to process.
        thicknes: The thicknes of the croped boundaries

    Returns:
        list[offset_image]: The cropped boundaries with offset.
    '''
    shape = gray_img.shape
    top = get_top_border(gray_img, thickness)
    bottom = get_bottom_border(gray_img, thickness)
    
    boundaries = [
    OffsetImage((0,0), top[1]),  # append top
    OffsetImage((shape[0]-thickness,0), bottom[1])]  # append bottom
    return boundaries

def feature_extraction(img : cv.typing.MatLike)->list[BoxFeature]:
    '''
    Converts the image to an gray scale image.
    Crops the boundareis and computes the persitence diagram of them, to  convert them to features vectors.

    Parmeter:
        img: The image to analyse.
    Returns:
        vec: The homology feature vector.
    '''
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_image = cv.bitwise_not(gray_image)
    boundaries = _get_boundaries(gray_image)
    features : list[BoxFeature] = []
    for b in boundaries:
        persitence = compute_persistence(b.img)
        persitence_diag = persistence_diagram_to_image(persistence=persitence)
        persitence_features = persitence_diag.flatten()
        features.append(BoxFeature(top_left=b.top_left,bottom_right=b.bottom_right,feature=persitence_features))

    return features

def compute_persistence(gray_image : cv.typing.MatLike) -> any:
    '''
    Compute persistent homology.

    Attention:
        The Shape of the image must be in fortran order!

    Parameter:
        gray_image: The gray scale image.

    Returns:
        cc: The persitence model
    '''
    #dim = (gray_image.shape[1], gray_image.shape[0])
    cc = gd.CubicalComplex(dimensions=gray_image.shape, top_dimensional_cells=gray_image.flatten(order='F'))
    persistence = cc.persistence()
    return persistence

 
def persistence_diagram_to_image(
        persistence : list[tuple[int,tuple[float,float]]],
        resolution : int = 50) -> tuple[bool,cv.typing.MatLike]:
    '''
    Compute the persitence diagram.

    Parameter:
        persitence: The persitence of the image.
        resolution: The resolution of the output feature image.
    '''    
    max_value = np.max([point[1][1] for point in persistence if point[1][1] < np.inf])
    img = np.zeros((resolution, resolution))
    for dimension, (birth, death) in persistence:
        if death < np.inf:
            x = int((birth / max_value) * (resolution - 1))
            y = int((death / max_value) * (resolution - 1))
            img[y, x] += 1

    return img



