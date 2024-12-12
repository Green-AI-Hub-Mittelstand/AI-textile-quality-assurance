"""
Created on Mon Sept 16 15:01:00 2024

@author: Kai Krämer
"""

import cv2 as cv
import numpy as np
import numpy.typing as npt
import typing

class BoundingBoxes(object):
    def __init__(self,
                 top_left_cv : tuple[int,int],
                 bottom_right_cv : tuple[int,int]) -> None:
        super().__init__()
        self.top_left = top_left_cv
        self.bottom_right = bottom_right_cv
    
    def get_top_left_px(self)->tuple[int,int]:
        return (self.top_left[1],self.top_left[0])
    def get_bottom_left(self)->tuple[int,int]:
       return (self.bottom_right[1],self.bottom_right[0])
    def get_height(self):
        return self.bottom_right[1] - self.top_left[1]
    def get_width(self):
        return self.bottom_right[0] - self.top_left[0]
       

class OffsetImage(BoundingBoxes):
    '''
    Store a cropped image with offset in the original image.
    '''
    def __init__(self, offset_px : tuple, img : cv.typing.MatLike) -> None:
        top_left = (offset_px[1],offset_px[0])
        bottom_right = (offset_px[1] + img.shape[1],offset_px[0] + img.shape[0])
        super().__init__(top_left,bottom_right)
        self.img = img
        self.offset = offset_px
        

class BoxFeature(BoundingBoxes):
    '''
    Store a cropped image features.
    '''
    def __init__(self, top_left, bottom_right, feature : npt.ArrayLike ) -> None:
        super().__init__(top_left,bottom_right)
        self.feature = feature

class EvalBox(BoundingBoxes):
    '''
    The bounding box coordinates with precision value.
    '''
    def __init__(self,
                 top_left : tuple,
                 bottom_right : tuple,
                 precision: float,
                 label: str) -> None:
        super().__init__(top_left,bottom_right)
        self.precision = precision
        self.label = label

    
class DistanceMeasurement(object):
    '''
    distance in mm of two points in cv coordinates and label if the distance is in the allowed range (is_ok)
    '''
    
    def __init__(self, name: str, 
                 p1_cv : tuple[int, int], 
                 p2_cv : tuple[int, int], 
                 distance : float,
                 variance : tuple[float, float],
                 trust_variance: tuple[float,float] = (0.0,10000.0),
                 has_ground_trust: bool = True ):
        self.name = name
        self.p_1 = p1_cv
        self.p_2 = p2_cv
        self.distance = distance
        self.variance = variance
        self.is_ok = bool(variance[0] <= distance <= variance[1])
        self.is_trustful = bool(has_ground_trust 
                                and bool(variance[0]*trust_variance[0] <= distance <= variance[1]*trust_variance[1]))
        
    def toJSON(self):
        return {'name': self.name, 
                'p_1' : self.p_1,
                'p_2' : self.p_2,
                'distance' : self.distance,
                'variance': self.variance,
                'is_ok' : self.is_ok}
    
class TemplateMatchConfig(object):
    def __init__(self,
                 path_template : str,
                 path_weights : typing.Optional[str] = None,
                 rel_x_0 : float = 0.0,
                 rel_x_1 : float = 1.0,
                 rel_y_0 : float = 0.0,
                 rel_y_1 : float = 1.0,
                 match_type : int = cv.TM_CCOEFF) -> None:
        self.path_template = path_template
        self.path_weights = path_weights
        self.rel_top_0 = rel_x_0
        self.rel_top_1 = rel_x_1
        self.rel_bottom_0 = rel_y_0
        self.rel_bottom_1 = rel_y_1
        self.match_type =  match_type
    
    def from_json(json : dict[str,typing.Any]):
        return TemplateMatchConfig(json['path_template'],
                                     json.get('path_weights',None),
                                     json.get('rel_x_0',0.0),
                                     json.get('rel_x_1',1.0),
                                     json.get('rel_y_0',0.0),
                                     json.get('rel_y_1',1.0),
                                     json.get('match_type',cv.TM_CCOEFF)
                                     )