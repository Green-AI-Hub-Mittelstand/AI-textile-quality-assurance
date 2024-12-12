import cv2
from data_transfer import dtos
import numpy as np
import libs.distance_measurements as distance_measurements
import libs.polygon as polylib



def find_top_left(grey_image, boundary_thickness = 100) -> tuple[int, int]:
    '''
    finds top left corner of the material in a grey image

    Parameters:
        grey_image : grey image
        boundary_thickness : boundary thickness used by the cropper, default 100.

    Returns:
       top_left: point of the material edge in pixel coordinates as tuple.

    '''
    height = grey_image.shape[0]
    width = grey_image.shape[1]
    h = width//10
    for i in range(0, height):
        if sum(grey_image[i,boundary_thickness :  boundary_thickness + h])/h > 15:
            break
        
    for j in range(0, width):
        if sum(grey_image[boundary_thickness :  boundary_thickness + h,j])/h > 15:
            break
    top_left = (i,j)
    return top_left

def detect_square_corners_simple(image, boundary_thickness = 100) -> polylib.Square:
    '''
    Get all four corners of the material in a grey image

    Parameters
    ----------
    image : input image
    boundary_thickness: boundary thickness used by the cropper is by default 100 pixel

    Returns
    -------
    Square object with four corner points.

    '''
    #black background and black white conversion
    #image = replace_grey_with_black_hsv(image, lower_grey = np.array([94, 4, 100]), upper_grey = np.array([129, 50, 205]), morph_step=True)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height = grey_image.shape[0]
    width = grey_image.shape[1]
    top_left = find_top_left(grey_image, boundary_thickness)
    #top right
    top_right_help = find_top_left(cv2.rotate(grey_image, cv2.ROTATE_90_COUNTERCLOCKWISE), boundary_thickness)
    top_right = (top_right_help[1], width - top_right_help[0])
    #bottom left    
    bottom_left_help = find_top_left(cv2.rotate(grey_image, cv2.ROTATE_90_CLOCKWISE), boundary_thickness)
    bottom_left = (height - bottom_left_help[1], bottom_left_help[0])
    #bottom_right
    bottom_right_help = find_top_left(cv2.rotate(grey_image, cv2.ROTATE_180), boundary_thickness)
    bottom_right = (height - bottom_right_help[0], width - bottom_right_help[1])
    return polylib.Square(top_left, top_right, bottom_left, bottom_right)

    
def get_corner_min_area_rect(image, boundary_thickness = 100):
    '''
    Get all four corners of the material in a grey image

    Parameters
    ----------
    image : input image
    boundary_thickness: boundary thickness used by the cropper is by default 100 pixel

    Returns
    -------
    corners object with four corner points

    '''
    #black background and black white conversion
    #image = replace_grey_with_black_hsv(image, lower_grey = np.array([94, 4, 100]), upper_grey = np.array([129, 50, 205]), morph_step=True)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cnts, _ = cv2.findContours(grey_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if len(cnts) == 0:
        return 0, 0, [[0, 0], [0, 0], [0, 0], [0, 0]]
 
    # choose max contour
    c = max(cnts, key=cv2.contourArea)
    
    bbox = cv2.minAreaRect(c)
    bbox = cv2.boxPoints(bbox)
    (top_left, top_right, bottom_left, bottom_right) = bbox
    top_left = distance_measurements.switch_axes(top_left)
    top_right = distance_measurements.switch_axes(top_right)
    bottom_left = distance_measurements.switch_axes(bottom_left)
    bottom_right = distance_measurements.switch_axes(bottom_right)
    return polylib.Square(top_left, top_right, bottom_right, bottom_left)

def template_matching(binary_img : cv2.typing.MatLike,
                    config : dtos.TemplateMatchConfig ):
    height = binary_img.shape[0]
    width = binary_img.shape[1]
    label = 'HIGHEST'
    x_0 = round(width * config.rel_top_0)
    x_1 = round(width * config.rel_top_1)
    y_0 = round(height * config.rel_bottom_0)
    y_1 = round(height * config.rel_bottom_1)
    match_method =config.match_type
    img = binary_img[y_0:y_1,x_0:x_1]
    template = np.load(config.path_template)
    mat_weights = None
    if config.path_weights != None:
        mat_weights = np.load(config.path_weights )
    result = cv2.matchTemplate(image=img,templ=template,method=match_method,mask=mat_weights)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        val = min_val
        label = 'LOWEST'
    else:
        top_left = max_loc
        val = max_val
    top_left = (top_left[0] + x_0, top_left[1] + y_0)
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    return dtos.EvalBox(top_left, bottom_right, val, label)
    
def measure_circle_dist_trafo(binary_img : cv2.typing.MatLike,
                         search_area : dtos.EvalBox,
                         variance : tuple,
                         trust_variance,
                         dpi : int) ->dtos.DistanceMeasurement:
    """_summary_

    Args:
        binary_img (cv2.typing.MatLike): The preprocessed binary image.
        search_area (dtos.eval_box): The search area in cv coordinates.
        variance (tuple): The variance of measurements.
        dpi (int): The dpi.

    Returns:
        dtos.distance_measurement: The circle measurements.
    """    
    h = search_area.bottom_right[1] - search_area.top_left[1]
    w = search_area.bottom_right[0] - search_area.top_left[0]
    crop = binary_img[search_area.top_left[1]:search_area.top_left[1] + h,search_area.top_left[0]: search_area.top_left[0] + w]


    distance_transform = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    min_val, radiant, min_loc, center_point = cv2.minMaxLoc(distance_transform)
    c_point = (center_point[0] + search_area.top_left[0], center_point[1] + search_area.top_left[1])
    circle_point_1 = (int(c_point[0] - radiant),c_point[1])
    measurement = dtos.DistanceMeasurement('CIRCLE_CENTER_BOUNDARY',c_point,circle_point_1,distance_measurements.distance_pixel_to_mm(radiant,dpi),variance,trust_variance)
    return measurement