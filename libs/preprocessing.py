import cv2
import copy
import numpy as np
import err_detection.utils.helper as helper


def delete_white_bottom(scanned_image : cv2.typing.MatLike) -> cv2.typing.MatLike:
    '''
    Deletes the bottom white part of the scanned image

    Parameters:
        scanned_image: image from the scanner

    Returns:
        output: image without bottom white part

    '''
    gray_image = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2GRAY)
    i_end = scanned_image.shape[0]
    j_end = scanned_image.shape[1]
    for i in reversed(range(0, i_end)):
        if sum(gray_image[i,:(j_end//50)])/(j_end//50) < 253:
            output = scanned_image[:i,:]
            break
    return output



def image_crop(image, thickness : int = 100) -> cv2.typing.MatLike:
    '''
    Crops the scanned image with a thickness thick border.

    Parameters:
        image: image from the scanner
        thickness: thickness of black edge

    Returns:
        output: cropped image
    '''
    #scaling factor to scale image to work on down for speed up.
    scale_factor = 20
    thickness = thickness//scale_factor    
    image = delete_white_bottom(image)
    #scale image down for speed up
    copy_image = cv2.resize(image, (image.shape[1]//scale_factor, image.shape[0]//scale_factor), interpolation=cv2.INTER_AREA) 
    #convert grey background to black
    copy_image = replace_grey_with_black_hsv(copy_image, lower_grey = np.array([94, 4, 160]), upper_grey = np.array([129, 50, 205])) 
    copy_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY) #convert to black white
    height = copy_image.shape[0]
    width = copy_image.shape[1]
    #search edges by looking at the mean grey value in rows/columns.
    #top border
    top = 0
    for i in range(0, height):
        if sum(copy_image[i,:])/height > 15:
            break
    if i > thickness:
        top = i - thickness
    #bottom border
    bottom = height
    for i in reversed(range(0, height)):
        if sum(copy_image[i,:])/height > 15:
            break
    if height - i > thickness:
        bottom = i + thickness
    #left border
    left = 0
    for j in range(0, width):
        if sum(copy_image[:,j])/width > 15:
            break
    if j > thickness:
        left = j - thickness
    #right border
    right = width
    for j in reversed(range(0, width)):
        if sum(copy_image[:,j])/width > 15:
            break
    if width - j > thickness:
        right = j + thickness
    #rescale the boundaries for the original image
    return image[top*scale_factor:bottom*scale_factor, left*scale_factor:right*scale_factor]

def replace_grey_with_black_hsv(image : cv2.typing.MatLike, 
                                lower_grey : np.array = np.array([95, 5, 100]), 
                                upper_grey : np.array = np.array([125, 47, 203],),
                                morph_step = False) -> cv2.typing.MatLike:
    '''
    convert grey into black by using hsv color range. 

    Parameters:
        image: image from the scanner
        lower_grey: lower bound of hsv color range
        upper_grey: upper bound of hsv color range
        morph_step: can help to denoise the black

    Returns:
        output: image with black instead of grey
    '''
    copy_image = copy.copy(image)
    #BGR2HSV
    hsv_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2HSV)
    # Create a mask that isolates the grey background and erode + delate after
    grey_mask = cv2.inRange(hsv_image, lower_grey, upper_grey)
    #th, grey_mask = cv2.threshold(grey_mask, 255/2,255.0,cv2.THRESH_BINARY)
    if morph_step:
        grey_mask = morph(grey_mask, 0, 1)
    # Create a 3-channel version of the background mask
    background_mask_3channel = cv2.cvtColor(grey_mask, cv2.COLOR_GRAY2BGR)
    # Change the background color to black where the mask is true
    output = np.where(background_mask_3channel != 0, [0, 0, 0], copy_image)
    output = output.astype(np.uint8)
    return output

def morph(im_gray, num_erode = 1, num_dilate = 1):
    """
    
    Morphologic filtering erode and dilate the image.
    
    Arguments:

    im_gray - the image

    num_erode - number iterations of erode

    num_dilate - number iterations of dilate
    
    """
    kernel = np.ones((5, 5), np.uint8)
    im_gray = cv2.erode(im_gray, kernel, iterations=num_erode) 
    im_gray = cv2.dilate(im_gray, kernel, iterations=num_dilate)
    return im_gray

def color_to_binary(img,threash =255/2, inverse= True):
    """
    
    Convert to gray and  create an threashold image.
    
    Arguments:

    img - the image to filter

    threash - the threash

    inverse - if true the gry image is inverted else not

    Returns:
    
    th - threashold

    img - the threshed image

    """
    img_type = cv2.THRESH_BINARY
    if inverse:
        img_type =cv2.THRESH_BINARY_INV
    im_gray = cv2.cvtColor( img,cv2.COLOR_BGR2GRAY )
    th, im_gray = cv2.threshold(im_gray, threash,255.0,img_type)
    return th,im_gray

def preprocessing_resnet50v2(image):
    normalized = image/ 127.5
    normalized -= 1.0
    return normalized

