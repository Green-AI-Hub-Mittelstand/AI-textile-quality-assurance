import cv2 as cv
import numpy as np
import random2 as rand

def build_shape(
        point : tuple [int,int],
        width : int,
        height : int,
        line_dense : int,
        variation : int,
        shape):
    """

    Build an mass distorted random contour from an rectangle.

    Arguments:

    point - the corner of the rectangle
    width - the width of the rectangle
    height - the height of the rectangle
    line_dense - the divisor of the rectangle
    variation - the variation of the point oof the rectangle
    shape - (width,height) of the image

    """
    horizontal = max(width//line_dense,1)
    vertical = max(height//line_dense,1)
    top_left = point
    top_right = (point[0],point[1] + width)
    bottom_right = (point[0] + height,point[1] + width)
    bottom_left = (point[0] + height, point[1])

    p_line_vertical = range(top_left[1] + vertical,top_right[1],vertical)
    p_line_horizontal = range(top_left[0] + horizontal,bottom_left[0],horizontal)
    segments : list[tuple[int,int]] = []
    segments.append([top_left[0] +  rand.randint(-variation,0),top_left[1] + rand.randint(-variation,0)])

    for l in p_line_vertical:
        segments.append([top_left[0] + rand.randint(-variation,0),l])

    # append top right corner
    segments.append([top_right[0] + rand.randint(-variation,0), top_right[1] + rand.randint(0,variation)])

    for l in p_line_horizontal:
        segments.append([l, top_right[1] + rand.randint(0,variation)])

    # append bottom right corner
    segments.append([bottom_right[0] + rand.randint(0,variation),bottom_right[1] + rand.randint(0,variation)])
    for l in reversed(p_line_vertical):
        segments.append([bottom_right[0] + rand.randint(0,variation),l])

    # append bottom left corner
    segments.append([bottom_left[0] + rand.randint(0,variation), bottom_left[1] + rand.randint(-variation,0)])
    for l in reversed(p_line_horizontal):
        segments.append([l, bottom_left[1] + rand.randint(-variation,0)])
    cv_segments = []
    for p in segments:
        cv_segments.append((p[1],p[0]))
    contour = [np.array(cv_segments, np.int32 )]
    zeros = np.zeros(shape,np.uint8)

    image = cv.drawContours(image=zeros,contours=contour,contourIdx=0, color= 255,thickness=cv.FILLED)
    # blur the contour
    for i in range(0,rand.randint(1,4)):
        image = cv.GaussianBlur(image,(5,5),1.)
        image = cv.GaussianBlur(image,(7,7),(7.-1.)/4.)
        image = cv.GaussianBlur(image,(9,9),(9-1)/4.)

    th, im_gray = cv.threshold(image, 255/2,255.0,cv.THRESH_BINARY)

    return im_gray


def fill_distorted_mass(
        texture : cv.typing.MatLike,
        image : cv.typing.MatLike,
        boundary_dist : int = 120 ):
    img_height = image.shape[0]
    img_width = image.shape[1]

    max_h = texture.shape[0] - img_height - 1
    max_w = texture.shape[1] - img_width - 1
    i = rand.randint(0,max_h)
    j = rand.randint(0,max_w)
    tex_crop = texture[i:i+img_height,j:j+img_width]
        
    ph = rand.randint(boundary_dist,img_height - boundary_dist - 1)
    pw = rand.randint(boundary_dist,img_width - boundary_dist - 1)
    b_point = (ph,pw)

    err_shape = build_shape(b_point,rand.randint(50,120),rand.randint(50,120),rand.randint(5,8),120,(img_height,img_width))
    _, max_val, _, _ = cv.minMaxLoc(err_shape)

    for c_i in range(0,img_height):
        for c_j in range(0,img_width):
            if err_shape[c_i,c_j] > 1:
                image[c_i,c_j] = tex_crop[c_i,c_j]
