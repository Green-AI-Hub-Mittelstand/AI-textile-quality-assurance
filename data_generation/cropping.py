import cv2

def split_image(image : cv2.typing.MatLike,
                size = (1024,1024),
                stride = (1024//2,1024//2))->tuple[list[cv2.typing.MatLike],list[tuple[int,int]]]:
    """
    Split the image in patches of given size with an overlapping defined by stride.
    The splitting strategy is boundary based.
    This algorithm crops first the corners, then the boundaries without corners.
    Then the inner of an image is splitted.

    Args:
        image (cv2.typing.MatLike): the image to split.
        size (tuple, optional): the size of patches. Defaults to (1024,1024).
        stride (tuple, optional): the stride. Defaults to (1024//2,1024//2).

    Returns:
        tuple[list[cv2.typing.MatLike],list[tuple[int,int]]]: The cropped patches with left points in pixel coordinates.
    """    
    images,points = crop_corners(image=image,size=size)
    new_img, new_p  = crop_inner_boundaries(image =image,size=size,stride=stride)
    images.extend(new_img)
    points.extend(new_p)
    new_img, new_p = crop_inner_image(image=image,size=size,stride=stride)
    images.extend(new_img)
    points.extend(new_p)
    return images, points

def crop_inner_image(image : cv2.typing.MatLike,
                size = (1024,1024),
                stride = (1024//2,1024//2))->tuple[list[cv2.typing.MatLike],list[tuple[int,int]]]:
    """
    Split the inner of an images in patches.

    Args:
        image (cv2.typing.MatLike): the image.
        size (tuple, optional): the size of patches. Defaults to (1024,1024).
        stride (tuple, optional): the stride of the split. Defaults to (1024//2,1024//2).

    Returns:
        tuple[list[cv2.typing.MatLike],list[tuple[int,int]]]: the returned patches and top left points.
    """    
    
    height = image.shape[0]
    width = image.shape[1]
    points = []
    for i in range(stride[0], height - size[0] - 1, stride[0]):
        for j in range(stride[1], width - size[1] - 1, stride[1]):
            points.append((i,j))
    return crop_image(image=image,size=size,top_left_positions=points),points

def crop_inner_boundaries(image : cv2.typing.MatLike,
                           size : tuple[int,int] = (1024,1024),
                           stride : tuple[int,int] = (1024//2,1024//2))->tuple[list[cv2.typing.MatLike],list[tuple[int,int]]]:
    height = image.shape[0]
    width = image.shape[1]
    steps_h = range(stride[0], height - size[1] - 1, stride[0])
    steps_w = range(stride[1], width - size[1] - 1, stride[1])
    points = []
    for h in steps_h:
        points.extend([(h,0),(h, width -1 - size[1])])
    for w in steps_w:
        points.extend([(0,w),(height -1 - size[1],w)])
    
    return crop_image(image,size,points),points

def crop_corners(image : cv2.typing.MatLike,
                  size = (1024,1024))->tuple[list[cv2.typing.MatLike],list[tuple[int,int]]]:
    """
    Crops the corners of an image.

    Args:
        image (cv2.typing.MatLike): the image.
        size (tuple, optional): the size of the patches. Defaults to (1024,1024).

    Returns:
        list[cv2.typing.MatLike]: the returned corner patches.
    """    
    image = image.copy()
    height = image.shape[0] - 1
    width = image.shape[1] - 1
    tl = (0,0)
    tr = (0,width - size[1])
    br = (height - size[0], width - size[1])
    bl = (height - size[0], 0)
    top_left_positions : list[tuple[int,int]] = [tl,tr,br,bl]
    images = crop_image(image, size, top_left_positions)
    return images,top_left_positions

def crop_image(image: cv2.typing.MatLike,
                size : tuple[int,int],
                top_left_positions : list[tuple[int,int]])->list[cv2.typing.MatLike]:
    """
    Crops the image by top left positions and size. 
    Args:
        image (cv2.typing.MatLike): the image to split.
        size (tuple[int,int]): the size of the cropped patches.
        top_left_positions (tuple[int,int]): the top left position of the cropped patches.

    Returns:
        list[cv2.typing.MatLike]: the cropped patches.
    """    
    images = []
    for p in top_left_positions:
        crop = image[p[0]:p[0] + size[0],p[1]:p[1] + size[1]]
        images.append(crop)
    return images
