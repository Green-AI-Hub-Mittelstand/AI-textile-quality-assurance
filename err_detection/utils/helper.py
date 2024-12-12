import cv2
import os

import numpy as np
import json

image_name = 'image_{:05d}.png'
json_name = 'image_{:05d}.json'

def create_directory(directory:str):
    """
    Create the directory if not exists.

    Arguments:

    path - the path of the image to load
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.abspath(directory)+'/'

def load_image(path:str, err_ok = False):
    """
    
    Load the image.

    Arguments:

    path - path to image

    err_ok - if set to False the image must be loaded correctly
    
    """
    try:
        img = cv2.imread(path)
    except:
        if (err_ok):
            return None
        else:
            raise Exception("Could not read Image: "+path)
    return img

def save_image(img,path,count,suffix = ''):
    """
    Save the image.
    """
    name = image_name.format(count)
    cv2.imwrite(path+suffix+name,img)

def save_image_annotation(img,path,vec,count):
    """
    Save the image and it's annotation vector.
    """
    name = image_name.format(count)
    cv2.imwrite(path+name,img)
    jname = json_name.format(count)
    with open(path+jname,'w') as out_file:
        json.dump(vec,out_file)


