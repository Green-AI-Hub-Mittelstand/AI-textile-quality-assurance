"""
Created on Mon Sept 17 10:53:00 2024

@author: Thomas Schmeyer
"""
import tensorflow as tf
import cv2 as cv
import numpy as np 
from data_transfer.dtos import EvalBox
import onnxruntime
import err_detection.utils.helper as h
import libs.preprocessing as pre

class MaterialErrorDetector():
    def __init__(self,
                 path_to_model = './err_detection/models/res_net/resmodel50.onnx',
                 size : int = 1024,
                 r_scale: int = 224) -> None:
        '''
        Initialize the adapted model.

        Parameter:
            path_to_model: the model to load.
            size: The size of the cropped models
            r_scale: The rescaling pixel size.
        '''
        try:
            self.session = onnxruntime.InferenceSession(path_to_model)
            #self.model.summary()
        except Exception as e:
            self.session = None
            print(e)
        self.stride = size//2
        self.size = size
        self.r_scale = r_scale

    def reinit(self,
               path_to_model: str,
               size: int = 1024,
               r_scale: int = 224):
        '''
        Reinitialize the adapted model.

        Parameter:
            path_to_model: the model to load.
            size: The size of the cropped models
            r_scale: The rescaling pixel size.
        '''
        old = self.session
        try:
            self.session = onnxruntime.InferenceSession(path_to_model)
            self.size = size
            self.stride = size // 2
            self.r_scale = r_scale
        except:
            self.session = old
    
    def analyse(self,
                image: cv.typing.MatLike,
                precision: float = 0.5,
                show_result_img=False) -> list[EvalBox]:
        '''
        Analyse the image on material errors.

        Parameter:
            image: The image to analyse.
            precision: The threshold to detect a error.
        
        Returns:
            list[eval_box]: A list of bounding boxes with detected errors with a probability
            greater then the precision.
        '''

        image = pre.replace_grey_with_black_hsv(image=image,morph_step=True)
        height, width, _ = image.shape
        width_stride = width // self.stride
        height_stride = height // self.stride
        imgs = []
        eval_boxes: list[EvalBox] = []

        self.get_preprocessed_imgs(image, width_stride, height_stride, imgs, eval_boxes)
        # right boundary
        self.get_preprocessed_right_boundary_imgs(image, height_stride, imgs, eval_boxes)
        # bottom
        self.get_preprocessed_bottom_boundary_imgs(image, width_stride, imgs, eval_boxes)
        # corner image
        self.get_preprocessed_corner_img(image,imgs,eval_boxes)
        
        x = np.array(imgs)
        x = x.astype(np.float32)
        session = self.session
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        res = session.run([output_name], {input_name: x})
        prediction = res[0]
        results : list[EvalBox] = []
        for i in range(len(eval_boxes)):
            eb = eval_boxes[i]
            pred = prediction[i]
            #idx = np.argmax(pred)
            probability = pred[1]
            if probability > precision:
                eb.precision = pred[1]
                eb.label = "material_error"
                results.append(eb)
                print('error found on crop index:' + str(i) + ' prob:' + str(probability))

        if show_result_img:
            for r in results:
                cv.rectangle(image, r.top_left, r.bottom_right, (255, 0, 0), 10)
            cv.imshow("result", image)
            cv.waitKey(0)
        return results

    def append_preprocessed_img(self,
                                imgs: list[cv.typing.MatLike],
                                new_size: tuple[int, int],
                                image: cv.typing.MatLike) -> bool:
        '''
        Preprocess the image and append the result to imgs. 
        It returns true if the dimension of the crop is valid.

        Parameters:
            imgs: The image list to append the image.
            new_size: The new resolution of the image in both directions.
            image: The processed image.
        '''
        # enable to debug
        # cv.imshow("crop",image)
        # cv.waitKey(0)
        if image.shape[0] < self.size or image.shape[1] < self.size:
            return False
        img_preprocessed = cv.resize(image, new_size)
        img_preprocessed = pre.preprocessing_resnet50v2(img_preprocessed)
        imgs.append(img_preprocessed)
        return True

    def get_preprocessed_corner_img(self,
                                    image: cv.typing.MatLike,
                                    imgs: list[cv.typing.MatLike],
                                    eval_boxes: list[EvalBox]):
        '''
        Crop the bottom right corner of the image.

        Parameters:
            image: The original image.
            imgs: The list to store the processed cropped image.
            eval_boxes: The list to store the default evaluated bounding box without precision.
        '''
        new_size = (self.r_scale, self.r_scale)
        crop = image[-self.size:, -self.size:]
        valid = self.append_preprocessed_img(imgs, new_size, crop)

        if valid:
            eval_boxes.append(EvalBox(
                top_left=(
                    image.shape[1] - self.size, image.shape[0] - self.size),
                bottom_right=(image.shape[1], image.shape[0]),
                precision=0.0,
                label=''))

    def get_preprocessed_bottom_boundary_imgs(self,
                                              image: cv.typing.MatLike,
                                              width_stride: int,
                                              imgs: list[cv.typing.MatLike],
                                              eval_boxes: list[EvalBox]):
        '''
        Divide the image's bottom into the parts to analyse.

        Parameters:
            image: The original image.
            width_stride: The max number of steps in width direction.
            imgs: The list to store the processed cropped image.
            eval_boxes: The list to store the default evaluated bounding box without precision.
        '''
        stride = [self.stride, self.stride]
        new_size = (self.r_scale, self.r_scale)
        for j in range(width_stride):
            w_0 = j * stride[1]
            w_1 = w_0 + self.size
            crop = image[-self.size:, w_0:w_1]
            valid = self.append_preprocessed_img(imgs, new_size, crop)
            if valid:
                eval_boxes.append(EvalBox(
                    top_left=(w_0, image.shape[0] - self.size),
                    bottom_right=(w_1, image.shape[0]),
                    precision=0.0,
                    label=''))

    def get_preprocessed_right_boundary_imgs(self,
                                             image: cv.typing.MatLike,
                                             height_stride: int,
                                             imgs: list[cv.typing.MatLike],
                                             eval_boxes: list[EvalBox]):
        '''
        Divide the image's right boundary into the parts to analyse. 

        Parameters:
            image: The original image.
            height_stride: The max number of steps in height direction.
            imgs: The list to store the processed cropped image.
            eval_boxes: The list to store the default evaluated bounding box without precision.
        '''
        stride = [self.stride, self.stride]
        new_size = (self.r_scale, self.r_scale)
        for i in range(height_stride):
            h_0 = i * stride[0]
            h_1 = h_0 + self.size
            crop = image[h_0:h_1, -self.size:]
            valid = self.append_preprocessed_img(imgs, new_size, crop)
            if valid:
                eval_boxes.append(EvalBox(
                    top_left=(image.shape[1] - self.size, h_0),
                    bottom_right=(image.shape[1], h_1),
                    precision=0.0,
                    label=''))

    def get_preprocessed_imgs(self,
                              image: cv.typing.MatLike,
                              width_stride: int,
                              height_stride: int,
                              imgs: list[cv.typing.MatLike],
                              eval_boxes: list[EvalBox]):
        '''
        Divide the image into the parts to analyse. Ignores the right and bottom boundary.

        Parameters:
            image: The original image.
            width_stride: The max number of steps in width direction.
            height_stride: The max number of steps in height direction.
            imgs: The list to store the processed cropped image.
            eval_boxes: The list to store the default evaluated bounding box without precision.
        '''
        stride = [self.stride, self.stride]
        new_size = (self.r_scale, self.r_scale)
        for i in range(height_stride):
            for j in range(width_stride):
                h_0 = i * stride[0]
                h_1 = h_0 + self.size
                w_0 = j * stride[1]
                w_1 = w_0 + self.size
                crop = image[h_0:h_1, w_0:w_1]
                valid = self.append_preprocessed_img(imgs, new_size, crop)
                if valid:
                    eval_boxes.append(EvalBox(
                        top_left=(w_0, h_0),
                        bottom_right=(w_1, h_1),
                        precision=0.0,
                        label=''))
