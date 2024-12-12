"""
Created on Mon Sept 16 13:53:00 2024

@author: Kai  Krämer
"""
import joblib

import cv2 as cv

import err_detection.homology_ai.feature_extraction as fe
from data_transfer.dtos import EvalBox


class HomologyDetector(object):
    '''
    The boundary evaluation.
    '''
    model_not_init = 'MODEL_NOT_INIT'
    boundary_okey = 'OKEY'
    boundary_cut_bg_2 = 'ERR_CUT_BG_2'
    boundary_not_valid = 'NOT_VALID'

    def __init__(self, path_to_model='./err_detection/models/pers_hom/homo_randomforest_border.joblib') -> None:
        self.model = joblib.load(path_to_model)


    def reinit(self, path_to_model : str):
        old = self.model
        try:
            self.model = joblib.load(path_to_model)
        except:
            self.model = old

    def analyse(self,img: cv.typing.MatLike, precission = 0.7):
        '''
        Get the boundary cutting error.

        Parameters:
            img: The image to analyse.
            precission: Work in Progres, only on labels.

        Returns:
            list[eval_box]: The boundary boxes and the label if it is an error.
        '''
        new_width = int(img.shape[1] * 0.5) 
        new_height = int(img.shape[0] * 0.5)
        # Bild skalieren
        img_resized = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
        if self.model == None:
            return [EvalBox(
                top_left=(0,0),
                bottom_right=(img.shape[1],img.shape[0]),
                precision=0.0,
                label=HomologyDetector.model_not_init)]
        off_features = fe.feature_extraction(img=img)
        boundaries = []
        for entry in off_features:
            f = entry.feature
            tl = entry.top_left
            br = entry.bottom_right
            label = self.model.predict([f])[0]
            print(label)
            if label > precission:
                boundaries.append(EvalBox(top_left=tl, bottom_right=br, precision=1.0, label=HomologyDetector.boundary_cut_bg_2))
        return boundaries