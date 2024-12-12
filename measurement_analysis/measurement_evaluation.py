from data_transfer import dtos
from err_detection.utils import helper as err_helper
import cv2
import libs.object_detection as ob_detection
import libs.distance_measurements as distance_measurements 
import libs.preprocessing as pre
import numpy as np
import json
import typing

class MeasurementEvaluator(object):

    def __init__(self, config_path : str= './measurement_analysis/configurations/measurement.json' ):
        # ToDo: Configuration
        self._front_weft_circle_name = 'front_weft_circle'
        self._back_weft_circle_name = 'back_weft_circle'
        self._warp_edge = 'warp_edge'
        self._weft_edge = 'weft_edge'
        self._square = 'square'
        self._measure_variance = 'measure_variance'
        self._circle = 'circle'
        self._c2c = 'circle_to_circle'
        self._weft_edge_projection = '{}_to_weft_cut'
        self._warp_edge_projection = '{}_to_warp_cut'
        with open(config_path) as f:
            self.config = json.load(f)

        self.circleMeasurementEvaluator = CircleMeasurementEvaluator(None,self.config)

        super().__init__()

   
    def analyse(self,image : cv2.typing.MatLike, dpi = 600)->list[dtos.DistanceMeasurement]:
        '''
        Analyse the geometry.

        Arguments:
            image: The cropped image.
            dpi: scanned dpi.
        '''
        measurements : list[dtos.DistanceMeasurement] = []
        preprocessed = pre.replace_grey_with_black_hsv(image=image,morph_step=True)
        _, width = preprocessed.shape[:2]
        square = ob_detection.detect_square_corners_simple(preprocessed, 100)
        self._get_square_measurements(measurements, square,dpi)

        circle_measurement = self.circleMeasurementEvaluator.analyse(preprocessed,dpi)
        circle_measurement.sort(key=lambda x: x.p_1[0] if x.p_1[0]/width < 0.5 else width - x.p_1[0],reverse=True)
        frontWeftCircle,backWeftCircle = circle_measurement[:]
        frontWeftCircle.name = self._front_weft_circle_name
        backWeftCircle.name = self._back_weft_circle_name

        measurements.extend(circle_measurement)

        warp_edge_projections : list[dtos.DistanceMeasurement] = []
        weft_edge_projections : list[dtos.DistanceMeasurement] = []
        for b in circle_measurement:
            measurement_config = self.config[b.name]
            
            warp_projection_config = None 
            if self._warp_edge in measurement_config:
                warp_projection_config = measurement_config[self._warp_edge]
            
            weft_projection_config = None
            if self._weft_edge in measurement_config:
                weft_projection_config = measurement_config[self._weft_edge]

            center_cv = b.p_1
            circle_center = distance_measurements.switch_axes(center_cv)
            warp_edge_projection, weft_edge_projection = self._get_projection_edges(square, circle_center)
            measurement_name = self._warp_edge_projection.format(b.name)
            self._get_edge_projections(warp_edge_projections,
                                       warp_projection_config,
                                       circle_center,
                                       warp_edge_projection,
                                       measurement_name,
                                       b.is_trustful,
                                       dpi)
            
            measurement_name = self._weft_edge_projection.format(b.name)
            self._get_edge_projections(weft_edge_projections,
                                       weft_projection_config,
                                       circle_center,
                                       weft_edge_projection,
                                       measurement_name,
                                       b.is_trustful,
                                       dpi)
        measurements.extend(warp_edge_projections)
        measurements.extend(weft_edge_projections)
        if len(warp_edge_projections) >= 2:
            c0,c1 = warp_edge_projections[0:2]
            c2c_variance = self.config[self._c2c][self._measure_variance]
            c2c = dtos.DistanceMeasurement(
                self._c2c,
                c0.p_2,
                c1.p_2,
                distance_measurements.get_distance(c0.p_2,c1.p_2,dpi),
                c2c_variance,
                has_ground_trust=bool(c0.is_trustful and c1.is_trustful))
            measurements.append(c2c)
        return measurements

    def _get_projection_edges(self, square, circle_center):
        projections, _ = square.get_all_projections(circle_center)
        warp_edge_proj_points = [projections[1],projections[3]]
        weft_edge_proj_points = [projections[0],projections[2]]
        warp_edge_proj_points.sort(key=lambda x: x[1])
        weft_edge_proj_points.sort(key=lambda x: x[1])
        warp_edge_projection = warp_edge_proj_points[0]
        weft_edge_projection = weft_edge_proj_points[0]
        return warp_edge_projection,weft_edge_projection

    def _get_square_measurements(self, measurements, square, dpi):
        square_config = self.config[self._square]
        edge_name = self._weft_edge
        next_edge_name  = self._warp_edge
        for edge,name in zip(square.get_line_distances(),['top','right','bottom','left']):
            p1,p2,d = edge
            
            sq_line = dtos.DistanceMeasurement(name + '_' + edge_name,
                                                distance_measurements.switch_axes(p1),
                                                distance_measurements.switch_axes(p2),
                                                distance_measurements.get_distance(p1,p2,dpi),
                                                square_config[edge_name][self._measure_variance]
                                                )
            old = edge_name
            edge_name = next_edge_name
            next_edge_name = old
            measurements.append(sq_line)

    def _get_edge_projections(self,
                              edge_projections, 
                              edge_projection_config, 
                              measurement_point, 
                              edge_point_projection, 
                              measurement_name,
                              ground_truth_value,
                              dpi):
        if edge_projection_config is not None:
            f_p = edge_point_projection[0]
            edge_point_projection = (int(f_p[1]),int(f_p[0]))
            mp = (measurement_point[1],measurement_point[0])
            variance = edge_projection_config[self._measure_variance]
            dm = dtos.DistanceMeasurement(name=measurement_name,
                                            p1_cv=mp,
                                            p2_cv=edge_point_projection,
                                            distance=distance_measurements.get_distance(mp,edge_point_projection,dpi),
                                            variance=variance,has_ground_trust=ground_truth_value)
            edge_projections.append(dm)
        return



class CircleMeasurementEvaluator(object):
    

    def __init__(self,
                 path_to_config : typing.Optional[str] = None, config: typing.Optional[dict] = None ) -> None:
            self._object_detection = 'object_detection'
            self._template_matching = 'template_matching'
            self._boundary_variance = 'boundary_variance'
            self._measure_variance = 'measure_variance'
            self._trust_variance = 'trust_variance'
            self._name = 'name'

            if path_to_config is not None:
                with open(path_to_config) as f:
                    config = json.load(f)
       
            if config == None:
                raise ValueError('Circle configuration not initialized')
            self._config : dict = config
            self._configure()
            super().__init__()

    def _configure(self):
        detection_config : list[dict[str,typing.Any]] = self._config[self._object_detection]
        self.template_matching_configs : list[dtos.TemplateMatchConfig] = []
        self._boundary_variances : list[tuple[int,int]] = []
        self._measures : list[tuple[float,float]] = []
        self._trusts : list[tuple[float,float]] = []
        self._names : list[str] = []
        
        for c in detection_config:
            t_matching : dict[str,typing.Any] = c[self._template_matching]
            self.template_matching_configs.append(dtos.TemplateMatchConfig.from_json(t_matching))
            self._boundary_variances.append(c[self._boundary_variance])
            self._measures.append(c[self._measure_variance])
            self._names.append(c[self._name])
            self._trusts.append(c[self._trust_variance])
  
        return 
    
    def analyse(self,image : cv2.typing.MatLike, dpi = 600)->list[dtos.DistanceMeasurement]:
        """
        Analyse the image to find the configured cycles.

        Args:
            image (cv2.typing.MatLike): The image to analyse.
            dpi (int, optional): The dpi of the image. Defaults to 600.

        Returns:
            list[dtos.DistanceMeasurement]: The radiant measurement of the cycles.
        """            
        _,detection_image = pre.color_to_binary(image.copy(),150,True)
        detected_circles = self._detect_objects(detection_image)
        detected_circles = self._eval_box_processing(detected_circles)
        measurements = []
        for area,v,t,name in zip(detected_circles,self._measures,self._trusts,self._names):
            measurement = ob_detection.measure_circle_dist_trafo(detection_image,area,v,t,dpi)
            measurement.name = name
            measurements.append(measurement)
        return measurements


    def _detect_objects(self, detection_image):
        detected_circles : list[dtos.EvalBox] = []
        for template_match_config in self.template_matching_configs:
            detected_circles.append(
                    ob_detection.template_matching(detection_image,config=template_match_config))
        return detected_circles

    def _eval_box_processing(self, detected_cycles):
        processed_boxes = []
        for d,b in zip(detected_cycles,self._boundary_variances):
            processed_boxes.append(self._expand_box(d,b))
        return processed_boxes
    
    def _expand_box(self,
                   box : dtos.EvalBox,
                    expansion : tuple[float,float]):
        tl_0 = box.top_left[0] - expansion[0]
        tl_1 = box.top_left[1] - expansion[1]
        br_0 = box.bottom_right[0] + expansion[0]
        br_1 = box.bottom_right[1] + expansion[1] 
        tl_0 = max(0.0,tl_0)
        tl_1 = max(0.0,tl_1)
        return dtos.EvalBox((tl_0,tl_1),(br_0,br_1),box.precision,box.label)


            
                

   

            
    