import libs.projections as projections
import numpy as np

class AbstractPolygon:
    def get_cv_coordinates(self,pixel : tuple[int,int]):
        '''
        switches axes, sind cv and matrixes use different notation.

        Parameters:
            pixel: coordinates of a pixel

        Returns:
            cv_pixel: tuple with cv coordinates
        '''
        cv_pixel = (pixel[1], pixel[0])
        return cv_pixel

    def get_line_vertices(self) -> list[tuple[int,int]]:
        """
        Get the polygon vertices in cycle direction from left to right in pixel coordinates.
        Returns:
            list[tuple[int,int]]: The polygon vertices.
        """        
        return []
    
    def get_all_cv(self):
        """
        Get the polygon vertices in cycle direction from left to right in cv coordinates.
        Returns:
            list[tuple[int,int]]: The polygon vertices.
        """   
        cvs = []
        for coordinate in  self.get_line_vertices():
            cvs.append(self.get_cv_coordinates(coordinate))
        return cvs
    
    def get_all_projections(self,point : tuple[int,int])->tuple[list[tuple[tuple[float,float],float]],list[tuple[int,int]]]:
        """
        Get all orthogonal projections of one point to all line segments.

        Args:
            point (tuple[int,int]): The point to project to the line segments.

        Returns:
            tuple[list[tuple[tuple[float,float],float]],list[tuple[int,int]]]: The projection points and the corresponding line segment vertices indices.
        """          
        segments = self.get_line_vertices()
        indices = self._natural_vertices_indices()
        
        projection = projections.get_line_segment_to_point_projection(point,segments,indices)
        
        return projection,indices

    def _natural_vertices_indices(self):
        segments = self.get_line_vertices()
        indices : list[tuple[int,int]] = []
        for idx in zip(range(0,len(segments) - 1),range(1,len(segments))):
            indices.append(idx)
        indices.append((len(segments) - 1,0))
        return indices
    
    def get_line_distances(self)->list[tuple[tuple[int,int],tuple[int,int],float]]:
        """
        Get the line vertices and distances.

        Returns:
            list[tuple[tuple[int,int],tuple[int,int],float]]: The line vertices with distances.
        """        
        vertices = self.get_line_vertices()
        indices = self._natural_vertices_indices()
        res = []
        for i,j in indices:
            p1 = vertices[i]
            p2 = vertices[j]
            d = float(np.linalg.norm(np.array(p1)-np.array(p2),2))
            res.append((p1,p2,d))
        return res
            


class Square(AbstractPolygon):
    '''
    Corners of the material with pixel coordinates.
    '''
    def __init__(self, 
                 top_left_px : tuple[int, int], 
                 top_right_px : tuple[int, int],
                 bottom_left_px : tuple[int, int], 
                 bottom_right_px : tuple[int, int]):
        self.top_left_px = top_left_px
        self.top_right_px = top_right_px
        self.bottom_left_px = bottom_left_px
        self.bottom_right_px = bottom_right_px
        
    def get_tl(self):
        return self.get_cv_coordinates(self.top_left_px)
    
    def get_tr(self):
        return self.get_cv_coordinates(self.top_right_px)
    
    def get_bl(self):
        return self.get_cv_coordinates(self.bottom_left_px)
    
    def get_br(self):
        return self.get_cv_coordinates(self.bottom_right_px)
        
    
    def get_line_vertices(self) -> list[tuple[int,int]]:
        return [self.top_left_px,self.top_right_px,self.bottom_right_px,self.bottom_left_px]