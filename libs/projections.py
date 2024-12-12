import numpy as np


def get_orthogonal_projection(g_1, g_2, x, clamp_to_line_segment = False)->tuple[float,float]:
    r_0 = np.array(g_1)
    u_vector = np.array(g_2) - np.array(g_1)
    p_x = np.array(x)
    scaler = np.dot((p_x - r_0), u_vector) / np.dot(u_vector, u_vector) 
    if clamp_to_line_segment:
        scaler = max(0.0,scaler)
        scaler = min(1.0,scaler)

    projection = r_0 + scaler  * u_vector
    return (projection[0],projection[1])

def get_line_segment_to_point_projection(
        point : tuple[int,int],
        vertices : list[tuple[int,int]],
        indices : list[tuple[int,int]] = [],
        clamp_to_line_segment = True) ->list[tuple[tuple[float,float],float]]:
    """
    The point projections to the line.

    Args:
        point (tuple[int,int]): The point projected to the line segment.
        vertices (list[tuple[int,int]]): The line segments vertices.
        indices (list[tuple[int,int]], optional): The indices of connected segments. Defaults to [] leads to an indices that all the vertices connect to the next.
        clamp_to_line_segment (bool, optional): Clamps the projection point between the line segment if true, else it is the projection oan an infinite line. Defaults to True.

    Returns:
        list[tuple[tuple[float,float],float]]: The projection on the line segments with the distance to the projected point.
    """    
    if indices == None or len(indices) <= 0:
        indices = []
        for idx in zip(range(0,len(vertices) - 1),range(1,len(vertices))):
            indices.append(idx)
    projections = []
    for i,j in indices:
        p1 = vertices[i]
        p2 = vertices[j]
        p = get_orthogonal_projection(p1,p2,point,clamp_to_line_segment)
        dist = np.linalg.norm(np.array(p) - np.array(point))
        projections.append((p,dist))
    return projections