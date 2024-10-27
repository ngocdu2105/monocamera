
import numpy as np
import cv2
class PointCollection:
    """ Find the vector that translates the center point of the top face to the bottom face of the object. 
    The syntax top face edge minus bottom face edge is initialized in the PointCollection class.
    Args:
        edges (np.ndarray as image): input edges
    Example:
        >>> vector_point_top = PointCollection(edges_top).
        >>> vector_point_below = PointCollection(edges_below).
        >>> point_topcenter_move_below = vector_point_top | vector_point_below.
    """
    def __init__(self,edges):
        if not isinstance(edges,np.ndarray):
            raise ValueError("This is not a cv::Mat (numpy.ndarray) !!!")
        assert cv2.findNonZero(edges) is not None, "input edges is not None, please check again !!! "
        self.point_edges = np.array(np.where(edges>0))[::-1].T
    def __or__(self,other):
        if not isinstance(other,PointCollection):
            return ValueError("The data types are not the same")

        distance_edges_top_below = np.linalg.norm(other.point_edges-self.point_edges[:,None],axis=-1)
        index_point_of_intersection = np.array(list(set(np.argmax(distance_edges_top_below <4,axis=-1))))[1:]
        index_non_intersection_point=np.setxor1d(index_point_of_intersection,np.arange(len(other.point_edges)))

        vector_move_from_POI_to_NIP=(other.point_edges[index_point_of_intersection] - other.point_edges[index_non_intersection_point][:,None])

        magnitude_POI_and_NIP = np.linalg.norm(vector_move_from_POI_to_NIP,axis=-1)
        #get index max min magnitude_POI_and_NIP
        index_max_min_magnitude_NIP=np.argmax(np.sort(magnitude_POI_and_NIP)[:,1])
        index_max_min_magnitude_POI=np.argsort(magnitude_POI_and_NIP)[:,1][index_max_min_magnitude_NIP]
        
        value_point_NB_POI=np.sum(abs(other.point_edges[index_point_of_intersection]-other.point_edges[index_point_of_intersection][index_max_min_magnitude_POI]),axis=-1)
        point_NB_POI=other.point_edges[index_point_of_intersection][np.where(value_point_NB_POI <100)]
        vector_move_NBPOI_NIP=other.point_edges[index_non_intersection_point][index_max_min_magnitude_NIP]-point_NB_POI
        point_move_NBPOI_to_NIP=point_NB_POI+vector_move_NBPOI_NIP[:,None]
        
        magnitude_point_move_and_NIP=np.linalg.norm(other.point_edges[index_non_intersection_point]-point_move_NBPOI_to_NIP[:,:,None],axis=-1)
        index_vector_move_top_center=np.argmin(np.sum(np.min(magnitude_point_move_and_NIP,axis=-1),axis=-1))
        vector_move_top_center= -point_NB_POI[index_vector_move_top_center]+other.point_edges[index_non_intersection_point][index_max_min_magnitude_NIP]
        return vector_move_top_center
        