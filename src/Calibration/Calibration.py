import cv2
import numpy as np
import random
import os


class Calibrator:
    """Calibration adjusts a specified pixel point based on its corresponding checkerboard coordinates.
    Args:
        path_calibration (string): the file path to the specifies the location where the image is using for calibration.
    """

    def __init__(self,path_calibration):
        assert os.path.exists(path_calibration), "File path calibration not exists!!"
        self.img=cv2.imread(path_calibration)
        self.CHECKERBOARD = (13,9)
        self.__refercence_point = (0,0)
        self.__build()
    
    def __build(self):
        color = [(0,0,155),(0,255,0),(0,244,155)]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            refined_corners = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            self.imgpoints = refined_corners.reshape(-1, 2)
        
            refined_corners=refined_corners.reshape(-1,2)
            pointOxy=refined_corners[3]

            self.index_min= np.argmin(abs(self.imgpoints-pointOxy),axis=0)[0]
            self.pointOxy=pointOxy

            self.dis_oxy=np.array([
                self.imgpoints[self.index_min + 1],
                self.imgpoints[self.index_min + self.CHECKERBOARD[0] * 2]
                ])
            self.oxy=[
                pointOxy, refined_corners[self.index_min + 1],
                refined_corners[self.index_min + self.CHECKERBOARD[0] * 2]
                ]
            self.appr_edge_length=self.__size_of_chessboad_square(refined_corners)
            
            print(self.appr_edge_length)
            print('Loading Success Calibation from path')
        else:
            raise NotImplementedError("Fail Calibration calculation failed, please check the path again !!!")
    

    def reference_point_Oxy(self,point3D):
        """ The reference point is converted to chessboard coordinates.
        Args:
            point3D (np.narray): the input of the reference point.
        """
        refercence_point_no_direction_angle = np.array([
            self.__distance_from_point_to_others(p2=self.pointOxy,
                                                 p1=self.dis_oxy[0],
                                                 p3=point3D)/self.appr_edge_length,
        self.__distance_from_point_to_others(p2=self.pointOxy,p1=self.dis_oxy[1],p3=point3D)/self.appr_edge_length])
        refercence_point_no_direction_angle= np.array([refercence_point_no_direction_angle[1],refercence_point_no_direction_angle[0]]) \
              if self.CHECKERBOARD[0] > self.CHECKERBOARD[1] else refercence_point_no_direction_angle
        self.point3D=point3D
        self.__refercence_point = refercence_point_no_direction_angle * self.__coordinate_direction()

        # return self.__refercence_point
    def draw_results(self, img, radius = 5, color = (0,0,255), thickness = -1):
        """Show the final result including coordinates and reference points"""
        cv2.putText(img, f'Point P({self.__refercence_point[0]:0.2f},{self.__refercence_point[1]:0.2f})', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        for point in self.oxy:
            _x, _y = point
            cv2.circle(img, (int(_x),int(_y)), radius, color, thickness)
        return img

    def __direction_angle(self,a,b):
        """Calculate the direction of the point along the angle"""
        angleoxy=np.array([np.arccos(sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))) for a in a])*180/np.pi < [90,90]
        return angleoxy
    def __coordinate_direction(self):
        """Calculate the direction of the point along the Oxy axis"""
        angleoxy = self.__direction_angle(np.array(self.dis_oxy-self.pointOxy),np.array(self.point3D)-self.pointOxy)
        return [1 if i else -1 for i in angleoxy]

    def __size_of_chessboad_square(self,corners):
        """Size of a chessboard square"""
        idx=np.random.randint(len(corners),size=(len(corners),1))
        return np.mean(np.sort(np.sqrt(np.sum((np.array(corners[idx])-np.array(corners))**2,axis=-1)))[:,1])
        
    @classmethod
    def __distance_from_point_to_others(cls,p1,p2,p3):
        """ The distance of one point to the other two points"""
        return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
    @classmethod
    def distance(cls,p1,p2):
        """Calculate the Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))