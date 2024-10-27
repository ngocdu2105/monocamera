import cv2
import os
import time
from functools import wraps
import numpy as np
def timeit(func):
    """class calculates the execution"""
    def wraps(*args, **kwargs):
        startTime = time.time()
        results = func(*args, **kwargs)
        endTime = time.time()
        execution_time = endTime - startTime
        print(f" executed in {execution_time: .6f} seconds")
        return results
    return wraps




class LoadIMG:
    """load input image from a directory
    Args:
        directory (string): load input image from a directory where images are saved
    """
    def __init__(self,directory):
        assert os.path.exists(directory), "No imgs in directory!!"
        self.directory = directory
        self.imgs = []
    @timeit
    def loadImg(self):
        valif_extensions = ["jpg","png"]
        self.imgs = [cv2.imread(os.path.join(self.directory,filename)) \
                     for filename in os.listdir(self.directory)]
        self.imgs = [img for img in self.imgs if img is not None]
        print("Load success IMG in directory ",end ='')
    def getImgs(self):
        return self.imgs






    
def find_center_top(mask_top):
    """Find the coordinates of the center of the top surface"""
    assert mask_top is not None, "Input image is empty, please check again"
    # point_center_pixel_coordinates = np.array(np.where(mask_top>0))
    # point_center_x = min(point_center_pixel_coordinates[0])+abs(min(point_center_pixel_coordinates[0])-max(point_center_pixel_coordinates[0]))/2
    # point_center_y = min(point_center_pixel_coordinates[1])+abs(min(point_center_pixel_coordinates[1])-max(point_center_pixel_coordinates[1]))/2
    # return np.array([point_center_x,point_center_y]).astype("int32")
    contours, _ = cv2.findContours(mask_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    if M["m00"] != 5000:  # Tránh chia cho 0
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return np.array([cX,cY]).astype("int32")

def draw_point(image, point,radius = 5, color = (0,0,255),thickness = -1):
    """Draw point from previous input point"""
    cv2.circle(image, point, radius, color, thickness)
    return image
        
        
        
