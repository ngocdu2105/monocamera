import numpy as np
import cv2
from src.RCNN.rcnn import RCNN
from src.YOLOV5.yolov5 import YOLOV5
from src.Calibration.Calibration import Calibrator
from src.PointCollection.PointCollection import PointCollection
from src.utils import LoadIMG, find_center_top, draw_point
import matplotlib.pyplot as plt




if __name__ == "__main__":
    path = {
        'path_img' : "dataset/img",
        'path_model_yolo': "model/model_yolov5.onnx",
        'path_model_rcnn': 
        {
            'path': "model/mask_rcnn_dataset_fix.onnx",
            'cfg': "model/infer_cfg_dataset.yml"
        },
        'path_calibration': "dataset/calibration/cab.jpg"
    }
    imgs = LoadIMG(path["path_img"])
    yolo = YOLOV5(path["path_model_yolo"])
    rcnn = RCNN(path["path_model_rcnn"]["path"],path["path_model_rcnn"]["cfg"])
    cal = Calibrator(path["path_calibration"])
    imgs.loadImg()
    for img in imgs.getImgs():
        results_yolo = yolo.predict(img)
        for _,cof,box in zip(results_yolo["ids"],results_yolo["confidences"], results_yolo["box"]):
            if cof > 0.9:
                img_crop = img[box[1]-20:box[3]+box[1]+20,box[0]-20:box[0]+box[2]+20]
                print(img_crop.shape)
                if (0 in img_crop.shape):
                    print("Yolov5 predict fail !!!")
                    continue 
                mask_top, mask_below = rcnn.predict(img_crop)
                img_crop[mask_below > 0] = (0,255,0)
                img_crop[mask_top > 0] = (255,255,0)
                mask_top_smooth=cv2.bilateralFilter(mask_top,20,100,2)
                mask_below_smooth=cv2.bilateralFilter(mask_below,20,100,2)
                edged_top = cv2.Canny(mask_top_smooth, 0,15)
                point_center = find_center_top(edged_top)
                print(f"Point top center: {point_center}")
                edged_below=cv2.Canny(mask_below_smooth, 0, 100)
                
                #In case only the top surface
                if cv2.findNonZero(edged_below) is None:
                    top_center_move = point_center
                else:
                    point_top = PointCollection(edged_top)
                    point_below = PointCollection(edged_below)
                    point_move = point_top | point_below
                    top_center_move = point_center + point_move
                
                print("top_center_move :", top_center_move)
                print("top_move ", top_center_move + [box[0]-20, box[1]-20])
                img[box[1]-20:box[3]+box[1]+20,box[0]-20:box[0]+box[2]+20] = draw_point(img_crop,top_center_move)
                cal.reference_point_Oxy(top_center_move + [box[0]-20, box[1]-20])
                img = cal.draw_results(img)

                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.imshow(img[box[1]:box[3]+box[1],box[0]:box[0]+box[2]])
                plt.show()




        


    

