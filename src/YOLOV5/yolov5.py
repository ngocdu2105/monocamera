
import cv2
import os
import numpy as np
from ..utils import timeit

class YOLOV5:
    """ Yolov5 model predicts objects on the chessboard
    Args:
        path_onnx: model path with .onnx extension

    """
    @timeit
    def __init__(self,path_onnx):
        assert os.path.exists(path_onnx), f"Path does not exists {path_onnx}"
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.path_model_yolo = path_onnx
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.SCORE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4
        self.CONFIDENCE_THRESHOLD = 0.4
        self.class_list = ['obj_1','obj_2']
        self.is_cuda = False
        self.value_time=[]        
        self.__build_model()
        print("Loading suscces model yolov5",end=" ")
    @timeit
    def predict(self,img):
        """
        Args:
            img: input image for object prediction
        Returns:
            result_class_ids:
            confidences_ids:
            results_box_ids:
        """
        self.input_image = self.__square_img(img)
        self.outs = self.__forward_model(self.input_image)
        self.result_class_ids = []
        self.result_confidences = []
        self.result_boxes = []

        self.__wrap_get_box(self.input_image, self.outs[0])
        # self.mask_object()
        print("Model Yolov5 predict img ",end='')
        return {"ids": self.result_class_ids, "confidences":self.result_confidences, "box": self.result_boxes}

    def __build_model(self):

        self.net = cv2.dnn.readNet(self.path_model_yolo)
        
        if self.is_cuda:
            print("Attempty to use CUDA")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



    def __forward_model(self,image):
        """calculate the output of a model based on given inputs."""
        self.blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        self.net.setInput(self.blob)
        preds = self.net.forward()
        return preds
    def __wrap_get_box(self,input_image, output_data):
        """Identifying Objects from Model Output, provides information such as bounding boxes, class labels, and confidence scores for each detected object
        Args:
            input_image (np.narray): input image
            output_data (np.narray): which is output model onnx in dnn opencv
        """
        self.class_ids = []
        self.confidences = []
        self.boxes = []

        self.rows = output_data.shape[0]
        self.image_width, self.image_height, _ = input_image.shape

        self.x_factor = self.image_width / self.INPUT_WIDTH
        self.y_factor =  self.image_height / self.INPUT_HEIGHT
        # print(output_data.shape)
        for r in range(self.rows):
            row = output_data[r]
            confidence = row[4]

            if confidence >= 0.4:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)

                class_id = max_indx[1]

                if ( classes_scores[class_id] > .25):

                    self.confidences.append(confidence)

                    self.class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * self.x_factor)
                    top = int((y - 0.5 * h) * self.y_factor)
                    width = int(w * self.x_factor)
                    height = int(h * self.y_factor)
                    box = np.array([left, top, width, height])

                    self.boxes.append(box)

        self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.25, 0.45) 


            
        for i in self.indexes:
            self.result_confidences.append(self.confidences[i])
            self.result_class_ids.append(self.class_ids[i])
            self.result_boxes.append(self.boxes[i])
        





    def __square_img(self,frame):
        """ Convert the input image to a square image similar to the yolo model input """
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result



