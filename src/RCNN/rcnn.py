from ..RCNN.rcnn_preprocess import Compose, PredictConfig
from onnxruntime import InferenceSession
from ..utils import timeit
import numpy as np
import matplotlib.pyplot as plt
class RCNN:
    """Implements R-CNN.

    The input to the model is expected to be be path_model and config model
    Args:
        path_model (string): The file path to the ONNX model specifies the location where the model RCNN is stored in the ONNX.
        cfg_RCNN (yaml): The default config file stores the initial parameters of the model.
    """
    @timeit
    def __init__(self, path_model, cfg_RCNN):
        if not isinstance(path_model,str):
            raise ValueError("path_model must be a string")
        if not isinstance(cfg_RCNN,str):
            raise ValueError("cfg_RCNN must be a string")
        self.predictor = InferenceSession(path_model)
        # load infer config
        self.infer_config = PredictConfig(cfg_RCNN)
        # load preprocess transforms
        self.transforms = Compose(self.infer_config.preprocess_infos)
        print("Loading success RCNN ",end='')

    @timeit
    def predict(self,img_seg):
        """ Predicting segmentation from the input image, where the input consists of cropped images obtained from the YOLO model.
        
        Args:
            img_seg (np.ndarray): image cropped from model yolo.
        Returns:
            mask_top (np.ndarray): top surface segmentation result on object from input image.
            mask_below (np.ndarray): below surface segmentation result on object from input image.
        """

        inputs = self.transforms(img_seg)
        print(inputs['image'].shape)
        inputs_name = [var.name for var in self.predictor.get_inputs()]
        inputs = {k: inputs[k][None, ] for k in inputs_name}
        outputs = self.predictor.run(output_names=None, input_feed=inputs)
        return self.get_mask(outputs)
    def get_mask(self,outputs):
        outputsT=outputs[0].T[0]

        id1=outputs[2][np.where(outputsT==1)]
        id2=outputs[2][np.where(outputsT==0)]

        mask_below=np.where(np.sum(id1,axis=0) >0,75,0).astype('uint8')
        mask_top=np.where(np.sum(id2,axis=0) >0,15,0).astype('uint8')
        print("ONNXRuntime RCNN predict img crop ", end="")
        return mask_top , mask_below

        
        
    