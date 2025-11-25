from .model.rcnn.rcnn import RCNN
from .model.yolov5.yolov5 import YOLOV5
from .handler.calibration_handler.calibration import Calibrator
from .handler.handler import (
    YoloHandler,
    RCNNHandler,
    PointCalculationHandler,
    DrawHandler,
)


def build_pipeline(path):
    yolo = YOLOV5(path["yolo_model_path"])
    rcnn = RCNN(path["rcnn_model_path"]["path"], path["rcnn_model_path"]["cfg"])
    cal = Calibrator(path["calibration_path"])

    # Chain setup
    yolo_step = YoloHandler(yolo)
    rcnn_step = RCNNHandler(rcnn)
    point_step = PointCalculationHandler()
    draw_step = DrawHandler(cal)

    # Chain of Responsibility
    (yolo_step.set_next(rcnn_step)
     .set_next(point_step)
     .set_next(draw_step))
    return yolo_step
