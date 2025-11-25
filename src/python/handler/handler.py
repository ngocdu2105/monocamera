from abc import ABC, abstractmethod

from .points_handler.points_handler import PointProcessorHandler
from ..utils.utils import find_center_top, draw_point
import matplotlib.pyplot as plt
import cv2


class Handler(ABC):
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler  # hỗ trợ chain fluent

    @abstractmethod
    def handle(self, data: dict):
        pass


class YoloHandler(Handler):
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model

    def handle(self, data: dict):
        img = data["img"]
        results = self.yolo_model.predict(img)
        data["results"]["yolo_results"] = results
        if self._next_handler:
            return self._next_handler.handle(data)
        return data


class RCNNHandler(Handler):
    def __init__(self, rcnn_model):
        super().__init__()
        self.rcnn_model = rcnn_model

    def handle(self, data: dict):
        results_yolo = data["results"]["yolo_results"]
        img = data["img"]
        bboxes_yolo = zip(
            results_yolo["ids"], results_yolo["confidences"], results_yolo["box"]
        )
        crop_imgs = []
        masks_top_below = []
        for _, cof, box in bboxes_yolo:
            if cof <= 0.9:
                continue
            crop_yolo = img[
                box[1] - 20 : box[3] + box[1] + 20, box[0] - 20 : box[0] + box[2] + 20
            ]
            if 0 in crop_yolo.shape:
                continue
            mask_top, mask_below = self.rcnn_model.predict(crop_yolo)
            crop_imgs.append((box, crop_yolo))
            masks_top_below.append((mask_top, mask_below))
        data["results"]["crop_imgs"] = crop_imgs
        data["results"]["masks_top_below"] = masks_top_below
        if self._next_handler:
            return self._next_handler.handle(data)
        return data


class PointCalculationHandler(Handler):
    def handle(self, data: dict):
        result_points = []
        for (box, crop), (mask_top, mask_below) in zip(
            data["results"]["crop_imgs"], data["results"]["masks_top_below"]
        ):
            mask_top_smooth = cv2.bilateralFilter(mask_top, 20, 100, 2)
            mask_below_smooth = cv2.bilateralFilter(mask_below, 20, 100, 2)
            edged_top = cv2.Canny(mask_top_smooth, 0, 15)
            center = find_center_top(edged_top)

            edged_below = cv2.Canny(mask_below_smooth, 0, 100)
            if cv2.findNonZero(edged_below) is None:
                move = center
            else:
                top = PointProcessorHandler(edged_top)
                below = PointProcessorHandler(edged_below)
                move = center + (top | below)

            result_points.append((box, move, crop))
        data["results"]["points"] = result_points
        data["log"]["points"] = result_points
        if self._next_handler:
            return self._next_handler.handle(data)
        return data


class DrawHandler(Handler):
    def __init__(self, calibrator):
        super().__init__()
        self.cal = calibrator

    def handle(self, data: dict):
        img = data["img"]
        for (box, point, crop), (mask_top, mask_below) in zip(
            data["results"]["points"], data["results"]["masks_top_below"]
        ):
            offset_point = point + [box[0] - 20, box[1] - 20]
            crop[mask_below > 0] = (0, 255, 0)
            crop[mask_top > 0] = (255, 255, 0)
            crop = draw_point(crop, point)
            img[
                box[1] - 20 : box[3] + box[1] + 20, box[0] - 20 : box[0] + box[2] + 20
            ] = crop
            self.cal.reference_point_oxy(offset_point)
            img = self.cal.draw_results(img)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        return data
