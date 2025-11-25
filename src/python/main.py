from .utils.utils import LoadIMG
from .pipeline import build_pipeline


if __name__ == "__main__":
    path = {
        "img_path": "dataset/img",
        "yolo_model_path": "ckpt/model_yolov5.onnx",
        "rcnn_model_path": {
            "path": "ckpt/mask_rcnn_dataset_fix.onnx",
            "cfg": "ckpt/infer_cfg_dataset.yml",
        },
        "calibration_path": "dataset/calibration/cab.jpg",
    }
    pipeline = build_pipeline(path)
    imgs = LoadIMG(path["img_path"])
    imgs.loadImg()
    logs_all_images = []
    for id, img in enumerate(imgs.getImgs()[:4]):
        data = {"img": img, "img_id": f"image_{id}", "results": {}, "log": {}}
        pipeline.handle(data)
        print(data["log"])
        logs_all_images.append(data)
