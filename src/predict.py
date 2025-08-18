from posixpath import basename
import torch
import cv2
from nn import Yolov8n
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from utils import class_names


class YoloInference:
    def __init__(
        self,
        model_path,
        version="n",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Yolov8n(version=version).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()

    def preprocess(self, image: str, imgsize: int = 640):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_shape = image.shape[:2]

        # Resize image
        h, w = image.shape[:2]
        r = imgsize / max(h, w)
        if r != 1:
            new_w, new_h = int(w * r), int(h * r)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        dh, dw = imgsize - new_h, imgsize - new_w
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert to tensor
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = torch.from_numpy(image).float() / 255.0
        image = image.unsqueeze(0).to(self.device)

        return image, orig_shape

    def postprocess(
        self, predictions, orig_shape, conf_threshold=0.25, iou_threshold=0.45
    ):
        """Postprocess YOLO predictions"""
        # Apply NMS
        predictions = non_max_suppression(
            predictions,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=None,
            agnostic=False,
            max_det=10,  # Further reduced to 10 for cleaner output
        )

        detections = []
        for pred in predictions:
            if pred is not None and len(pred):
                # Scale boxes back to original image size
                pred[:, :4] = scale_boxes((640, 640), pred[:, :4], orig_shape)

                # Extract boxes, scores, and classes
                boxes = pred[:, :4].cpu().numpy()
                scores = pred[:, 4].cpu().numpy()
                classes = pred[:, 5].cpu().numpy().astype(int)

                detections.append(
                    {"boxes": boxes, "scores": scores, "classes": classes}
                )

        return detections

    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        with torch.no_grad():
            input_tensor, orig_shape = self.preprocess(image)
            predictions = self.model(input_tensor)
            detections = self.postprocess(
                predictions, orig_shape, conf_threshold, iou_threshold
            )
            return detections

    def visualize(self, image: str, detections, save_path="./assets/result.jpg"):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotator = Annotator(image, line_width=2, font_size=12)

        if detections and len(detections) > 0:
            det = detections[0]  # Take first image detections
            boxes = det["boxes"]
            scores = det["scores"]
            classes = det["classes"]

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                label = f"{class_names[cls]} {score:.2f}"
                color = colors(cls, True)
                annotator.box_label((x1, y1, x2, y2), label, color=color)
        result = annotator.result()
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    model = YoloInference(model_path="./weights/yolov8n.pth", version="n")
    imgpath = "./assets/people.png"
    result = model.predict(imgpath)
    basename = imgpath.split("/")[-1]
    result_image = model.visualize(
        imgpath, result, save_path=f"./assets/result_{basename}"
    )
