import cv2
import os 
import numpy as np
import onnxruntime
from tqdm import tqdm
import torch


class YOLOV8:
    
    def __init__(self, path, conf_thres=0.4, iou_thres=0.5, imgsz=(800, 800), device=None):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.imgsz = imgsz

        # Initialize model
        self.initialize_model(path,device)

    def __call__(self, image):
        return self.box_objects(image)

    def initialize_model(self, path, device):
        if device == torch.device('cuda'):
            self.session = onnxruntime.InferenceSession(path,providers=['CUDAExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession(path,providers=['CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def box_objects(self, image):
        input_tensor,h,w = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_box_output(outputs[0],h,w)
        
        return self.boxes, self.scores, self.class_ids

    # def prepare_input(self, image):
    #     self.img_height, self.img_width = image.shape[:2]

    #     input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # Resize input image
    #     scaled_img = cv2.resize(input_img, (self.imgsz[1], self.imgsz[0]))
    #     input_img = scaled_img

    #     h,w = scaled_img.shape[:2]

    #     # Scale input pixel values to 0 to 1
    #     input_img = input_img / 255.0
    #     input_img = input_img.transpose(2, 0, 1)
    #     input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    #     return input_tensor,h,w

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        scale = min(self.imgsz[0] / self.img_height, self.imgsz[1] / self.img_width)
        scaled_img = cv2.resize(input_img, (int(round(scale * self.img_width)), int(round(scale * self.img_height))))
        dw, dh = self.imgsz[1] - scaled_img.shape[1], self.imgsz[0] - scaled_img.shape[0]    
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        scaled_img = cv2.copyMakeBorder(scaled_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)) 
        input_img = scaled_img

        h,w = scaled_img.shape[:2]

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor,h,w
    
    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes
    
    def clip_boxes(self, boxes, shape):
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs

    def process_box_output(self, box_output,h,w):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - 4
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        if len(scores) == 0:
            return [], [], []

        box_predictions = predictions[..., :num_classes+4]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions,h,w)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        #indices = nms_rc(boxes, scores, self.iou_threshold)
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]


    def extract_boxes(self, box_predictions,h,w):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        # boxes = self.rescale_boxes(boxes,
        #                            (h, w),
        #                            (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        # Scale boxes to original image dimensions
        boxes = self.scale_boxes((h, w),
                                 boxes,
                                 (self.img_height, self.img_width))

        # Check the boxes are within the image
        # boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        # boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        # boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        # boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        # self.input_shape = model_inputs[0].shape
        # self.input_height = self.input_shape[2]
        # self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    # print(boxes.shape)
    # print(scores.shape)
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



            # if self.class_names[class_ids[idx]] == 'Caption':










# model_path = '/ssd7/exec/huangjy/ultralytics/runs/detect/train81/weights//best.onnx'
# yolov8 = YOLOV8(model_path, conf_thres=0.4, iou_thres=0.65,imgsz=(800, 800))

# test_dir = '/ssd8/exec/huangjy/ultralytics/'
# files = [item for item in os.listdir(test_dir) if item.endswith('.jpg')]

# class_names = (
#      'Text', 'Title', 'Figure', 'Equation', 'Table',
#     'Caption', 'Header', 'Footer', 'BibInfo', 'Reference',
#     'Content', 'Code', 'Other', 'Item', 'Author', 
# )
# for file in tqdm(files):
#     img_path = os.path.join(test_dir,file)
#     img = cv2.imread(file)
#     h, w = img.shape[:2]
#     # Detect Objects
#     boxes, scores, class_ids = yolov8(img)
#     for idx, box in enumerate(boxes):
#         color = np.random.rand(3) * 160
#         cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color)
#         cv2.putText(img, '%.3f %s' % (scores[idx], class_names[class_ids[idx]]), 
#             (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#     cv2.imwrite('{}_res.jpg'.format(os.path.basename(file)[:-4]), img)
#     # 输出的结果
#     """
#     boxes: 输出的bounding box，正矩形。N*4
#     scored: 每个框的置信度。N
#     class_ids: 每个框的类别
#     """

