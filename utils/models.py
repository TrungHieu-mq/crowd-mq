
import cv2
import numpy as np
import threading
from utils.base_trt import BaseEngine
import os
import warnings
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def blob(im):
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    return im
    
def letterbox(im, new_shape,color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)

class YOLOV8TRT:
    def __init__(self, p, conf=0.5, nms=0.7):
        self.Engine = BaseEngine(p)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]

    def pre(self, bgr):
        bgr, ratio, dwdh = letterbox(bgr, (self.W, self.H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        return tensor, ratio, dwdh

    def post(self, data, ratio, dwdh):
        assert len(data) == 4
        num_dets, bboxes, scores, labels = (i[0] for i in data)
        idx = [i for i, box in enumerate(bboxes) if np.sum(box) != 0] 
        if len(idx) <= 0:
            return np.empty((0, 4), dtype=np.float32), np.empty(
                (0, ), dtype=np.float32), np.empty((0, ), dtype=np.float32)
        # check score negative
        bboxes, scores, labels = bboxes[idx], scores[idx], labels[idx]
        scores[scores < 0] = 1 + scores[scores < 0]
        bboxes -= dwdh
        bboxes /= ratio
        return bboxes, scores, labels

    def infer(self, tensor):
        data = self.Engine(tensor)
        return data

    def __call__(self, bgr):
        tensor, ratio, dwdh = self.pre(bgr)
        data = self.infer(tensor)
        bboxes, scores, labels = self.post(data, ratio, dwdh)
        return bboxes, scores, labels


class BaseOnnx(threading.Thread):
    def __init__(self, path):
        threading.Thread.__init__(self)
        import onnxruntime
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])

    def pre(self, input_feed):
        for key in input_feed:
            input_feed[key] = input_feed[key].astype(np.float32)
        return input_feed

    def pos(self, out):
        return out[0]
    
    def infer(self, input_feed):
        out = self.session.run(None, input_feed=input_feed)
        return out
    
    def __call__(self, input_feed):

        input_feed = self.pre(input_feed)
        out = self.infer(input_feed)
        out = self.pos(out)
        return out

class SlowonlyTrt(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.mean = np.float64([123.675, 116.28, 103.53])
        self.norm = np.float64([58.395, 57.12, 57.375])
        self.frame_inds = [0,  2,  4,  6,  8, 10, 12, 15]
        self.backbone = BaseEngine("../weights/slowonly/slowonly_backbone.trt")
        # self.bbox_roi_extractor = BaseEngine("../weights/slowonly/slowonly_bbox_roi_extractor.trt")
        # self.bbox_head = BaseEngine("../weights/slowonly/slowonly_bbox_head.trt")
        self.bbox_roi_extractor = BaseOnnx("../weights/slowonly/slowonly_bbox_roi_extractor.onnx")
        self.bbox_head = BaseOnnx("../weights/slowonly/slowonly_bbox_head.onnx")

        self.size = 256

    def infer(self, input_tensor, rois):
        feat = self.backbone(input_tensor)[0]
        feat = feat.reshape(1, 2048, 8, 16, 16)
        bbox_feats = self.bbox_roi_extractor({"feat":feat, "rois":rois})
        bbox_feats = bbox_feats.reshape(-1, 4096, 1, 8, 8)
        cls_score = self.bbox_head({"bbox_feats": bbox_feats})
        cls_score = cls_score.reshape(-1, 81)
        return cls_score

    def normalize(self, x):
        x = x.transpose(-1, 0, 1)
        x = (x - self.mean[:, None, None]) / self.norm[:, None, None]
        return x
    
    def resize_bbox(self, bbox, original_size, target_size):
        x1, y1, x2, y2 = bbox
        orig_width, orig_height = original_size
        target_width, target_height = target_size
        
        # Calculate scaling factors for width and height
        width_scale = target_width / orig_width
        height_scale = target_height / orig_height
        
        # Resize bounding box coordinates
        resized_x1 = int(x1 * width_scale)
        resized_y1 = int(y1 * height_scale)
        resized_x2 = int(x2 * width_scale)
        resized_y2 = int(y2 * height_scale)
        
        return resized_x1, resized_y1, resized_x2, resized_y2

    def pre(self, frames_data, bboxs):
        
        boxs = [self.resize_bbox(box, (frames_data[0].shape[1],frames_data[0].shape[0]), (self.size,self.size)) for box in bboxs]
        imgs = [cv2.resize(frames_data[ind], (self.size, self.size)).astype(np.float32) for ind in self.frame_inds]
        imgs = [self.normalize(img) for img in imgs]
        input_array = np.stack(imgs).transpose((1, 0, 2, 3))[np.newaxis]
        rois = np.insert(boxs, 0, 0, axis=1)
        return input_array, rois


    def pos(self, scores):
        scores = 1 / (1 + np.exp(-scores))
        return scores
    
    def __call__(self, frames_data, bboxs):
        if len(bboxs) <= 0:
            return []

        input_tensor, rois = self.pre(frames_data, bboxs)
        scores = self.infer(input_tensor, rois)
        scores = self.pos(scores)
        return scores

class OpticalFlow():
    def __init__(self, s_opt=320):
        self.prev = None
        self.s_opt = s_opt
    

    def resize_bbox(self, box, d_size):
        x1, y1, x2, y2 = box
        x_scale = self.s_opt / d_size[1]
        y_scale = self.s_opt / d_size[0]
        x1 = int(np.round(x1 * x_scale))
        y1 = int(np.round(y1 * y_scale))
        x2 = int(np.round(x2 * x_scale))
        y2 = int(np.round(y2 * y_scale))
        return x1, y1, x2, y2 
        
    def get(self, frame):

        frame = cv2.resize(frame, (self.s_opt, self.s_opt))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev is None:
            self.prev = frame

        flow = cv2.calcOpticalFlowFarneback(self.prev, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = frame
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        return mag




