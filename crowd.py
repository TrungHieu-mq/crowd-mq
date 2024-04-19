from utils.common import VideoManager, viz_text_bg
from utils.models import YOLOV8TRT, OpticalFlow
import numpy as np
import cv2
from time import time
from utils.bytetrack.byte_tracker import BYTETracker
from ultralytics import YOLO
import json
import os
from tqdm import tqdm

class ABNORMAL:
    min_person = 1.4
    min_grad = 2
    live = 5
    min_ab_frames = 3

    def __init__(self):
        self.h_tracks = {}
        self.c_fs = []
    
    def save(self, tid, speed):
        if tid not in self.h_tracks:
            self.h_tracks[tid] = {"value": [], "last":time()}
        self.h_tracks[tid]["value"].append(speed)
        self.h_tracks[tid]["last"] = time()

    def remove_unuse_id(self):
        keys = [k for k in self.h_tracks]
        for k in keys:
            if time() - self.h_tracks[k]["last"] >= self.live:
                del self.h_tracks[k]

    def is_abnormal(self):
        c_f = []
        self.remove_unuse_id()
        for k in self.h_tracks:
            val = self.h_tracks[k]["value"]
            # chưa thu thập đủ min_ab_frames
            if len(val) <= self.min_ab_frames:
                continue
            i_get = -self.min_ab_frames if len(val) <= self.min_ab_frames*2 else (-self.min_ab_frames*2)
            grad = np.gradient(val[i_get:])
            grad = np.absolute(grad)
            c_f.append(grad.mean())
        num_person = sum([p >= self.min_grad for p in c_f])
        self.c_fs = self.c_fs[1:] if len(self.c_fs) == self.min_ab_frames else self.c_fs
        self.c_fs.append(num_person)
        return np.mean(self.c_fs) >= self.min_person



if __name__ == "__main__":
    dir_video = '/home/hieu/sambashare/disk2T/Data_De_Tai_Org/MQ_03_04'
    l = []
    for dir in os.listdir(dir_video):
        list_video = os.listdir(os.path.join(dir_video,dir))
        list_video = [f"{dir}/{item}" for item in list_video]
        l.append(list_video)
    l = [item for sublist in l for item in sublist]
 
    # model = YOLOV8TRT("../weights/weapon_personv8.trt",
    #                     conf=0.5, nms=0.7)
    model = YOLO("../weights/weapon_v8s_89e.pt")

    anno_vids = {"annotations": [],
                "categories":
                    {
                        "knife": 0,
                        "gun": 1,
                        "sword": 2,
                        "dao": 3,
                        "matau": 4,
                        "con": 5,
                        "person": 6
                    }
                    }
    sampling = 2
    for name in tqdm(l):
        vid = VideoManager(dir_video+f"/{name}")
        tracker = BYTETracker()
        otp = OpticalFlow(320)
        ab = ABNORMAL()
        anno_vid = {"video_name": name,
                    "annotation": []
                    }

        for i, frame in vid.iter(skip=sampling):
            bboxs = model(frame, conf=0.45, 
                          imgsz = (736,1280),
                                verbose= False)
            bboxs =[pred.boxes.data.cpu().numpy() for pred in bboxs][0]
            
            bboxs_person = bboxs[bboxs[:, -1] == 6]
            # bboxs_weapon = bboxs[bboxs[:, -1] != 6]
            print(bboxs_person)
            

            tracks = tracker.update(bboxs_person)

            flow = otp.get(frame)
            for track in tracks:

                # update ABNORMAL
                tid = int(track[4])
                x1, y1, x2, y2 = otp.resize_bbox(track[:4], frame.shape[:2])
                speed = flow[y1:y2, x1:x2]
                speed = speed[speed>1]
                speed = 0 if len(speed) <= 0 else speed.mean()
                ab.save(tid, speed)

            anno_frame = {"image_id": i,
                            "people_count": len(bboxs_person),
                            "is_abnormal":1 if ab.is_abnormal() else 0,
                            "bboxs": np.delete(bboxs.astype(int), 4, axis=1).tolist()
                            }
            anno_vid["annotation"].append(anno_frame)

            # frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
            # vid.write(frame, f"../inp/det/infer_{name}".replace("dav", "avi"))
            
        
        
        anno_vids["annotations"].append(anno_vid)
    # save json file
    with open("MQ03_04.json", "w") as json_file:
        json.dump(anno_vids, json_file, indent=4)    
