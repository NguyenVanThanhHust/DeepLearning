# https://github.com/a8252525/detectron2_example_PCBdata/blob/master/PCBdata_fasterRCNN_30000.ipynb

import os
import numpy as np
import json
from detectron2.structures import BoxMode
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import csv

train, test = [], []
def take_path(x, d):
    with open(d) as t:
        tmp = csv.reader(t,delimiter=' ')
        for i in tmp:
            x.append(i)
        for ele in x:

            ele[0] = '../../data/DeepPCB-master/PCBData/'+ele[0][:-4]+'_test.jpg'
            ele[1] = '../../data/DeepPCB-master/PCBData/'+ele[1]

take_path(train, '../../data/DeepPCB-master/PCBData/test.txt')
take_path(test, '../../data/DeepPCB-master/PCBData/trainval.txt')

def get_PCB_dict(data_list):
    dataset_dicts = []
    for i, path in enumerate(data_list):
        filename = path[0]
        height, width = cv2.imread(filename).shape[:2]
        record = {}
        record['file_name'] = filename
        record['image_id'] = i
        record['height'] = height
        record['width'] = width

        objs = []
        with open(path[1]) as t:
            lines = t.readlines()
            for line in lines:
                box = line[:-1].split(' ')
                boxes = list(map(float,[box[0],box[1],box[2],box[3]]))
                category = int(box[4])

                obj = {
                    "bbox": boxes,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    #"segmentation": [poly], To draw a line, along to ballon
                    "category_id": category-1,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d,x in [("train",train), ("test",test)]:
    DatasetCatalog.register("PCB_" +d, lambda x=x: get_PCB_dict(x))
    MetadataCatalog.get("PCB_"+d).set(thing_classes=["open", "short", "mousebite", "spur", "copper", "pin-hole"],
    thing_colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)])

PCB_metadata = MetadataCatalog.get("PCB_train")
MetadataCatalog.get("PCB_test")

dataset_dicts = get_PCB_dict(train)
print(dataset_dicts[0])

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("PCB_train",)
cfg.DATASETS.TEST = ("PCB_test")
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 30000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4096   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("PCB_test", )
predictor = DefaultPredictor(cfg)
