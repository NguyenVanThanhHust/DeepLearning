import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image
import re

class CustomDataset:

    def __init__(self, data_dir, split='trainval',
                 ):
        id_list_file = os.path.join(
            data_dir, 'listImg.txt')

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.ground_truth_dir = os.path.join(self.data_dir, "ground truth")

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        file_path = os.path.join(self.ground_truth_dir, id_ + '.txt')
        with open(file_path) as f:
            content = f.readlines()
        content = [x.rstrip('\n') for x in content]
        bbox = list()
        label = list() 
        scale = list()
        for eachline in content:
            list_number = re.findall(r'\d+', str(eachline))
            if len(list_number) > 0:
                x1, y1, x2, y2, number = list_number
                bbox.append([y1, x1, y2, x2])
                label.append(number)
                scale.append(1)
            else:
                print("filepath: ", file_path, " list number: ", list_number)

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        scale = np.stack(scale).astype(np.int32)

        # Load a image
        img_file = os.path.join(self.data_dir, 'positive image set', id_ + '.jpg')
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, scale

    __getitem__ = get_example


# VOC_BBOX_LABEL_NAMES = (
#     'aeroplane',
#     'bicycle',
#     'bird',
#     'boat',
#     'bottle',
#     'bus',
#     'car',
#     'cat',
#     'chair',
#     'cow',
#     'diningtable',
#     'dog',
#     'horse',
#     'motorbike',
#     'person',
#     'pottedplant',
#     'sheep',
#     'sofa',
#     'train',
#     'tvmonitor')