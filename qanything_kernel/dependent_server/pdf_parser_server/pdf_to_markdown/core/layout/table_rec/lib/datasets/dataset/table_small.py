from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import random
import os

import torch.utils.data as data

class Table(data.Dataset):
  num_classes = 2
  table_size = 512
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(Table, self).__init__()
    self.split = split
    self.data_dir = os.path.join(opt.data_dir, opt.dataset_name)
    print(self.data_dir)
    self.img_dir = opt.image_dir
    _ann_name = {'train':'train', 'val':'test','test':'test'}
    self.annot_path = os.path.join(self.data_dir, 'json', '{}.json').format(_ann_name[split])
    print(self.annot_path)
 
    self.max_objs = 300
    self.max_pairs = 900
    self.max_cors = 1200 #800 #1200 #800
     
    self.class_name = [
      '__background__', 'center','corner']
    self._valid_ids = [1,2]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing table {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes, thresh):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          if bbox[4] > float(thresh):
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            score = bbox[4]
            bbox_out  = list(map(self._to_float, bbox[0:4]))

            detection = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": bbox_out,
                "score": float("{:.2f}".format(score))
            }
            if len(bbox) > 5:
                extreme_points = list(map(self._to_float, bbox[5:13]))
                detection["extreme_points"] = extreme_points
            detections.append(detection)
    print('total:',len(detections))
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir, thresh):
    json.dump(self.convert_eval_format(results, thresh), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir, thresh):
    self.save_results(results, save_dir, thresh)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
