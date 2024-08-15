from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform, get_affine_transform_upper_left
from utils.post_process import ctdet_4ps_post_process
from utils.image import gaussian_radius, draw_umich_gaussian, draw_umich_gaussian_wh, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.adjacency import adjacency, h_adjacency, v_adjacency, same_col, same_row
import math
import time
import random
import imgaug.augmenters as iaa
import time 

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def _get_border_upper_left(self, border, size):
    i = 1
    while size/2 - border // i <= border // i:
        i *= 2
    return border // i

  def _get_radius(self,r,w,h):
    if w > h:
        k = float(w)/float(h)
    else:
        k = float(h)/float(w)
    ratio = k**0.5
    if w>h:
        r_w = r*ratio
        r_h = r
    else:
        r_h = r*ratio
        r_w = r
    return int(r_w),int(r_h)

  def color(self,image,p,magnitude):
    if np.random.randint(0,10) > p*10:
      return image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    img_float,bgr_img_float = img.astype(float), bgr_img.astype(float)
    diff = img_float - bgr_img_float
    diff = diff*magnitude
    diff_img_ = diff + bgr_img_float
    diff_img_ = diff_img_.astype(np.uint8)
    diff_img_ = np.array(diff_img_)
    diff_img_ = np.clip(diff_img_,0,255)
    diff_img_ = cv2.cvtColor(diff_img_,cv2.COLOR_BGR2RGB)
    diff_img_ = cv2.cvtColor(diff_img_,cv2.COLOR_RGB2BGR)
    return diff_img_

  def rotate(self,p,magnitude):
    if np.random.randint(0,10) > p*10:
      return 0
    rot = np.random.randint(magnitude[0],magnitude[1])
    return rot

  def hisEqulColor(self,img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return img

  def _judge(self,box):
    countx = len(list(set([box[0],box[2],box[4],box[6]]))) 
    county = len(list(set([box[1],box[3],box[5],box[7]]))) 
    if countx<2 or county<2:
        return False
    
    return True

  def _get_Center(self, point):
    x1 = point[0]
    y1 = point[1]
    x3 = point[2]
    y3 = point[3]
    x2 = point[4]
    y2 = point[5]
    x4 = point[6]
    y4 = point[7]
    w1 = math.sqrt((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3))
    w2 = math.sqrt((x2-x4)*(x2-x4)+(y2-y4)*(y2-y4))
    h1 = math.sqrt((x1-x4)*(x1-x4)+(y1-y4)*(y1-y4))
    h2 = math.sqrt((x2-x3)*(x2-x3)+(y2-y3)*(y2-y3))
    nw = min(w1,w2)
    nh = min(h1,h2)
    x_dev = x4*y2-x4*y1-x3*y2+x3*y1-x2*y4+x2*y3+x1*y4-x1*y3
    y_dev = y4*x2-y4*x1-y3*x2+x1*y3-y2*x4+y2*x3+y1*x4-y1*x3
    c_x = 0
    c_y = 0
    if x_dev != 0:
      c_x = (y3*x4*x2-y4*x3*x2-y3*x4*x1+y4*x3*x1-y1*x2*x4+y2*x1*x4+y1*x2*x3-y2*x1*x3)/x_dev
    if y_dev != 0:
      c_y = (-y3*x4*y2+y4*x3*y2+y3*x4*y1-y4*x3*y1+y1*x2*y4-y1*x2*y3-y2*x1*y4+y2*x1*y3)/y_dev
    return nw,nh,c_x,c_y

  def _rank(self,bbox,cter,file_name):
    init_bbox = bbox
    #bbox = list(map(float,bbox))
    continue_sign = False
    bbox = [bbox[0:2],bbox[2:4],bbox[4:6],bbox[6:8]]
    bbox_= np.array(bbox) - np.array(cter)
    i,box_y,sign= 0,[],'LT'
    choice = []
    for box in bbox_:
        if box[0]<0 and box[1]<0:
            box_y.append(box)
            choice.append(i)
        i = i + 1
    if len(choice)==0: 
        i,box_y,sign = 0,[],'RT'
        for box in bbox_:
            if box[0]>0 and box[1]<0:  
                box_y.append(box)
                choice.append(i)
            i = i + 1
    if sign=='LT':
        ylist = np.array(box_y)[:,1]
        #index = list(ylist).index(max(ylist))  
        index = list(ylist).index(min(ylist))  
    elif sign=='RT':
        try:
            xlist = np.array(box_y)[:,0]
        except Exception as e: 
            print("center:",cter,"box:",init_bbox,"box_y:",box_y)
            return True,bbox
        index = list(xlist).index(min(xlist))  
    
    index = choice[index]
    p = []
    for i in range(4):
        if i + index < 4:
            p.append(bbox[index+i])
        else:
            p.append(bbox[index+i-4])
    return continue_sign,[p[0][0],p[0][1],p[1][0],p[1][1],p[2][0],p[2][1],p[3][0],p[3][1]]

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    if self.opt.dataset_name == 'ICDAR19':
      if self.split == 'train':
        img_path = os.path.join(self.img_dir, 'train_images' ,file_name)
      else:
        img_path = os.path.join(self.img_dir, 'test_images' ,file_name)
    else:
      img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
    num_cors = self.max_cors
    if self.opt.dataset_name == 'TG24K':
      img_path = img_path.replace('.jpg', '_org.png')
    elif self.opt.dataset_name == 'SciTSR':
      img_path = img_path.replace('.jpg', '.png')
    elif self.opt.dataset_name == 'PTN':
      img_path = img_path.replace('.jpg', '.png')
    elif self.opt.dataset_name == 'bankdata_june':
      img_path = img_path[:-4]
   
    img = cv2.imread(img_path)
    img_size = img.shape

    height, width = img.shape[0], img.shape[1]

    if self.opt.upper_left:
      c = np.array([0, 0], dtype=np.float32)
    else:
      c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

    if self.opt.keep_res:
      input_h = (height | self.opt.pad)# + 1
      input_w = (width | self.opt.pad)# + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
   
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        if self.opt.upper_left:
          c = np.array([0, 0], dtype=np.float32)
        else:
          s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
          w_border = self._get_border(128, img.shape[1])
          h_border = self._get_border(128, img.shape[0])
          c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
          c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
    
    rot = 0
    if self.opt.rotate==1:
      print('----rotate----')
      rot = np.random.randint(-15,15) 

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio

    if self.opt.upper_left:
      trans_input = get_affine_transform_upper_left(c, s, rot, [input_w, input_h])
      trans_output = get_affine_transform_upper_left(c, s, rot, [output_w, output_h])
      trans_output_mk = get_affine_transform_upper_left(c, s, rot, [output_w, output_h])
    else:
     
      trans_input = get_affine_transform(c, s, rot, [input_w, input_h])
      trans_output = get_affine_transform(c, s, rot, [output_w, output_h])
      trans_output_mk = get_affine_transform(c, s, rot, [output_w, output_h])
      
    num_classes = self.num_classes
    
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 8), dtype=np.float32)
    reg = np.zeros((self.max_objs*5, 2), dtype=np.float32)
    st = np.zeros((self.max_cors, 8), dtype=np.float32)
    hm_ctxy = np.zeros((self.max_objs, 2), dtype=np.float32)
    hm_ind = np.zeros((self.max_objs), dtype=np.int64)
    hm_mask = np.zeros((self.max_objs), dtype=np.uint8)
    mk_ind = np.zeros((self.max_cors), dtype=np.int64)
    mk_mask = np.zeros((self.max_cors), dtype=np.uint8)
    reg_ind = np.zeros((self.max_objs*5), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs*5), dtype=np.uint8)
    ctr_cro_ind = np.zeros((self.max_objs*4), dtype=np.int64)
    log_ax = np.zeros((self.max_objs, 4), dtype=np.float32)
    cc_match = np.zeros((self.max_objs, 4), dtype=np.int64)
    h_pair_ind = np.zeros((self.max_pairs), dtype=np.int64)
    v_pair_ind = np.zeros((self.max_pairs), dtype=np.int64)
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    gt_det = []
    corList = []
    point = []
    pair_mark = 0
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h),flags=cv2.INTER_LINEAR)
    
    for k in range(num_objs):
      ann = anns[k]
      
      seg_mask = ann['segmentation'][0] #[[351.0, 73.0, 172.0, 70.0, 174.0, 127.0, 351.0, 129.0, 351.0, 73.0]]
      x1,y1 = seg_mask[0],seg_mask[1]
      x2,y2 = seg_mask[2],seg_mask[3]
      x3,y3 = seg_mask[4],seg_mask[5]
      x4,y4 = seg_mask[6],seg_mask[7]
   
      CorNer = np.array([x1,y1,x2,y2,x3,y3,x4,y4])
      boxes = [[CorNer[0],CorNer[1]],[CorNer[2],CorNer[3]],\
               [CorNer[4],CorNer[5]],[CorNer[6],CorNer[7]]]
      cls_id = int(self.cat_ids[ann['category_id']])

      if flipped:
       
        CorNer[[0,2,4,6]] = width - CorNer[[2,0,6,4]] - 1

      CorNer[0:2] = affine_transform(CorNer[0:2], trans_output_mk)
      CorNer[2:4] = affine_transform(CorNer[2:4], trans_output_mk)
      CorNer[4:6] = affine_transform(CorNer[4:6], trans_output_mk)
      CorNer[6:8] = affine_transform(CorNer[6:8], trans_output_mk)
      CorNer[[0,2,4,6]] = np.clip(CorNer[[0,2,4,6]], 0, output_w - 1)
      CorNer[[1,3,5,7]] = np.clip(CorNer[[1,3,5,7]], 0, output_h - 1)
      if not self._judge(CorNer):
          continue
 
      maxx = max([CorNer[2*I] for I in range(0,4)])
      minx = min([CorNer[2*I] for I in range(0,4)])
      maxy = max([CorNer[2*I+1] for I in range(0,4)])
      miny = min([CorNer[2*I+1] for I in range(0,4)])
      h, w = maxy-miny, maxx-minx #bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
       
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius

        ct = np.array([(maxx+minx)/2.0,(maxy+miny)/2.0], dtype=np.float32)
        ct_int = ct.astype(np.int32)
       
        draw_gaussian(hm[cls_id], ct_int, radius)

        for i in range(4):
          Cor = np.array([CorNer[2*i],CorNer[2*i+1]], dtype=np.float32)
          Cor_int = Cor.astype(np.int32)
          Cor_key = str(Cor_int[0])+"_"+str(Cor_int[1])
          if Cor_key not in corList:
            
            corNum = len(corList)
            
            corList.append(Cor_key)
            reg[self.max_objs+corNum] = np.array([abs(Cor[0]-Cor_int[0]),abs(Cor[1]-Cor_int[1])])
            mk_ind[corNum] = Cor_int[1]*output_w + Cor_int[0]
            cc_match[k][i] = mk_ind[corNum]
            reg_ind[self.max_objs+corNum] = Cor_int[1]*output_w + Cor_int[0]
            mk_mask[corNum] = 1
            reg_mask[self.max_objs+corNum] = 1
            draw_gaussian(hm[num_classes-1], Cor_int, 2)
            st[corNum][i*2:(i+1)*2] = np.array([Cor[0]-ct[0],Cor[1]-ct[1]])
            ctr_cro_ind[4*k+i] = corNum*4 + i
        
          else:
            index_of_key = corList.index(Cor_key)
            cc_match[k][i] = mk_ind[index_of_key]
            st[index_of_key][i*2:(i+1)*2] = np.array([Cor[0]-ct[0],Cor[1]-ct[1]])
            ctr_cro_ind[4*k+i] = index_of_key*4 + i
            
        wh[k] = ct[0] - 1. * CorNer[0], ct[1] - 1. * CorNer[1], \
                ct[0] - 1. * CorNer[2], ct[1] - 1. * CorNer[3], \
                ct[0] - 1. * CorNer[4], ct[1] - 1. * CorNer[5], \
                ct[0] - 1. * CorNer[6], ct[1] - 1. * CorNer[7]
        
        hm_ind[k] = ct_int[1] * output_w + ct_int[0]
        hm_mask[k] = 1
        reg_ind[k] = ct_int[1] * output_w + ct_int[0]
        reg_mask[k] = 1
        reg[k] = ct - ct_int
        hm_ctxy[k] = ct[0],ct[1]

        log_ax[k] = ann['logic_axis'][0][0], ann['logic_axis'][0][1], ann['logic_axis'][0][2], ann['logic_axis'][0][3]

     
        gt_det.append([ct[0] - 1. * CorNer[0], ct[1] - 1. * CorNer[1],
                       ct[0] - 1. * CorNer[2], ct[1] - 1. * CorNer[3],
                       ct[0] - 1. * CorNer[4], ct[1] - 1. * CorNer[5], 
                       ct[0] - 1. * CorNer[6], ct[1] - 1. * CorNer[7], 1, cls_id])
        
    hm_mask_v = hm_mask.reshape(1, hm_mask.shape[0])
  
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    ret = {'input': inp, 'hm': hm, 'hm_ind':hm_ind, 'hm_mask':hm_mask, 'mk_ind':mk_ind, 'mk_mask':mk_mask, 'reg':reg,'reg_ind':reg_ind,'reg_mask': reg_mask, \
           'wh': wh,'st':st, 'ctr_cro_ind':ctr_cro_ind, 'cc_match': cc_match, 'hm_ctxy':hm_ctxy, 'logic': log_ax, 'h_pair_ind': h_pair_ind, 'v_pair_ind': v_pair_ind}

    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 10), dtype=np.float32)
      meta = {'c': c, 's': s, 'rot':rot, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret
