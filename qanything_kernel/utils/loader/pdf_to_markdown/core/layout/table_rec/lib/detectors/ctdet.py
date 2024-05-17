from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from external.shapelyNMS import pnms
from models.decode import ctdet_decode,corner_decode,ctdet_4ps_decode,ctdet_st_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process,ctdet_4ps_post_process,ctdet_4ps_post_process_upper_left
from utils.post_process import ctdet_corner_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector
from PIL import Image

from matplotlib import cm


class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)

  def process_logi(self, logi):
    logi_floor = logi.floor()
    dev = logi - logi_floor
    logi = torch.where(dev>0.5, logi_floor+1, logi_floor)
  
    logi0 = logi[:,:,0].unsqueeze(2)
    logi2 = logi[:,:,2].unsqueeze(2)

    logi_st = torch.cat((logi0, logi0, logi2, logi2), dim=2)
    logi = torch.where(logi<logi_st, logi_st, logi)
    return logi

  def process(self, images, origin, return_time=False, batch=None):
 
    with torch.no_grad():
      #outputs, feature_maps = self.model(images)

      outputs = self.model(images)
      output = outputs[-1]

      if batch is None :
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if self.opt.reg_offset else None
        
      else:
        print('This results is generated from ground truth detection boxes.')
        hm = torch.Tensor(batch['hm']).unsqueeze(0).cuda()

        wh_ind = torch.tensor(batch['hm_ind']).expand(output['wh'].size(0), output['wh'].size(1), len(batch['hm_ind']))
        batchwh = torch.Tensor(batch['wh']).transpose(0,1).unsqueeze(0)
        wh = torch.zeros(size = output['wh'].size()).view(output['wh'].size(0), output['wh'].size(1), -1).scatter(2, wh_ind, batchwh)
        wh = wh.view(output['wh'].size(0), output['wh'].size(1), output['wh'].size(2), output['wh'].size(3)).cuda()
        #wh = wh + 2 * torch.rand(size = wh.shape).cuda()

        reg_ind = torch.tensor(batch['reg_ind']).expand(output['reg'].size(0), output['reg'].size(1), len(batch['reg_ind']))
        batchreg = torch.Tensor(batch['reg']).transpose(0,1).unsqueeze(0)
        reg = torch.zeros(size = output['reg'].size()).view(output['reg'].size(0), output['reg'].size(1), -1).scatter(2, reg_ind, batchreg)
        reg = reg.view(output['reg'].size(0), output['reg'].size(1), output['reg'].size(2), output['reg'].size(3)).cuda()
      
      st = output['st']
      ax = output['ax']
      cr = output['cr']
        
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None

      torch.cuda.synchronize()
      forward_time = time.time()

      #return dets [bboxes, scores, clses]
    
      scores, inds, ys, xs, st_reg, corner_dict = corner_decode(hm[:,1:2,:,:], st, reg, K=int(self.opt.MK))
      dets, keep, logi, cr = ctdet_4ps_decode(hm[:,0:1,:,:], wh, ax, cr, corner_dict, reg=reg, K=self.opt.K, wiz_rev = self.opt.wiz_rev)
      corner_output = np.concatenate((np.transpose(xs.cpu()),np.transpose(ys.cpu()),np.array(st_reg.cpu()),np.transpose(scores.cpu())), axis=2)
     
      #logi = self.process_logi(logi)

    if return_time:
      return outputs, output, dets, corner_output, forward_time, logi, cr, keep#, overlayed_map
    else:
      return outputs, output, dets, logi, cr, keep#, corner_output

  def post_process(self, dets, meta, corner_st, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    #return dets is list and what in dets is dict. key of dict is classes, value of dict is [bbox,score]
    if self.opt.upper_left:
      dets = ctdet_4ps_post_process_upper_left(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'], self.opt.num_classes)
    else:
      dets = ctdet_4ps_post_process(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'], self.opt.num_classes)
    corner_st = ctdet_corner_post_process(
        corner_st.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 9)
      dets[0][j][:, :8] /= scale
    return dets[0],corner_st[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         #soft_nms(results[j], Nt=0.5, method=2)
         results[j] = pnms(results[j],self.opt.thresh_min,self.opt.thresh_conf)
    scores = np.hstack(
      [results[j][:, 8] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 8] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :8] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 8] > self.opt.center_thresh:
          debugger.add_4ps_coco_bbox(detection[i, k, :8], detection[i, k, -1],
                                 detection[i, k, 8], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results, corner, image_name, logi=None):
    debugger.add_img(image, img_id='ctdet')
    m,n = corner.shape
    
    count = 0
 
    fc = open(self.opt.output_dir + self.opt.demo_name +'/center/'+image_name+'.txt','w') #bounding boxes saved
    fv = open(self.opt.output_dir + self.opt.demo_name +'/corner/'+image_name+'.txt','w')
    fl = open(self.opt.output_dir + self.opt.demo_name +'/logi/'+image_name+'.txt','w') #logic axis saved 
    for j in range(1, self.num_classes + 1):
      k = 0
      for m in range(len(results[j])):
        bbox = results[j][m]
        k = k + 1
        if bbox[8] > self.opt.vis_thresh:
       
          if len(logi.shape) == 1:
            debugger.add_4ps_coco_bbox(bbox[:8], j-1, bbox[8], logi, show_txt=True, img_id='ctdet')
          else:
            debugger.add_4ps_coco_bbox(bbox[:8], j-1, bbox[8], logi[m,:], show_txt=True, img_id='ctdet')
          for i in range(0,3):
            position_holder = 1
            fc.write(str(bbox[2*i])+','+str(bbox[2*i+1])+';')
            if not logi is None:
              if len(logi.shape) == 1:
                fl.write(str(int(logi[i]))+',')
              else:
                fl.write(str(int(logi[m,:][i]))+',')
          fc.write(str(bbox[6])+','+str(bbox[7])+'\n')

          if not logi is None:
            if len(logi.shape) == 1:
              fl.write(str(int(logi[3]))+'\n')
            else:
              fl.write(str(int(logi[m,:][3]))+'\n')

      if self.opt.vis_corner==1:
        for i in range(m):
          if corner[i,10] > self.opt.vis_thresh_corner:
            for w in range(0,4):
              position_holder = 1
              fv.write(str(corner[i,2*w])+','+str(corner[i,2*w+1])+';')
            fv.write(str(corner[i,8])+','+str(corner[i,9])+'\n')
            count+=1
            
    fc.close()
    fv.close() 
    debugger.save_all_imgs(image_name, self.opt.demo_dir)
      
 