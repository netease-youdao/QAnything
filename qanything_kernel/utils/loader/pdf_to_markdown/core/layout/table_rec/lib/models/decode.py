from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat, _get_4ps_feat
import numpy as np 
import shapely
import time
from shapely.geometry import Polygon, MultiPoint, Point

def _nms(heat, name, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    #save_map(hmax.cpu().numpy()[0],name)
    keep = (hmax == heat).float()
    return heat * keep,keep

def _topk_channel(scores, K=40):
     
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40, device=None):
    #import ipdb
    #ipdb.set_trace()
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (torch.Tensor([height]).to(torch.int64).to(device) * torch.Tensor([width]).to(torch.int64).to(device))
    topk_ys   = (topk_inds / torch.Tensor([width]).to(device)).int().float()
    topk_xs   = (topk_inds % torch.Tensor([width]).to(torch.int64).to(device)).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def corner_decode(mk, st_reg, mk_reg=None, K=400,device=None):
    batch, cat, height, width = mk.size()
    mk,keep = _nms(mk,'mk.0.maxpool')
    scores, inds, clses, ys, xs = _topk(mk, K=K, device=device)
    if mk_reg is not None:
      reg = _tranpose_and_gather_feat(mk_reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    scores = scores.view(batch, K, 1)
    st_Reg = _tranpose_and_gather_feat(st_reg, inds)
    bboxes = torch.cat([xs - st_Reg[..., 0:1], 
                        ys - st_Reg[..., 1:2],
                        xs - st_Reg[..., 2:3], 
                        ys - st_Reg[..., 3:4],
                        xs - st_Reg[..., 4:5],
                        ys - st_Reg[..., 5:6],
                        xs - st_Reg[..., 6:7],
                        ys - st_Reg[..., 7:8]], dim=2)
    corner_dict = {'scores': scores, 'inds': inds, 'ys': ys, 'xs': xs, 'gboxes': bboxes}
    return scores, inds, ys, xs, bboxes, corner_dict

def ctdet_4ps_decode(heat, wh, ax, cr, corner_dict=None, reg=None, cat_spec_wh=False, K=100, wiz_rev = False):
    
    # if wiz_rev :
    #     print('Grouping and Parsing ...')
    batch, cat, height, width = heat.size()
    device = heat.device
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat,keep = _nms(heat,'hm.0.maxpool')
  
    scores, inds, clses, ys, xs = _topk(heat, K=K,device=device)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    ax = _tranpose_and_gather_feat(ax, inds)
    
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 8)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 8).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 8)
    else:
      wh = wh.view(batch, K, 8)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    '''
    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs + wh[..., 2:3], 
                        ys - wh[..., 3:4],
                        xs + wh[..., 4:5],
                        ys + wh[..., 5:6],
                        xs - wh[..., 6:7],
                        ys + wh[..., 7:8]], dim=2)
    '''

    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs - wh[..., 2:3], 
                        ys - wh[..., 3:4],
                        xs - wh[..., 4:5],
                        ys - wh[..., 5:6],
                        xs - wh[..., 6:7],
                        ys - wh[..., 7:8]], dim=2)

    rev_time_s1 = time.time()
    if wiz_rev :
        bboxes_rev = bboxes.clone()
        bboxes_cpu = bboxes.clone().cpu()

        gboxes = corner_dict['gboxes']
        gboxes_cpu = gboxes.cpu()

        num_bboxes = bboxes.shape[1]
        num_gboxes = gboxes.shape[1]

        corner_xs = corner_dict['xs'] 
        corner_ys = corner_dict['ys'] 
        corner_scores = corner_dict['scores'] 
        
       
        for i in range(num_bboxes):
            if scores[0,i,0] >= 0.2 :
                count = 0 # counting the number of ends of st head in bbox i
                for j in range(num_gboxes):
                    if corner_scores[0,j,0] >= 0.3:
                        #here comes to one pair of valid bbox and gbox
                        #step1 is there an overlap
                      
                        bbox = bboxes_cpu[0,i,:]
                        gbox = gboxes_cpu[0,j,:]
                        #rev_time_s3 = time.time()
                        if is_group_faster_faster(bbox, gbox): 
                            #step2 find which corner point to refine, and do refine
                            cr_x = corner_xs[0,j,0]
                            cr_y = corner_ys[0,j,0]
                        
                            ind4ps = find4ps(bbox, cr_x, cr_y, device)
                            if bboxes_rev[0, i, 2*ind4ps] == bboxes[0, i, 2*ind4ps] and bboxes_rev[0, i, 2*ind4ps+1] == bboxes[0, i, 2*ind4ps+1]:
                                #first_shift
                                count = count + 1
                                bboxes_rev[0, i, 2*ind4ps] = cr_x
                                bboxes_rev[0, i, 2*ind4ps + 1] = cr_y
                            else:
                                origin_x = bboxes[0, i, 2*ind4ps]
                                origin_y = bboxes[0, i, 2*ind4ps+1]

                                old_x = bboxes_rev[0, i, 2*ind4ps]
                                old_y = bboxes_rev[0, i, 2*ind4ps+1]

                                if dist(origin_x, origin_y, old_x, old_y) >= dist(origin_x, origin_y, cr_x, cr_y):
                                    count = count + 1
                                    bboxes_rev[0, i, 2*ind4ps] = cr_x
                                    bboxes_rev[0, i, 2*ind4ps + 1] = cr_y
                                else:
                                    continue
                        else:
                           
                            continue
                    else:
                        break        
                if count <= 2:
                    scores[0,i,0] = scores[0,i,0] * 0.4
            else :
                break

    if wiz_rev:

        cc_match = torch.cat([(bboxes_rev[:,:,0:1]) + width * torch.round(bboxes_rev[:,:,1:2]),
                            (bboxes_rev[:,:,2:3]) + width * torch.round(bboxes_rev[:,:,3:4]),
                            (bboxes_rev[:,:,4:5]) + width * torch.round(bboxes_rev[:,:,5:6]),
                            (bboxes_rev[:,:,6:7]) + width * torch.round(bboxes_rev[:,:,7:8])], dim=2)
    
    else:    
        cc_match = torch.cat([(xs - wh[..., 0:1]) + width * torch.round(ys - wh[..., 1:2]),
                            (xs - wh[..., 2:3]) + width * torch.round(ys - wh[..., 3:4]),
                            (xs - wh[..., 4:5]) + width * torch.round(ys - wh[..., 5:6]),
                            (xs - wh[..., 6:7]) + width * torch.round(ys - wh[..., 7:8])], dim=2)
    
    cc_match = torch.round(cc_match).to(torch.int64)

    cr_feat = _get_4ps_feat(cc_match, cr)
    cr_feat = cr_feat.sum(axis = 3)
    if wiz_rev:
        detections = torch.cat([bboxes_rev, scores, clses], dim=2)
        _, sorted_ind = torch.sort(scores, descending=True, dim=1)
        sorted_inds = sorted_ind.expand(detections.size(0), detections.size(1), detections.size(2))
        detections = detections.gather(1, sorted_inds)
        sorted_inds2 = sorted_ind.expand(detections.size(0), detections.size(1), ax.size(2))
        ax =  ax.gather(1, sorted_inds2)
    else:
       
        detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, keep, ax, cr_feat

def wireless_decode(heat, wh, ax, cr, reg=None, cat_spec_wh=False, K=100):
    
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat,keep = _nms(heat,'hm.0.maxpool')
  
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    ax = _tranpose_and_gather_feat(ax, inds)
    
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 8)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 8).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 8)
    else:
      wh = wh.view(batch, K, 8)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    '''
    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs + wh[..., 2:3], 
                        ys - wh[..., 3:4],
                        xs + wh[..., 4:5],
                        ys + wh[..., 5:6],
                        xs - wh[..., 6:7],
                        ys + wh[..., 7:8]], dim=2)
    '''
    bboxes = torch.cat([xs - wh[..., 0:1], 
                        ys - wh[..., 1:2],
                        xs - wh[..., 2:3], 
                        ys - wh[..., 3:4],
                        xs - wh[..., 4:5],
                        ys - wh[..., 5:6],
                        xs - wh[..., 6:7],
                        ys - wh[..., 7:8]], dim=2)
  
    cc_match = torch.cat([(xs - wh[..., 0:1]) + width * torch.round(ys - wh[..., 1:2]),
                        (xs - wh[..., 2:3]) + width * torch.round(ys - wh[..., 3:4]),
                        (xs - wh[..., 4:5]) + width * torch.round(ys - wh[..., 5:6]),
                        (xs - wh[..., 6:7]) + width * torch.round(ys - wh[..., 7:8])], dim=2)

    cc_match = torch.round(cc_match).to(torch.int64)

    cr_feat = _get_4ps_feat(cc_match, cr)
    cr_feat = cr_feat.sum(axis = 3)
    
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections, keep, ax, cr_feat

def find4ps(bbox, x, y,device):
    xs = torch.Tensor([bbox[0],bbox[2],bbox[4],bbox[6]]).to(device)
    ys = torch.Tensor([bbox[1],bbox[3],bbox[5],bbox[7]]).to(device)

    dx = xs - x
    dy = ys - y

    l = dx**2 + dy**2
    return torch.argmin(l)

def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    l = dx**2 + dy**2
    return l

def rect_inter(b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2):
    if (b1_x1 <= b2_x1  and b2_x1 <= b1_x2) or (b1_x1 <= b2_x2  and b2_x2 <= b1_x2):
        if (b1_y1 <= b2_y1  and b2_y1 <= b1_y2) or (b1_y1 <= b2_y2  and b2_y2 <= b1_y2):
            return True
        else:
            return False
    else:
        return False

def is_group_faster_faster(bbox, gbox):
    bbox = bbox.view(4,2)
    gbox = gbox.view(4,2)
  
    bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax = bbox[:,0].min(), bbox[:,0].max(), bbox[:,1].min(), bbox[:,1].max()#min(bbox_xs), max(bbox_xs), min(bbox_ys), max(bbox_ys)
    gbox_xmin, gbox_xmax, gbox_ymin, gbox_ymax = gbox[:,0].min(), gbox[:,0].max(), gbox[:,1].min(), gbox[:,1].max()

    if bbox_xmin > gbox_xmax or gbox_xmin > bbox_xmax or bbox_ymin > gbox_ymax or gbox_ymin > bbox_ymax:
        return False
    else:
        bpoly = Polygon(bbox)

        flag = 0
        for i in range(4):
            p = Point(gbox[i])
            if p.within(bpoly):
                flag = 1
                break
        if flag == 0:
            return False
        else :
            return True

def ctdet_st_decode(heat, st, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    heat,keep = _nms(heat,'hm.0.maxpool')
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    st = _tranpose_and_gather_feat(st, inds)
    if cat_spec_wh:
      st = st.view(batch, K, cat, 4)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 4).long()
      st = st.gather(2, clses_ind).view(batch, K, 4)
    else:
      st = st.view(batch, K, 4)
      
    return st

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections

