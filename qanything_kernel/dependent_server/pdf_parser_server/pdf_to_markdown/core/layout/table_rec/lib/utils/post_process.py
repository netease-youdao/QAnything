from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds, transform_preds_upper_left


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ctdet_4ps_post_process(dets, c, s, h, w, num_classes,rot=0):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []

  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, 0:2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h),rot)
    dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h),rot)
    dets[i, :, 4:6] = transform_preds(dets[i, :, 4:6], c[i], s[i], (w, h),rot)
    dets[i, :, 6:8] = transform_preds(dets[i, :, 6:8], c[i], s[i], (w, h),rot)
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :8].astype(np.float32),
        dets[i, inds, 8:9].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def ctdet_4ps_post_process_upper_left(dets, c, s, h, w, num_classes,rot=0):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []

  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, 0:2] = transform_preds_upper_left(dets[i, :, 0:2], c[i], s[i], (w, h),rot)
    dets[i, :, 2:4] = transform_preds_upper_left(dets[i, :, 2:4], c[i], s[i], (w, h),rot)
    dets[i, :, 4:6] = transform_preds_upper_left(dets[i, :, 4:6], c[i], s[i], (w, h),rot)
    dets[i, :, 6:8] = transform_preds_upper_left(dets[i, :, 6:8], c[i], s[i], (w, h),rot)
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :8].astype(np.float32),
        dets[i, inds, 8:9].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def ctdet_corner_post_process(corner_st_reg, c, s, h, w, num_classes):
  for i in range(corner_st_reg.shape[0]):
    corner_st_reg[i, :, 0:2] = transform_preds(corner_st_reg[i, :, 0:2], c[i], s[i], (w, h))
    corner_st_reg[i, :, 2:4] = transform_preds(corner_st_reg[i, :, 2:4], c[i], s[i], (w, h))
    corner_st_reg[i, :, 4:6] = transform_preds(corner_st_reg[i, :, 4:6], c[i], s[i], (w, h))
    corner_st_reg[i, :, 6:8] = transform_preds(corner_st_reg[i, :, 6:8], c[i], s[i], (w, h))
    corner_st_reg[i, :, 8:10] = transform_preds(corner_st_reg[i, :, 8:10], c[i], s[i], (w, h))
  return corner_st_reg

