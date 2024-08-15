from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _h_dist_feat(output, width):
  feat =  (output[:,:,0] + output[:,:,1])/(2*(width+1))
  return feat

def _make_pair_feat(output):
  if len(output.shape) == 2:
    output = output.unsqueeze(2)

  output1 = output.unsqueeze(1).expand(output.size(0), output.size(1), output.size(1), output.size(2))
  output2 = output.unsqueeze(2).expand(output.size(0), output.size(1), output.size(1), output.size(2))
  output_paired = torch.cat((output1, output2), 3)

  return output_paired

def _v_dist_feat(output, height):
  feat =  (output[:,:,2] + output[:,:,3])/(2*(height + 1))
  return feat

def _gather_feat(feat, ind, mask=None):
  dim  = feat.size(2)
  ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  if mask is not None:
      mask = mask.unsqueeze(2).expand_as(feat)
      feat = feat[mask]
      feat = feat.view(-1, dim)
  return feat

def _flatten_and_gather_feat(output, ind):
  dim = output.size(3)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  output = output.contiguous().view(output.size(0), -1, output.size(3))
  output1 = output.gather(1, ind)

  return output1

def _get_4ps_feat(cc_match, output):
  if isinstance(output, dict):
    feat = output['cr']
  else :
    feat = output
  device = feat.device
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.contiguous().view(feat.size(0), -1, feat.size(3))
  feat = feat.unsqueeze(3).expand(feat.size(0), feat.size(1), feat.size(2), 4)

  dim = feat.size(2)
  cc_match = cc_match.unsqueeze(2).expand(cc_match.size(0), cc_match.size(1), dim, cc_match.size(2))
  if not(isinstance(output, dict)):
    cc_match = torch.where(cc_match<feat.shape[1], cc_match, (feat.shape[0]-1)* torch.ones(cc_match.shape).to(torch.int64).to(device))
    cc_match = torch.where(cc_match>=0, cc_match, torch.zeros(cc_match.shape).to(torch.int64).to(device))
  feat = feat.gather(1, cc_match)
  return feat

def _get_wh_feat(ind, output, ttype):
 
  width = output['hm'].shape[2]
  xs = (ind % width).unsqueeze(2).int().float()
  ys = (ind // width).unsqueeze(2).int().float()
  if ttype == 'gt':
    wh = output['wh']
  elif ttype == 'pred':
    wh = _tranpose_and_gather_feat(output['wh'], ind)
  ct = torch.cat([xs, ys, xs, ys, xs, ys, xs, ys], dim=2)
  bbx = ct - wh

  return bbx

def _normalized_ps(ps, vocab_size):
  device = ps.device
  ps = torch.round(ps).to(torch.int64)
  ps = torch.where(ps < vocab_size, ps, (vocab_size-1) * torch.ones(ps.shape).to(torch.int64).to(device))
  ps = torch.where(ps >= 0, ps, torch.zeros(ps.shape).to(torch.int64).to(device))
  return ps

def _tranpose_and_gather_feat(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feat(feat, ind)
  return feat

def flip_tensor(x):
    return torch.flip(x, [3])

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)