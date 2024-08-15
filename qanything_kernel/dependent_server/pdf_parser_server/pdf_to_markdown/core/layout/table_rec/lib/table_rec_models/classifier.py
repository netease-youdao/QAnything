from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat, _get_wh_feat, _get_4ps_feat, _normalized_ps
import torch.nn.functional as F

import json
import cv2
import os
from .transformer import Transformer
import math
import time
import random
import imgaug.augmenters as iaa
import time 
import copy

class Stacker(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, heads=8, dropout=0.1):
        super(Stacker, self).__init__()
        self.logi_encoder =  nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True) #newly added
        )
        self.tsfm = Transformer(2 * hidden_size, hidden_size, output_size, layers, heads, dropout)

    def forward(self, outputs, logi, mask = None, require_att = False):
      logi_embeddings = self.logi_encoder(logi)

      cat_embeddings = torch.cat((logi_embeddings, outputs), dim=2)

      if mask is None:
        if require_att:
          stacked_axis, att = self.tsfm(cat_embeddings)
        else:
          stacked_axis = self.tsfm(cat_embeddings)
      else:
        stacked_axis = self.tsfm(cat_embeddings, mask=mask)

      if require_att:
        return stacked_axis, att
      else:
        return stacked_axis

class Processor(nn.Module):
    def __init__(self, opt):
        super(Processor, self).__init__()
       
        if opt.wiz_stacking:
          self.stacker = Stacker(opt.output_size, opt.hidden_size, opt.output_size, opt.stacking_layers)

        #input_state, hidden_state, output_state, layers, heads, dropout
        self.tsfm_axis = Transformer(opt.input_size, opt.hidden_size, opt.output_size, opt.tsfm_layers, opt.num_heads, opt.att_dropout) #original version
        self.x_position_embeddings = nn.Embedding(opt.max_fmp_size, opt.hidden_size)
        self.y_position_embeddings = nn.Embedding(opt.max_fmp_size, opt.hidden_size)
        
        self.opt = opt
    
    def forward(self, outputs, dets = None, batch = None, cc_match = None): #training version forward
      # 'outputs' stands for the feature of cells
      # mask = None
      # att = None

      '''
        Constructing Features:
      '''
      if batch is None:
        # Inference Mode, the four corner features are gathered 
        # during bounding boxes decoding for simplicity (See ctdet_4ps_decode() in ./src/lib/model/decode.py).
        
        vis_feat = outputs
        if dets is None:
          feat = vis_feat

        else:
          left_pe = self.x_position_embeddings(dets[:, :, 0])
          upper_pe = self.y_position_embeddings(dets[:, :, 1])
          right_pe = self.x_position_embeddings(dets[:, :, 2])
          lower_pe = self.y_position_embeddings(dets[:, :, 5])
          feat = vis_feat + left_pe + upper_pe + right_pe + lower_pe

        # !TODO: moving the processings here and uniform the feature construction code for training and inference.
      
      else:
        #Training Mode
        ind = batch['hm_ind']
        mask = batch['hm_mask'] #during training, the attention mask will be applied
        output = outputs[-1]
        pred = output['ax']
        ct_feat = _tranpose_and_gather_feat(pred, ind)

        if self.opt.wiz_2dpe:        
          cr_feat = _get_4ps_feat(batch['cc_match'], output)
          cr_feat = cr_feat.sum(axis = 3)
          vis_feat = ct_feat + cr_feat
          
          ps = _get_wh_feat(ind, batch, 'gt')
          ps = _normalized_ps(ps, self.opt.max_fmp_size)

          left_pe = self.x_position_embeddings(ps[:, :, 0])
          upper_pe = self.y_position_embeddings(ps[:, :, 1])
          right_pe = self.x_position_embeddings(ps[:, :, 2])
          lower_pe = self.y_position_embeddings(ps[:, :, 5])

          feat = vis_feat + left_pe + upper_pe + right_pe + lower_pe

        elif self.opt.wiz_4ps:
          cr_feat = _get_4ps_feat(batch['cc_match'], output)
          cr_feat = cr_feat.sum(axis = 3)
          feat = ct_feat + cr_feat

        elif self.opt.wiz_vanilla:
          feat = ct_feat

      '''
        Put Features into TSFM:
      '''

      if batch is None:
        #Inference Mode
        logic_axis = self.tsfm_axis(feat) 
        if self.opt.wiz_stacking:
            stacked_axis = self.stacker(feat, logic_axis)
      else:
        #Training Mode
        logic_axis = self.tsfm_axis(feat, mask = mask)   
        if self.opt.wiz_stacking:
          stacked_axis = self.stacker(feat, logic_axis, mask = mask)

      if self.opt.wiz_stacking:
        return logic_axis, stacked_axis
      else:
        return logic_axis 

def load_processor(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    return model

def _judge(box):
    countx = len(list(set([box[0],box[2],box[4],box[6]]))) 
    county = len(list(set([box[1],box[3],box[5],box[7]]))) 
    if countx<2 or county<2:
        return False
    
    return True

