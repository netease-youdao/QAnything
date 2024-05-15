from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, PairLoss, AxisLoss, _axis_eval
from models.decode import ctdet_decode
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_mk = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
                   NormRegL1Loss() if opt.norm_wh else \
                   RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.crit_st = self.crit_reg
    self.crit_ax = AxisLoss()
    self.pair_loss = PairLoss()
    self.opt = opt

  def forward(self, epoch, outputs, batch, logi=None, slogi=None):
    
    opt = self.opt
    """hm, re, off, wh losses are original losses of CenterNet detector, and the st loss is the loss for parsing-grouping in Cycle-CenterNet."""
    hm_loss, st_loss, re_loss, off_loss, wh_loss, lo_loss, ax_loss, sax_loss, sm_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0
 
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['hm_ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(batch['reg'].detach().cpu().numpy(), \
                        batch['reg_ind'].detach().cpu().numpy(), output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
      
      """LOSS FOR DETECTION MODULE"""
    
      if self.opt.wiz_pairloss:
        hm_loss += self.crit(output['hm'] , batch['hm'] ) / opt.num_stacks

        loss1, loss2 = \
        self.pair_loss(output['wh'],batch['hm_ind'],output['st'],batch['mk_ind'],batch['hm_mask'], \
        batch['mk_mask'],batch['ctr_cro_ind'],batch['wh'],batch['st'],batch['hm_ctxy'])
        
        wh_loss += loss1 / opt.num_stacks
        st_loss += loss2 / opt.num_stacks
      else:
        hm_loss += self.crit(output['hm'][:,0,:,:], batch['hm'][:,0,:,:]) / opt.num_stacks # only supervision on centers
        wh_loss += self.crit_wh(output['wh'], batch['hm_mask'], batch['hm_ind'], batch['wh'])
      
      if opt.reg_offset and opt.off_weight > 0:
          off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['reg_ind'], batch['reg']) / opt.num_stacks     

      """LOSS FOR RECONSTRUCTION MODULE"""
     
      ax_loss = self.crit_ax(output['ax'], batch['hm_mask'], batch['hm_ind'], batch['logic'], logi)
     
      '''COMBINING LOSSES'''
      
      loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
              opt.off_weight * off_loss + 2 * ax_loss 
      
      if self.opt.wiz_pairloss:
        loss = loss + st_loss
      
      if self.opt.wiz_stacking:
        sax_loss = self.crit_ax(output['ax'], batch['hm_mask'], batch['hm_ind'], batch['logic'], slogi)
        loss = loss + 2 * sax_loss
        sacc = _axis_eval(output['ax'], batch['hm_mask'], batch['hm_ind'], batch['logic'], slogi)
    '''CONSTRUCTING LOSS STATUS'''
    
    #weather asking for grouping
    if self.opt.wiz_pairloss :
      loss_stats = {'loss': loss, 'hm_l': hm_loss,  'wh_l': wh_loss, "st_l": st_loss, "ax_l": ax_loss}
    else:
      loss_stats = {'loss': loss, 'hm_l': hm_loss,  'wh_l': wh_loss, "ax_l": ax_loss} 

    #weather asking for stacking
    if self.opt.wiz_stacking:
      loss_stats['sax_l'] = sax_loss

    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None, processor=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer, processor)
  
  def _get_losses(self, opt):
    
    if opt.wiz_pairloss:
      loss_stats = ['loss', 'hm_l', 'wh_l',  'st_l', 'ax_l']
    else:
      loss_stats = ['loss', 'hm_l', 'wh_l',  'ax_l']

    if opt.wiz_stacking:
      loss_stats.append('sax_l')

    loss = CtdetLoss(opt)
    return loss_stats, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    mk_reg = output['mk_reg']
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :8] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :8] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 8] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :8], dets[i, k, -1],
                                 dets[i, k, 8], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 8] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :8], dets_gt[i, k, -1],
                                 dets_gt[i, k, 8], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]