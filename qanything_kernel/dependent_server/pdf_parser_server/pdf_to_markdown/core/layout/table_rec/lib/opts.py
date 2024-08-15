from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('task', default='ctdet',
                             help='ctdet | ddd | multi_pose | exdet')
    self.parser.add_argument('--dataset', default='table',
                             help='coco | kitti | coco_hp | pascal')
    self.parser.add_argument('--dataset_name', default='WTW',
                             help='WTW | TG24K | PTN | ICDAR | SciTSR')
    self.parser.add_argument('--exp_id', default='default', 
                             help='name of this experiment')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--hold_det', action='store_true')
    self.parser.add_argument('--vis_corner', type=int, default=0,
                             help='vis corner or not'
                                  '0: do not vis corner'
                                  '1: vis corner')
    self.parser.add_argument('--convert_onnx', type=int, default=0,
                             help='0: donot convert'
                                  '1: convert pytorch model to onnx')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk')
    self.parser.add_argument('--demo', default='./demo_wtw/', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--demo_name', default='demo_results', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--output_dir', default='../data/table/', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')   
    self.parser.add_argument('--demo_dir', default='../demo/', 
                             help='path to demo output. '
                                  'or "webcam"')      
    self.parser.add_argument('--demo_output', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')                              
    self.parser.add_argument('--image_dir', default='../data/WTW/full_images', 
                             help='path to images of training set')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--load_processor', default='', 
                             help='path to pretrained processor')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 

    # system
    self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--anno_path', default='',
                             help='path to annotations of coco data')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--vis_thresh_corner', type=float, default=0.1,
                             help='visualization threshold for corner.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    
    # model
    self.parser.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested'
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')
    self.parser.add_argument('--load_mode', type=str, default='single',
                             help='single or multiple')
    self.parser.add_argument('--wiz_2dpe', action='store_true', default=False, 
                             help='whether to use the 2-dimensional position embeddings. ')
    self.parser.add_argument('--wiz_4ps', action='store_true', default=False, 
                             help='whether to use the feature of the four corner points. ')
    self.parser.add_argument('--wiz_stacking', action='store_true', default=False, 
                             help='whether to use the cascading regressor or the vinilla regressor. ')
    
    # model logical prediction head
    self.parser.add_argument('--input_size', type=int, default=256,
                             help='size of inputs of the logical location head')
    self.parser.add_argument('--hidden_size', type=int, default=256,
                             help='size of hidden states of the logical location head')
    self.parser.add_argument('--output_size', type=int, default=4,
                             help='size of output, i.e., the four logical indices')
    self.parser.add_argument('--tsfm_layers', type=int, default=6,
                             help='layers of base transformers')
    self.parser.add_argument('--stacking_layers', type=int, default=3,
                             help='single or multiple')
    self.parser.add_argument('--num_heads', type=int, default=8,
                             help='number of attention heads')
    self.parser.add_argument('--att_dropout', type=float, default=0.1,
                             help='dropout used in attention')
    self.parser.add_argument('--max_fmp_size', type=float, default=256,
                             help='max size of feature map, which corresponding to image input of 1024*1024')

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')

    # train
    self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--lr_step', type=str, default='80',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=90,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')
    self.parser.add_argument('--full_loss', default=False)
    self.parser.add_argument('--upper_left', action='store_true',
                             help='changing the padding pattern for on-line cloud service. ')
    self.parser.add_argument('--wiz_pairloss', action='store_true', default=False, 
                             help='using pairloss for wired table only to apply the parsing-grouping process. ')
    self.parser.add_argument('--wiz_dsloss', action='store_true', default=False, 
                             help='using the inter-cell loss. ')

    # inference
    self.parser.add_argument('--flip_test', action='store_true',
                             help='flip data augmentation.')
    self.parser.add_argument('--save_path', default='../data/dingding_shiyu/',
                             help='path for save test res.')
    self.parser.add_argument('--test_scales', type=str, default='1',
                             help='multi scale test augmentation.')
    self.parser.add_argument('--nms', action='store_false',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=100,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--thresh_min', type=float, default=0.5,
                             help='max number of output objects.') 
    self.parser.add_argument('--thresh_max', type=float, default=0.7,
                             help='max number of output objects.') 
    self.parser.add_argument('--thresh_conf', type=float, default=0.1,
                             help='max number of output objects.') 
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')
    self.parser.add_argument('--wiz_rev',  action='store_true', 
                             help='using the pasing-grouping mechanism for wired table only. ')
    self.parser.add_argument('--wiz_detect',  action='store_true', default=False, 
                             help='using the ground-truth bounding boxes and only infering the logical locations. ')

    # dataset
    self.parser.add_argument('--not_rand_crop', action='store_true',
                             help='not use the random crop data augmentation'
                                  'from CornerNet.')
    self.parser.add_argument('--shift', type=float, default=0.1,
                             help='when not using random crop'
                                  'apply shift augmentation.')
    self.parser.add_argument('--scale', type=float, default=0.4,
                             help='when not using random crop'
                                  'apply scale augmentation.')
    self.parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    self.parser.add_argument('--flip', type = float, default=0.5,
                             help='probability of applying flip augmentation.')
    self.parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')
    self.parser.add_argument('--MK', default=700,
                             help='max corner number')
    # multi_pose
    self.parser.add_argument('--aug_rot', type=float, default=0, 
                             help='probability of applying '
                                  'rotation augmentation.')
    # ddd
    self.parser.add_argument('--aug_ddd', type=float, default=0.5,
                             help='probability of applying crop augmentation.')
    self.parser.add_argument('--rect_mask', action='store_true',
                             help='for ignored object, apply mask on the '
                                  'rectangular region or just center point.')
    self.parser.add_argument('--kitti_split', default='3dop',
                             help='different validation split for kitti: '
                                  '3dop | subcnn')

    # loss
    self.parser.add_argument('--mse_loss', action='store_true',
                             help='use mse loss or focal loss to train '
                                  'keypoint heatmaps.')
    # ctdet
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--mk_weight', type=float, default=1,
                             help='loss weight for corner keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=1,
                             help='loss weight for bounding box size.')
    self.parser.add_argument('--st_weight', type=float, default=1,
                             help='loss weight for cell coor reg.')
    # multi_pose
    self.parser.add_argument('--hp_weight', type=float, default=1,
                             help='loss weight for human pose offset.')
    self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                             help='loss weight for human keypoint heatmap.')
    # ddd
    self.parser.add_argument('--dep_weight', type=float, default=1,
                             help='loss weight for depth.')
    self.parser.add_argument('--dim_weight', type=float, default=1,
                             help='loss weight for 3d bounding box size.')
    self.parser.add_argument('--rot_weight', type=float, default=1,
                             help='loss weight for orientation.')
    self.parser.add_argument('--peak_thresh', type=float, default=0.1)
    
    # task
    # ctdet
    self.parser.add_argument('--norm_wh', action='store_true',
                             help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
    self.parser.add_argument('--dense_wh', action='store_true',
                             help='apply weighted regression near center or '
                                  'just apply regression on center point.')
    self.parser.add_argument('--cat_spec_wh', action='store_true',
                             help='category specific bounding box size.')
    self.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')
    # exdet
    self.parser.add_argument('--agnostic_ex', action='store_true',
                             help='use category agnostic extreme points.')
    self.parser.add_argument('--scores_thresh', type=float, default=0.1,
                             help='threshold for extreme point heatmap.')
    self.parser.add_argument('--center_thresh', type=float, default=0.1,
                             help='threshold for centermap.')
    self.parser.add_argument('--aggr_weight', type=float, default=0.0,
                             help='edge aggregation weight.')
    # multi_pose
    self.parser.add_argument('--dense_hp', action='store_true',
                             help='apply weighted pose regression near center '
                                  'or just apply regression on center point.')
    self.parser.add_argument('--not_hm_hp', action='store_true',
                             help='not estimate human joint heatmap, '
                                  'directly use the joint offset from center.')
    self.parser.add_argument('--not_reg_hp_offset', action='store_true',
                             help='not regress local offset for '
                                  'human joint heatmaps.')
    self.parser.add_argument('--not_reg_bbox', action='store_true',
                             help='not regression bounding box size.')
    
    # ground truth validation
    self.parser.add_argument('--eval_oracle_hm', action='store_true', 
                             help='use ground center heatmap.')
    self.parser.add_argument('--eval_oracle_mk', action='store_true', 
                             help='use ground corner heatmap.')
    self.parser.add_argument('--eval_oracle_wh', action='store_true', 
                             help='use ground truth bounding box size.')
    self.parser.add_argument('--eval_oracle_offset', action='store_true', 
                             help='use ground truth local heatmap offset.')
    self.parser.add_argument('--eval_oracle_kps', action='store_true', 
                             help='use ground truth human pose offset.')
    self.parser.add_argument('--eval_oracle_hmhp', action='store_true', 
                             help='use ground truth human joint heatmaps.')
    self.parser.add_argument('--eval_oracle_hp_offset', action='store_true', 
                             help='use ground truth human joint local offset.')
    self.parser.add_argument('--eval_oracle_dep', action='store_true', 
                             help='use ground truth depth.')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    if 'resfpn' in opt.arch:
      opt.pad = 0
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    
    if opt.task == 'exdet':
      # assert opt.dataset in ['coco']
      num_hm = 1 if opt.agnostic_ex else opt.num_classes
      opt.heads = {'hm_t': num_hm, 'hm_l': num_hm, 
                   'hm_b': num_hm, 'hm_r': num_hm,
                   'hm_c': opt.num_classes}
      if opt.reg_offset:
        opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
    elif opt.task == 'ddd':
      # assert opt.dataset in ['gta', 'kitti', 'viper']
      opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
      if opt.reg_bbox:
        opt.heads.update(
          {'wh': 2})
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'ctdet':
      # assert opt.dataset in ['pascal', 'coco']
      opt.heads = {'hm': opt.num_classes,'st':8,
                   'wh': 8 if not opt.cat_spec_wh else 8 * opt.num_classes,
                   'ax': 256, 'cr': 256}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'ctdet_mid':
      # assert opt.dataset in ['pascal', 'coco']
      opt.heads = {'hm': opt.num_classes,'st':8,
                   'wh': 8 if not opt.cat_spec_wh else 8 * opt.num_classes,
                   'ax': 256, 'cr': 256}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'ctdet_small':
      # assert opt.dataset in ['pascal', 'coco']
      opt.heads = {'hm': opt.num_classes,'st':8,
                   'wh': 8 if not opt.cat_spec_wh else 8 * opt.num_classes,
                   'ax': 256, 'cr': 256}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'multi_pose':
      # assert opt.dataset in ['coco_hp']
      opt.flip_idx = dataset.flip_idx
      opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
      if opt.hm_hp:
        opt.heads.update({'hm_hp': 17})
      if opt.reg_hp_offset:
        opt.heads.update({'hp_offset': 2})
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    default_dataset_info = {
      'ctdet': {'default_resolution': [1024, 1024], 'num_classes': 2, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'table'},
      'ctdet_mid': {'default_resolution': [768, 768], 'num_classes': 2, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'table'},
      'ctdet_small': {'default_resolution': [512, 512], 'num_classes': 2, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'table'},
      'exdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'multi_pose': {
        'default_resolution': [512, 512], 'num_classes': 1, 
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                     [11, 12], [13, 14], [15, 16]]},
      'ddd': {'default_resolution': [384, 1280], 'num_classes': 3, 
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'kitti'},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
