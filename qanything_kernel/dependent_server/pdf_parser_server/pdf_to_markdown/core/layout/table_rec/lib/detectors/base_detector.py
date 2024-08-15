from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import torch

from models.model import create_model, load_model
from models.classifier import Processor, load_processor
from utils.utils import make_batch
from utils.image import get_affine_transform, get_affine_transform_upper_left

from utils.debugger import Debugger

class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')

    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.processor = Processor(opt)
    self.processor = load_model(self.processor, opt.load_processor)
    self.processor.cuda()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = opt.K
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) #+ 1
      inp_width = (new_width | self.opt.pad) #+ 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    if self.opt.upper_left:
      c = np.array([0, 0], dtype=np.float32)
      s = max(height, width) * 1.0
      trans_input = get_affine_transform_upper_left(c, s, 0, [inp_width, inp_height])
    else:
      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)

    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
    
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'input_height':inp_height,
            'input_width':inp_width,
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def save_img_txt(self,img):
    shape = list(img.shape)
    f1 = open('/home/rujiao.lrj/CenterNet_cell_Coord/src/img.txt','w')
    for i in range(shape[0]):
      for j in range(shape[1]):
        for k in range(shape[2]):
          data = img[i][j][k].item()
          f1.write(str(data)+'\n')
    f1.close()

  def Duplicate_removal(self, results, corners):
    bbox = []
    for j in range(len(results)):
      box = results[j]
      if box[-1] > self.opt.scores_thresh:
        for i in range(8):
          if box[i]<0:
            box[i]=0
          if box[i]>1024:
            box[i]=1024
        def dist(p1,p2):
            return ((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))**0.5
        p1,p2,p3,p4 = [box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]
        if dist(p1,p2)>3 and dist(p2,p3)>3 and dist(p3,p4)>3 and dist(p4,p1)>3:
            bbox.append(box)
        else:
            continue

    corner = []
    for i in range(len(corners)):
        if corners[i][-1] > self.opt.vis_thresh_corner:
            corner.append(corners[i])
    return np.array(bbox),np.array(corner)
   
  def filter(self, image_name, results, logi, ps):
    # this function select boxes
    batch_size, feat_dim = logi.shape[0], logi.shape[2]
    num_valid = sum(results[1][:,8] >= self.opt.vis_thresh)
   
    #if num_valid <= 900 : #opt.max_objs
    slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
    slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.int32)
    for i in range(batch_size):
      for j in range(num_valid):
        slct_logi[i,j,:] = logi[i,j,:].cpu()
        slct_dets[i,j,:] = ps[i,j,:].cpu()
    #else:
      #print('Error: Number of Detected Boxes Exceed the Model Defaults.')
      #quit()

    return torch.Tensor(slct_logi).cuda(), torch.Tensor(slct_dets).cuda()

  def process_logi(self, logi):
    logi_floor = logi.floor()
    dev = logi - logi_floor
    logi = torch.where(dev>0.5, logi_floor+1, logi_floor)
    
    return logi

  def _normalized_ps(self, ps, vocab_size):
    ps = torch.round(ps).to(torch.int64)
    ps = torch.where(ps < vocab_size, ps, (vocab_size-1) * torch.ones(ps.shape).to(torch.int64).cuda())
    ps = torch.where(ps >= 0, ps, torch.zeros(ps.shape).to(torch.int64).cuda())
    return ps

  def resize(self,image):
    h,w,_ = image.shape
    scale = 1024/(max(w,h)+1e-4)
    image = cv2.resize(image,(int(w*scale),int(h*scale)))
    image = cv2.copyMakeBorder(image,0,1024 - int(h*scale), 0, 1024 - int(w*scale),cv2.BORDER_CONSTANT, value=[0,0,0])
    return image,scale

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def ps_convert_minmax(self,results):
    detection = {}
    for j in range(1,self.num_classes+1):
      detection[j]=[]
    for j in range(1,self.num_classes+1):
      for bbox in results[j]:
        minx = min(bbox[0],bbox[2],bbox[4],bbox[6])
        miny = min(bbox[1],bbox[3],bbox[5],bbox[7])
        maxx = max(bbox[0],bbox[2],bbox[4],bbox[6])
        maxy = max(bbox[1],bbox[3],bbox[5],bbox[7])
        detection[j].append([minx,miny,maxx,maxy,bbox[-1]])
    for j in range(1,self.num_classes+1):
      detection[j] = np.array(detection[j])
    return detection

  def run(self, opt, image_or_path_or_tensor, image_anno=None, meta=None):
   
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
 
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
 

    if not opt.wiz_detect:
      batch = make_batch(opt, image_or_path_or_tensor, image_anno)
  
    detections = []
    hm = []
    corner_st = []
    if self.opt.demo!='':
      image_name = image_or_path_or_tensor.split('/')[-1]
      
    for scale in self.scales:
      
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
    
      else:
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        
      images = images.to(self.opt.device)
     
      torch.cuda.synchronize()

      if self.opt.wiz_detect:
        outputs, output, dets, corner_st_reg, forward_time, logi, cr, keep = self.process(images, image, return_time=True)
      else:
        outputs, output, dets, corner_st_reg, forward_time, logi, cr, keep = self.process(images,  image, return_time=True, batch=batch)

      raw_dets = dets

      torch.cuda.synchronize()
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)

      dets,corner_st_reg = self.post_process(dets, meta, corner_st_reg, scale)
      torch.cuda.synchronize()

      detections.append(dets)
      hm.append(keep)
       
    if self.opt.wiz_4ps or self.opt.wiz_2dpe:
      logi = logi + cr

    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
  
    slct_logi, slct_dets = self.filter(image_or_path_or_tensor, results, logi, raw_dets[:,:,:8])
    slct_dets = self._normalized_ps(slct_dets, 256)

    if self.opt.wiz_2dpe:
      if self.opt.wiz_stacking:
        _, slct_logi = self.processor(slct_logi, dets = slct_dets)
      else:
        slct_logi = self.processor(slct_logi, dets = slct_dets)
    else:
      if self.opt.wiz_stacking:
        _, slct_logi = self.processor(slct_logi)
      else:
        slct_logi = self.processor(slct_logi)

    slct_logi = self.process_logi(slct_logi)

    
    if self.opt.debug >= 1:
      self.show_results(debugger, image, results, corner_st_reg, image_name, slct_logi.squeeze())
    
    Results = self.ps_convert_minmax(results)
    return {'results': Results,'4ps':results,'corner_st_reg':corner_st_reg, 'hm': hm}