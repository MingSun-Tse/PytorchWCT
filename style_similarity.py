import os
import numpy as np
import sys
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import torch
import torchvision.transforms as transforms
from model import Encoder5
from PIL import Image
pjoin = os.path.join
from utils import logprint

def L2_loss(x, y):
  x = np.array(x).flatten()
  y = np.array(y).flatten()
  return np.linalg.norm(x - y)

@torch.no_grad()
def get_feats(stylized_img_fp, stylized_img_wct, stylized_img_ours, style_img):
  pair_number = style_img.split("pair")[1].split("_")[0]
  img_fp    = transforms.ToTensor()(Image.open(stylized_img_fp).convert("RGB")).cuda().unsqueeze(0)
  img_wct   = transforms.ToTensor()(Image.open(stylized_img_wct).convert("RGB")).cuda().unsqueeze(0)
  img_ours  = transforms.ToTensor()(Image.open(stylized_img_ours).convert("RGB")).cuda().unsqueeze(0)
  img_style = transforms.ToTensor()(Image.open(style_img).convert("RGB")).cuda().unsqueeze(0)
  global e5_wct, log; model = Encoder5(e5_wct).cuda()
  _ = model.forward_style_similarity(img_fp, "images/Picked_UserStudy_Samples/feats/feats_fp_pair" + str(pair_number) + "_stage%s.npy")[0]; torch.cuda.empty_cache()
  logprint("get_feats done: img_fp", log)
  _ = model.forward_style_similarity(img_wct, "images/Picked_UserStudy_Samples/feats/feats_wct_pair" + str(pair_number) + "_stage%s.npy")[0]; torch.cuda.empty_cache() 
  logprint("get_feats done: img_wct", log)
  _ = model.forward_style_similarity(img_ours, "images/Picked_UserStudy_Samples/feats/feats_ours_pair" + str(pair_number) + "_stage%s.npy")[0]; torch.cuda.empty_cache() 
  logprint("get_feats done: img_ours", log)
  _ = model.forward_style_similarity(img_style, "images/Picked_UserStudy_Samples/feats/feats_style_pair" + str(pair_number) + "_stage%s.npy")[0]; torch.cuda.empty_cache() 
  logprint("get_feats done: img_style", log)  

def get_feats():
  global log, stylized_img_dir
  for cnt in range(1, int(sys.argv[1]) + 1):
    logprint("====> Processing pair %d:" % cnt, log)
    style_img         = [pjoin(stylized_img_dir, i) for i in os.listdir(stylized_img_dir) if "pair%d_style" % cnt in i][0]
    stylized_img_fp   = pjoin(stylized_img_dir, "pair%d_stylised_fp.jpg"   % cnt)
    stylized_img_wct  = pjoin(stylized_img_dir, "pair%d_stylised_original.jpg"  % cnt)
    stylized_img_ours = pjoin(stylized_img_dir, "pair%d_stylised_ours.jpg" % cnt)
    get_feats(stylized_img_fp, stylized_img_wct, stylized_img_ours, style_img)
  
def get_style_loss():
  style_loss_fp   = np.zeros([int(sys.argv[1]), 5]) # shape: num pair x num stage (20 x 5)
  style_loss_wct  = np.zeros([int(sys.argv[1]), 5])
  style_loss_ours = np.zeros([int(sys.argv[1]), 5]) 
  gram_dir = "images/Picked_UserStudy_Samples/feats"
  for pair in range(1, int(sys.argv[1]) + 1):
    print("get style loss for pair: %s" % pair)
    for stage in range(1, 6):
      gram_fp = np.load(pjoin(gram_dir, "feats_fp_pair%s_stage%s.npy" % (pair, stage)))
      gram_wct = np.load(pjoin(gram_dir, "feats_wct_pair%s_stage%s.npy" % (pair, stage)))
      gram_ours = np.load(pjoin(gram_dir, "feats_ours_pair%s_stage%s.npy" % (pair, stage)))
      gram_style = np.load(pjoin(gram_dir, "feats_style_pair%s_stage%s.npy" % (pair, stage)))
      style_loss_fp[pair-1, stage-1] = L2_loss(gram_fp, gram_style)
      style_loss_wct[pair-1, stage-1] = L2_loss(gram_wct, gram_style)
      style_loss_ours[pair-1, stage-1] = L2_loss(gram_ours, gram_style)
  print(np.average(style_loss_fp, axis = 0))
  print(np.average(style_loss_wct, axis = 0))
  print(np.average(style_loss_ours, axis = 0))
  
stylized_img_dir = "images/Picked_UserStudy_Samples"
e5_wct = "models/vgg_normalised_conv5_1.t7"
log = open("log_style_similarity_%s.txt" % time.strftime("%Y%m%d-%H%M"), "w+")  
if __name__ == "__main__":
  if sys.argv[2] == "for_loss":
    get_style_loss()
  elif sys.argv[2] == "for_feat":
    get_feats()
