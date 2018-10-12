import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
from util import *
import scipy.misc
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import time

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='images/content')
parser.add_argument('--stylePath',default='images/style')
parser.add_argument('--texturePath',default='images/texture')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')

#### original model
parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7', help='Path to the decoder1')

#### 16x model
# parser.add_argument('--e5', default='../KD/Bin/models/small16x_encoder/5SE_16x_QA_E30S10000.pth')
# parser.add_argument('--e4', default='../KD/Bin/models/small16x_encoder/e4.pth')
# parser.add_argument('--e3', default='../KD/Bin/models/small16x_encoder/e3.pth')
# parser.add_argument('--e2', default='../KD/Bin/models/small16x_encoder/e2.pth')
# parser.add_argument('--e1', default='models/vgg_normalised_conv1_1.pth')
# parser.add_argument('--d5', default='../KD/Experiments/Small16xDecoder/e5/weights/5SD_16x_QA_E30S10000.pth')
# parser.add_argument('--d4', default='../KD/Experiments/Small16xDecoder/e4/weights/4SD_16x_QA_E15S7000.pth')
# parser.add_argument('--d3', default='../KD/Experiments/Small16xDecoder/e3/weights/3SD_16x_QA_E16S0.pth')
# parser.add_argument('--d2', default='../KD/Experiments/Small16xDecoder/e2/weights/2SD_16x_QA_E16S6000.pth')
# parser.add_argument('--d1', default='../KD/Bin/models/my_decoder/1BD_E30S0.pth')

# #### 16x model -- first UHD model
# parser.add_argument('--e5', default='../KD/Bin/models/small16x_encoder/5SE_16x_QA_E20S0.pth')
# parser.add_argument('--e4', default='../KD/Bin/models/small16x_encoder/4SE_16x_QA_E20S0.pth')
# parser.add_argument('--e3', default='../KD/Bin/models/small16x_encoder/3SE_16x_QA_E17S9000.pth')
# parser.add_argument('--e2', default='../KD/Bin/models/small16x_encoder/2SE_16x_QA_E19S6000.pth')
# parser.add_argument('--e1', default='models/vgg_normalised_conv1_1.pth')
# parser.add_argument('--d5', default='../KD/Experiments/Small16xDecoder_2/e5/weights/5SD_16x_QA_E13S9000.pth')
# parser.add_argument('--d4', default='../KD/Experiments/Small16xDecoder_2/e4/weights/4SD_16x_QA_E14S5000.pth')
# parser.add_argument('--d3', default='../KD/Experiments/Small16xDecoder_2/e3/weights/3SD_16x_QA_E25S0.pth')
# parser.add_argument('--d2', default='../KD/Experiments/Small16xDecoder_2/e2/weights/2SD_16x_QA_E26S2000.pth')
# parser.add_argument('--d1', default='../KD/Bin/models/my_decoder/1BD_E30S0.pth')

# #### 16x model
# parser.add_argument('--e5', default='../KD/Bin/models/small16x_encoder/5SE_16x_QA_E20S0.pth')
# parser.add_argument('--e4', default='../KD/Bin/models/small16x_encoder/4SE_16x_QA_E20S0.pth')
# parser.add_argument('--e3', default='../KD/Bin/models/small16x_encoder/3SE_16x_QA_E17S9000.pth')
# parser.add_argument('--e2', default='../KD/Bin/models/small16x_encoder/2SE_16x_QA_E19S6000.pth')
# parser.add_argument('--e1', default='models/vgg_normalised_conv1_1.pth')
# parser.add_argument('--d5', default='../KD/Experiments/Small16xDecoder_2/e5/weights/5SD_16x_QA_E30S10000.pth')
# parser.add_argument('--d4', default='../KD/Experiments/Small16xDecoder_2/e4/weights/4SD_16x_QA_E30S10000.pth')
# parser.add_argument('--d3', default='../KD/Experiments/Small16xDecoder_2/e3/weights/3SD_16x_QA_E30S10000.pth')
# parser.add_argument('--d2', default='../KD/Experiments/Small16xDecoder_2/e2/weights/2SD_16x_QA_E30S10000.pth')
# parser.add_argument('--d1', default='../KD/Bin/models/my_decoder/1BD_E30S0.pth')

#### 16x model -- split
parser.add_argument('--e5', default='../KD/Bin/models/small16x_encoder/5SE_16x_QA_E20S0.pth')
parser.add_argument('--e4', default='../KD/Bin/models/small16x_encoder/4SE_16x_QA_E20S0.pth')
parser.add_argument('--e3', default='E3_faked_by_E5.pth') # replaced
parser.add_argument('--e2', default='E2_faked_by_E5.pth') # replaced
parser.add_argument('--e1', default='models/vgg_normalised_conv1_1.pth')
parser.add_argument('--d5', default='../KD/Experiments/Small16xDecoder_2/e5/weights/5SD_16x_QA_E13S9000.pth')
parser.add_argument('--d4', default='../KD/Experiments/Small16xDecoder_2/e4/weights/4SD_16x_QA_E14S5000.pth')
parser.add_argument('--d3', default='../KD/Experiments/Small16xDecoder_2/e3/weights/3SD_16x_QA_E25S0.pth')
parser.add_argument('--d2', default='D2_faked_by_D5.pth')
parser.add_argument('--d1', default='../KD/Bin/models/my_decoder/1BD_E30S0.pth')

#### 10x model
# parser.add_argument('--e5', default='../KD/Bin/models/small10x_encoder/5SE_10x_E24S0.pth')
# parser.add_argument('--e4', default='../KD/Bin/models/small10x_encoder/4SE_10x_E24S0.pth')
# parser.add_argument('--e3', default='../KD/Bin/models/small10x_encoder/3SE_10x_E24S0.pth')
# parser.add_argument('--e2', default='models/vgg_normalised_conv2_1.pth')
# parser.add_argument('--e1', default='models/vgg_normalised_conv1_1.pth')
# parser.add_argument('--d5', default='../KD/Bin/models/small10x_decoder/5SD_10x_E29S5000.pth')
# parser.add_argument('--d4', default='../KD/Bin/models/small10x_decoder/4SD_10x_E29S5000.pth')
# parser.add_argument('--d3', default='../KD/Bin/models/small10x_decoder/3SD_10x_E29S5000.pth')
# parser.add_argument('--d2', default='../KD/Bin/models/my_decoder/2BD_E30S0.pth')
# parser.add_argument('--d1', default='../KD/Bin/models/my_decoder/1BD_E30S0.pth')

parser.add_argument('-m', '--mode', type=str, help='training mode')
parser.add_argument('--fineSize', type=int, default=0, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on. default is 0")
parser.add_argument('--cImg', type=str)
parser.add_argument('--sImg', type=str)
parser.add_argument('--oImg', type=str)
parser.add_argument('--AE', type=str)
args = parser.parse_args()

def resize_and_crop(img, size):
  # w, h = img.size
  # if w <= h:
    # neww = size
    # newh = int(h * size / w)
  # else:
    # newh = size
    # neww = int(w * size / h)
  # img = img.resize([neww, newh])
  return img.resize([size, size])

wct = WCT(args)
@torch.no_grad()
def styleTransfer(encoder, decoder, contentImg, styleImg, csF):
    sF  = encoder(styleImg); torch.cuda.empty_cache()
    cF  = encoder(contentImg); torch.cuda.empty_cache()
    sF  = sF.data.cpu().squeeze(0)
    cF  = cF.data.cpu().squeeze(0)
    csF = wct.transform(cF, sF, csF, args.alpha)
    Img = decoder(csF); torch.cuda.empty_cache()
    return Img

wct.cuda(args.gpu)
start_time = time.time();
csF = torch.Tensor()
csF = Variable(csF)
csF = csF.cuda(args.gpu)
    
# WCT Style Transfer
cImg = Image.open(args.cImg).convert('RGB')
sImg = Image.open(args.sImg).convert('RGB')
if args.fineSize:
  cImg = resize_and_crop(cImg, args.fineSize)
  sImg = resize_and_crop(sImg, args.fineSize)
cImg = transforms.ToTensor()(cImg).unsqueeze(0).cuda(args.gpu)
sImg = transforms.ToTensor()(sImg).unsqueeze(0).cuda(args.gpu)

if args.AE == "5":
  cImg = styleTransfer(wct.e5, wct.d5, cImg, sImg, csF)
elif args.AE == "4":
  cImg = styleTransfer(wct.e4, wct.d4, cImg, sImg, csF)
elif args.AE == "3":
  cImg = styleTransfer(wct.e3, wct.d3, cImg, sImg, csF)
elif args.AE == "2":
  cImg = styleTransfer(wct.e2, wct.d2, cImg, sImg, csF)
elif args.AE == "1":
  cImg = styleTransfer(wct.e1, wct.d1, cImg, sImg, csF)
vutils.save_image(cImg.data.cpu().float(), "afterAE%s.jpg" % args.AE)

end_time = time.time()
print('Elapsed time is: %f' % (end_time - start_time))