import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
import time
from utils import logprint

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--UHD_contentPath',default='images/UHD_content')
parser.add_argument('--UHD_stylePath',default='images/UHD_style')
parser.add_argument('--contentPath',default='images/content')
parser.add_argument('--stylePath',default='images/style')
parser.add_argument('--texturePath',default='images/texture')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--mode', type=str)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--UHD', action='store_true')
parser.add_argument('--synthesis', action="store_true")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=0, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on. default is 0")
parser.add_argument('--log_mark', type=str, default=time.strftime("%Y%m%d-%H%M") )
parser.add_argument('--picked_content_mark', type=str, default=".")
parser.add_argument('--picked_style_mark', type=str, default=".")
parser.add_argument('--num_run', type=int, default = 1)

#### original WCT models
parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.pth', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.pth', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.pth', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.pth', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.pth', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.pth', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.pth', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.pth', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.pth', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.pth', help='Path to the decoder1')

#### 16x model -- training failed model
# note the decoder model is from Decoder
# parser.add_argument('--e5', default='../KD/Bin/models/small16x_encoder/old_models_wo_styleloss/5SE_16x_QA_E30S10000.pth')
# parser.add_argument('--e4', default='../KD/Bin/models/small16x_encoder/e4.pth')
# parser.add_argument('--e3', default='../KD/Bin/models/small16x_encoder/e3.pth')
# parser.add_argument('--e2', default='../KD/Bin/models/small16x_encoder/e2.pth')
# parser.add_argument('--e1', default='models/vgg_normalised_conv1_1.pth')
# parser.add_argument('--d5', default='../KD/Experiments/Small16xDecoder/e5/weights/5SD_16x_QA_E30S10000.pth')
# parser.add_argument('--d4', default='../KD/Experiments/Small16xDecoder/e4/weights/4SD_16x_QA_E15S7000.pth')
# parser.add_argument('--d3', default='../KD/Experiments/Small16xDecoder/e3/weights/3SD_16x_QA_E16S0.pth')
# parser.add_argument('--d2', default='../KD/Experiments/Small16xDecoder/e2/weights/2SD_16x_QA_E16S6000.pth')
# parser.add_argument('--d1', default='../KD/Bin/models/my_decoder/1BD_E30S0.pth')

#### FP16x model
# parser.add_argument('--e5', default='../KD/Bin/convert_caffemodel_to_pth/normalise_vgg/fp16x_subset1000/fp16x_normalised_FP16x_5E.pth')
# parser.add_argument('--e4', default='../KD/Bin/convert_caffemodel_to_pth/normalise_vgg/fp16x_subset1000/fp16x_normalised_FP16x_4E.pth')
# parser.add_argument('--e3', default='../KD/Bin/convert_caffemodel_to_pth/normalise_vgg/fp16x_subset1000/fp16x_normalised_FP16x_3E.pth')
# parser.add_argument('--e2', default='../KD/Bin/convert_caffemodel_to_pth/normalise_vgg/fp16x_subset1000/fp16x_normalised_FP16x_2E.pth')
# parser.add_argument('--e1', default='../KD/Bin/convert_caffemodel_to_pth/normalise_vgg/fp16x_subset1000/fp16x_normalised_FP16x_1E.pth')
# parser.add_argument('--d5', default='../KD/Experiments/SmallFP16xDecoder/e5/weights/12-20181022-0801_5SD_FP16x_E25S0-3.pth')
# parser.add_argument('--d4', default='../KD/Experiments/SmallFP16xDecoder/e4_reimpl/weights/12-20181023-0551_4SD_FP16x_E25S0-3.pth')
# parser.add_argument('--d3', default='../KD/Experiments/SmallFP16xDecoder/e3/weights/12-20181022-1528_3SD_FP16x_E25S0-3.pth')
# parser.add_argument('--d2', default='../KD/Experiments/SmallFP16xDecoder/e2/weights/12-20181022-1529_2SD_FP16x_E25S0-3.pth')
# parser.add_argument('--d1', default='../KD/Experiments/SmallFP16xDecoder/e1/weights/12-20181022-1529_1SD_FP16x_E25S0-3.pth')

# #### JointED 16x model
# parser.add_argument('--e5', default='../KD/Experiments/Small16xEncoder/e5_JointED/weights/12-20181022-1654_5SED_16x_E30S0-2.pth')
# parser.add_argument('--d5', default='../KD/Experiments/Small16xEncoder/e5_JointED/weights/12-20181022-1654_5SED_16x_E30S0-3.pth')
# parser.add_argument('--e4', default='../KD/Experiments/Small16xEncoder/e4_JointED/weights/12-20181023-0618_4SED_16x_E15S0-2.pth')
# parser.add_argument('--d4', default='../KD/Experiments/Small16xEncoder/e4_JointED/weights/12-20181023-0618_4SED_16x_E15S0-3.pth')
# parser.add_argument('--e3', default='../KD/Experiments/Small16xEncoder/e3_JointED/weights/12-20181023-0619_3SED_16x_E16S0-2.pth')
# parser.add_argument('--d3', default='../KD/Experiments/Small16xEncoder/e3_JointED/weights/12-20181023-0619_3SED_16x_E16S0-3.pth')
# parser.add_argument('--e2', default='../KD/Experiments/Small16xEncoder/e2_JointED/weights/12-20181023-0620_2SED_16x_E17S0-2.pth')
# parser.add_argument('--d2', default='../KD/Experiments/Small16xEncoder/e2_JointED/weights/12-20181023-0620_2SED_16x_E17S0-3.pth')
# parser.add_argument('--e1', default='../KD/Experiments/Small16xEncoder/e1_JointED/weights/12-20181023-0641_1SED_16x_E19S0-2.pth')
# parser.add_argument('--d1', default='../KD/Experiments/Small16xEncoder/e1_JointED/weights/12-20181023-0641_1SED_16x_E19S0-3.pth')

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

# #### 16x model -- first UHD model, decoder trained more
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

# #### 16x model -- UHD model, Conv1_1 pruned
parser.add_argument('--e5', default='../KD/Bin/models/small16x_encoder/5SE_16x_QA_E20S0.pth')
parser.add_argument('--e4', default='../KD/Bin/models/small16x_encoder/4SE_16x_QA_E20S0.pth')
parser.add_argument('--e3', default='../KD/Bin/models/small16x_encoder/3SE_16x_QA_E17S9000.pth')
parser.add_argument('--e2', default='../KD/Bin/models/small16x_encoder/2SE_16x_QA_E19S6000.pth')
parser.add_argument('--e1', default='../KD/Bin/models/small16x_encoder/1SE_16x_QA_E18S1000.pth')
# parser.add_argument('--d5', default='../KD/Experiments/Small16xDecoder_2/e5/weights/5SD_16x_QA_E13S9000.pth')
# parser.add_argument('--d4', default='../KD/Experiments/Small16xDecoder_2/e4/weights/4SD_16x_QA_E14S5000.pth')
# parser.add_argument('--d3', default='../KD/Experiments/Small16xDecoder_2/e3/weights/3SD_16x_QA_E25S0.pth')
# parser.add_argument('--d2', default='../KD/Experiments/Small16xDecoder_2/e2/weights/2SD_16x_QA_E26S2000.pth')
# parser.add_argument('--d1', default='../KD/Experiments/Small16xDecoder_2/e1/weights/1SD_16x_QA_E11S8000.pth')
parser.add_argument('--d5', default='../KD/Experiments/Small16xDecoder_2/e5/weights/5SD_16x_QA_E25S0.pth')
parser.add_argument('--d4', default='../KD/Experiments/Small16xDecoder_2/e4/weights/4SD_16x_QA_E25S0.pth')
parser.add_argument('--d3', default='../KD/Experiments/Small16xDecoder_2/e3/weights/3SD_16x_QA_E25S0.pth')
parser.add_argument('--d2', default='../KD/Experiments/Small16xDecoder_2/e2/weights/2SD_16x_QA_E25S0.pth')
parser.add_argument('--d1', default='../KD/Experiments/Small16xDecoder_2/e1/weights/1SD_16x_QA_E25S0.pth')

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

args = parser.parse_args()

try:
  os.makedirs(args.outf)
except OSError:
  pass
  
# Data loading code
contentPath = args.UHD_contentPath if args.UHD else args.contentPath
stylePath   = args.UHD_stylePath   if args.UHD else args.stylePath
dataset = Dataset(contentPath, stylePath, args.texturePath, args.fineSize, args.picked_content_mark, args.picked_style_mark, args.synthesis)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

logprint(args.log_mark)
log = open("samples/log_%s_%s.txt" % (args.mode, args.log_mark), "w+")
logprint(args._get_kwargs(), log)

wct = WCT(args)
@torch.no_grad()
def styleTransfer(encoder, decoder, contentImg, styleImg, csF):
    # sF  = encoder.forward_aux(styleImg)[-1]; torch.cuda.empty_cache()
    # cF  = encoder.forward_aux(contentImg)[-1]; torch.cuda.empty_cache()
    sF  = encoder(styleImg); torch.cuda.empty_cache()
    cF  = encoder(contentImg); torch.cuda.empty_cache()
    sF  = sF.data.cpu().squeeze(0)
    cF  = cF.data.cpu().squeeze(0)
    # sF  = sF.squeeze(0) # to use GPU SVD
    # cF  = cF.squeeze(0)
    csF = wct.transform(cF, sF, csF, args.alpha)
    Img = decoder(csF); torch.cuda.empty_cache()
    return Img

avgTime = 0
csF = torch.Tensor()
if(args.cuda):
    csF = csF.cuda(args.gpu)
    wct.cuda(args.gpu)

logprint("number of pairs: %s" % len(loader), log)
for i, (cImg, sImg, imname) in enumerate(loader):
    imname = imname[0]
    logprint('\n' + '*' * 30 + '#%s: Transferring "%s"' % (i, imname), log)
    if (args.cuda):
        cImg = cImg.cuda(args.gpu)
        sImg = sImg.cuda(args.gpu)
    start_time = time.time()
    
    # WCT Style Transfer
    for k in range(args.num_run):
      cImg = styleTransfer(wct.e5, wct.d5, cImg, sImg, csF)
      # cImg = styleTransfer(wct.e4, wct.d4, cImg, sImg, csF)
      # cImg = styleTransfer(wct.e3, wct.d3, cImg, sImg, csF)
      # cImg = styleTransfer(wct.e2, wct.d2, cImg, sImg, csF)
      # cImg = styleTransfer(wct.e1, wct.d1, cImg, sImg, csF)
    vutils.save_image(cImg.data.cpu().float(), os.path.join(args.outf, args.log_mark + "_" + str(args.alpha) + "_" + imname))
    
    end_time = time.time()
    logprint('Elapsed time is: %f' % (end_time - start_time), log)
    avgTime += (end_time - start_time)

logprint('Processed %d images. Averaged time is %f' % ((i+1),avgTime/(i+1)), log)
log.close()
