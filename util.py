from __future__ import division
import torch
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import numpy as np
import argparse
import time
import os
from PIL import Image
from modelsNIPS import decoder1,decoder2,decoder3,decoder4,decoder5
from modelsNIPS import encoder1,encoder2,encoder3,encoder4,encoder5
import torch.nn as nn
from model import Encoder1, Encoder2, Encoder3, Encoder4, Encoder5
from model import Decoder1, Decoder2, Decoder3, Decoder4, Decoder5
# 10x model
from model import SmallEncoder3_10x, SmallEncoder4_10x, SmallEncoder5_10x
from model import SmallDecoder3_10x, SmallDecoder4_10x, SmallDecoder5_10x
# 16x model
from model import SmallEncoder2_16x_plus, SmallEncoder3_16x_plus, SmallEncoder4_16x_plus, SmallEncoder5_16x_plus
from model import SmallDecoder2_16x,      SmallDecoder3_16x,       SmallDecoder4_16x,     SmallDecoder5_16x
       
class WCT(nn.Module):
    def __init__(self, args):
        super(WCT, self).__init__()
        self.gpu = args.gpu
        # load pre-trained network
        if args.mode == None:
          vgg1 = load_lua(args.vgg1)
          decoder1_torch = load_lua(args.decoder1)
          vgg2 = load_lua(args.vgg2)
          decoder2_torch = load_lua(args.decoder2)
          vgg3 = load_lua(args.vgg3)
          decoder3_torch = load_lua(args.decoder3)
          vgg4 = load_lua(args.vgg4)
          decoder4_torch = load_lua(args.decoder4)
          vgg5 = load_lua(args.vgg5)
          decoder5_torch = load_lua(args.decoder5)
          self.e1 = encoder1(vgg1)
          self.d1 = decoder1(decoder1_torch)
          self.e2 = encoder2(vgg2)
          self.d2 = decoder2(decoder2_torch)
          self.e3 = encoder3(vgg3)
          self.d3 = decoder3(decoder3_torch)
          self.e4 = encoder4(vgg4)
          self.d4 = decoder4(decoder4_torch)
          self.e5 = encoder5(vgg5)
          self.d5 = decoder5(decoder5_torch)
        else:
          if "10x" in args.mode:
            self.e5 = SmallEncoder5_10x(args.e5)
            self.d5 = SmallDecoder5_10x(args.d5)
            self.e4 = SmallEncoder4_10x(args.e4)
            self.d4 = SmallDecoder4_10x(args.d4)
            self.e3 = SmallEncoder3_10x(args.e3)
            self.d3 = SmallDecoder3_10x(args.d3)
            self.e2 = Encoder2(args.e2)
            self.d2 = Decoder2(args.d2)
          elif "16x" in args.mode:
            self.e5 = SmallEncoder5_16x_plus(args.e5)
            self.d5 = SmallDecoder5_16x(args.d5)
            self.e4 = SmallEncoder4_16x_plus(args.e4)
            self.d4 = SmallDecoder4_16x(args.d4)
            self.e3 = SmallEncoder3_16x_plus(args.e3)
            self.d3 = SmallDecoder3_16x(args.d3)
            self.e2 = SmallEncoder2_16x_plus(args.e2)
            self.d2 = SmallDecoder2_16x(args.d2)
          else:
            print("mode wrong")
            exit(1)
          self.e1 = Encoder1(args.e1)
          self.d1 = Decoder1(args.d1)
    
    def whiten_and_color(self, cF, sF):
        print("\n" + "-" * 10 + " whiten_and_color")
        cFSize = cF.size()
        c_mean = torch.mean(cF, 1) # c * (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean
        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).double().cuda(self.gpu)
        c_u, c_e, c_v = torch.svd(contentConv, some=False);

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break
        print("k_c = %s" % k_c)
        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False);
        
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break
        print("k_s = %s" % k_s)
        
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
        step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
        whiten_cF = torch.mm(step2, cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s], torch.diag(s_d)), (s_v[:,0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        torch.cuda.empty_cache()
        return targetFeature
    
    def transform(self,cF,sF,csF,alpha):
        cF = cF.double()
        sF = sF.double()
        C, W,  H  = cF.size(0), cF.size(1), cF.size(2)
        _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cFView = cF.view(C,-1)
        sFView = sF.view(C,-1)
        targetFeature = self.whiten_and_color(cFView, sFView)
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        csF.data.resize_(ccsF.size()).copy_(ccsF)
        torch.cuda.empty_cache()
        return csF
