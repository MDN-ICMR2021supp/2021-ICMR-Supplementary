# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mdn_lib.basenet import BaseNet

class MDU(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, dila=1):
        super(MDU, self).__init__()

        self.kernel_size = 3
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(in_xC, out_C, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.unfold = nn.Unfold(kernel_size=3, dilation=dila, padding=1, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])

        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)

        return self.fuse(result) 
class MDM(nn.Module):

    def __init__(self, in_xC, in_yC, out_C, kernel_size=3):

        super(MDM, self).__init__()
        self.kernel_size = kernel_size
        self.mid_c = out_C // 1
        self.down_input = nn.Conv2d(in_xC, self.mid_c, 1)
        self.branch_1 = DepthDC(self.mid_c, in_yC, self.mid_c, dila=1)
        self.branch_3 = DepthDC(self.mid_c, in_yC, self.mid_c, dila=3)
        self.branch_5 = DepthDC(self.mid_c, in_yC, self.mid_c, dila=5)
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * self.mid_c, out_C, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        fusion =  x + y  

        result_1 = self.branch_1(x, fusion)
        result_3 = self.branch_3(x, fusion)
        result_5 = self.branch_5(x, fusion)

        return self.fuse(torch.cat((x, result_1, result_3, result_5), dim=1)) + x

class MDN(nn.Module):
    def __init__(self, opt):
        super(MDN, self).__init__()

        self.base = BaseNet(opt)
        
        self.mdm1 = DDPM(128, 128, 128)
        self.mdm2 = DDPM(128, 128, 128)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True)
        )

        self.avg = nn.AdaptiveAvgPool1d(1)
       
        self.pre = nn.Sequential(
            nn.Conv2d(self.h_dim_2d, 2, kernel_size=1), nn.Sigmoid()
        )

    
    def forward(self, x):

        seg, start, end, enrich = self.base(x)
        
        avg_feature = self.avg(enrich).unsqueeze(-1).repeat(1, 1, 100, 100)
        avg_feature = self.reduce(avg_feature)

        l_feature = self.ddpm1(seg, avg_feature)
        
        iou_map = self.pre(l_feature)

        return iou_map, start, start
