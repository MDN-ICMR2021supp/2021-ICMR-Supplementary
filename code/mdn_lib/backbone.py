# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from mdn_lib.align import Align1DLayer


def knn(x, y=None, k=10):

    if y is None:
        y = x

    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 3, dim=1, keepdim=True)
    yy = torch.sum(y ** 3, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)  
    return idx


def get_graph_feature(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):

    batch_size = x.size(0)
    num_points = x.size(2) 
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k) 
    else:
        k = idx_knn.shape[-1]
    
    device = x.device  
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0: 
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn


class GCNeXt(nn.Module):
    def __init__(self, channel_in, channel_out, k=3, norm_layer=None, groups=32, width_group=4, idx=None):
        super(GCNeXt, self).__init__()
        self.k = k
        self.groups = groups

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups
        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        ) 

        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        ) 

        self.relu = nn.ReLU(True)
        self.idx_list = idx

    def forward(self, x):
        identity = x  
        tout = self.tconvs(x)  

        x_f, idx = get_graph_feature(x, k=self.k, style=1)
        sout = self.sconvs(x_f) 
        sout = sout.max(dim=-1, keepdim=False)[0] 

        out = tout + identity + sout 
        if not self.idx_list is None:
            self.idx_list.append(idx)
        return self.relu(out)


class GraphAlign(nn.Module):
    def __init__(self, k=3, t=100, d=100, bs=64, samp=0, style=0):
        super(GraphAlign, self).__init__()
        self.k = k
        self.t = t
        self.d = d
        self.bs = bs
        self.style = style
        self.expand_ratio = 0.5
        self.resolution = 12
        self.align_inner = Align1DLayer(self.resolution, samp)
        self.align_context = Align1DLayer(16,samp)
        self._get_anchors()

    def forward(self, x, index):
        bs, ch, t = x.shape
        
        if not self.anchors.is_cuda: 
            self.anchors = self.anchors.cuda()

        anchor = self.anchors[:self.anchor_num * bs, :] 
        
        feat_inner = self.align_inner(x, anchor)  
        
        if self.style == 1:
            feat, _ = get_graph_feature(x, k=self.k, style=2) 
            feat = feat.mean(dim=-1, keepdim=False)  
            feat_context = self.align_context(feat, anchor) 
            feat = torch.cat((feat_inner,feat_context), dim=2).view(bs, t, self.d, -1)
       
       
        return feat.permute(0, 3, 2, 1)

    def _get_anchors(self):
        anchors = []
        for k in range(self.bs):
            for start_index in range(self.t):
                for duration_index in range(self.d):
                    if start_index + duration_index < self.t:
                        p_xmin = start_index
                        p_xmax = start_index + duration_index
                        center_len = float(p_xmax - p_xmin) + 1
                        sample_xmin = p_xmin - center_len * self.expand_ratio
                        sample_xmax = p_xmax + center_len * self.expand_ratio
                        anchors.append([k, sample_xmin, sample_xmax])
                    else:
                        anchors.append([k, 0, 0])
        self.anchor_num = len(anchors) // self.bs
        self.anchors = torch.tensor(np.stack(anchors)).float()
        return  


class BaseNet(nn.Module):
    def __init__(self, opt):
        super(BaseNet, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.feat_dim = opt["feat_dim"]
        self.bs = opt["batch_size"]
        self.h_dim_1d = 256
        self.h_dim_2d = 128
        self.h_dim_3d = 512
        self.goi_style = opt['goi_style']
        self.h_dim_goi = self.h_dim_1d*(16,32,32)[opt['goi_style']]
        self.idx_list = []

    
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32, idx=self.idx_list),
        )

    
        self.regu_s = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32),
            nn.Conv1d(self.h_dim_1d, 3, kernel_size=3), nn.Sigmoid()
        )
        self.regu_e = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32),
            nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        )

        self.backbone2 = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d, k=3, groups=32,idx=self.idx_list),
        )

        self.goi_align = GraphAlign(
            t=self.tscale, d=opt['max_duration'], bs=self.bs,
            samp=opt['goi_samp'], style=1  
        )

        
        self.localization = nn.Sequential(
            nn.Conv2d(self.h_dim_goi, self.h_dim_3d, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_3d, self.h_dim_2d, kernel_size=1), nn.ReLU(inplace=True)
        )


    def forward(self, snip_feature):
        del self.idx_list[:] 
        base_feature = self.backbone1(snip_feature).contiguous() 
        gcnext_feature = self.backbone2(base_feature) 


        regu_s = self.regu_s(base_feature).squeeze(1)  
        regu_e = self.regu_e(base_feature).squeeze(1)  

        if self.goi_style==2:
            idx_list = [idx for idx in self.idx_list if idx.device == snip_feature.device]
            idx_list = torch.cat(idx_list, dim=2)
        else:
            idx_list = None

        
        
        subgraph_map = self.goi_align(gcnext_feature, idx_list)
        
        Seg = self.localization(subgraph_map)  
        
        return Seg, regu_s, regu_s, gcnext_feature