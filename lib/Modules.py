# -*- coding: utf-8 -*-
import torch.nn as nn

affine_par = True
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import time
import einops
from torch.nn.parameter import Parameter
from thop import profile
from lib.Slot import SoftPositionEmbed,spatial_broadcast,spatial_flatten,spatial_broadcast2,unstack_and_split
# from lib.IAM import IAM
# import numpy as np #借助numpy模块的set_printoptions()函数，将打印上限设置为无限即可
# np.set_printoptions(threshold=np.inf)
# Low-level feature extraction module
class LFE(nn.Module):
    def __init__(self):
        super(LFE, self).__init__()
        self.conv1 = ConvBR(3, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBR(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBR(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBR(64, 1, kernel_size=1, stride=1, padding=0)

        self.E1 = ETM(128, 64)
        self.E2 = ETM(64, 64)

        self.dwt_hh = DWT_hh()
        
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.E1(feat)
        xg = self.E2(feat)
        xg = self.dwt_hh(feat) 
        pg = self.conv_out(xg)
        return xg, pg 
    
# Discrete Wavelet Transformation
class DWT_hh(nn.Module):
    def __init__(self):
        super(DWT_hh, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        # ll = x1 + x2 + x3 + x4
        # lh = -x1 + x2 - x3 + x4
        # hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return hh

# high-level feature enhancement module 
class HFE(nn.Module):
    def __init__(self, dilation_series=[3, 5, 7], padding_series=[3, 5, 7], depth=128):
        super(HFE, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBR(512, depth, kernel_size=1, stride=1)
        )
        self.branch0 = ConvBR(512, depth, kernel_size=1, stride=1)
        self.branch1 = ConvBR(512, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                   dilation=dilation_series[0])
        self.branch2 = ConvBR(512, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                   dilation=dilation_series[1])
        self.branch3 = ConvBR(512, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                   dilation=dilation_series[2])
        self.head = nn.Sequential(
            ConvBR(depth * 5, 256, kernel_size=3, padding=1),
            PAM(256)
        )
        self.out = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=True),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 32, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3], 1)
        out = self.head(out)
        out = self.out(out)
        return out


# feature enhancement module
class FeatureFusionModule(nn.Module):
    def __init__(self, channel=64, M=[8, 8, 8], N=[4, 8, 16]):
        super(FeatureFusionModule, self).__init__()

        self.T1 = ETM(64, channel)
        self.T2 = ETM(128, channel)
        self.T3 = ETM(320, channel)
        self.ODE = ODE(channel,num_slots_N=4,num_slots_M=2,iters=3,resolutions=24)

        # transmit xg cues into x1 x2 x3
        self.interFA1 = InterFA(channel)
        self.interFA2 = InterFA(channel)
        self.interFA3 = InterFA(channel)

        # transmit x4 cues into x1 x2 x3
        self.M = M

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.sgs3 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs4 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs5 = SoftGroupingStrategy(channel + 32, channel, N=N)

    def forward(self, xg, x1, x2, x3, gp4):
        f1 = self.T1(x1)  # 64,96,96
        f2 = self.T2(x2)  # 64,48,48
        f3 = self.T3(x3)  # 64,24,24
        f3 = self.ODE(f3) # 64,24,24

        # transmit xg cues into x1 x2 x3
        temp1, fg1 = self.interFA1(f1, xg)
        temp2, fg2 = self.interFA2(f2, temp1)
        temp3, fg3 = self.interFA3(f3, temp2)

        # transmit the x4 cues into the x1 x2 x3
        q1 = self.gradient_induced_feature_grouping(fg1, self.upsample8(gp4), M=self.M[0])
        q2 = self.gradient_induced_feature_grouping(fg2, self.upsample4(gp4), M=self.M[1])
        q3 = self.gradient_induced_feature_grouping(fg3, self.upsample2(gp4), M=self.M[2])
        # attention residual learning
        fgp1 = fg1 + self.sgs3(q1)
        fgp2 = fg2 + self.sgs4(q2)
        fgp3 = fg3 + self.sgs5(q3)

        return fgp1, fgp2, fgp3

    def gradient_induced_feature_grouping(self, xr, xg, M):
        if not M in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Invalid Group Number!: must be one of [1, 2, 4, 8, 16, 32]")

        if M == 1:
            return torch.cat((xr, xg), 1)

        xr_g = torch.chunk(xr, M, dim=1)
        xg_g = torch.chunk(xg, M, dim=1)
        foo = list()
        for i in range(M):
            foo.extend([xr_g[i], xg_g[i]])

        return torch.cat(foo, 1)

# NCD
# class NeighborConnectionDecoder(nn.Module):
#     def __init__(self, channel):
#         super(NeighborConnectionDecoder, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
#         self.conv_upsample2 = ConvBR(channel, channel, 3, padding=1)
#         self.conv_upsample3 = ConvBR(channel, channel, 3, padding=1)
#         self.conv_upsample4 = ConvBR(channel, channel, 3, padding=1)
#         self.conv_upsample5 = ConvBR(2 * channel, 2 * channel, 3, padding=1)

#         self.conv_concat2 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
#         self.conv_concat3 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
#         self.conv4 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
#         self.conv5 = nn.Conv2d(3 * channel, 1, 1)

#     def forward(self, zt5, zt4, zt3):
#         zt5_1 = zt5
#         zt4_1 = self.conv_upsample1(self.upsample(zt5)) * zt4
#         zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) * zt3

#         zt4_2 = torch.cat((zt4_1, self.conv_upsample4(self.upsample(zt5_1))), 1)
#         zt4_2 = self.conv_concat2(zt4_2)

#         zt3_2 = torch.cat((zt3_1, self.conv_upsample5(self.upsample(zt4_2))), 1)
#         zt3_2 = self.conv_concat3(zt3_2)

#         pc = self.conv4(zt3_2)
#         pc = self.conv5(pc)

#         return pc
    

# 4 input NCD
class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):     # high-->low
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2)) * x3
        x3_2 = self.conv_upsample3(self.upsample(x2_1)) * x3_1
        x4_1 = self.conv_upsample4(self.upsample(x3)) * x4
        x4_2 = self.conv_upsample5(self.upsample(x3_1)) * x4_1
        x4_3 = self.conv_upsample6(self.upsample(x3_2)) * x4_2

        x2_o = torch.cat((x2_1, self.conv_upsample7(self.upsample(x1_1))), 1)
        x2_o = self.conv_concat2(x2_o)

        x3_o = torch.cat((x3_2, self.conv_upsample8(self.upsample(x2_o))), 1)
        x3_o = self.conv_concat3(x3_o)

        x4_o = torch.cat((x4_3, self.conv_upsample9(self.upsample(x3_o))), 1)
        x4_o = self.conv_concat4(x4_o)

        x = self.conv4(x4_o)
        x = self.conv5(x)

        return x1, x2_o, x3_o, x4_o, x


#  texture modules
class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



"""
    position attention module
"""
class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out



"""
    enhance texture module
"""
class ETM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ETM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = ConvBR(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            ConvBR(in_channels, out_channels, 1),
            ConvBR(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            ConvBR(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            ConvBR(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            ConvBR(in_channels, out_channels, 1),
            ConvBR(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            ConvBR(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            ConvBR(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            ConvBR(in_channels, out_channels, 1),
            ConvBR(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            ConvBR(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            ConvBR(out_channels, out_channels, 3, padding=7, dilation=7)
        )
        self.conv_cat = ConvBR(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = ConvBR(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


"""
InterFA+ODE+slot
"""

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class InterFA(nn.Module):
    def __init__(self, in_channels):
        super(InterFA, self).__init__()
        self.conv3x3 = ConvBR(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(in_channels)
        self.conv1x1 = ConvBR(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f1, f2):
        f2_up = F.interpolate(f2, size=f1.size()[2:], mode='bilinear', align_corners=True)
        cat = torch.cat([f1, f2_up], dim=1)
        f = self.conv3x3(cat)
        f = self.cbam(f)
        cat2 = torch.cat([f, f1], dim=1)
        out = self.conv1x1(cat2)
        return f, out



class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # Parameters for Gaussian init (shared by all slots).
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # random slots initialization,
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        # Multiple rounds of attention.
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttentionModule(nn.Module):
    def __init__(self, encoder_dims, resolution, num_slots, iters):
        super(SlotAttentionModule, self).__init__()
        self.resolution = resolution
        self.encoder_pos = SoftPositionEmbed(encoder_dims, ((int(resolution), int(resolution))))
        self.layer_norm = nn.LayerNorm(encoder_dims)
        self.mlp = nn.Sequential(
                nn.Linear(encoder_dims, encoder_dims),
                nn.ReLU(inplace=True),
                nn.Linear(encoder_dims, encoder_dims)
            )
        self.slot_attention = SlotAttention(iters=iters,
                                            num_slots=num_slots,
                                            encoder_dims=encoder_dims,
                                            hidden_dim=encoder_dims)
        self.decoder_pos = SoftPositionEmbed(encoder_dims, (int(resolution), int(resolution)))
        self.conv = nn.Conv2d(encoder_dims*num_slots, encoder_dims, kernel_size=1, padding=0, stride=1)
        
    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.encoder_pos(x) 
        x = spatial_flatten(x)
        x = self.mlp(self.layer_norm(x))
        slots = self.slot_attention(x)
        x = spatial_broadcast(slots, (int(self.resolution), int(self.resolution)))
        x = self.decoder_pos(x)
        x = einops.rearrange(x, 'b n h w c -> b (n c) h w')
        out = self.conv(x)
        return out


class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels*2,in_channels,kernel_size =1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels,1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ODE(nn.Module):
    def __init__(self, in_channels,num_slots_N,num_slots_M,iters,resolutions):
        super(ODE, self).__init__()
        self.SAn = SlotAttentionModule(in_channels, resolutions, num_slots_N, iters)
        self.SAm = SlotAttentionModule(in_channels, resolutions, num_slots_M, iters)
        self.getalpha = getAlpha(in_channels)

    def forward(self, feature_map):
        f1 = self.SAn(feature_map)
        f2 = self.SAm(f1+feature_map)
        alpha = self.getalpha(torch.cat([f1,f2],dim=1))
        out = feature_map+f1*alpha+f2*(1-alpha)
        return  out


# git
class SoftGroupingStrategy(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(SoftGroupingStrategy, self).__init__()
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)

class GradientInducedTransition(nn.Module):
    def __init__(self, channel=64, M=[8, 8, 8], N=[4, 8, 16]):
        super(GradientInducedTransition, self).__init__()
        self.T2 = ETM(channel * 2, channel)
        self.T3 = ETM(channel * 5, channel)
        self.T4 = ETM(channel * 8, channel)
        self.M = M

        self.downsample2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)

        self.sgs3 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs4 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs5 = SoftGroupingStrategy(channel + 32, channel, N=N)

    def forward(self, xr3, xr4, xr5, xg):
        xr3 = self.T2(xr3) 
        xr4 = self.T3(xr4) 
        xr5 = self.T4(xr5)
        # transmit the gradient cues into the context embeddings
        q3 = self.gradient_induced_feature_grouping(xr3, xg, M=self.M[0])
        q4 = self.gradient_induced_feature_grouping(xr4, self.downsample2(xg), M=self.M[1])
        q5 = self.gradient_induced_feature_grouping(xr5, self.downsample4(xg), M=self.M[2])

        # attention residual learning
        zt3 = xr3 + self.sgs3(q3)
        zt4 = xr4 + self.sgs4(q4)
        zt5 = xr5 + self.sgs5(q5)

        return zt3, zt4, zt5

    def gradient_induced_feature_grouping(self, xr, xg, M):
        if not M in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Invalid Group Number!: must be one of [1, 2, 4, 8, 16, 32]")

        if M == 1:
            return torch.cat((xr, xg), 1)

        xr_g = torch.chunk(xr, M, dim=1)
        xg_g = torch.chunk(xg, M, dim=1)
        foo = list()
        for i in range(M):
            foo.extend([xr_g[i], xg_g[i]])

        return torch.cat(foo, 1)


# # decoder
# class PPM(nn.ModuleList):
#     def __init__(self, pool_sizes, in_channels, out_channels):
#         super(PPM, self).__init__()
#         self.pool_sizes = pool_sizes
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         for pool_size in pool_sizes:
#             self.append(
#                 nn.Sequential(
#                     nn.AdaptiveMaxPool2d(pool_size),
#                     nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
#                 )
#             )     
            
#     def forward(self, x):
#         out_puts = []
#         for ppm in self:
#             ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
#             out_puts.append(ppm_out)
#         return out_puts
 
# decoder  改
# class PPM(nn.ModuleList):
#     def __init__(self, pool_sizes, in_channels, out_channels):
#         super(PPM, self).__init__()
#         self.pool_sizes = pool_sizes
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.branch0 = nn.Sequential(
#             nn.AdaptiveMaxPool2d(pool_sizes[0]),
#             nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
#         )
#         self.branch1 = nn.Sequential(
#             nn.AdaptiveMaxPool2d(pool_sizes[1]),
#             nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
#             nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
#         )
#         self.branch2 = nn.Sequential(
#             nn.AdaptiveMaxPool2d(pool_sizes[2]),
#             nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
#             nn.Conv2d(self.out_channels, self.out_channels, kernel_size=5, padding=2)
#         )
#         self.branch3 = nn.Sequential(
#             nn.AdaptiveMaxPool2d(pool_sizes[3]),
#             nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
#             nn.Conv2d(self.out_channels, self.out_channels, kernel_size=7, padding=3)
#         )
            
#     def forward(self, x):
#         out_puts = []
#         ppm_out0 = nn.functional.interpolate(self.branch0(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
#         out_puts.append(ppm_out0)
#         ppm_out1 = nn.functional.interpolate(self.branch1(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
#         out_puts.append(ppm_out1)
#         ppm_out2 = nn.functional.interpolate(self.branch2(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
#         out_puts.append(ppm_out2)
#         ppm_out3 = nn.functional.interpolate(self.branch3(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
#         out_puts.append(ppm_out3)
#         return out_puts


# decoder 加 etm
class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    ETM(in_channels, out_channels)
                    #IAM(in_channels, out_channels)
                )
            )     
            
    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts
    
class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes = [1, 2, 3, 6]):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
#             nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=1),
            nn.Conv2d(len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.psp_modules(x)
#         out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out
 

"""
    enhance texture module
"""
# class ETM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ETM, self).__init__()
#         self.relu = nn.ReLU(True)
#         self.branch0 = ConvBR(in_channels, out_channels, 1)
#         self.branch1 = nn.Sequential(
#             ConvBR(in_channels, out_channels, 1),
#             ConvBR(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
#             ConvBR(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
#             ConvBR(out_channels, out_channels, 3, padding=3, dilation=3)
#         )
#         self.branch2 = nn.Sequential(
#             ConvBR(in_channels, out_channels, 1),
#             ConvBR(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
#             ConvBR(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
#             ConvBR(out_channels, out_channels, 3, padding=5, dilation=5)
#         )
#         self.branch3 = nn.Sequential(
#             ConvBR(in_channels, out_channels, 1),
#             ConvBR(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
#             ConvBR(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
#             ConvBR(out_channels, out_channels, 3, padding=7, dilation=7)
#         )
#         self.conv_cat = ConvBR(4 * out_channels, out_channels, 3, padding=1)
#         self.conv_res = ConvBR(in_channels, out_channels, 1)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

#         x = self.relu(x_cat + self.conv_res(x))
#         return x


 
class FPNHEAD(nn.Module):
    def __init__(self, channels=512, out_channels=64):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)
        
        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )    
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ) 
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)
 
    def forward(self, input_fpn):
        # b, 512, 7, 7
        x1 = self.PPMHead(input_fpn[-1])
 
        x = nn.functional.interpolate(x1, size=(x1.size(2)*2, x1.size(3)*2),mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)
        
        x = nn.functional.interpolate(x2, size=(x2.size(2)*2, x2.size(3)*2),mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)  
 
        x = nn.functional.interpolate(x3, size=(x3.size(2)*2, x3.size(3)*2),mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)
 
        x1 = F.interpolate(x1, x4.size()[-2:],mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:],mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:],mode='bilinear', align_corners=True)
 
        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))
        
        return x





# test
if __name__ == '__main__':
    # f3 = torch.randn(2, 64, 96, 96).cuda()
    # net = SlotAttentionModule2(encoder_dims=64, resolution=96, num_slots =4, iters =3).cuda()
    # y = net(f3)
    # print(y[0].shape)
    # print(y[1].shape)
    f0 = torch.randn(2, 256, 96, 96).cuda()
    # f1 = torch.randn(2, 512, 48, 48).cuda()
    # f2 = torch.randn(2, 1024, 24, 24).cuda()
    # f3 = torch.randn(2, 2048, 12, 12).cuda()
    # gcm4 = GCM(256, 64).cuda()
    # f1, f2, f3, f4 = gcm4(f0, f1, f2, f3)


    E1 = ETM(256, 256)
    print(E1(f0).shape)
    # print(f1.shape)
    # print(f2.shape)
    # print(f3.shape)
    # print(f4.shape)
    # f0 = torch.randn(2, 1, 12, 12).cuda()
    # pict = torch.randn(2, 3, 384, 384).cuda()
    # rem = REM_decoder(96).cuda()
    # prior_cam, p4_s_out, p3_s_out, p2_s_out, p1_s_out, p4_e_out, p3_e_out, p2_e_out, p1_e_out = rem([f1, f2, f3, f4],f0,pict)

    # print(prior_cam.shape)
    # print(p4_s_out.shape)
    # print(p3_s_out.shape)
    # print(p2_s_out.shape)
    # print(p1_s_out.shape)
    # print(p4_e_out.shape)
    # print(p3_e_out.shape)
    # print(p2_e_out.shape)
    # print(p1_e_out.shape)


    # f0 = torch.randn(2, 1, 12, 12).cuda()
    # f4 = torch.randn(2, 96, 12, 12).cuda()
    # f3 = torch.randn(2, 96, 24, 24).cuda()
    # f2 = torch.randn(2, 96, 48, 48).cuda()
    # f1 = torch.randn(2, 96, 96, 96).cuda()
    # pict = torch.randn(2, 3, 384, 384).cuda()
    # x = [f1, f2, f3, f4]
    # sr = SR(96, 96).cuda()
    # f4, f3, f2, f1, bound_f1 = sr(x, f0, pict)
    # print(f4.shape)
    # print(f3.shape)
    # print(f2.shape)
    # print(f1.shape)
    # print(bound_f1.shape)
    # f1 = torch.randn(2, 1, 12, 12).cuda()
    # ll = torch.randn(2, 64, 96, 96).cuda()
    # lh = torch.randn(2, 64, 48, 48).cuda()
    # hl = torch.randn(2, 64, 24, 24).cuda()
    # hh = torch.randn(2, 64, 12, 12).cuda()
    # pict = torch.randn(2, 3, 384, 384).cuda()
    # x = [ll, lh, hl, hh]
    # rem = REM12(64,64).cuda()
    # f4,f3,f2,f1,bound_f4,bound_f3,bound_f2,bound_f1 = rem(x, f1, pict)
    # print(f4.shape)
    # print(f3.shape)
    # print(f2.shape)
    # print(f1.shape)
    # print(bound_f4.shape)
    # print(bound_f3.shape)
    # print(bound_f2.shape)
    # print(bound_f1.shape)

    # input = torch.randn(2, 2048, 16, 16)
    # model = GPM(depth=128)
    # total = sum([param.nelement() for param in model.parameters()])
    # print('Number of parameter: %.2fM' % (total/1e6))
    # flops, params = profile(model, inputs=(input, ))
    # print('flops:{}'.format(flops*2))
    # print('params:{}'.format(params))
