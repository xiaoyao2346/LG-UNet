import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvt_v2 import pvt_v2_b4
from lib.LocalUnet import LocalUnet


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            BasicConv2d(in_channel, out_channel, 1)
        )
        self.conv_cat = BasicConv2d(5*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        size = x.shape[-2:]
        x4 = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        # x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4), 1))
        # xm = self.conv_cat(torch.cat((x-x0, x-x1, x-x2, x-x3, x-x4), 1))
        # x = self.relu(x_cat + xm + self.conv_res(x))
        x = self.relu(x*self.conv_res(x0 + x1 + x2 + x3 + x4))
        return x


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

        return x1, x2_o, x3_o, x4_o, x   # # high-->low


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- PVT Backbone ----
        self.shared_encoder = pvt_v2_b4()
        pretrained_dict = torch.load('/root/autodl-tmp/pre_train_pth/pvt_v2_b4.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.shared_encoder.state_dict()}
        self.shared_encoder.load_state_dict(pretrained_dict)


        self.conv1 = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=1),nn.BatchNorm2d(channel),nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, channel, kernel_size=1),nn.BatchNorm2d(channel),nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(320, channel, kernel_size=1),nn.BatchNorm2d(channel),nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, channel, kernel_size=1),nn.BatchNorm2d(channel),nn.ReLU(True)
        )

        self.rfb1 = RFB_modified(channel, channel)
        self.rfb2 = RFB_modified(channel, channel)
        self.rfb3 = RFB_modified(channel, channel)
        self.rfb4 = RFB_modified(channel, channel)
        

        self.lunet1 = LocalUnet(channel)
        self.lunet2 = LocalUnet(channel)
        self.lunet3 = LocalUnet(channel)
        self.lunet4 = LocalUnet(channel)

        # 4stage NCD
        self.NCD = NeighborConnectionDecoder(channel)


    def forward(self, x):
        # Feature Extraction

        x1, x2, x3, x4 = self.shared_encoder(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        x1 = self.rfb1(x1) + x1
        x2 = self.rfb2(x2) 
        x3 = self.rfb3(x3) 
        x4 = self.rfb4(x4) + x4


        u3_1, u3_2 = self.lunet3(x4, x3)
        u2_1, u2_2 = self.lunet2(x3 + u3_2, x2)
        u1_1, u1_2 = self.lunet1(x2 + u2_2, x1)
        
        x4 = u3_1
        x3 = u2_1
        x2 = u1_1
        x1 = u1_2

        x4, x1 = self.lunet4(x4, x1)
        
        # Neighbourhood Connected Decoder
        x4, x3, x2, x1, S_g = self.NCD(x4, x3, x2, x1)

        S_g_pred = F.interpolate(S_g, scale_factor=4, mode='bilinear')    # Sup-1 (bs, 1, 88, 88) -> (bs, 1, 352, 352)
        return S_g_pred


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)