import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.pvt_v2 import pvt_v2_b4


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

    def forward(self, x1, x2, x3, x4):
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

        return x


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- PVT Backbone ----
        self.shared_encoder = pvt_v2_b4()
        pretrained_dict = torch.load('/home/user2/xiaoyao/pre_train_pth/pvt_v2_b4.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.shared_encoder.state_dict()}
        self.shared_encoder.load_state_dict(pretrained_dict)

        # 4stage NCD
        self.NCD = NeighborConnectionDecoder(channel)

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
    def forward(self, x):
        # Feature Extraction

        x1, x2, x3, x4 = self.shared_encoder(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4, x3, x2, x1)
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