import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        # Using depthwise separable convolution to save VRAM
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in, bias=False),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out, bias=False),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionUNetDehaze(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, base_ch=32):
        super(AttentionUNetDehaze, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=base_ch)
        self.Conv2 = ConvBlock(ch_in=base_ch, ch_out=base_ch*2)
        self.Conv3 = ConvBlock(ch_in=base_ch*2, ch_out=base_ch*4)
        self.Conv4 = ConvBlock(ch_in=base_ch*4, ch_out=base_ch*8)

        self.Up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att4 = AttentionBlock(F_g=base_ch*8, F_l=base_ch*4, F_int=base_ch*4)
        self.Up_conv4 = ConvBlock(ch_in=base_ch*8, ch_out=base_ch*4)

        self.Up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att3 = AttentionBlock(F_g=base_ch*4, F_l=base_ch*2, F_int=base_ch*2)
        self.Up_conv3 = ConvBlock(ch_in=base_ch*4, ch_out=base_ch*2)

        self.Up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Att2 = AttentionBlock(F_g=base_ch*2, F_l=base_ch, F_int=base_ch)
        self.Up_conv2 = ConvBlock(ch_in=base_ch*2, ch_out=base_ch)

        # Head for Transmission Map
        self.Conv_1x1 = nn.Conv2d(base_ch, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        # Head for Atmospheric Light (Global)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_light = nn.Sequential(
            nn.Linear(base_ch*8, base_ch*2),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch*2, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        # Global Atmospheric Light Prediction from bottleneck
        a_pool = self.avgpool(x4).view(x4.size(0), -1)
        pred_a = self.fc_light(a_pool)

        # decoding + concat path
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        pred_t = self.sigmoid(self.Conv_1x1(d2))

        return pred_t, pred_a
