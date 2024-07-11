import torch
import torch.nn as nn

class NestedUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NestedUNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        def CBR2d(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.conv0_0 = CBR2d(in_channels, 64)
        self.conv1_0 = CBR2d(64, 128)
        self.conv2_0 = CBR2d(128, 256)
        self.conv3_0 = CBR2d(256, 512)
        self.conv4_0 = CBR2d(512, 1024)

        self.conv0_1 = CBR2d(64+128, 64)
        self.conv1_1 = CBR2d(128+256, 128)
        self.conv2_1 = CBR2d(256+512, 256)
        self.conv3_1 = CBR2d(512+1024, 512)

        self.conv0_2 = CBR2d(64*2+128, 64)
        self.conv1_2 = CBR2d(128*2+256, 128)
        self.conv2_2 = CBR2d(256*2+512, 256)

        self.conv0_3 = CBR2d(64*3+128, 64)
        self.conv1_3 = CBR2d(128*3+256, 128)

        self.conv0_4 = CBR2d(64*4+128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3)], 1))

        output = self.final(x0_4)
        return output
