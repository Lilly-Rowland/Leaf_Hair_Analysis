#To be deleted

# import torch
# import torch.nn as nn
# import torch
# torch.autograd.set_detect_anomaly(True)


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),  # Removed inplace=True
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()  # Removed inplace=True
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet, self).__init__()
#         self.dconv_down1 = DoubleConv(in_channels, 64)
#         self.dconv_down2 = DoubleConv(64, 128)
#         self.dconv_down3 = DoubleConv(128, 256)
#         self.dconv_down4 = DoubleConv(256, 512)

#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.dconv_up3 = DoubleConv(256 + 512, 256)
#         self.dconv_up2 = DoubleConv(128 + 256, 128)
#         self.dconv_up1 = DoubleConv(128 + 64, 64)

#         self.conv_last = nn.Conv2d(64, out_channels, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Downward path
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)

#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)

#         x = self.dconv_down4(x)

#         # Upward path
#         x = self.upsample(x)
#         x = torch.cat([x, conv3], dim=1)
#         x = self.dconv_up3(x)

#         x = self.upsample(x)
#         x = torch.cat([x, conv2], dim=1)
#         x = self.dconv_up2(x)

#         x = self.upsample(x)
#         x = torch.cat([x, conv1], dim=1)
#         x = self.dconv_up1(x)

#         out = self.conv_last(x)
#         out = self.sigmoid(out)
#         return out
